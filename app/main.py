import asyncio
import hashlib
import hmac
import json
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiofiles
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse

from app.archive import (
    ArchiveError,
    assemble_archive,
    safe_extract_zip,
    validate_chunk_index,
)
from app.config import Settings
from app.database import RelayDb
from app.kaggle_adapter import KaggleAdapter
from app.schemas import ChunkResponse, CreateJobRequest, HealthResponse, JobProgressRequest, JobResponse
from app.security import redact_secrets
from app.worker import has_ready_dataset_cache, process_job, validate_kernel_payload, validate_payloads

VERSION = "0.1.0"


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> RelayDb:
    return request.app.state.db


async def require_auth(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def job_response(db: RelayDb, job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    dataset_cache_hit = has_ready_dataset_cache(db, job["dataset_ref"], job["payload_hash"])
    return JobResponse(
        **RelayDb.to_response(
            {
                **job,
                "callback_enabled": bool(job.get("callback_token_sha256")),
                "dataset_cache_hit": dataset_cache_hit,
                "dataset_upload_required": not dataset_cache_hit,
            },
            db.accepted_chunks(job_id),
            db.recent_logs(job_id),
        )
    )


def bearer_token(authorization: str) -> str:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return ""
    return authorization[len(prefix):].strip()


def authorize_job_callback(job: dict, authorization: str, settings: Settings) -> bool:
    token = bearer_token(authorization)
    if not token:
        return False
    if hmac.compare_digest(token, settings.api_token):
        return True
    expected_hash = (job.get("callback_token_sha256") or "").strip().lower()
    if not expected_hash:
        return False
    actual_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return hmac.compare_digest(actual_hash, expected_hash)


def progress_from_callback(job: dict, payload: JobProgressRequest) -> float:
    remote_progress = payload.remote_progress
    if remote_progress is None and payload.epoch is not None and payload.epochs:
        remote_progress = min(100.0, max(0.0, payload.epoch / payload.epochs * 100))
    if remote_progress is None:
        return float(job["progress"])
    callback_progress = min(80.0, 60.0 + remote_progress / 100.0 * 20.0)
    return max(float(job["progress"]), callback_progress)


def callback_log_message(data: dict) -> str:
    message = str(data.get("message") or data.get("log") or "").strip()
    if message:
        return message
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def apply_progress_callback(db: RelayDb, job: dict, payload: JobProgressRequest) -> None:
    data = payload.model_dump()
    clean_message = redact_secrets(callback_log_message(data))[-8000:]
    if clean_message:
        db.append_log(job["job_id"], clean_message)
    updates = {
        "kernel_status": json.dumps(data, ensure_ascii=False, sort_keys=True),
        "kaggle_output": clean_message[-4000:],
        "progress": progress_from_callback(job, payload),
    }
    if job["status"] not in {"complete", "failed"}:
        updates["status"] = "waiting_kernel"
    db.update_job(job["job_id"], **updates)


async def worker_loop(app: FastAPI) -> None:
    while True:
        job_id = await app.state.queue.get()
        try:
            await asyncio.to_thread(process_job, app.state.settings, app.state.db, job_id)
        finally:
            app.state.queue.task_done()


def cleanup_expired(settings: Settings, db: RelayDb) -> None:
    cutoff = time.time() - settings.retention_hours * 60 * 60
    for job_id in db.completed_before(cutoff):
        shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
        shutil.rmtree(settings.artifacts_dir / job_id, ignore_errors=True)
        db.update_job(
            job_id,
            artifact_path="",
            kaggle_output="expired by relay retention cleanup",
        )
        db.append_log(job_id, "expired by relay retention cleanup")


async def cleanup_loop(app: FastAPI) -> None:
    while True:
        await asyncio.to_thread(cleanup_expired, app.state.settings, app.state.db)
        await asyncio.sleep(60 * 60)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_task = asyncio.create_task(worker_loop(app))
        app.state.cleanup_task = asyncio.create_task(cleanup_loop(app))
        try:
            yield
        finally:
            app.state.worker_task.cancel()
            app.state.cleanup_task.cancel()
            try:
                await app.state.worker_task
            except asyncio.CancelledError:
                pass
            try:
                await app.state.cleanup_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Kaggle Relay", version=VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = RelayDb(settings.db_path)
    app.state.queue = asyncio.Queue()

    @app.get("/v1/health", response_model=HealthResponse, dependencies=[Depends(require_auth)])
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        usage = shutil.disk_usage(settings.storage_dir)
        return HealthResponse(
            status="ok",
            version=VERSION,
            storage_dir=str(settings.storage_dir),
            free_bytes=usage.free,
        )

    @app.get("/v1/kaggle/account", dependencies=[Depends(require_auth)])
    def kaggle_account(settings: Settings = Depends(get_settings)) -> dict:
        adapter = KaggleAdapter(settings, lambda _message: None)
        return adapter.account()

    @app.post("/v1/jobs", response_model=JobResponse, dependencies=[Depends(require_auth)])
    def create_job(payload: CreateJobRequest, settings: Settings = Depends(get_settings), db: RelayDb = Depends(get_db)) -> JobResponse:
        job_id = uuid.uuid4().hex
        (settings.jobs_dir / job_id / "chunks" / "dataset").mkdir(parents=True, exist_ok=True)
        (settings.jobs_dir / job_id / "chunks" / "kernel").mkdir(parents=True, exist_ok=True)
        db.create_job({"job_id": job_id, **payload.model_dump()})
        return job_response(db, job_id)

    @app.put(
        "/v1/jobs/{job_id}/archives/{archive_type}/chunks/{index}",
        response_model=ChunkResponse,
        dependencies=[Depends(require_auth)],
    )
    async def upload_chunk(
        job_id: str,
        archive_type: Literal["dataset", "kernel"],
        index: int,
        request: Request,
        x_chunk_sha256: str = Header(alias="X-Chunk-Sha256"),
        x_chunk_size: int = Header(alias="X-Chunk-Size"),
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
    ) -> ChunkResponse:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        total_size = job[f"{archive_type}_size"]
        try:
            validate_chunk_index(index, total_size, job["chunk_size"])
        except ArchiveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        existing = db.get_chunk(job_id, archive_type, index)
        chunk_dir = settings.jobs_dir / job_id / "chunks" / archive_type
        chunk_path = chunk_dir / f"{index}.part"
        if existing:
            if existing["sha256"] == x_chunk_sha256 and existing["size"] == x_chunk_size and chunk_path.exists():
                return ChunkResponse(
                    job_id=job_id,
                    archive_type=archive_type,
                    index=index,
                    size=x_chunk_size,
                    sha256=x_chunk_sha256,
                    duplicate=True,
                )
            raise HTTPException(status_code=409, detail="chunk already exists with different checksum")

        chunk_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = chunk_dir / f"{index}.tmp"
        digest = hashlib.sha256()
        size = 0
        async with aiofiles.open(tmp_path, "wb") as handle:
            async for part in request.stream():
                size += len(part)
                if size > x_chunk_size:
                    await handle.close()
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="chunk larger than X-Chunk-Size")
                digest.update(part)
                await handle.write(part)
        actual_sha = digest.hexdigest()
        if size != x_chunk_size:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="chunk size mismatch")
        if actual_sha.lower() != x_chunk_sha256.lower():
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="chunk sha256 mismatch")
        tmp_path.replace(chunk_path)
        db.add_chunk(job_id, archive_type, index, size, actual_sha)
        return ChunkResponse(
            job_id=job_id,
            archive_type=archive_type,
            index=index,
            size=size,
            sha256=actual_sha,
        )

    @app.post("/v1/jobs/{job_id}/complete", response_model=JobResponse, dependencies=[Depends(require_auth)])
    async def complete_job(
        job_id: str,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        request: Request = None,
    ) -> JobResponse:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] in {"queued", "uploading_dataset", "waiting_dataset", "pushing_kernel", "waiting_kernel", "downloading_output", "complete"}:
            return job_response(db, job_id)

        job_dir = settings.jobs_dir / job_id
        archives_dir = job_dir / "archives"
        extracted_dir = job_dir / "extracted"
        db.update_job(job_id, status="assembling", progress=10)
        try:
            kernel_zip = archives_dir / "kernel.zip"
            dataset_cache_hit = has_ready_dataset_cache(db, job["dataset_ref"], job["payload_hash"])
            if not dataset_cache_hit:
                dataset_zip = archives_dir / "dataset.zip"
                assemble_archive(
                    job_dir / "chunks" / "dataset",
                    dataset_zip,
                    job["dataset_size"],
                    job["chunk_size"],
                    job["dataset_archive_sha256"],
                )
                safe_extract_zip(dataset_zip, extracted_dir / "dataset", job["dataset_size"])
            assemble_archive(
                job_dir / "chunks" / "kernel",
                kernel_zip,
                job["kernel_size"],
                job["chunk_size"],
                job["kernel_archive_sha256"],
            )
            safe_extract_zip(kernel_zip, extracted_dir / "kernel", job["kernel_size"])
            if dataset_cache_hit:
                validate_kernel_payload(extracted_dir / "kernel")
            else:
                validate_payloads(extracted_dir / "dataset", extracted_dir / "kernel")
        except Exception as exc:
            db.update_job(job_id, status="failed", progress=0, error=redact_secrets(str(exc)))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.update_job(job_id, status="queued", progress=15)
        await request.app.state.queue.put(job_id)
        return job_response(db, job_id)

    @app.post("/v1/jobs/by-kernel/progress", response_model=JobResponse)
    def update_job_progress_by_kernel(
        payload: JobProgressRequest,
        authorization: str = Header(default=""),
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
    ) -> JobResponse:
        kernel_ref = str(payload.model_extra.get("kernel_ref") or "").strip() if payload.model_extra else ""
        if not kernel_ref:
            raise HTTPException(status_code=400, detail="kernel_ref is required")
        job = db.get_latest_job_by_kernel_ref(kernel_ref)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if not authorize_job_callback(job, authorization, settings):
            raise HTTPException(status_code=401, detail="unauthorized")

        apply_progress_callback(db, job, payload)
        return job_response(db, job["job_id"])

    @app.post("/v1/jobs/{job_id}/progress", response_model=JobResponse)
    def update_job_progress(
        job_id: str,
        payload: JobProgressRequest,
        authorization: str = Header(default=""),
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
    ) -> JobResponse:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if not authorize_job_callback(job, authorization, settings):
            raise HTTPException(status_code=401, detail="unauthorized")

        apply_progress_callback(db, job, payload)
        return job_response(db, job_id)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_auth)])
    def get_job(job_id: str, db: RelayDb = Depends(get_db)) -> JobResponse:
        return job_response(db, job_id)

    @app.get("/v1/jobs/{job_id}/artifacts.zip", dependencies=[Depends(require_auth)])
    def download_artifacts(job_id: str, db: RelayDb = Depends(get_db)) -> FileResponse:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "complete" or not job["artifact_path"]:
            raise HTTPException(status_code=409, detail="job is not complete")
        artifact_path = Path(job["artifact_path"])
        if not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(artifact_path, media_type="application/zip", filename="artifacts.zip")

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def delete_job(job_id: str, settings: Settings = Depends(get_settings), db: RelayDb = Depends(get_db)) -> Response:
        if not db.get_job(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
        shutil.rmtree(settings.artifacts_dir / job_id, ignore_errors=True)
        db.update_job(job_id, status="failed", error="deleted")
        return Response(status_code=204)

    return app


app = create_app()

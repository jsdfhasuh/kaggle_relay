import json
from pathlib import Path

from app.archive import ArchiveError, require_file
from app.database import RelayDb
from app.kaggle_adapter import KaggleAdapter
from app.security import redact_secrets

READY_DATASET_STATUSES = {"ready", "complete", "ok"}


def is_ready_dataset_status(status: str) -> bool:
    return (status or "").strip().lower() in READY_DATASET_STATUSES


def has_ready_dataset_cache(db: RelayDb, dataset_ref: str, payload_hash: str) -> bool:
    if not payload_hash:
        return False
    cache = db.get_dataset_cache(dataset_ref, payload_hash)
    if cache and cache["status"] == "ready":
        return True
    last_job = db.get_last_dataset_job(dataset_ref, payload_hash)
    if last_job and is_ready_dataset_status(last_job["dataset_status"]):
        db.upsert_dataset_cache(
            dataset_ref=dataset_ref,
            payload_hash=payload_hash,
            status="ready",
            dataset_status=last_job["dataset_status"],
            source_job_id=last_job["job_id"],
        )
        return True
    return False


def validate_kernel_payload(kernel_dir: Path) -> str:
    metadata_path = require_file(kernel_dir, "kernel-metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveError("kernel-metadata.json is not valid JSON") from exc
    code_file = str(metadata.get("code_file", "train.py") or "train.py")
    require_file(kernel_dir, code_file)
    return code_file


def validate_payloads(dataset_dir: Path, kernel_dir: Path) -> str:
    require_file(dataset_dir, "dataset-metadata.json")
    return validate_kernel_payload(kernel_dir)


def process_job(settings, db: RelayDb, job_id: str) -> None:
    job = db.get_job(job_id)
    if not job:
        return

    job_dir = settings.jobs_dir / job_id
    dataset_dir = job_dir / "extracted" / "dataset"
    kernel_dir = job_dir / "extracted" / "kernel"
    output_dir = job_dir / "kaggle_output"
    artifact_zip = settings.artifacts_dir / job_id / "artifacts.zip"

    def log(message: str) -> None:
        clean = redact_secrets(message)
        db.append_log(job_id, clean)
        db.update_job(job_id, kaggle_output=clean[-4000:])

    adapter = KaggleAdapter(settings, log)
    try:
        dataset_cache_hit = has_ready_dataset_cache(db, job["dataset_ref"], job["payload_hash"])
        if dataset_cache_hit:
            validate_kernel_payload(kernel_dir)
            db.update_job(job_id, dataset_status="ready", progress=40)
            log(f"Reusing ready dataset cache for {job['dataset_ref']}")
        else:
            validate_payloads(dataset_dir, kernel_dir)
            db.update_job(job_id, status="uploading_dataset", progress=20)
            adapter.upload_dataset(
                dataset_dir,
                job["dataset_ref"],
                update_message=f"update relay dataset for {job['kernel_ref']}",
            )

            db.update_job(job_id, status="waiting_dataset", progress=35)
            dataset_status = adapter.wait_dataset(job["dataset_ref"])
            db.update_job(job_id, dataset_status=dataset_status, progress=40)
            if is_ready_dataset_status(dataset_status):
                db.upsert_dataset_cache(
                    dataset_ref=job["dataset_ref"],
                    payload_hash=job["payload_hash"],
                    status="ready",
                    dataset_status=dataset_status,
                    source_job_id=job_id,
                )
                db.upsert_last_dataset_job(
                    dataset_ref=job["dataset_ref"],
                    payload_hash=job["payload_hash"],
                    dataset_status=dataset_status,
                    job_id=job_id,
                )

        db.update_job(job_id, status="pushing_kernel", progress=45)
        push_output = adapter.push_kernel(kernel_dir)
        log(push_output)

        def progress_callback(progress_data: dict) -> None:
            remote_progress = float(progress_data.get("remote_progress", 0) or 0)
            db.update_job(
                job_id,
                status="waiting_kernel",
                progress=min(80, 60 + int(remote_progress / 100 * 20)),
                kernel_status=json.dumps(progress_data, ensure_ascii=False),
            )

        db.update_job(job_id, status="waiting_kernel", progress=60)
        kernel_status = adapter.wait_kernel(job["kernel_ref"], progress_callback)
        db.update_job(job_id, kernel_status=kernel_status, progress=82)

        db.update_job(job_id, status="downloading_output", progress=85)
        output = adapter.download_output(job["kernel_ref"], output_dir)
        log(output)
        adapter.package_artifacts(output_dir, artifact_zip)
        db.update_job(
            job_id,
            status="complete",
            progress=100,
            artifact_path=str(artifact_zip),
            error="",
        )
    except Exception as exc:
        db.update_job(job_id, status="failed", progress=0, error=redact_secrets(str(exc)))
        db.append_log(job_id, redact_secrets(str(exc)))

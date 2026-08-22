import json
from pathlib import Path

from app.archive import ArchiveError, require_file
from app.database import RelayDb
from app.auth_config import AuthStore
from app.kaggle_adapter import KaggleAdapter, is_ready_kaggle_status
from app.security import redact_secrets

TERMINAL_JOB_STATUSES = {"complete", "failed", "canceled"}


class JobCanceled(Exception):
    pass


def is_ready_dataset_status(status: str) -> bool:
    return is_ready_kaggle_status(status)


def has_ready_dataset_cache(
    db: RelayDb,
    dataset_ref: str,
    payload_hash: str,
    kaggle_key_id: str = "",
) -> bool:
    if not payload_hash:
        return False
    cache = db.get_dataset_cache(dataset_ref, payload_hash, kaggle_key_id=kaggle_key_id)
    if cache and cache["status"] == "ready":
        return True
    last_job = db.get_last_dataset_job(dataset_ref, payload_hash, kaggle_key_id=kaggle_key_id)
    if last_job and is_ready_dataset_status(last_job["dataset_status"]):
        db.upsert_dataset_cache(
            dataset_ref=dataset_ref,
            payload_hash=payload_hash,
            status="ready",
            dataset_status=last_job["dataset_status"],
            source_job_id=last_job["job_id"],
            kaggle_key_id=kaggle_key_id,
        )
        return True
    return False


def read_metadata_json(path: Path, name: str) -> dict:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveError(f"{name} is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ArchiveError(f"{name} must be a JSON object")
    return metadata


def write_metadata_json(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_kaggle_ref(value: str, name: str) -> tuple[str, str]:
    ref = str(value or "").strip()
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ArchiveError(f"{name} must be in owner/slug format")
    return parts[0].strip(), parts[1].strip()


def kaggle_ref_slug(value: str) -> str:
    ref = str(value or "").strip()
    parts = ref.split("/", 1)
    if len(parts) != 2:
        return ""
    return parts[1].strip()


def rewrite_ref_owner(value: str, owner: str) -> str:
    owner = str(owner or "").strip()
    if not owner:
        return str(value or "").strip()
    _current_owner, slug = split_kaggle_ref(value, "kaggle ref")
    return f"{owner}/{slug}"


def job_cancel_requested(job: dict | None) -> bool:
    if not job:
        return False
    return bool(job.get("cancel_requested_at")) or job.get("status") == "cancel_requested"


def job_paths(settings, job_id: str) -> dict[str, Path]:
    job_dir = settings.jobs_dir / job_id
    return {
        "job_dir": job_dir,
        "dataset_dir": job_dir / "extracted" / "dataset",
        "kernel_dir": job_dir / "extracted" / "kernel",
        "output_dir": job_dir / "kaggle_output",
        "artifact_zip": settings.artifacts_dir / job_id / "artifacts.zip",
    }


def job_log(db: RelayDb, job_id: str):
    def log(message: str) -> None:
        clean = redact_secrets(message)
        db.append_log(job_id, clean)
        db.update_job(job_id, kaggle_output=clean[-4000:])

    return log


def job_adapter(settings, db: RelayDb, job: dict, auth_store: AuthStore | None = None) -> KaggleAdapter:
    kaggle_key_id = job.get("kaggle_key_id", "")
    credentials = auth_store.credentials_for(kaggle_key_id) if auth_store else None
    return KaggleAdapter(settings, job_log(db, job["job_id"]), credentials=credentials)


def rewrite_dataset_sources(metadata: dict, final_dataset_ref: str) -> bool:
    final_slug = kaggle_ref_slug(final_dataset_ref).lower()
    if not final_slug:
        return False
    dataset_sources = metadata.get("dataset_sources")
    if not isinstance(dataset_sources, list):
        return False

    changed = False
    rewritten = []
    for source in dataset_sources:
        if isinstance(source, str) and kaggle_ref_slug(source).lower() == final_slug:
            if source != final_dataset_ref:
                changed = True
            rewritten.append(final_dataset_ref)
        else:
            rewritten.append(source)
    if changed:
        metadata["dataset_sources"] = rewritten
    return changed


def prepare_metadata_ref(
    path: Path,
    metadata_name: str,
    expected_ref: str,
    final_dataset_ref: str = "",
) -> dict:
    metadata = read_metadata_json(path, metadata_name)
    changed = False
    if expected_ref:
        metadata_ref = str(metadata.get("id") or "").strip()
        if metadata_ref:
            _metadata_owner, metadata_slug = split_kaggle_ref(metadata_ref, f"{metadata_name} id")
            _expected_owner, expected_slug = split_kaggle_ref(expected_ref, f"{metadata_name} requested ref")
            if metadata_slug.lower() != expected_slug.lower():
                raise ArchiveError(
                    f"{metadata_name} id {metadata_ref} does not match requested ref {expected_ref}"
                )
        if metadata_ref != expected_ref:
            metadata["id"] = expected_ref
            changed = True
    if metadata_name == "kernel-metadata.json" and rewrite_dataset_sources(metadata, final_dataset_ref):
        changed = True
    if changed:
        write_metadata_json(path, metadata)
    return metadata


def validate_metadata_ref(
    metadata: dict,
    metadata_name: str,
    expected_ref: str,
    credentials=None,
) -> None:
    metadata_ref = str(metadata.get("id") or "").strip()
    if not metadata_ref:
        raise ArchiveError(f"{metadata_name} requires id")
    metadata_owner, _metadata_slug = split_kaggle_ref(metadata_ref, f"{metadata_name} id")

    if expected_ref and metadata_ref.lower() != expected_ref.lower():
        raise ArchiveError(
            f"{metadata_name} id {metadata_ref} does not match requested ref {expected_ref}"
        )

    username = str(getattr(credentials, "username", "") or "").strip()
    if username and metadata_owner.lower() != username.lower():
        raise ArchiveError(
            f"{metadata_name} owner {metadata_owner} does not match Kaggle key username {username}"
        )


def validate_kernel_payload(
    kernel_dir: Path,
    kernel_ref: str = "",
    credentials=None,
    dataset_ref: str = "",
) -> str:
    metadata_path = require_file(kernel_dir, "kernel-metadata.json")
    metadata = prepare_metadata_ref(
        metadata_path,
        "kernel-metadata.json",
        kernel_ref,
        final_dataset_ref=dataset_ref,
    )
    validate_metadata_ref(metadata, "kernel-metadata.json", kernel_ref, credentials)
    code_file = str(metadata.get("code_file", "train.py") or "train.py")
    require_file(kernel_dir, code_file)
    return code_file


def validate_payloads(
    dataset_dir: Path,
    kernel_dir: Path,
    dataset_ref: str = "",
    kernel_ref: str = "",
    credentials=None,
) -> str:
    metadata_path = require_file(dataset_dir, "dataset-metadata.json")
    metadata = prepare_metadata_ref(metadata_path, "dataset-metadata.json", dataset_ref)
    validate_metadata_ref(metadata, "dataset-metadata.json", dataset_ref, credentials)
    return validate_kernel_payload(kernel_dir, kernel_ref, credentials, dataset_ref=dataset_ref)


def structured_kernel_failure_error(kernel_status) -> str:
    if isinstance(kernel_status, dict):
        payload = kernel_status
    elif isinstance(kernel_status, str):
        try:
            payload = json.loads(kernel_status)
        except (TypeError, json.JSONDecodeError):
            return ""
    else:
        return ""
    if not isinstance(payload, dict):
        return ""
    if str(payload.get("status") or "").strip().lower() not in {
        "failed",
        "error",
        "canceled",
    }:
        return ""
    error = str(payload.get("error") or "").strip()
    if error:
        return error
    error_code = str(payload.get("error_code") or "").strip()
    message = str(payload.get("message") or "").strip()
    if error_code and message:
        return f"{error_code}: {message}"
    return error_code or message


def finish_kernel_job(
    settings,
    db: RelayDb,
    job_id: str,
    adapter: KaggleAdapter,
    final_status: str | None = None,
) -> None:
    paths = job_paths(settings, job_id)
    log = job_log(db, job_id)

    def latest_job() -> dict:
        return db.get_job(job_id) or {}

    def progress_callback(progress_data: dict) -> None:
        current = latest_job()
        remote_progress = float(progress_data.get("remote_progress", 0) or 0)
        status = "cancel_requested" if job_cancel_requested(current) else "waiting_kernel"
        db.update_job(
            job_id,
            status=status,
            progress=max(float(current.get("progress") or 0), min(80, 60 + int(remote_progress / 100 * 20))),
            kernel_status=json.dumps(progress_data, ensure_ascii=False),
        )

    current = latest_job()
    db.update_job(
        job_id,
        status="cancel_requested" if job_cancel_requested(current) else "waiting_kernel",
        progress=max(float(current.get("progress") or 0), 60),
    )
    kernel_ref = str(current.get("kernel_ref") or "")
    if not kernel_ref:
        raise RuntimeError(f"job {job_id} is missing kernel_ref")
    kernel_status = adapter.wait_kernel(kernel_ref, progress_callback)
    current = latest_job()
    db.update_job(job_id, kernel_status=kernel_status, progress=max(float(current.get("progress") or 0), 82))

    current = latest_job()
    db.update_job(job_id, status="downloading_output", progress=max(float(current.get("progress") or 0), 85))
    output = adapter.download_output(kernel_ref, paths["output_dir"])
    log(output)
    adapter.package_artifacts(paths["output_dir"], paths["artifact_zip"])
    current = latest_job()
    status = final_status or ("canceled" if job_cancel_requested(current) else "complete")
    db.update_job(
        job_id,
        status=status,
        progress=100,
        artifact_path=str(paths["artifact_zip"]),
        error="",
    )


def resume_kernel_job(
    settings,
    db: RelayDb,
    job_id: str,
    auth_store: AuthStore | None = None,
    final_status: str | None = None,
) -> None:
    job = db.get_job(job_id)
    if not job:
        return
    adapter = job_adapter(settings, db, job, auth_store)
    finish_kernel_job(settings, db, job_id, adapter, final_status=final_status)


def process_job(settings, db: RelayDb, job_id: str, auth_store: AuthStore | None = None) -> None:
    job = db.get_job(job_id)
    if not job:
        return
    kaggle_key_id = job.get("kaggle_key_id", "")

    paths = job_paths(settings, job_id)
    dataset_dir = paths["dataset_dir"]
    kernel_dir = paths["kernel_dir"]
    log = job_log(db, job_id)

    def latest_job() -> dict:
        return db.get_job(job_id) or job

    def cancel_requested() -> bool:
        return job_cancel_requested(latest_job())

    def cancel_reason() -> str:
        return str(latest_job().get("cancel_reason") or "cancel requested")

    def stop_if_cancel_requested(message: str = "canceled before Kaggle kernel submission") -> None:
        if cancel_requested():
            raise JobCanceled(message)

    try:
        credentials = auth_store.credentials_for(kaggle_key_id) if auth_store else None
        adapter = KaggleAdapter(settings, log, credentials=credentials)
        stop_if_cancel_requested()
        dataset_cache_hit = has_ready_dataset_cache(
            db,
            job["dataset_ref"],
            job["payload_hash"],
            kaggle_key_id=kaggle_key_id,
        )
        if dataset_cache_hit:
            validate_kernel_payload(
                kernel_dir,
                job["kernel_ref"],
                credentials,
                dataset_ref=job["dataset_ref"],
            )
            db.update_job(job_id, dataset_status="ready", progress=40)
            log(f"Reusing ready dataset cache for {job['dataset_ref']}")
        else:
            stop_if_cancel_requested()
            validate_payloads(
                dataset_dir,
                kernel_dir,
                job["dataset_ref"],
                job["kernel_ref"],
                credentials,
            )
            stop_if_cancel_requested()
            db.update_job(job_id, status="uploading_dataset", progress=20)
            adapter.upload_dataset(
                dataset_dir,
                job["dataset_ref"],
                update_message=f"update relay dataset for {job['kernel_ref']}",
            )

            stop_if_cancel_requested()
            db.update_job(job_id, status="waiting_dataset", progress=35)
            dataset_status = adapter.wait_dataset(
                job["dataset_ref"],
                permission_grace_seconds=settings.dataset_status_permission_grace_seconds,
            )
            db.update_job(job_id, dataset_status=dataset_status, progress=40)
            stop_if_cancel_requested()
            if is_ready_dataset_status(dataset_status):
                db.upsert_dataset_cache(
                    dataset_ref=job["dataset_ref"],
                    payload_hash=job["payload_hash"],
                    status="ready",
                    dataset_status=dataset_status,
                    source_job_id=job_id,
                    kaggle_key_id=kaggle_key_id,
                )
                db.upsert_last_dataset_job(
                    dataset_ref=job["dataset_ref"],
                    payload_hash=job["payload_hash"],
                    dataset_status=dataset_status,
                    job_id=job_id,
                    kaggle_key_id=kaggle_key_id,
                )

        stop_if_cancel_requested()
        db.update_job(job_id, status="pushing_kernel", progress=45)
        push_output = adapter.push_kernel(kernel_dir)
        log(push_output)
        finish_kernel_job(settings, db, job_id, adapter)
    except JobCanceled as exc:
        message = redact_secrets(str(exc) or cancel_reason())
        db.update_job(job_id, status="canceled", error=message)
        db.append_log(job_id, message)
    except Exception as exc:
        current = latest_job()
        message = structured_kernel_failure_error(current.get("kernel_status")) or str(exc)
        message = redact_secrets(message)
        db.update_job(job_id, status="failed", progress=0, error=message)
        db.append_log(job_id, message)

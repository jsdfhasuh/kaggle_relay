import json
from pathlib import Path

from app.archive import ArchiveError, require_file
from app.database import RelayDb
from app.auth_config import AuthStore
from app.kaggle_adapter import KaggleAdapter
from app.security import redact_secrets

READY_DATASET_STATUSES = {"ready", "complete", "ok"}


def is_ready_dataset_status(status: str) -> bool:
    return (status or "").strip().lower() in READY_DATASET_STATUSES


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


def split_kaggle_ref(value: str, name: str) -> tuple[str, str]:
    ref = str(value or "").strip()
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ArchiveError(f"{name} must be in owner/slug format")
    return parts[0].strip(), parts[1].strip()


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
) -> str:
    metadata_path = require_file(kernel_dir, "kernel-metadata.json")
    metadata = read_metadata_json(metadata_path, "kernel-metadata.json")
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
    metadata = read_metadata_json(metadata_path, "dataset-metadata.json")
    validate_metadata_ref(metadata, "dataset-metadata.json", dataset_ref, credentials)
    return validate_kernel_payload(kernel_dir, kernel_ref, credentials)


def process_job(settings, db: RelayDb, job_id: str, auth_store: AuthStore | None = None) -> None:
    job = db.get_job(job_id)
    if not job:
        return
    kaggle_key_id = job.get("kaggle_key_id", "")

    job_dir = settings.jobs_dir / job_id
    dataset_dir = job_dir / "extracted" / "dataset"
    kernel_dir = job_dir / "extracted" / "kernel"
    output_dir = job_dir / "kaggle_output"
    artifact_zip = settings.artifacts_dir / job_id / "artifacts.zip"

    def log(message: str) -> None:
        clean = redact_secrets(message)
        db.append_log(job_id, clean)
        db.update_job(job_id, kaggle_output=clean[-4000:])

    credentials = auth_store.credentials_for(kaggle_key_id) if auth_store else None
    adapter = KaggleAdapter(settings, log, credentials=credentials)
    try:
        dataset_cache_hit = has_ready_dataset_cache(
            db,
            job["dataset_ref"],
            job["payload_hash"],
            kaggle_key_id=kaggle_key_id,
        )
        if dataset_cache_hit:
            validate_kernel_payload(kernel_dir, job["kernel_ref"], credentials)
            db.update_job(job_id, dataset_status="ready", progress=40)
            log(f"Reusing ready dataset cache for {job['dataset_ref']}")
        else:
            validate_payloads(
                dataset_dir,
                kernel_dir,
                job["dataset_ref"],
                job["kernel_ref"],
                credentials,
            )
            db.update_job(job_id, status="uploading_dataset", progress=20)
            adapter.upload_dataset(
                dataset_dir,
                job["dataset_ref"],
                update_message=f"update relay dataset for {job['kernel_ref']}",
            )

            db.update_job(job_id, status="waiting_dataset", progress=35)
            dataset_status = adapter.wait_dataset(
                job["dataset_ref"],
                permission_grace_seconds=settings.dataset_status_permission_grace_seconds,
            )
            db.update_job(job_id, dataset_status=dataset_status, progress=40)
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

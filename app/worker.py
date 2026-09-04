import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from app.archive import ArchiveError, require_file
from app.database import RelayDb
from app.auth_config import AuthStore
from app.kaggle_adapter import (
    DatasetUploadReceipt,
    KaggleAdapter,
    KaggleAdapterInterrupted,
    is_ready_kaggle_status,
    parse_kaggle_dataset_status,
)
from app.security import redact_secrets

TERMINAL_JOB_STATUSES = {"complete", "failed", "canceled"}
_DATASET_SUBMISSION_LOCKS: dict[str, list] = {}
_DATASET_SUBMISSION_LOCKS_GUARD = threading.Lock()


class JobCanceled(Exception):
    pass


@contextmanager
def dataset_submission_lock(
    dataset_ref: str,
    shutdown_event: threading.Event | None = None,
):
    key = str(dataset_ref or "").strip().lower()
    with _DATASET_SUBMISSION_LOCKS_GUARD:
        entry = _DATASET_SUBMISSION_LOCKS.get(key)
        if entry is None:
            entry = [threading.Lock(), 0]
            _DATASET_SUBMISSION_LOCKS[key] = entry
        entry[1] += 1
    lock = entry[0]
    acquired = False
    try:
        while not acquired:
            if shutdown_event and shutdown_event.is_set():
                raise KaggleAdapterInterrupted(
                    "relay shutdown interrupted dataset submission wait"
                )
            acquired = lock.acquire(timeout=0.5) if shutdown_event else lock.acquire()
        yield
    finally:
        if acquired:
            lock.release()
        with _DATASET_SUBMISSION_LOCKS_GUARD:
            entry[1] -= 1
            if entry[1] == 0 and _DATASET_SUBMISSION_LOCKS.get(key) is entry:
                _DATASET_SUBMISSION_LOCKS.pop(key, None)


def is_ready_dataset_status(status: str) -> bool:
    return is_ready_kaggle_status(status)


def ready_dataset_version(status: str) -> int | None:
    status_name, version_number = parse_kaggle_dataset_status(status)
    if (
        status_name not in {"ready", "complete", "ok"}
        or version_number is None
        or version_number <= 0
    ):
        return None
    return version_number


def get_ready_dataset_cache(
    db: RelayDb,
    dataset_ref: str,
    payload_hash: str,
    kaggle_key_id: str = "",
) -> dict | None:
    if not payload_hash:
        return None
    cache = db.get_current_dataset_cache(dataset_ref, kaggle_key_id=kaggle_key_id)
    if (
        cache
        and cache["payload_hash"] == payload_hash
        and cache["status"] == "ready"
        and ready_dataset_version(cache["dataset_status"]) is not None
    ):
        return cache
    if cache is not None:
        return None
    last_job = db.get_current_last_dataset_job(
        dataset_ref,
        kaggle_key_id=kaggle_key_id,
    )
    if (
        last_job
        and last_job["payload_hash"] == payload_hash
        and ready_dataset_version(last_job["dataset_status"]) is not None
    ):
        db.upsert_dataset_cache(
            dataset_ref=dataset_ref,
            payload_hash=payload_hash,
            status="ready",
            dataset_status=last_job["dataset_status"],
            source_job_id=last_job["job_id"],
            kaggle_key_id=kaggle_key_id,
        )
        return db.get_dataset_cache(
            dataset_ref,
            payload_hash,
            kaggle_key_id=kaggle_key_id,
        )
    return None


def has_ready_dataset_cache(
    db: RelayDb,
    dataset_ref: str,
    payload_hash: str,
    kaggle_key_id: str = "",
) -> bool:
    return get_ready_dataset_cache(
        db,
        dataset_ref,
        payload_hash,
        kaggle_key_id=kaggle_key_id,
    ) is not None


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
    parts = ref.split("/")
    if len(parts) < 2:
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


def job_adapter(
    settings,
    db: RelayDb,
    job: dict,
    auth_store: AuthStore | None = None,
) -> KaggleAdapter:
    kaggle_key_id = job.get("kaggle_key_id", "")
    credentials = auth_store.credentials_for(kaggle_key_id) if auth_store else None
    adapter = KaggleAdapter(
        settings,
        job_log(db, job["job_id"]),
        credentials=credentials,
    )
    shutdown_event = getattr(settings, "_shutdown_event", None)
    if shutdown_event is not None:
        adapter.shutdown_event = shutdown_event
    return adapter


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


def pin_kernel_dataset_version(
    kernel_dir: Path,
    dataset_ref: str,
    version_number: int | None,
) -> None:
    if version_number is None or version_number <= 0:
        return
    metadata_path = require_file(kernel_dir, "kernel-metadata.json")
    metadata = read_metadata_json(metadata_path, "kernel-metadata.json")
    versioned_ref = f"{dataset_ref}/{version_number}"
    if rewrite_dataset_sources(metadata, versioned_ref):
        write_metadata_json(metadata_path, metadata)


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

    def update_finish_status(
        running_status: str,
        values_for_job: Callable[[dict], dict],
    ) -> dict:
        for _attempt in range(5):
            current = latest_job()
            current_status = str(current.get("status") or "")
            if not current_status or current_status in TERMINAL_JOB_STATUSES:
                return current
            status = (
                "cancel_requested"
                if job_cancel_requested(current)
                else running_status
            )
            if db.update_job_if_status(
                job_id,
                {current_status},
                status=status,
                **values_for_job(current),
            ):
                return latest_job()
        raise RuntimeError(
            f"job {job_id} status changed repeatedly during {running_status} transition"
        )

    def progress_callback(progress_data: dict) -> None:
        remote_progress = float(progress_data.get("remote_progress", 0) or 0)
        update_finish_status(
            "waiting_kernel",
            lambda current: {
                "progress": max(
                    float(current.get("progress") or 0),
                    min(80, 60 + int(remote_progress / 100 * 20)),
                ),
                "kernel_status": json.dumps(progress_data, ensure_ascii=False),
            },
        )

    current = update_finish_status(
        "waiting_kernel",
        lambda job: {
            "progress": max(float(job.get("progress") or 0), 60),
        },
    )
    kernel_ref = str(current.get("kernel_ref") or "")
    if not kernel_ref:
        raise RuntimeError(f"job {job_id} is missing kernel_ref")
    kernel_status = adapter.wait_kernel(kernel_ref, progress_callback)
    current = latest_job()
    db.update_job(job_id, kernel_status=kernel_status, progress=max(float(current.get("progress") or 0), 82))

    current = update_finish_status(
        "downloading_output",
        lambda job: {
            "progress": max(float(job.get("progress") or 0), 85),
        },
    )
    artifact_contract = str(current.get("artifact_contract") or "yolo")
    output = adapter.download_output(
        kernel_ref,
        paths["output_dir"],
        artifact_contract=artifact_contract,
    )
    log(output)
    adapter.package_artifacts(
        paths["output_dir"],
        paths["artifact_zip"],
        artifact_contract=artifact_contract,
    )
    db.finalize_job(
        job_id,
        final_status or "complete",
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


def process_job(
    settings,
    db: RelayDb,
    job_id: str,
    auth_store: AuthStore | None = None,
) -> None:
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

    def transition_before_submission(
        expected_statuses: set[str],
        status: str,
        progress: float,
    ) -> None:
        if db.update_job_if_status(
            job_id,
            expected_statuses,
            require_not_canceled=True,
            status=status,
            progress=progress,
        ):
            return
        stop_if_cancel_requested()
        current_status = str(latest_job().get("status") or "missing")
        raise RuntimeError(
            f"job {job_id} changed to {current_status} before {status} transition"
        )

    try:
        credentials = auth_store.credentials_for(kaggle_key_id) if auth_store else None
        adapter = KaggleAdapter(settings, log, credentials=credentials)
        shutdown_event = getattr(settings, "_shutdown_event", None)
        if shutdown_event is not None:
            adapter.shutdown_event = shutdown_event
        with dataset_submission_lock(job["dataset_ref"], shutdown_event):
            stop_if_cancel_requested()
            dataset_cache = get_ready_dataset_cache(
                db,
                job["dataset_ref"],
                job["payload_hash"],
                kaggle_key_id=kaggle_key_id,
            )
            if dataset_cache:
                validate_kernel_payload(
                    kernel_dir,
                    job["kernel_ref"],
                    credentials,
                    dataset_ref=job["dataset_ref"],
                )
                cached_version_number = ready_dataset_version(
                    dataset_cache["dataset_status"]
                )
                dataset_status = adapter.wait_dataset(
                    job["dataset_ref"],
                    permission_grace_seconds=(
                        settings.dataset_status_permission_grace_seconds
                    ),
                    upload_receipt=DatasetUploadReceipt(
                        expected_version_number=cached_version_number,
                    ),
                )
                pin_kernel_dataset_version(
                    kernel_dir,
                    job["dataset_ref"],
                    cached_version_number,
                )
                db.update_job(job_id, dataset_status=dataset_status, progress=40)
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
                transition_before_submission({"queued"}, "uploading_dataset", 20)
                upload_receipt = adapter.upload_dataset(
                    dataset_dir,
                    job["dataset_ref"],
                    update_message=f"update relay dataset for {job['kernel_ref']}",
                )

                transition_before_submission({"uploading_dataset"}, "waiting_dataset", 35)
                wait_dataset_kwargs = {
                    "permission_grace_seconds": (
                        settings.dataset_status_permission_grace_seconds
                    ),
                }
                if upload_receipt is not None:
                    wait_dataset_kwargs["upload_receipt"] = upload_receipt
                dataset_status = adapter.wait_dataset(
                    job["dataset_ref"],
                    **wait_dataset_kwargs,
                )
                db.update_job(job_id, dataset_status=dataset_status, progress=40)
                stop_if_cancel_requested()
                dataset_version_number = ready_dataset_version(dataset_status)
                if upload_receipt is not None and dataset_version_number is None:
                    raise KaggleAdapterError(
                        "Dataset wait completed without a verified version number"
                    )
                pin_kernel_dataset_version(
                    kernel_dir,
                    job["dataset_ref"],
                    dataset_version_number,
                )
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

            transition_before_submission(
                {"queued", "waiting_dataset"},
                "pushing_kernel",
                45,
            )
            push_output = adapter.push_kernel(kernel_dir)
            log(push_output)
        finish_kernel_job(settings, db, job_id, adapter)
    except KaggleAdapterInterrupted as exc:
        db.append_log(job_id, redact_secrets(str(exc)))
    except JobCanceled as exc:
        message = redact_secrets(str(exc) or cancel_reason())
        db.finalize_job(job_id, "canceled", error=message)
        db.append_log(job_id, message)
    except Exception as exc:
        current = latest_job()
        message = structured_kernel_failure_error(current.get("kernel_status")) or str(exc)
        message = redact_secrets(message)
        db.finalize_job(job_id, "failed", progress=0, error=message)
        db.append_log(job_id, message)

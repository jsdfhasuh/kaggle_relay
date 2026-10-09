"""Durable, bounded rechecks of an already submitted Dataset version."""

import asyncio
import time
from datetime import datetime, timezone

from app.dataset_verification_process import verification_error
from app.upload_intent import intent_path, read_intent
from app.dataset_verification_state import PROGRESS_RECHECK_SECONDS, VerificationStore


RECHECK_INITIAL_SECONDS = 300
RECHECK_MAX_SECONDS = 1800
RECHECK_WINDOW_SECONDS = 24 * 3600
RETAINED_PREFIX = "dataset_upload_outcome_unknown:"


def retryable_verification(exc) -> bool:
    error = verification_error(exc)
    return (error["category"] in {"publication", "transport", "progress"}
            or error["category"] == "http" and error["http_status"] in {408, 409, 429, 500, 502, 503, 504})


def schedule_recheck(db, job_id, *, retry_after=0, now=None, progress=False) -> None:
    now = time.time() if now is None else now
    job = db.get_job(job_id)
    if not job or job["status"] != "waiting_dataset" or job.get("cancel_requested_at") is not None:
        return
    started = job.get("dataset_recheck_started_at") or now
    failures = job.get("dataset_recheck_failures", 0)
    delay = max(retry_after or 0, PROGRESS_RECHECK_SECONDS if progress else
                min(RECHECK_MAX_SECONDS, RECHECK_INITIAL_SECONDS * 2 ** min(failures, 4)))
    due = now + delay
    exhausted = due >= started + RECHECK_WINDOW_SECONDS
    state = "exhausted" if exhausted else "scheduled"
    message = ("Dataset automatic recheck window exhausted; original version retained; manual recovery required"
               if exhausted else "Dataset automatic recheck scheduled at "
               + datetime.fromtimestamp(due, timezone.utc).isoformat()
               + f" (in {delay:.0f}s); original version retained, no reupload")
    if db.update_job_if_status(job_id, {"waiting_dataset"}, require_not_canceled=True,
                              dataset_recheck_started_at=started, dataset_recheck_state=state,
                              dataset_recheck_failures=0 if progress else failures + 1,
                              dataset_recheck_at=None if exhausted else due, queue_reason=message):
        db.append_log(job_id, message)


def block_recheck(db, job_id, reason) -> None:
    message = "Dataset automatic recheck blocked; manual inspection required: " + reason
    if db.update_job_if_status(job_id, {"waiting_dataset"}, require_not_canceled=True,
                              dataset_recheck_state="blocked", dataset_recheck_at=None,
                              queue_reason=message):
        db.append_log(job_id, message)


async def schedule_dataset_rechecks(app) -> None:
    db, settings = app.state.db, app.state.settings
    now = time.time()
    for snapshot in db.list_jobs_by_status({"waiting_dataset"}):
        job_id = snapshot["job_id"]
        async with app.state.job_submission_locks.setdefault(job_id, asyncio.Lock()):
            job = db.get_job(job_id)
            if (not job or job["status"] != "waiting_dataset" or job_id in app.state.active_job_ids
                    or job.get("cancel_requested_at") is not None
                    or not job.get("error", "").startswith(RETAINED_PREFIX)):
                continue
            state = job.get("dataset_recheck_state", "")
            if state in {"blocked", "exhausted"}:
                continue
            if not state:
                # Only migrate known transient outcomes from older releases.
                if any(code in job["error"] for code in (
                        "dataset_verification_retry_exhausted:", "payload_publication_timeout:")):
                    schedule_recheck(db, job_id, now=now)
                else:
                    block_recheck(db, job_id, "previous failure is not a classified transient verification error")
                continue
            if state != "scheduled" or not job.get("dataset_recheck_at") or job["dataset_recheck_at"] > now:
                continue
            if now >= job["dataset_recheck_started_at"] + RECHECK_WINDOW_SECONDS:
                schedule_recheck(db, job_id, now=now)
                continue
            store_path = settings.storage_dir / "dataset-verification.sqlite3"
            if store_path.exists():
                cooldown = VerificationStore(settings.storage_dir).cooldown_remaining(job["dataset_ref"].split("/")[0], now)
                if cooldown:
                    db.update_job_if_status(job_id, {"waiting_dataset"}, require_not_canceled=True,
                                            dataset_recheck_at=now + cooldown)
                    continue
            auth = app.state.auth_store
            if not auth.legacy:
                principal = next((p for _, p in auth._tokens if p.id == job.get("relay_token_id")), None)
                credentials = auth._kaggle_keys.get(job.get("kaggle_key_id"))
                if (not principal or not credentials or not credentials.enabled
                        or not auth.can_access_key(principal, job.get("kaggle_key_id", ""))):
                    block_recheck(db, job_id, "original account is disabled or no longer authorized")
                    continue
            dataset_dir = settings.jobs_dir / job_id / "extracted" / "dataset"
            try:
                intent = read_intent(intent_path(settings.storage_dir, dataset_dir, job["dataset_ref"]))
                valid = (intent and intent["state"] in {"accepted", "unknown"}
                         and intent.get("dataset_ref") == job["dataset_ref"]
                         and intent.get("dataset_dir") == str(dataset_dir.absolute()))
            except (OSError, ValueError):
                valid = False
            if not valid:
                block_recheck(db, job_id, "original upload intent is missing, invalid or rejected")
                continue
            if db.update_job_if_status(job_id, {"waiting_dataset"}, require_not_canceled=True,
                                       status="queued", error="", dataset_recheck_state="checking",
                                       dataset_recheck_at=None,
                                       dataset_recheck_count=job["dataset_recheck_count"] + 1,
                                       queue_reason="waiting for a Dataset recheck worker"):
                db.append_log(job_id, f"Dataset automatic recheck {job['dataset_recheck_count'] + 1}: "
                              f"queued original version {intent['version_number']}; no reupload")
                app.state.queue.put_nowait({"action": "recheck_dataset", "job_id": job_id})

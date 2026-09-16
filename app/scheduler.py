"""Bind uploaded jobs once, when an allowed account becomes available."""

import asyncio
import json
import logging

from fastapi import FastAPI

from app.auth_config import AuthStore
from app.kaggle_adapter import KaggleAdapter
from app.security import redact_secrets
from app.worker import rewrite_ref_owner


LOGGER = logging.getLogger("uvicorn.error")
OCCUPIED_STATUSES = {
    "queued", "uploading_dataset", "waiting_dataset", "pushing_kernel",
    "waiting_kernel", "cancel_requested", "downloading_output",
}


def awaiting_assignment(job: dict) -> bool:
    return job.get("scheduling_mode") == "dynamic" and job.get("assignment_state") == "pending"


def eligible_accounts(job: dict) -> dict[str, str]:
    return json.loads(job.get("eligible_accounts") or "{}")


def current_accounts(job: dict, auth_store: AuthStore) -> dict[str, str]:
    principal = next((p for _, p in auth_store._tokens if p.id == job.get("relay_token_id")), None)
    if principal is None:
        return {}
    allowed = set(auth_store.allowed_key_ids(principal))
    return {
        key: username for key, username in eligible_accounts(job).items()
        if key in allowed and auth_store.credentials_for(key).username.lower() == username.lower()
    }


def occupied_accounts(app: FastAPI) -> dict[str, int]:
    counts = {}
    for job in app.state.db.list_jobs_by_status(OCCUPIED_STATUSES):
        if awaiting_assignment(job):
            continue
        credentials = app.state.auth_store._kaggle_keys.get(job.get("kaggle_key_id"))
        if credentials and credentials.username:
            username = credentials.username.lower()
            counts[username] = counts.get(username, 0) + 1
    return counts


async def schedule_pending_jobs(app: FastAPI) -> None:
    db, settings = app.state.db, app.state.settings
    pending = [job for job in db.list_jobs_by_status({"queued"}) if awaiting_assignment(job)]
    if not pending:
        return
    # Fetch each account once, outside both the event loop and upload I/O executor.
    occupied = occupied_accounts(app)
    lookups = {}
    for job in pending:
        for key, username in current_accounts(job, app.state.auth_store).items():
            if occupied.get(username.lower(), 0) >= settings.account_concurrency or key in lookups:
                continue
            credentials = app.state.auth_store.credentials_for(key)
            adapter = KaggleAdapter(settings, lambda _: None, credentials=credentials,
                                    shutdown_event=app.state.shutdown_event)
            future = settings._quota_cache.submit((credentials, settings.kaggle_cmd), adapter.quota)
            waiting = not future.done()
            lookups[key] = asyncio.wrap_future(future)
            if waiting:
                lookups[key].add_done_callback(lambda _: app.state.scheduler_event.set())
    # One slow or unreachable account must not hold back other idle accounts.
    if lookups:
        await asyncio.wait(lookups.values(), timeout=1)
    quotas = {}
    for key, future in lookups.items():
        if future.done() and not future.cancelled():
            try:
                quotas[key] = future.result()
            except Exception as exc:
                LOGGER.warning("quota unavailable for dynamic account %s: %s", key, redact_secrets(str(exc))[-300:])
    for job in pending:
        # The same lock also serializes create-time reservations. No network I/O inside it.
        with app.state.storage_budget.lock:
            current = db.get_job(job["job_id"])
            if not current or current["status"] != "queued" or not awaiting_assignment(current):
                continue
            allowed = current_accounts(current, app.state.auth_store)
            if not allowed:
                db.update_job_if_status(current["job_id"], {"queued"}, status="failed",
                                        error="no originally allowed Kaggle account remains authorized")
                continue
            occupied = occupied_accounts(app)
            available = []
            for key, username in allowed.items():
                quota = quotas.get(key)
                if occupied.get(username.lower(), 0) >= settings.account_concurrency or not isinstance(quota, dict):
                    continue
                if not quota.get("available"):
                    continue
                remaining = next((float(item.get("remaining_hours") or 0)
                                  for item in quota.get("accelerators", [])
                                  if item.get("resource", "").upper() == "GPU"), 0)
                if remaining > 0:
                    available.append((-remaining, key, username))
            if not available:
                db.update_job_if_status(current["job_id"], {"queued"},
                                        queue_reason="waiting for an idle authorized account with GPU quota")
                continue
            _, key, username = min(available)
            bound = db.update_job_if_status(
                current["job_id"], {"queued"}, require_not_canceled=True,
                kaggle_key_id=key, assignment_state="bound",
                dataset_ref=rewrite_ref_owner(current["dataset_ref"], username),
                kernel_ref=rewrite_ref_owner(current["kernel_ref"], username),
                queue_reason="waiting for an available worker",
            )
            if bound:
                db.append_log(current["job_id"], f"scheduler bound job to idle account {key}")
                app.state.queue.put_nowait({"action": "process", "job_id": current["job_id"]})


async def scheduler_loop(app: FastAPI) -> None:
    while not app.state.shutdown_event.is_set():
        app.state.scheduler_event.clear()
        try:
            await schedule_pending_jobs(app)
        except Exception:
            LOGGER.exception("dynamic account scheduling failed; pending jobs will be retried")
        if app.state.shutdown_event.is_set():
            return
        try:
            async with asyncio.timeout(15):
                await app.state.scheduler_event.wait()
        except TimeoutError:
            pass

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import weakref
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args

import aiofiles
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app.archive import (
    ArchiveError,
    assemble_archive,
    expected_chunk_count,
    safe_extract_zip,
    sha256_file,
    validate_chunk_index,
)
from app.auth_config import AuthConfigError, AuthSelectionError, AuthStore, RelayPrincipal, bearer_token
from app.config import Settings
from app.capacity import CapacityError, StorageBudget
from app.quota_cache import QuotaCache
from app.scheduler import awaiting_assignment, eligible_accounts, scheduler_loop
from app.database import RelayDb
from app.kaggle_adapter import KaggleAdapter, KaggleAdapterInterrupted
from app.schemas import (
    ChunkResponse,
    CreateKaggleKeyRequest,
    CreateJobRequest,
    CreateRelayTokenRequest,
    HealthResponse,
    JobProgressRequest,
    JobResponse,
    JobStatus,
    UiLoginRequest,
    UpdateKaggleKeyRequest,
    UpdateRelayTokenPermissionsRequest,
)
from app.security import (
    AuthFailureLimiter,
    auth_source,
    is_same_origin_request,
    redact_secrets,
)
from app.ui_auth import (
    authenticate_ui_session,
    create_ui_session_cookie,
    delete_ui_session_cookie,
    set_ui_session_cookie,
    ui_session_max_age_seconds,
)
from app.worker import (
    TERMINAL_JOB_STATUSES,
    has_ready_dataset_cache,
    process_job,
    resume_kernel_job,
    rewrite_ref_owner,
    validate_kernel_payload,
    validate_payloads,
)

VERSION = "0.1.0"
LOGGER = logging.getLogger("uvicorn.error")
AUTH_CONFIG_LOCK = threading.RLock()
KAGGLE_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
JOB_STATUS_VALUES = set(get_args(JobStatus))
RUNNING_JOB_STATUSES = {
    "assembling",
    "queued",
    "uploading_dataset",
    "waiting_dataset",
    "pushing_kernel",
    "waiting_kernel",
    "cancel_requested",
    "downloading_output",
}
ACTIVE_JOB_STATUSES = JOB_STATUS_VALUES - TERMINAL_JOB_STATUSES
UNSAFE_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class JobLockedFileResponse(FileResponse):
    def __init__(self, *args, job_lock: asyncio.Lock, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_lock = job_lock

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job_lock.release()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> RelayDb:
    return request.app.state.db


def get_auth_store(request: Request) -> AuthStore:
    return request.app.state.auth_store


def auth_limit_key(request: Request, channel: str) -> str:
    trusted_proxy_ips = request.app.state.settings.trusted_proxy_ips
    return f"{channel}:{auth_source(request, trusted_proxy_ips)}"


def reject_if_auth_blocked(limiter: AuthFailureLimiter, key: str) -> None:
    retry_after = limiter.retry_after(key)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail="too many authentication failures",
            headers={"Retry-After": str(retry_after)},
        )


def record_auth_failure(limiter: AuthFailureLimiter, key: str) -> None:
    retry_after = limiter.record_failure(key)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail="too many authentication failures",
            headers={"Retry-After": str(retry_after)},
        )


async def require_auth(
    request: Request,
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
    auth_store: AuthStore = Depends(get_auth_store),
) -> RelayPrincipal:
    token = bearer_token(authorization)
    if token:
        limiter = request.app.state.auth_failure_limiter
        limit_key = auth_limit_key(request, "bearer")
        reject_if_auth_blocked(limiter, limit_key)
        principal = auth_store.authenticate_token(token)
        if principal:
            limiter.clear(limit_key)
            return principal
        record_auth_failure(limiter, limit_key)
        raise HTTPException(status_code=401, detail="unauthorized")

    principal = authenticate_ui_session(request, settings, auth_store)
    if principal:
        if request.method.upper() in UNSAFE_HTTP_METHODS and not is_same_origin_request(
            request,
            settings.public_origin,
        ):
            raise HTTPException(status_code=403, detail="same-origin request required")
        return principal
    raise HTTPException(status_code=401, detail="unauthorized")


def selection_error(exc: AuthSelectionError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def can_access_job(job: dict, principal: RelayPrincipal, auth_store: AuthStore) -> bool:
    key_access = auth_store.can_access_key(principal, job.get("kaggle_key_id", ""))
    if awaiting_assignment(job):
        key_access = any(auth_store.can_access_key(principal, key) for key in eligible_accounts(job))
    if not key_access:
        return False

    job_owner = str(job.get("relay_token_id") or "").strip()
    if auth_store.legacy:
        return not job_owner or job_owner == principal.id
    if job_owner:
        return principal.allow_all_keys or job_owner == principal.id
    return principal.allow_all_keys


def require_job_access(job: dict, principal: RelayPrincipal, auth_store: AuthStore) -> None:
    if not can_access_job(job, principal, auth_store):
        raise HTTPException(status_code=404, detail="job not found")


def get_authorized_job(
    db: RelayDb,
    job_id: str,
    principal: RelayPrincipal,
    auth_store: AuthStore,
) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    require_job_access(job, principal, auth_store)
    return job


def split_query_values(values: list[str] | None) -> set[str]:
    result: set[str] = set()
    for value in values or []:
        for item in str(value or "").split(","):
            item = item.strip()
            if item:
                result.add(item)
    return result


def status_filter_for_list(status_values: list[str] | None, active: bool) -> set[str] | None:
    statuses = split_query_values(status_values)
    invalid = sorted(statuses - JOB_STATUS_VALUES)
    if invalid:
        raise HTTPException(status_code=400, detail=f"unknown job status: {', '.join(invalid)}")
    if active:
        return statuses & ACTIVE_JOB_STATUSES if statuses else set(ACTIVE_JOB_STATUSES)
    return statuses or None


def job_response(db: RelayDb, job_id: str, retention_hours: int = 168) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_response(db, job, retention_hours)


def artifact_download_metadata(job: dict, retention_hours: int = 168) -> dict:
    filename = f"{job['job_id']}-artifacts.zip"
    metadata = {
        "can_download": False,
        "artifact_size": None,
        "artifact_filename": filename,
        "download_unavailable_reason": "",
        "download_unavailable_code": "",
        "artifact_expires_at": None,
    }
    if job["status"] not in {"complete", "canceled"}:
        metadata["download_unavailable_reason"] = "job is not complete"
        metadata["download_unavailable_code"] = "not_ready"
        return metadata
    if not job.get("artifact_path"):
        metadata["download_unavailable_reason"] = "artifact path is missing"
        metadata["download_unavailable_code"] = "missing"
        if job.get("kaggle_output") == "expired by relay retention cleanup":
            metadata["download_unavailable_reason"] = "expired by relay retention cleanup"
            metadata["download_unavailable_code"] = "expired"
        return metadata

    artifact_path = Path(job["artifact_path"])
    try:
        stat = artifact_path.stat()
    except FileNotFoundError:
        metadata["download_unavailable_reason"] = "artifact file is missing"
        metadata["download_unavailable_code"] = "missing"
        return metadata
    except OSError:
        metadata["download_unavailable_reason"] = "artifact file is inaccessible"
        metadata["download_unavailable_code"] = "inaccessible"
        return metadata
    if not artifact_path.is_file():
        metadata["download_unavailable_reason"] = "artifact path is not a file"
        metadata["download_unavailable_code"] = "missing"
        return metadata

    metadata["can_download"] = True
    metadata["artifact_size"] = stat.st_size
    if job.get("completed_at") is not None:
        metadata["artifact_expires_at"] = job["completed_at"] + retention_hours * 3600
    return metadata


def dataset_download_metadata(job: dict, jobs_dir: Path) -> dict:
    code = ""
    if job.get("cleaned_at") is not None:
        code = "expired"
    elif job["status"] in {"receiving", "assembling"}:
        code = "not_ready"
    else:
        path = jobs_dir / job["job_id"] / "archives" / "dataset.zip"
        try:
            if not path.is_file():
                code = "missing"
            elif path.stat().st_size != job["dataset_size"]:
                code = "invalid"
        except OSError:
            code = "inaccessible"
    return {
        "can_download_dataset": not code,
        "dataset_download_unavailable_code": code,
    }


def job_to_response(db: RelayDb, job: dict, retention_hours: int = 168) -> JobResponse:
    job_id = job["job_id"]
    dataset_cache_hit = has_ready_dataset_cache(
        db,
        job["dataset_ref"],
        job["payload_hash"],
        kaggle_key_id=job.get("kaggle_key_id", ""),
    )
    if awaiting_assignment(job):
        dataset_cache_hit = False
    return JobResponse(
        **RelayDb.to_response(
            {
                **job,
                "eligible_accounts": eligible_accounts(job),
                "callback_enabled": bool(job.get("callback_token_sha256")),
                "cancel_requested": bool(job.get("cancel_requested_at")),
                "upload_expires_at": (
                    min(
                        float(job.get("upload_activity_at") or job["created_at"])
                        + getattr(db, "receiving_retention_hours", 168) * 3600,
                        float(job["created_at"]) + getattr(db, "receiving_timeout_hours", 3) * 3600,
                    ) if job["status"] == "receiving" else None
                ),
                **artifact_download_metadata(job, retention_hours),
                **dataset_download_metadata(job, db.path.parent / "jobs"),
                "dataset_cache_hit": dataset_cache_hit,
                "dataset_upload_required": not dataset_cache_hit,
            },
            db.accepted_chunks(job_id),
            db.recent_logs(job_id),
        )
    )


def public_allowed_key_ids(auth_store: AuthStore, principal: RelayPrincipal) -> list[str]:
    return auth_store.allowed_key_ids(principal)


def public_kaggle_keys(auth_store: AuthStore, principal: RelayPrincipal) -> list[dict]:
    if not principal.can_view_keys:
        return []
    allowed_key_ids = public_allowed_key_ids(auth_store, principal)
    if auth_store.legacy:
        return [{"id": "", "username": "", "credential_source": "environment"}]

    summaries = []
    kaggle_keys = getattr(auth_store, "_kaggle_keys", {})
    for key_id in allowed_key_ids:
        credentials = kaggle_keys.get(key_id)
        if not credentials:
            continue
        if credentials.config_dir:
            credential_source = "config_dir"
        elif credentials.api_token:
            credential_source = "api_token"
        elif credentials.username and credentials.key:
            credential_source = "username_key"
        else:
            credential_source = "unknown"
        summaries.append(
            {
                "id": credentials.id,
                "username": credentials.username,
                "credential_source": credential_source,
            }
        )
    return summaries


def public_relay_tokens(auth_store: AuthStore, principal: RelayPrincipal) -> list[dict]:
    tokens = []
    for _token_value, token_principal in getattr(auth_store, "_tokens", []):
        if not principal.management_admin and token_principal.id != principal.id:
            continue
        allowed = (
            "*"
            if token_principal.allow_all_keys
            else sorted(token_principal.allowed_kaggle_key_ids or [])
        )
        tokens.append(
            {
                "id": token_principal.id,
                "allowed_kaggle_key_ids": allowed if principal.can_view_keys else [],
                "can_view_keys": token_principal.can_view_keys,
                "current": token_principal.id == principal.id,
                "management": token_principal.management_admin,
            }
        )
    return tokens


def auth_config_summary(auth_store: AuthStore, principal: RelayPrincipal) -> dict:
    allowed_key_ids = public_allowed_key_ids(auth_store, principal) if principal.can_view_keys else []
    return {
        "mode": "legacy" if auth_store.legacy else "multi_key",
        "principal_id": principal.id,
        "current_token_id": principal.id,
        "allowed_kaggle_key_ids": allowed_key_ids,
        "can_manage_auth": principal.management_admin and not auth_store.legacy,
        "can_view_keys": principal.can_view_keys,
        "management_token_configured": auth_store.management_token_configured,
        "relay_tokens": public_relay_tokens(auth_store, principal),
        "kaggle_keys": public_kaggle_keys(auth_store, principal),
    }


def quota_unavailable(error: str) -> dict:
    return {
        "available": False,
        "refresh_at": "",
        "accelerators": [],
        "error": redact_secrets(error)[-2000:],
    }


def kaggle_account_status(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    kaggle_key_id: str = "",
) -> dict:
    try:
        allowed = public_allowed_key_ids(auth_store, principal)
        if not kaggle_key_id.strip() and not auth_store.legacy and len(allowed) > 1:
            # Account discovery is not job admission. Dynamic clients need an
            # owner for provisional refs even when the whole pool must wait.
            named = [key for key in allowed if auth_store.credentials_for(key).username]
            choices = named or allowed
            candidates, _exhausted, _unavailable = quota_key_candidates(settings, auth_store, choices)
            resolved_key_id = max(candidates)[1] if candidates else choices[0]
        else:
            resolved_key_id = auth_store.resolve_kaggle_key_id(principal, kaggle_key_id)
        credentials = auth_store.credentials_for(resolved_key_id)
    except AuthSelectionError as exc:
        raise selection_error(exc) from exc

    adapter = KaggleAdapter(settings, lambda _message: None, credentials=credentials)
    account = adapter.account()
    if account.get("authenticated"):
        try:
            quota = adapter.quota()
        except Exception as exc:
            quota = quota_unavailable(str(exc))
    else:
        quota = quota_unavailable("kaggle authentication failed")
    return {"kaggle_key_id": resolved_key_id, **account, "quota": quota}


def kaggle_account_probe(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    kaggle_key_id: str = "",
) -> dict:
    try:
        resolved_key_id = auth_store.resolve_kaggle_key_id(principal, kaggle_key_id)
        credentials = auth_store.credentials_for(resolved_key_id)
    except AuthSelectionError as exc:
        raise selection_error(exc) from exc

    adapter = KaggleAdapter(settings, lambda _message: None, credentials=credentials)
    return {"kaggle_key_id": resolved_key_id, **adapter.probe_username_write_access()}


def quota_remaining_hours(quota: dict, preferred_resource: str = "GPU") -> float | None:
    if not quota.get("available"):
        return None
    accelerators = quota.get("accelerators") or []
    preferred = next(
        (
            item
            for item in accelerators
            if str(item.get("resource", "")).upper() == preferred_resource.upper()
        ),
        None,
    )
    if preferred is not None:
        return float(preferred.get("remaining_hours") or 0)
    if not accelerators:
        return 0
    return max(float(item.get("remaining_hours") or 0) for item in accelerators)


def quota_key_candidates(
    settings: Settings,
    auth_store: AuthStore,
    key_ids: list[str],
) -> tuple[list[tuple[float, str]], list[str], list[str]]:
    candidates: list[tuple[float, str]] = []
    exhausted: list[str] = []
    unavailable: list[str] = []

    lookups = {}
    cache = getattr(settings, "_quota_cache", None)
    if cache:
        for key_id in key_ids:
            credentials = auth_store.credentials_for(key_id)
            adapter = KaggleAdapter(settings, lambda _message: None, credentials=credentials)
            lookups[key_id] = cache.submit((credentials, settings.kaggle_cmd), adapter.quota)
    for key_id in key_ids:
        try:
            credentials = auth_store.credentials_for(key_id)
            quota = (lookups[key_id].result() if cache else
                     KaggleAdapter(settings, lambda _message: None, credentials=credentials).quota())
            remaining = quota_remaining_hours(quota)
        except Exception as exc:
            unavailable.append(f"{key_id}: {redact_secrets(str(exc))[-300:]}")
            continue

        if remaining is None:
            unavailable.append(f"{key_id}: quota unavailable")
        elif remaining > 0:
            candidates.append((remaining, key_id))
        else:
            exhausted.append(key_id)

    return candidates, exhausted, unavailable


def select_kaggle_key_candidates(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    preferred_owner: str = "",
) -> list[tuple[float, str]]:
    allowed_key_ids = public_allowed_key_ids(auth_store, principal)
    preferred_owner = preferred_owner.strip()
    all_exhausted: list[str] = []
    all_unavailable: list[str] = []

    if preferred_owner:
        preferred_key_ids = []
        fallback_key_ids = []
        for key_id in allowed_key_ids:
            credentials = auth_store.credentials_for(key_id)
            username = str(getattr(credentials, "username", "") or "").strip()
            if username.lower() == preferred_owner.lower():
                preferred_key_ids.append(key_id)
            elif username:
                fallback_key_ids.append(key_id)
            else:
                all_unavailable.append(f"{key_id}: username required to rewrite owner")

        if preferred_key_ids:
            candidates, exhausted, unavailable = quota_key_candidates(settings, auth_store, preferred_key_ids)
            if candidates:
                return candidates
            all_exhausted.extend(exhausted)
            all_unavailable.extend(unavailable)
        allowed_key_ids = fallback_key_ids

    candidates, exhausted, unavailable = quota_key_candidates(settings, auth_store, allowed_key_ids)
    all_exhausted.extend(exhausted)
    all_unavailable.extend(unavailable)

    if candidates:
        return candidates
    if all_exhausted:
        raise HTTPException(
            status_code=409,
            detail="no allowed kaggle key has remaining GPU quota",
        )
    detail = "unable to read quota for allowed kaggle keys"
    if all_unavailable:
        detail = f"{detail}: {'; '.join(all_unavailable)}"
    raise HTTPException(status_code=503, detail=detail)


def resolve_job_kaggle_candidates(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    requested_key_id: str = "",
    dataset_ref: str = "",
    kernel_ref: str = "",
) -> list[tuple[float, str]]:
    requested = str(requested_key_id or "").strip()
    if requested:
        return [(0, auth_store.resolve_kaggle_key_id(principal, requested))]
    if auth_store.legacy or len(public_allowed_key_ids(auth_store, principal)) <= 1:
        return [(0, auth_store.resolve_kaggle_key_id(principal, requested))]
    return select_kaggle_key_candidates(
        settings,
        auth_store,
        principal,
        preferred_owner=requested_owner_from_refs(dataset_ref, kernel_ref),
    )


def resolve_job_kaggle_key_id(settings, auth_store, principal, requested_key_id="", dataset_ref="", kernel_ref="") -> str:
    return max(resolve_job_kaggle_candidates(settings, auth_store, principal, requested_key_id, dataset_ref, kernel_ref))[1]


def account_key(auth_store: AuthStore, key_id: str) -> str:
    try:
        credentials = auth_store.credentials_for(key_id)
    except AuthSelectionError:
        # The worker reports a removed credential as a job failure, not a scheduler failure.
        credentials = None
    username = str(getattr(credentials, "username", "") or "").strip().lower()
    return f"user:{username}" if username else f"key:{key_id}"


def least_loaded_key(db: RelayDb, auth_store: AuthStore, candidates: list[tuple[float, str]]) -> str:
    loads = {}
    for job in db.list_jobs_by_status(ACTIVE_JOB_STATUSES):
        key = account_key(auth_store, job.get("kaggle_key_id", ""))
        loads[key] = loads.get(key, 0) + 1
    return min(candidates, key=lambda candidate: (loads.get(account_key(auth_store, candidate[1]), 0),
                                                 -candidate[0], candidate[1]))[1]

def ref_owner(value: str) -> str:
    ref = str(value or "").strip()
    parts = ref.split("/", 1)
    if len(parts) != 2:
        return ""
    return parts[0].strip()


def requested_owner_from_refs(dataset_ref: str, kernel_ref: str) -> str:
    dataset_owner = ref_owner(dataset_ref)
    kernel_owner = ref_owner(kernel_ref)
    if dataset_owner and kernel_owner and dataset_owner.lower() != kernel_owner.lower():
        raise HTTPException(
            status_code=400,
            detail=f"dataset_ref owner {dataset_owner} does not match kernel_ref owner {kernel_owner}",
        )
    return dataset_owner or kernel_owner


def final_job_refs(dataset_ref: str, kernel_ref: str, username: str) -> tuple[str, str]:
    username = str(username or "").strip()
    if not username:
        return dataset_ref, kernel_ref
    try:
        return rewrite_ref_owner(dataset_ref, username), rewrite_ref_owner(kernel_ref, username)
    except ArchiveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def session_summary(auth_store: AuthStore, principal: RelayPrincipal) -> dict:
    return {
        "authenticated": True,
        "principal_id": principal.id,
        "allowed_kaggle_key_ids": public_allowed_key_ids(auth_store, principal) if principal.can_view_keys else [],
        "can_view_keys": principal.can_view_keys,
        "can_manage_auth": principal.management_admin and not auth_store.legacy,
    }


def require_config_admin(settings: Settings, principal: RelayPrincipal) -> Path:
    if not settings.auth_config_path:
        raise HTTPException(status_code=400, detail="RELAY_AUTH_CONFIG is required")
    if not principal.management_admin:
        raise HTTPException(status_code=403, detail="admin permission is required")
    return Path(settings.auth_config_path)


def read_auth_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="failed to read auth config") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="auth config is not valid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="auth config must be a JSON object")
    data.setdefault("relay_tokens", [])
    data.setdefault("kaggle_keys", [])
    if not isinstance(data["relay_tokens"], list) or not isinstance(data["kaggle_keys"], list):
        raise HTTPException(status_code=500, detail="auth config lists are invalid")
    return data


def validate_and_write_auth_config(path: Path, data: dict, admin_token: str = "") -> AuthStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp_path, 0o600)
        new_store = AuthStore.from_file(tmp_path, admin_token=admin_token)
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
        return new_store
    except AuthConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def validate_kaggle_username(username: str) -> str:
    username = username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="kaggle username is required")
    if not KAGGLE_USERNAME_RE.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="kaggle username must be the profile slug from kaggle.com, not the display name",
        )
    return username


def add_kaggle_key_config(settings: Settings, principal: RelayPrincipal, payload: CreateKaggleKeyRequest) -> AuthStore:
    path = require_config_admin(settings, principal)
    key_id = payload.id.strip()
    entry = {"id": key_id}
    username = validate_kaggle_username(payload.username)
    key = payload.key.strip()
    api_token = payload.api_token.strip()
    config_dir = payload.config_dir.strip()
    if key.upper().startswith("KGAT_"):
        raise HTTPException(status_code=400, detail="KGAT token must be provided as api_token, not key")
    entry["username"] = username
    if key:
        entry["key"] = key
    if api_token:
        entry["api_token"] = api_token
    if config_dir:
        entry["config_dir"] = config_dir
    if not (key or api_token or config_dir):
        raise HTTPException(status_code=400, detail="kaggle credentials are required")

    with AUTH_CONFIG_LOCK:
        data = read_auth_config(path)
        if any(str(item.get("id", "")).strip() == key_id for item in data["kaggle_keys"] if isinstance(item, dict)):
            raise HTTPException(status_code=409, detail="kaggle key id already exists")
        data["kaggle_keys"].append(entry)
        return validate_and_write_auth_config(path, data, settings.admin_token)


def update_kaggle_key_config(
    settings: Settings,
    principal: RelayPrincipal,
    key_id: str,
    payload: UpdateKaggleKeyRequest,
) -> AuthStore:
    path = require_config_admin(settings, principal)
    key_id = key_id.strip()
    if not key_id:
        raise HTTPException(status_code=400, detail="kaggle key id is required")
    username = validate_kaggle_username(payload.username)
    key = payload.key.strip()
    api_token = payload.api_token.strip()
    config_dir = payload.config_dir.strip()
    if key.upper().startswith("KGAT_"):
        raise HTTPException(status_code=400, detail="KGAT token must be provided as api_token, not key")

    with AUTH_CONFIG_LOCK:
        data = read_auth_config(path)
        index = -1
        existing: dict | None = None
        for candidate_index, item in enumerate(data["kaggle_keys"]):
            if isinstance(item, dict) and str(item.get("id", "")).strip() == key_id:
                index = candidate_index
                existing = dict(item)
                break
        if existing is None:
            raise HTTPException(status_code=404, detail="kaggle key id not found")

        existing["id"] = key_id
        existing["username"] = username
        if key or api_token or config_dir:
            for field in ("key", "api_token", "config_dir"):
                existing.pop(field, None)
            if key:
                existing["key"] = key
            if api_token:
                existing["api_token"] = api_token
            if config_dir:
                existing["config_dir"] = config_dir
        if not (
            str(existing.get("key", "") or "").strip()
            or str(existing.get("api_token", "") or "").strip()
            or str(existing.get("config_dir", "") or "").strip()
        ):
            raise HTTPException(status_code=400, detail="kaggle credentials are required")

        data["kaggle_keys"][index] = existing
        return validate_and_write_auth_config(path, data, settings.admin_token)


def add_relay_token_config(settings: Settings, principal: RelayPrincipal, payload: CreateRelayTokenRequest) -> AuthStore:
    path = require_config_admin(settings, principal)
    token_id = payload.id.strip()
    token = payload.token.strip()
    allowed_ids = [value.strip() for value in payload.allowed_kaggle_key_ids if value.strip()]
    allowed: str | list[str] = "*" if payload.allow_all_kaggle_keys else allowed_ids
    if not payload.allow_all_kaggle_keys and not allowed_ids:
        raise HTTPException(status_code=400, detail="allowed_kaggle_key_ids is required")

    with AUTH_CONFIG_LOCK:
        data = read_auth_config(path)
        if any(str(item.get("id", "")).strip() == token_id for item in data["relay_tokens"] if isinstance(item, dict)):
            raise HTTPException(status_code=409, detail="relay token id already exists")
        if any(str(item.get("token", "")).strip() == token for item in data["relay_tokens"] if isinstance(item, dict)):
            raise HTTPException(status_code=409, detail="relay token already exists")
        data["relay_tokens"].append(
            {
                "id": token_id,
                "token": token,
                "allowed_kaggle_key_ids": allowed,
                "can_view_keys": payload.can_view_keys,
            }
        )
        return validate_and_write_auth_config(path, data, settings.admin_token)


def update_relay_token_permissions(settings: Settings, principal: RelayPrincipal, token_id: str,
                                   payload: UpdateRelayTokenPermissionsRequest) -> AuthStore:
    path = require_config_admin(settings, principal)
    with AUTH_CONFIG_LOCK:
        data = read_auth_config(path)
        token = next((item for item in data["relay_tokens"] if item.get("id") == token_id), None)
        if token is None:
            raise HTTPException(status_code=404, detail="relay token id not found")
        token["can_view_keys"] = payload.can_view_keys
        return validate_and_write_auth_config(path, data, settings.admin_token)


def require_key_view(principal: RelayPrincipal) -> None:
    if not principal.can_view_keys:
        raise HTTPException(status_code=403, detail="key viewing permission is required")


def authorize_job_callback(job: dict, authorization: str, auth_store: AuthStore) -> bool:
    token = bearer_token(authorization)
    if not token:
        return False
    principal = auth_store.authenticate_token(token)
    if principal and can_access_job(job, principal, auth_store):
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
    if job.get("scheduling_mode") == "dynamic" and (
        awaiting_assignment(job) or job["status"] in {"receiving", "assembling", "queued"}
    ):
        raise HTTPException(status_code=409, detail="job has not been submitted to Kaggle")
    data = payload.model_dump()
    clean_message = redact_secrets(callback_log_message(data))[-8000:]
    if clean_message:
        db.append_log(job["job_id"], clean_message)
    for _attempt in range(5):
        current = db.get_job(job["job_id"])
        if not current:
            return
        updates = {
            "kernel_status": json.dumps(data, ensure_ascii=False, sort_keys=True),
            "kaggle_output": clean_message[-4000:],
            "progress": progress_from_callback(current, payload),
        }
        if current["status"] not in TERMINAL_JOB_STATUSES:
            updates["status"] = (
                "cancel_requested"
                if current.get("cancel_requested_at")
                else "waiting_kernel"
            )
        if db.update_job_if_status(
            job["job_id"],
            {current["status"]},
            **updates,
        ):
            return
    raise RuntimeError("job status changed repeatedly while applying progress callback")


def request_job_cancel(db: RelayDb, job: dict) -> None:
    for _attempt in range(5):
        status = str(job.get("status") or "")
        if status in {"complete", "failed"}:
            raise HTTPException(status_code=409, detail=f"job is already {status}")
        if status in {"cancel_requested", "canceled"}:
            return

        stamp = time.time()
        reason = "cancel requested"
        updates = {
            "cancel_requested_at": stamp,
            "cancel_reason": reason,
            "error": "",
        }
        if status in {"receiving", "assembling", "queued"}:
            updates.update(
                {
                    "status": "canceled",
                    "error": "canceled before submission",
                }
            )
        else:
            updates["status"] = "cancel_requested"
        if db.update_job_if_status(job["job_id"], {status}, **updates):
            db.append_log(job["job_id"], reason)
            return
        job = db.get_job(job["job_id"])
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
    raise HTTPException(status_code=409, detail="job status changed; retry cancellation")


def assemble_and_validate_job(settings: Settings, db: RelayDb, auth_store: AuthStore, job: dict) -> None:
    job_id = job["job_id"]
    job_dir = settings.jobs_dir / job_id
    archives_dir = job_dir / "archives"
    extracted_dir = job_dir / "extracted"
    credentials = None if awaiting_assignment(job) else auth_store.credentials_for(job.get("kaggle_key_id", ""))
    kernel_zip = archives_dir / "kernel.zip"
    dataset_cache_hit = has_ready_dataset_cache(
        db,
        job["dataset_ref"],
        job["payload_hash"],
        kaggle_key_id=job.get("kaggle_key_id", ""),
    )
    if awaiting_assignment(job):
        dataset_cache_hit = False

    budget = getattr(settings, "_storage_budget", None)
    inputs_size = job["kernel_size"] + (0 if dataset_cache_hit else job["dataset_size"])
    if budget:
        budget.reserve(job_id, inputs_size * 2)
    if not dataset_cache_hit:
        dataset_zip = archives_dir / "dataset.zip"
        assemble_archive(
            job_dir / "chunks" / "dataset",
            dataset_zip,
            job["dataset_size"],
            job["chunk_size"],
            job["dataset_archive_sha256"],
            space_check=budget.check_free if budget else None,
        )
        if budget:
            budget.consume(job_id, job["dataset_size"])
    assemble_archive(
        job_dir / "chunks" / "kernel",
        kernel_zip,
        job["kernel_size"],
        job["chunk_size"],
        job["kernel_archive_sha256"],
        space_check=budget.check_free if budget else None,
    )
    inputs = [("kernel", kernel_zip)]
    if not dataset_cache_hit:
        inputs.append(("dataset", dataset_zip))
    if budget:
        sizes = []
        for kind, path in inputs:
            with zipfile.ZipFile(path) as archive:
                sizes.append(sum(((info.file_size + 4095) // 4096 + 1) * 4096 for info in archive.infolist()))
        budget.reserve(job_id, sum(sizes))
    for kind, path in inputs:
        if budget:
            budget.check_free()
        safe_extract_zip(path, extracted_dir / kind, job[f"{kind}_size"],
                         space_check=budget.check_free if budget else None)
    if budget:
        budget.consume(job_id, sum(sizes))
    if dataset_cache_hit:
        validate_kernel_payload(
            extracted_dir / "kernel",
            job["kernel_ref"],
            credentials,
            dataset_ref=job["dataset_ref"],
        )
    else:
        validate_payloads(
            extracted_dir / "dataset",
            extracted_dir / "kernel",
            job["dataset_ref"],
            job["kernel_ref"],
            credentials,
        )


def append_internal_log(db: RelayDb, job_id: str, message: str) -> None:
    clean = redact_secrets(message)
    db.append_log(job_id, clean)
    db.update_job(job_id, kaggle_output=clean[-4000:])


def recovery_adapter(settings: Settings, db: RelayDb, auth_store: AuthStore, job: dict) -> KaggleAdapter:
    credentials = auth_store.credentials_for(job.get("kaggle_key_id", ""))
    return KaggleAdapter(settings, lambda message: append_internal_log(db, job["job_id"], message), credentials=credentials)


def not_found_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return any(marker in detail for marker in ["404", "not found", "no kernel", "not exist"])


def kernel_was_submitted(db: RelayDb, job: dict) -> bool:
    try:
        if float(job.get("progress") or 0) >= 60:
            return True
    except (TypeError, ValueError):
        pass
    logs = "\n".join(db.recent_logs(job["job_id"], limit=500)).lower()
    return "kernel version" in logs and "successfully pushed" in logs


def kernel_submission_may_be_in_flight(job: dict) -> bool:
    try:
        return float(job.get("progress") or 0) >= 45
    except (TypeError, ValueError):
        return False


def fail_recovered_job(db: RelayDb, job_id: str, message: str) -> None:
    clean = redact_secrets(message)
    LOGGER.warning("restart recovery failed job %s: %s", job_id, clean)
    db.update_job(job_id, status="failed", progress=0, error=clean)
    db.append_log(job_id, clean)


def recovery_item_description(item: dict) -> str:
    action = item.get("action", "process")
    if action == "resume_kernel" and item.get("final_status"):
        return f"{action} final_status={item['final_status']}"
    return action


def recover_job_after_restart(
    settings: Settings,
    db: RelayDb,
    auth_store: AuthStore,
    job: dict,
) -> dict | None:
    job_id = job["job_id"]
    status = str(job.get("status") or "")
    append_internal_log(db, job_id, f"recovering job after relay restart from status {status}")
    LOGGER.info("restart recovery inspecting job %s from status %s", job_id, status)

    if status == "queued":
        append_internal_log(db, job_id, "restart recovery action: requeue full process")
        return {"action": "process", "job_id": job_id}

    if status in {"waiting_kernel", "downloading_output"}:
        append_internal_log(db, job_id, "restart recovery action: resume kernel finish path")
        return {"action": "resume_kernel", "job_id": job_id}

    if status == "cancel_requested":
        if kernel_was_submitted(db, job):
            append_internal_log(
                db,
                job_id,
                "restart recovery action: resume kernel finish path with canceled final status",
            )
            return {"action": "resume_kernel", "job_id": job_id, "final_status": "canceled"}
        if kernel_submission_may_be_in_flight(job):
            append_internal_log(db, job_id, "restart recovery action: probe canceled kernel visibility")
            adapter = recovery_adapter(settings, db, auth_store, job)
            try:
                adapter.kernel_status(job["kernel_ref"])
            except Exception as exc:
                if not not_found_error(exc):
                    fail_recovered_job(
                        db,
                        job_id,
                        "restart after cancellation could not verify whether the kernel was submitted; "
                        f"inspect Kaggle before resubmitting: {exc}",
                    )
                    return None
            else:
                append_internal_log(
                    db,
                    job_id,
                    "restart recovery action: resume visible kernel with canceled final status",
                )
                return {"action": "resume_kernel", "job_id": job_id, "final_status": "canceled"}
        append_internal_log(db, job_id, "restart recovery action: mark canceled before Kaggle kernel submission")
        db.update_job(job_id, status="canceled", error=job.get("cancel_reason") or "cancel requested")
        db.append_log(job_id, "canceled during restart recovery before Kaggle kernel submission")
        return None

    if status == "pushing_kernel":
        append_internal_log(db, job_id, "restart recovery action: probe kernel visibility")
        adapter = recovery_adapter(settings, db, auth_store, job)
        try:
            adapter.kernel_status(job["kernel_ref"])
        except Exception as exc:
            if not_found_error(exc):
                fail_recovered_job(
                    db,
                    job_id,
                    "restart during kernel push before submission could be verified; resubmit job",
                )
            else:
                fail_recovered_job(
                    db,
                    job_id,
                    f"restart during kernel push and kernel status could not be verified; resubmit job: {exc}",
                )
            return None
        append_internal_log(
            db,
            job_id,
            "restart recovery action: resume kernel finish path after verified kernel submission",
        )
        return {"action": "resume_kernel", "job_id": job_id}

    if status in {"uploading_dataset", "waiting_dataset"}:
        fail_recovered_job(
            db,
            job_id,
            "restart during dataset upload/wait lost the exact uploaded version; "
            "resubmit job",
        )
        return None

    if status == "assembling":
        append_internal_log(db, job_id, "restart recovery action: resume archive assembly then process")
        try:
            assemble_and_validate_job(settings, db, auth_store, job)
        except Exception as exc:
            fail_recovered_job(db, job_id, redact_secrets(str(exc)))
            return None
        db.update_job(job_id, status="queued", progress=15, error="")
        return {"action": "process", "job_id": job_id}

    return None


async def recover_incomplete_jobs(app: FastAPI) -> None:
    jobs = app.state.db.list_jobs_by_status(RUNNING_JOB_STATUSES)
    LOGGER.info("startup recovery scan found %s incomplete job(s)", len(jobs))
    items = []
    for job in jobs:
        item = await asyncio.to_thread(
            recover_job_after_restart,
            app.state.settings,
            app.state.db,
            app.state.auth_store,
            job,
        )
        if item:
            LOGGER.info(
                "startup recovery queued job %s action %s",
                item["job_id"],
                recovery_item_description(item),
            )
            items.append(item)
    for item in sorted(items, key=lambda item: item.get("action") != "resume_kernel"):
        if not awaiting_assignment(app.state.db.get_job(item["job_id"])):
            await app.state.queue.put(item)


def normalize_queue_item(item) -> dict:
    if isinstance(item, dict):
        return item
    return {"action": "process", "job_id": item}


def job_submission_lock(app: FastAPI, job_id: str) -> asyncio.Lock:
    lock = app.state.job_submission_locks.get(job_id)
    if lock is None:
        lock = asyncio.Lock()
        app.state.job_submission_locks[job_id] = lock
    return lock


async def upload_body(request: Request, idle_seconds: int):
    stream = request.stream().__aiter__()
    while True:
        try:
            async with asyncio.timeout(idle_seconds):
                part = await anext(stream)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise HTTPException(status_code=408, detail="upload body timed out; retry this chunk") from exc
        yield part


def mark_worker_exception(db: RelayDb, job_id: str, exc: Exception) -> None:
    message = redact_secrets(str(exc))
    db.append_log(job_id, f"worker action failed: {message}")
    job = db.get_job(job_id)
    if job and job.get("status") not in TERMINAL_JOB_STATUSES:
        db.finalize_job(job_id, "failed", progress=0, error=message)


async def run_thread_to_completion(
    app: FastAPI,
    operation,
    *args,
    executor=None,
) -> tuple[bool, BaseException | None]:
    executor = executor or getattr(app.state, "maintenance_executor", None)
    thread_task = asyncio.get_running_loop().run_in_executor(executor, operation, *args)
    app.state.worker_thread_tasks.add(thread_task)
    was_cancelled = False
    try:
        while True:
            try:
                await asyncio.shield(thread_task)
                break
            except asyncio.CancelledError:
                was_cancelled = True
                LOGGER.info("waiting for in-flight thread operation to finish")
            except BaseException:
                break
        try:
            thread_task.result()
        except BaseException as exc:
            return was_cancelled, exc
        return was_cancelled, None
    finally:
        app.state.worker_thread_tasks.discard(thread_task)


async def run_worker_item(app: FastAPI, item: dict) -> None:
    job_id = item["job_id"]
    action = item.get("action", "process")
    if action == "resume_kernel":
        args = (
            app.state.settings,
            app.state.db,
            job_id,
            app.state.auth_store,
            item.get("final_status"),
        )
        operation = resume_kernel_job
    else:
        args = (
            app.state.settings,
            app.state.db,
            job_id,
            app.state.auth_store,
        )
        operation = process_job

    was_cancelled, operation_error = await run_thread_to_completion(
        app,
        operation,
        *args,
        executor=app.state.worker_executor,
    )
    if operation_error is not None:
        if was_cancelled and isinstance(operation_error, KaggleAdapterInterrupted):
            raise asyncio.CancelledError
        if was_cancelled and isinstance(operation_error, Exception):
            mark_worker_exception(app.state.db, job_id, operation_error)
            raise asyncio.CancelledError
        raise operation_error
    if was_cancelled:
        raise asyncio.CancelledError


def queue_item_expected_statuses(item: dict) -> set[str]:
    if item.get("action") == "resume_kernel":
        return {
            "pushing_kernel",
            "waiting_kernel",
            "downloading_output",
            "cancel_requested",
        }
    return {"queued"}


async def worker_loop(app: FastAPI, worker_index: int = 0) -> None:
    LOGGER.info("relay worker %s started", worker_index)
    while True:
        item = normalize_queue_item(await app.state.queue.get())
        job_id = item["job_id"]
        job = app.state.db.get_job(job_id)
        expected_statuses = queue_item_expected_statuses(item)
        if not job or job.get("status") not in expected_statuses:
            LOGGER.warning(
                "skipping stale queued item for job %s in status %s",
                job_id,
                job.get("status") if job else "missing",
            )
            app.state.queue.task_done()
            continue
        if job_id in app.state.active_job_ids:
            LOGGER.warning("skipping duplicate queued item for active job %s", job_id)
            app.state.queue.task_done()
            continue
        if awaiting_assignment(job):
            app.state.scheduler_event.set()
            app.state.queue.task_done()
            continue
        account = account_key(app.state.auth_store, job.get("kaggle_key_id", ""))
        active = app.state.active_accounts.get(account, 0)
        # Recovered remote runs already consume Kaggle slots and must all be monitored.
        if item.get("action") != "resume_kernel" and active >= app.state.settings.account_concurrency:
            app.state.account_pending.setdefault(account, {})[job_id] = item
            app.state.db.update_job(job_id, queue_reason="waiting for an available Kaggle account slot")
            app.state.queue.task_done()
            continue
        app.state.active_job_ids.add(job_id)
        app.state.active_accounts[account] = active + 1
        app.state.db.update_job(job_id, queue_reason="")
        try:
            await run_worker_item(app, item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.error(
                "worker action failed for job %s (%s): %s",
                job_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
            mark_worker_exception(app.state.db, job_id, exc)
        finally:
            app.state.active_job_ids.discard(job_id)
            app.state.active_accounts[account] -= 1
            for waiting in app.state.account_pending.pop(account, {}).values():
                app.state.queue.put_nowait(waiting)
            app.state.scheduler_event.set()
            app.state.queue.task_done()


def delete_job_files_and_record(settings: Settings, db: RelayDb, job_id: str) -> None:
    paths = []
    for root in (settings.jobs_dir, settings.artifacts_dir):
        path = root / job_id
        if path.resolve().parent != root.resolve() or path.is_symlink():
            raise OSError("job directory is outside its storage root")
        paths.append(path)
    for path in paths:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            if path.exists():
                raise
    # Keep the record on cleanup failure so the user can retry.
    db.delete_job_record(job_id)


def cleanup_expired_job(settings: Settings, db: RelayDb, job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job.get("cleaned_at") is not None:
        return
    for path in (settings.jobs_dir / job_id, settings.artifacts_dir / job_id):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
    db.update_job(
        job_id,
        artifact_path="",
        kaggle_output="expired by relay retention cleanup",
        cleaned_at=time.time(),
        cleanup_due_at=None,
        reserved_bytes=0,
    )
    db.append_log(job_id, "expired by relay retention cleanup")


def expire_receiving_upload(db: RelayDb, job: dict) -> str:
    hours = getattr(db, "receiving_timeout_hours", 3)
    if job["status"] != "receiving" or time.time() < float(job["created_at"]) + hours * 3600:
        return ""
    error = f"upload timed out: incomplete after {hours} hours from job creation; submit a new job"
    if db.update_job_if_status(job["job_id"], {"receiving"}, status="failed", error=error,
                               cleanup_due_at=time.time()):
        return error
    return ""


def reject_expired_upload(db: RelayDb, job: dict) -> None:
    error = expire_receiving_upload(db, job)
    if error:
        raise HTTPException(status_code=409, detail=error)


async def cleanup_expired_jobs(app: FastAPI) -> None:
    settings = app.state.settings
    db = app.state.db
    cutoff = time.time() - settings.retention_hours * 60 * 60
    receiving_cutoff = time.time() - settings.receiving_retention_hours * 3600
    stale = set(db.stale_receiving(receiving_cutoff, time.time() - settings.receiving_timeout_hours * 3600))
    for job_id in dict.fromkeys(db.completed_before(cutoff) + sorted(stale)):
        async with job_submission_lock(app, job_id):
            job = db.get_job(job_id)
            if not job or job.get("cleaned_at") is not None:
                continue
            timed_out = expire_receiving_upload(db, job)
            # Mark the deadline even during a body stream, but never remove files
            # until the stream has released its handles.
            if job_id in app.state.active_uploads or job_id in app.state.active_job_ids:
                continue
            if job_id in stale and not timed_out:
                if job["status"] != "receiving" or job["upload_activity_at"] >= receiving_cutoff:
                    continue
                if not db.update_job_if_status(job_id, {"receiving"}, status="failed", error="upload expired after inactivity",
                                               cleanup_due_at=time.time()):
                    continue
            was_cancelled, operation_error = await run_thread_to_completion(
                app,
                cleanup_expired_job,
                settings,
                db,
                job_id,
            )
        if operation_error is not None:
            if was_cancelled:
                LOGGER.error(
                    "cleanup failed during shutdown: %s",
                    redact_secrets(str(operation_error)),
                )
                raise asyncio.CancelledError
            LOGGER.error("retention cleanup failed for job %s: %s", job_id, redact_secrets(str(operation_error)))
            continue
        if was_cancelled:
            raise asyncio.CancelledError


async def cleanup_loop(app: FastAPI) -> None:
    while True:
        try:
            await cleanup_expired_jobs(app)
            was_cancelled, error = await run_thread_to_completion(app, app.state.db.trim_logs)
            if was_cancelled:
                raise asyncio.CancelledError
            if error:
                raise error
        except Exception as exc:
            LOGGER.error("retention cleanup failed: %s", redact_secrets(str(exc)))
        await asyncio.sleep(60)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await recover_incomplete_jobs(app)
        app.state.worker_tasks = [
            asyncio.create_task(worker_loop(app, worker_index=index))
            for index in range(settings.worker_count)
        ]
        app.state.worker_task = app.state.worker_tasks[0]
        app.state.cleanup_task = asyncio.create_task(cleanup_loop(app))
        app.state.scheduler_task = asyncio.create_task(scheduler_loop(app))
        try:
            yield
        finally:
            app.state.shutdown_event.set()
            app.state.scheduler_task.cancel()
            await asyncio.gather(app.state.scheduler_task, return_exceptions=True)
            for worker_task in app.state.worker_tasks:
                worker_task.cancel()
            app.state.cleanup_task.cancel()
            await asyncio.gather(*app.state.worker_tasks, return_exceptions=True)
            try:
                await app.state.cleanup_task
            except asyncio.CancelledError:
                pass
            app.state.worker_executor.shutdown(wait=True)
            app.state.maintenance_executor.shutdown(wait=True)
            settings._quota_cache.shutdown()

    app = FastAPI(title="Kaggle Relay", version=VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = RelayDb(settings.db_path)
    app.state.db.max_logs_per_job = settings.max_logs_per_job
    app.state.db.receiving_retention_hours = settings.receiving_retention_hours
    app.state.db.receiving_timeout_hours = settings.receiving_timeout_hours
    app.state.storage_budget = StorageBudget(settings, app.state.db)
    settings._storage_budget = app.state.storage_budget
    settings._quota_cache = QuotaCache()
    app.state.auth_store = AuthStore.from_settings(settings)
    app.state.auth_failure_limiter = AuthFailureLimiter(
        settings.auth_failure_limit,
        settings.auth_failure_window_seconds,
        settings.auth_lockout_seconds,
    )
    app.state.queue = asyncio.Queue()
    app.state.worker_tasks = []
    app.state.worker_task = None
    app.state.worker_thread_tasks = set()
    app.state.worker_executor = ThreadPoolExecutor(max_workers=settings.worker_count, thread_name_prefix="relay-job")
    app.state.maintenance_executor = ThreadPoolExecutor(max_workers=settings.assembly_workers, thread_name_prefix="relay-files")
    app.state.assembly_slots = asyncio.Semaphore(settings.assembly_workers)
    app.state.active_accounts = {}
    app.state.account_pending = {}
    app.state.scheduler_event = asyncio.Event()
    app.state.active_uploads = {}
    app.state.user_uploads = {}
    app.state.upload_count = 0
    app.state.active_job_ids = set()
    app.state.shutdown_event = threading.Event()
    settings._shutdown_event = app.state.shutdown_event
    app.state.job_submission_locks = weakref.WeakValueDictionary()

    def static_file(name: str) -> Path:
        return Path(__file__).parent / "static" / name

    def ui_response(
        request: Request,
        settings: Settings,
        auth_store: AuthStore,
    ):
        principal = authenticate_ui_session(request, settings, auth_store)
        if not principal:
            return RedirectResponse("/login", status_code=303)
        return FileResponse(static_file("index.html"))

    @app.get("/", include_in_schema=False)
    def ui_index(
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
    ):
        return ui_response(request, settings, auth_store)

    @app.get("/ui", include_in_schema=False)
    def ui_alias(
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
    ):
        return ui_response(request, settings, auth_store)

    @app.get("/admin", include_in_schema=False)
    def admin_alias(
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
    ):
        return ui_response(request, settings, auth_store)

    @app.get("/login", include_in_schema=False)
    def login_page() -> FileResponse:
        return FileResponse(static_file("login.html"))

    @app.post("/v1/ui/login")
    def ui_login(
        payload: UiLoginRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> JSONResponse:
        if not is_same_origin_request(request, settings.public_origin):
            raise HTTPException(status_code=403, detail="same-origin request required")
        limiter = request.app.state.auth_failure_limiter
        limit_key = auth_limit_key(request, "ui-login")
        reject_if_auth_blocked(limiter, limit_key)
        principal = auth_store.authenticate_token(payload.token.strip())
        if not principal:
            record_auth_failure(limiter, limit_key)
            raise HTTPException(status_code=401, detail="invalid token")
        limiter.clear(limit_key)
        max_age = ui_session_max_age_seconds()
        response = JSONResponse(
            {
                "ok": True,
                "principal_id": principal.id,
                "allowed_kaggle_key_ids": public_allowed_key_ids(auth_store, principal) if principal.can_view_keys else [],
                "can_view_keys": principal.can_view_keys,
            }
        )
        set_ui_session_cookie(
            response,
            create_ui_session_cookie(settings, auth_store, principal, max_age),
            max_age,
            settings,
        )
        return response

    @app.post("/v1/ui/logout")
    def ui_logout(
        settings: Settings = Depends(get_settings),
        _principal: RelayPrincipal = Depends(require_auth),
    ) -> JSONResponse:
        response = JSONResponse({"ok": True})
        delete_ui_session_cookie(response, settings)
        return response

    @app.get("/v1/ui/session")
    def ui_session(
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> dict:
        principal = authenticate_ui_session(request, settings, auth_store)
        if not principal:
            return {"authenticated": False}
        return session_summary(auth_store, principal)

    @app.get("/v1/health", response_model=HealthResponse)
    def health(
        settings: Settings = Depends(get_settings),
        _principal: RelayPrincipal = Depends(require_auth),
    ) -> HealthResponse:
        usage = shutil.disk_usage(settings.storage_dir)
        return HealthResponse(
            status="ok",
            version=VERSION,
            storage_dir=str(settings.storage_dir),
            free_bytes=usage.free,
        )

    @app.get("/v1/auth/config")
    def auth_config(
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        return auth_config_summary(auth_store, principal)

    @app.post("/v1/auth/kaggle-keys")
    def create_auth_kaggle_key(
        payload: CreateKaggleKeyRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        new_store = add_kaggle_key_config(settings, principal, payload)
        request.app.state.auth_store = new_store
        return auth_config_summary(new_store, principal)

    @app.patch("/v1/auth/kaggle-keys/{kaggle_key_id}")
    def update_auth_kaggle_key(
        kaggle_key_id: str,
        payload: UpdateKaggleKeyRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        new_store = update_kaggle_key_config(settings, principal, kaggle_key_id, payload)
        request.app.state.auth_store = new_store
        return auth_config_summary(new_store, principal)

    @app.post("/v1/auth/relay-tokens")
    def create_auth_relay_token(
        payload: CreateRelayTokenRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        new_store = add_relay_token_config(settings, principal, payload)
        request.app.state.auth_store = new_store
        return auth_config_summary(new_store, principal)

    @app.patch("/v1/auth/relay-tokens/{token_id}")
    def patch_relay_token_permissions(
        token_id: str,
        payload: UpdateRelayTokenPermissionsRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        new_store = update_relay_token_permissions(settings, principal, token_id, payload)
        request.app.state.auth_store = new_store
        return auth_config_summary(new_store, principal)

    @app.get("/v1/kaggle/account")
    def kaggle_account(
        kaggle_key_id: str = "",
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        if kaggle_key_id.strip():
            require_key_view(principal)
        return kaggle_account_status(settings, auth_store, principal, kaggle_key_id)

    @app.post("/v1/kaggle/account/probe")
    def kaggle_account_write_probe(
        kaggle_key_id: str = "",
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        require_key_view(principal)
        return kaggle_account_probe(settings, auth_store, principal, kaggle_key_id)

    @app.get("/v1/kaggle/accounts")
    def kaggle_accounts(
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        require_key_view(principal)
        return {
            "accounts": [
                kaggle_account_status(settings, auth_store, principal, key_id)
                for key_id in public_allowed_key_ids(auth_store, principal)
            ],
        }

    @app.post("/v1/jobs", response_model=JobResponse)
    def create_job(
        payload: CreateJobRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> JobResponse:
        dynamic = payload.scheduling_mode == "dynamic" and not payload.kaggle_key_id and not auth_store.legacy
        accounts = {}
        try:
            if dynamic:
                requested_owner_from_refs(payload.dataset_ref, payload.kernel_ref)
                accounts = {key: auth_store.credentials_for(key).username
                            for key in auth_store.allowed_key_ids(principal)
                            if auth_store.credentials_for(key).username}
                if not accounts:
                    raise HTTPException(status_code=409, detail="dynamic scheduling requires an allowed account with a username")
                candidates = [(0, key) for key in accounts]
            else:
                candidates = resolve_job_kaggle_candidates(
                    settings, auth_store, principal, payload.kaggle_key_id,
                    payload.dataset_ref, payload.kernel_ref,
                )
        except AuthSelectionError as exc:
            raise selection_error(exc) from exc
        if max(payload.dataset_size, payload.kernel_size) > settings.max_archive_bytes or payload.chunk_size > settings.chunk_size:
            raise HTTPException(status_code=413, detail="archive or chunk exceeds the configured size limit")
        if max(expected_chunk_count(size, payload.chunk_size) for size in (payload.dataset_size, payload.kernel_size)) > 65536:
            raise HTTPException(status_code=413, detail="archive has too many chunks; use a larger chunk size")
        budget = request.app.state.storage_budget
        with budget.lock:
            kaggle_key_id = least_loaded_key(db, auth_store, candidates)
            credentials = auth_store.credentials_for(kaggle_key_id)
            dataset_ref, kernel_ref = final_job_refs(
                payload.dataset_ref, payload.kernel_ref,
                str(getattr(credentials, "username", "") or ""),
            )
            reserved_bytes = 3 * (payload.dataset_size + payload.kernel_size)
            try:
                budget.check_admission(principal.id, reserved_bytes)
            except CapacityError as exc:
                raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "30"}) from exc
            job_id = uuid.uuid4().hex
            (settings.jobs_dir / job_id / "chunks" / "dataset").mkdir(parents=True, exist_ok=True)
            (settings.jobs_dir / job_id / "chunks" / "kernel").mkdir(parents=True, exist_ok=True)
            values = {
                **payload.model_dump(), "dataset_ref": dataset_ref, "kernel_ref": kernel_ref,
                "job_id": job_id, "relay_token_id": principal.id, "kaggle_key_id": kaggle_key_id,
                "reserved_bytes": reserved_bytes,
                "scheduling_mode": "dynamic" if dynamic else "fixed",
                "assignment_state": "pending" if dynamic else "bound",
                "eligible_accounts": json.dumps(accounts),
                "callback_kernel_ref": payload.kernel_ref if dynamic else "",
            }
            db.create_job(values)
        return job_response(db, job_id, settings.retention_hours)

    @app.get("/v1/jobs/summary", response_model=dict[str, int])
    def job_summary(
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict[str, int]:
        key_filter = None if principal.allow_all_keys else set(auth_store.allowed_key_ids(principal))
        owner_filter = None if auth_store.legacy or principal.allow_all_keys else principal.id
        counts = db.job_status_counts(key_filter, owner_filter)
        return {
            "total": sum(counts.values()),
            "in_progress": sum(counts.get(status, 0) for status in ACTIVE_JOB_STATUSES - {"queued"}),
            **{status: counts.get(status, 0) for status in ("queued", "failed", "complete", "canceled")},
        }

    @app.get("/v1/jobs", response_model=list[JobResponse])
    def list_jobs(
        limit: int = Query(default=50, ge=1, le=200),
        active: bool = Query(default=False),
        status: list[str] | None = Query(default=None),
        q: str = Query(default="", max_length=200),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> list[JobResponse]:
        key_filter = None if principal.allow_all_keys else set(auth_store.allowed_key_ids(principal))
        owner_filter = None if auth_store.legacy or principal.allow_all_keys else principal.id
        status_filter = status_filter_for_list(status, active)
        jobs = db.list_jobs(
            kaggle_key_ids=key_filter,
            relay_token_id=owner_filter,
            statuses=status_filter,
            limit=limit,
            search=q,
        )
        return [job_to_response(db, job, settings.retention_hours) for job in jobs]

    @app.put(
        "/v1/jobs/{job_id}/archives/{archive_type}/chunks/{index}",
        response_model=ChunkResponse,
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
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> ChunkResponse:
        async with job_submission_lock(request.app, job_id):
            job = get_authorized_job(db, job_id, principal, auth_store)
            reject_expired_upload(db, job)
            if job["status"] != "receiving":
                raise HTTPException(status_code=409, detail="job is no longer receiving chunks")
            total_size = job[f"{archive_type}_size"]
            try:
                validate_chunk_index(index, total_size, job["chunk_size"])
            except ArchiveError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            expected_size = min(job["chunk_size"], total_size - index * job["chunk_size"])
            if x_chunk_size != expected_size:
                raise HTTPException(status_code=400, detail="invalid declared chunk size")
            if not re.fullmatch(r"[a-fA-F0-9]{64}", x_chunk_sha256):
                raise HTTPException(status_code=400, detail="invalid chunk sha256")
            x_chunk_sha256 = x_chunk_sha256.lower()

            existing = db.get_chunk(job_id, archive_type, index)
            chunk_dir = settings.jobs_dir / job_id / "chunks" / archive_type
            chunk_path = chunk_dir / f"{index}.part"
            if existing:
                if existing["sha256"] == x_chunk_sha256 and existing["size"] == x_chunk_size and chunk_path.exists():
                    db.touch_upload(job_id)
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
            tmp_path = chunk_dir / f"{index}.{uuid.uuid4().hex}.tmp"
            state = request.app.state
            if state.upload_count >= settings.max_parallel_uploads or state.user_uploads.get(principal.id, 0) >= 4:
                raise HTTPException(status_code=429, detail="upload slots are busy; retry this chunk", headers={"Retry-After": "1"})
            try:
                state.storage_budget.check_free(x_chunk_size)
            except CapacityError as exc:
                raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "30"}) from exc
            db.touch_upload(job_id)
            state.upload_count += 1
            state.user_uploads[principal.id] = state.user_uploads.get(principal.id, 0) + 1
            state.active_uploads[job_id] = state.active_uploads.get(job_id, 0) + 1
        # Receiving a body must not hold the job-wide submission lock.
        digest = hashlib.sha256()
        size = 0
        last_space_check = time.monotonic()
        try:
            async with aiofiles.open(tmp_path, "wb") as handle:
                async for part in upload_body(request, settings.upload_idle_seconds):
                    if time.time() >= float(job["created_at"]) + settings.receiving_timeout_hours * 3600:
                        async with job_submission_lock(request.app, job_id):
                            reject_expired_upload(db, job)
                        raise HTTPException(status_code=409, detail="job upload deadline exceeded")
                    if time.monotonic() - last_space_check >= 1:
                        try:
                            state.storage_budget.check_free(len(part))
                        except CapacityError as exc:
                            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "30"}) from exc
                        last_space_check = time.monotonic()
                    size += len(part)
                    if size > x_chunk_size:
                        raise HTTPException(status_code=400, detail="chunk larger than X-Chunk-Size")
                    digest.update(part)
                    await handle.write(part)
            actual_sha = digest.hexdigest()
            if size != x_chunk_size:
                raise HTTPException(status_code=400, detail="chunk size mismatch")
            if actual_sha != x_chunk_sha256:
                raise HTTPException(status_code=400, detail="chunk sha256 mismatch")
            async with job_submission_lock(request.app, job_id):
                job = get_authorized_job(db, job_id, principal, auth_store)
                reject_expired_upload(db, job)
                if job["status"] != "receiving":
                    raise HTTPException(status_code=409, detail="job is no longer receiving chunks")
                existing = db.get_chunk(job_id, archive_type, index)
                if existing:
                    if existing["sha256"] != actual_sha or existing["size"] != size:
                        raise HTTPException(status_code=409, detail="chunk already exists with different checksum")
                    if chunk_path.is_file():
                        return ChunkResponse(
                            job_id=job_id, archive_type=archive_type, index=index,
                            size=size, sha256=actual_sha, duplicate=True,
                        )
                tmp_path.replace(chunk_path)
                db.add_chunk(job_id, archive_type, index, size, actual_sha)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            finally:
                state.upload_count -= 1
                state.user_uploads[principal.id] -= 1
                if not state.user_uploads[principal.id]:
                    state.user_uploads.pop(principal.id)
                state.active_uploads[job_id] -= 1
                if not state.active_uploads[job_id]:
                    state.active_uploads.pop(job_id)
        return ChunkResponse(
            job_id=job_id, archive_type=archive_type, index=index,
            size=size, sha256=actual_sha,
        )

    @app.post("/v1/jobs/{job_id}/complete", response_model=JobResponse)
    async def complete_job(
        job_id: str,
        request: Request,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> JobResponse:
        async with job_submission_lock(request.app, job_id):
            job = get_authorized_job(db, job_id, principal, auth_store)
            reject_expired_upload(db, job)
            if job["status"] != "receiving":
                return job_response(db, job_id, settings.retention_hours)

            cache_hit = has_ready_dataset_cache(
                db, job["dataset_ref"], job["payload_hash"],
                kaggle_key_id=job.get("kaggle_key_id", ""),
            )
            if awaiting_assignment(job):
                cache_hit = False
            for archive_type in (("kernel",) if cache_hit else ("dataset", "kernel")):
                count = expected_chunk_count(job[f"{archive_type}_size"], job["chunk_size"])
                rows = db.chunks_for(job_id, archive_type)
                chunk_dir = settings.jobs_dir / job_id / "chunks" / archive_type
                if ({row["chunk_index"] for row in rows} != set(range(count))
                        or any(not (chunk_dir / f"{i}.part").is_file() for i in range(count))):
                    raise HTTPException(status_code=409, detail="upload incomplete; resume missing chunks")

            if not db.update_job_if_status(
                job_id,
                {"receiving"},
                status="assembling",
                progress=10,
            ):
                return job_response(db, job_id, settings.retention_hours)
            was_cancelled = False
            operation_started = False
            try:
                async with request.app.state.assembly_slots:
                    operation_started = True
                    was_cancelled, operation_error = await run_thread_to_completion(
                        request.app, assemble_and_validate_job, settings, db, auth_store, job,
                    )
                if operation_error is not None:
                    raise operation_error
            except asyncio.CancelledError:
                if not operation_started:
                    db.update_job_if_status(job_id, {"assembling"}, status="receiving")
                raise
            except CapacityError as exc:
                db.update_job_if_status(job_id, {"assembling"}, status="receiving", error=str(exc))
                if was_cancelled:
                    raise asyncio.CancelledError from exc
                raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "30"}) from exc
            except Exception as exc:
                db.update_job_if_status(
                    job_id,
                    {"assembling"},
                    status="failed",
                    progress=0,
                    error=redact_secrets(str(exc)),
                )
                if was_cancelled:
                    raise asyncio.CancelledError from exc
                raise HTTPException(
                    status_code=400,
                    detail=redact_secrets(str(exc)),
                ) from exc
            queued = db.update_job_if_status(
                job_id,
                {"assembling"},
                status="queued",
                progress=15,
                queue_reason=("waiting for an idle authorized account with GPU quota"
                              if awaiting_assignment(job) else "waiting for an available worker"),
                error="",
            )
            if queued:
                if awaiting_assignment(job):
                    request.app.state.scheduler_event.set()
                else:
                    await request.app.state.queue.put({"action": "process", "job_id": job_id})
            response = job_response(db, job_id, settings.retention_hours)
            if was_cancelled:
                raise asyncio.CancelledError
            return response

    @app.post("/v1/jobs/by-kernel/progress", response_model=JobResponse)
    def update_job_progress_by_kernel(
        payload: JobProgressRequest,
        request: Request,
        authorization: str = Header(default=""),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> JobResponse:
        limiter = request.app.state.auth_failure_limiter
        limit_key = auth_limit_key(request, "callback")
        reject_if_auth_blocked(limiter, limit_key)
        kernel_ref = str(payload.model_extra.get("kernel_ref") or "").strip() if payload.model_extra else ""
        if not kernel_ref:
            raise HTTPException(status_code=400, detail="kernel_ref is required")

        principal = auth_store.authenticate_authorization(authorization)
        key_filter = None
        if principal and not principal.allow_all_keys:
            key_filter = set(auth_store.allowed_key_ids(principal))
        owner_filter = None if not principal or auth_store.legacy or principal.allow_all_keys else principal.id

        candidates = db.get_jobs_by_kernel_ref(
            kernel_ref,
            kaggle_key_ids=key_filter,
            relay_token_id=owner_filter,
            limit=50,
            include_callback_alias=True,
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="job not found")

        job = next(
            (candidate for candidate in candidates if authorize_job_callback(candidate, authorization, auth_store)),
            None,
        )
        if not job:
            record_auth_failure(limiter, limit_key)
            raise HTTPException(status_code=401, detail="unauthorized")

        limiter.clear(limit_key)
        apply_progress_callback(db, job, payload)
        return job_response(db, job["job_id"], settings.retention_hours)

    @app.post("/v1/jobs/{job_id}/progress", response_model=JobResponse)
    def update_job_progress(
        job_id: str,
        payload: JobProgressRequest,
        request: Request,
        authorization: str = Header(default=""),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> JobResponse:
        limiter = request.app.state.auth_failure_limiter
        limit_key = auth_limit_key(request, "callback")
        reject_if_auth_blocked(limiter, limit_key)
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if not authorize_job_callback(job, authorization, auth_store):
            record_auth_failure(limiter, limit_key)
            raise HTTPException(status_code=401, detail="unauthorized")

        limiter.clear(limit_key)
        apply_progress_callback(db, job, payload)
        return job_response(db, job_id, settings.retention_hours)

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    async def cancel_job(
        job_id: str,
        request: Request,
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> JobResponse:
        async with job_submission_lock(request.app, job_id):
            job = get_authorized_job(db, job_id, principal, auth_store)
            request_job_cancel(db, job)
            return job_response(db, job_id, settings.retention_hours)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: str,
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> JobResponse:
        get_authorized_job(db, job_id, principal, auth_store)
        return job_response(db, job_id, settings.retention_hours)

    @app.get("/v1/jobs/{job_id}/dataset.zip")
    async def download_dataset(
        job_id: str,
        request: Request,
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> FileResponse:
        get_authorized_job(db, job_id, principal, auth_store)
        lock = job_submission_lock(request.app, job_id)
        await lock.acquire()
        try:
            job = get_authorized_job(db, job_id, principal, auth_store)
            download = dataset_download_metadata(job, settings.jobs_dir)
            code = download["dataset_download_unavailable_code"]
            if code:
                raise HTTPException(
                    status_code={"expired": 410, "missing": 404, "inaccessible": 503}.get(code, 409),
                    detail=f"dataset archive unavailable: {code}",
                )
            path = settings.jobs_dir / job_id / "archives" / "dataset.zip"
            # Failed assembly can leave a full-size archive with the wrong digest.
            try:
                digest = await asyncio.to_thread(sha256_file, path)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="dataset archive unavailable: missing") from exc
            except OSError as exc:
                raise HTTPException(status_code=503, detail="dataset archive unavailable: inaccessible") from exc
            if not hmac.compare_digest(digest, job["dataset_archive_sha256"]):
                raise HTTPException(status_code=409, detail="dataset archive unavailable: invalid")
            return JobLockedFileResponse(
                path,
                job_lock=lock,
                media_type="application/zip",
                filename=f"{job_id}-dataset.zip",
            )
        except BaseException:
            lock.release()
            raise

    @app.get("/v1/jobs/{job_id}/artifacts.zip")
    async def download_artifacts(
        job_id: str,
        request: Request,
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> FileResponse:
        lock = job_submission_lock(request.app, job_id)
        await lock.acquire()
        try:
            job = get_authorized_job(db, job_id, principal, auth_store)
            download = artifact_download_metadata(job, settings.retention_hours)
            if not download["can_download"]:
                status_code = (
                    404
                    if job["status"] in {"complete", "canceled"}
                    and job.get("artifact_path")
                    else 409
                )
                raise HTTPException(
                    status_code=status_code,
                    detail=download["download_unavailable_reason"],
                )
            artifact_path = Path(job["artifact_path"])
            return JobLockedFileResponse(
                artifact_path,
                job_lock=lock,
                media_type="application/zip",
                filename=str(download["artifact_filename"]),
            )
        except BaseException:
            lock.release()
            raise

    @app.delete("/v1/jobs/{job_id}")
    async def delete_job(
        job_id: str,
        request: Request,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> Response:
        async with job_submission_lock(request.app, job_id):
            job = get_authorized_job(db, job_id, principal, auth_store)
            if job["status"] not in TERMINAL_JOB_STATUSES | {"receiving"}:
                raise HTTPException(
                    status_code=409,
                    detail="cancel the active job before deleting it",
                )
            if job_id in request.app.state.active_job_ids:
                raise HTTPException(status_code=409, detail="job worker is still active")
            if job_id in request.app.state.active_uploads:
                raise HTTPException(status_code=409, detail="job upload is still active")
            was_cancelled, error = await run_thread_to_completion(
                request.app, delete_job_files_and_record, settings, db, job_id,
            )
            if error is not None:
                LOGGER.error("job deletion failed for %s: %s", job_id, redact_secrets(str(error)))
            if was_cancelled:
                raise asyncio.CancelledError
            if error is not None:
                raise HTTPException(status_code=503, detail="job cleanup failed; retry deletion")
            return Response(status_code=204)

    return app


app = create_app()

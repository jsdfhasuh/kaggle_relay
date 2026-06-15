import asyncio
import hashlib
import hmac
import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiofiles
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app.archive import (
    ArchiveError,
    assemble_archive,
    safe_extract_zip,
    validate_chunk_index,
)
from app.auth_config import AuthConfigError, AuthSelectionError, AuthStore, RelayPrincipal, bearer_token
from app.config import Settings
from app.database import RelayDb
from app.kaggle_adapter import KaggleAdapter
from app.schemas import (
    ChunkResponse,
    CreateKaggleKeyRequest,
    CreateJobRequest,
    CreateRelayTokenRequest,
    HealthResponse,
    JobProgressRequest,
    JobResponse,
    UiLoginRequest,
    UpdateKaggleKeyRequest,
)
from app.security import redact_secrets
from app.ui_auth import (
    authenticate_ui_session,
    create_ui_session_cookie,
    delete_ui_session_cookie,
    set_ui_session_cookie,
    ui_session_max_age_seconds,
)
from app.worker import has_ready_dataset_cache, process_job, validate_kernel_payload, validate_payloads

VERSION = "0.1.0"
AUTH_CONFIG_LOCK = threading.RLock()
KAGGLE_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> RelayDb:
    return request.app.state.db


def get_auth_store(request: Request) -> AuthStore:
    return request.app.state.auth_store


async def require_auth(
    request: Request,
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
    auth_store: AuthStore = Depends(get_auth_store),
) -> RelayPrincipal:
    token = bearer_token(authorization)
    if token:
        principal = auth_store.authenticate_token(token)
        if principal:
            return principal
        raise HTTPException(status_code=401, detail="unauthorized")

    principal = authenticate_ui_session(request, settings, auth_store)
    if principal:
        return principal
    raise HTTPException(status_code=401, detail="unauthorized")


def selection_error(exc: AuthSelectionError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def can_access_job(job: dict, principal: RelayPrincipal, auth_store: AuthStore) -> bool:
    if not auth_store.can_access_key(principal, job.get("kaggle_key_id", "")):
        return False

    job_owner = str(job.get("relay_token_id") or "").strip()
    if auth_store.legacy:
        return not job_owner or job_owner == principal.id
    if job_owner:
        return job_owner == principal.id
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


def job_response(db: RelayDb, job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_response(db, job)


def artifact_download_metadata(job: dict) -> dict:
    filename = f"{job['job_id']}-artifacts.zip"
    metadata = {
        "can_download": False,
        "artifact_size": None,
        "artifact_filename": filename,
        "download_unavailable_reason": "",
    }
    if job["status"] != "complete":
        metadata["download_unavailable_reason"] = "job is not complete"
        return metadata
    if not job.get("artifact_path"):
        metadata["download_unavailable_reason"] = "artifact path is missing"
        return metadata

    artifact_path = Path(job["artifact_path"])
    try:
        stat = artifact_path.stat()
    except FileNotFoundError:
        metadata["download_unavailable_reason"] = "artifact file is missing"
        return metadata
    except OSError:
        metadata["download_unavailable_reason"] = "artifact file is inaccessible"
        return metadata
    if not artifact_path.is_file():
        metadata["download_unavailable_reason"] = "artifact path is not a file"
        return metadata

    metadata["can_download"] = True
    metadata["artifact_size"] = stat.st_size
    return metadata


def job_to_response(db: RelayDb, job: dict) -> JobResponse:
    job_id = job["job_id"]
    dataset_cache_hit = has_ready_dataset_cache(
        db,
        job["dataset_ref"],
        job["payload_hash"],
        kaggle_key_id=job.get("kaggle_key_id", ""),
    )
    return JobResponse(
        **RelayDb.to_response(
            {
                **job,
                "callback_enabled": bool(job.get("callback_token_sha256")),
                **artifact_download_metadata(job),
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
        if not principal.allow_all_keys and token_principal.id != principal.id:
            continue
        allowed = (
            "*"
            if token_principal.allow_all_keys
            else sorted(token_principal.allowed_kaggle_key_ids or [])
        )
        tokens.append(
            {
                "id": token_principal.id,
                "allowed_kaggle_key_ids": allowed,
                "current": token_principal.id == principal.id,
            }
        )
    return tokens


def auth_config_summary(auth_store: AuthStore, principal: RelayPrincipal) -> dict:
    allowed_key_ids = public_allowed_key_ids(auth_store, principal)
    return {
        "mode": "legacy" if auth_store.legacy else "multi_key",
        "principal_id": principal.id,
        "current_token_id": principal.id,
        "allowed_kaggle_key_ids": allowed_key_ids,
        "can_manage_auth": principal.allow_all_keys and not auth_store.legacy,
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
        resolved_key_id = resolve_job_kaggle_key_id(settings, auth_store, principal, kaggle_key_id)
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


def select_kaggle_key_by_quota(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    owner: str = "",
) -> str:
    allowed_key_ids = public_allowed_key_ids(auth_store, principal)
    owner = owner.strip()
    if owner:
        matching_key_ids = []
        for key_id in allowed_key_ids:
            credentials = auth_store.credentials_for(key_id)
            username = str(getattr(credentials, "username", "") or "").strip()
            if username.lower() == owner.lower():
                matching_key_ids.append(key_id)
        if not matching_key_ids:
            raise HTTPException(
                status_code=400,
                detail=f"no allowed kaggle key matches requested owner {owner}",
            )
        allowed_key_ids = matching_key_ids

    candidates: list[tuple[float, str]] = []
    exhausted: list[str] = []
    unavailable: list[str] = []

    for key_id in allowed_key_ids:
        try:
            credentials = auth_store.credentials_for(key_id)
            quota = KaggleAdapter(settings, lambda _message: None, credentials=credentials).quota()
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

    if candidates:
        return max(candidates)[1]
    if exhausted:
        raise HTTPException(
            status_code=409,
            detail="no allowed kaggle key has remaining GPU quota",
        )
    detail = "unable to read quota for allowed kaggle keys"
    if unavailable:
        detail = f"{detail}: {'; '.join(unavailable)}"
    raise HTTPException(status_code=503, detail=detail)


def resolve_job_kaggle_key_id(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    requested_key_id: str = "",
    dataset_ref: str = "",
    kernel_ref: str = "",
) -> str:
    requested = str(requested_key_id or "").strip()
    if requested:
        return auth_store.resolve_kaggle_key_id(principal, requested)
    if auth_store.legacy or len(public_allowed_key_ids(auth_store, principal)) <= 1:
        return auth_store.resolve_kaggle_key_id(principal, requested)
    return select_kaggle_key_by_quota(
        settings,
        auth_store,
        principal,
        owner=requested_owner_from_refs(dataset_ref, kernel_ref),
    )


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


def session_summary(auth_store: AuthStore, principal: RelayPrincipal) -> dict:
    return {
        "authenticated": True,
        "principal_id": principal.id,
        "allowed_kaggle_key_ids": public_allowed_key_ids(auth_store, principal),
    }


def require_config_admin(settings: Settings, principal: RelayPrincipal) -> Path:
    if not settings.auth_config_path:
        raise HTTPException(status_code=400, detail="RELAY_AUTH_CONFIG is required")
    if not principal.allow_all_keys:
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


def validate_and_write_auth_config(path: Path, data: dict) -> AuthStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp_path, 0o600)
        new_store = AuthStore.from_file(tmp_path)
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
        return validate_and_write_auth_config(path, data)


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
        return validate_and_write_auth_config(path, data)


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
            }
        )
        return validate_and_write_auth_config(path, data)


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
            await asyncio.to_thread(
                process_job,
                app.state.settings,
                app.state.db,
                job_id,
                app.state.auth_store,
            )
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
    app.state.auth_store = AuthStore.from_settings(settings)
    app.state.queue = asyncio.Queue()

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
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> JSONResponse:
        principal = auth_store.authenticate_token(payload.token.strip())
        if not principal:
            raise HTTPException(status_code=401, detail="invalid token")
        max_age = ui_session_max_age_seconds()
        response = JSONResponse(
            {
                "ok": True,
                "principal_id": principal.id,
                "allowed_kaggle_key_ids": public_allowed_key_ids(auth_store, principal),
            }
        )
        set_ui_session_cookie(
            response,
            create_ui_session_cookie(settings, auth_store, principal, max_age),
            max_age,
        )
        return response

    @app.post("/v1/ui/logout")
    def ui_logout() -> JSONResponse:
        response = JSONResponse({"ok": True})
        delete_ui_session_cookie(response)
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

    @app.get("/v1/kaggle/account")
    def kaggle_account(
        kaggle_key_id: str = "",
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        return kaggle_account_status(settings, auth_store, principal, kaggle_key_id)

    @app.post("/v1/kaggle/account/probe")
    def kaggle_account_write_probe(
        kaggle_key_id: str = "",
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        return kaggle_account_probe(settings, auth_store, principal, kaggle_key_id)

    @app.get("/v1/kaggle/accounts")
    def kaggle_accounts(
        settings: Settings = Depends(get_settings),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> dict:
        return {
            "accounts": [
                kaggle_account_status(settings, auth_store, principal, key_id)
                for key_id in public_allowed_key_ids(auth_store, principal)
            ],
        }

    @app.post("/v1/jobs", response_model=JobResponse)
    def create_job(
        payload: CreateJobRequest,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> JobResponse:
        try:
            kaggle_key_id = resolve_job_kaggle_key_id(
                settings,
                auth_store,
                principal,
                payload.kaggle_key_id,
                payload.dataset_ref,
                payload.kernel_ref,
            )
        except AuthSelectionError as exc:
            raise selection_error(exc) from exc
        job_id = uuid.uuid4().hex
        (settings.jobs_dir / job_id / "chunks" / "dataset").mkdir(parents=True, exist_ok=True)
        (settings.jobs_dir / job_id / "chunks" / "kernel").mkdir(parents=True, exist_ok=True)
        values = {
            **payload.model_dump(),
            "job_id": job_id,
            "relay_token_id": principal.id,
            "kaggle_key_id": kaggle_key_id,
        }
        db.create_job(values)
        return job_response(db, job_id)

    @app.get("/v1/jobs", response_model=list[JobResponse])
    def list_jobs(
        limit: int = Query(default=50, ge=1, le=200),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> list[JobResponse]:
        key_filter = None if principal.allow_all_keys else set(auth_store.allowed_key_ids(principal))
        owner_filter = None if auth_store.legacy else principal.id
        include_unowned = bool(owner_filter and principal.allow_all_keys)
        jobs = db.list_jobs(
            kaggle_key_ids=key_filter,
            relay_token_id=owner_filter,
            include_unowned=include_unowned,
            limit=limit,
        )
        return [job_to_response(db, job) for job in jobs]

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
        job = get_authorized_job(db, job_id, principal, auth_store)
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

    @app.post("/v1/jobs/{job_id}/complete", response_model=JobResponse)
    async def complete_job(
        job_id: str,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
        request: Request = None,
    ) -> JobResponse:
        job = get_authorized_job(db, job_id, principal, auth_store)
        if job["status"] in {"queued", "uploading_dataset", "waiting_dataset", "pushing_kernel", "waiting_kernel", "downloading_output", "complete"}:
            return job_response(db, job_id)

        job_dir = settings.jobs_dir / job_id
        archives_dir = job_dir / "archives"
        extracted_dir = job_dir / "extracted"
        credentials = auth_store.credentials_for(job.get("kaggle_key_id", ""))
        db.update_job(job_id, status="assembling", progress=10)
        try:
            kernel_zip = archives_dir / "kernel.zip"
            dataset_cache_hit = has_ready_dataset_cache(
                db,
                job["dataset_ref"],
                job["payload_hash"],
                kaggle_key_id=job.get("kaggle_key_id", ""),
            )
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
                validate_kernel_payload(
                    extracted_dir / "kernel",
                    job["kernel_ref"],
                    credentials,
                )
            else:
                validate_payloads(
                    extracted_dir / "dataset",
                    extracted_dir / "kernel",
                    job["dataset_ref"],
                    job["kernel_ref"],
                    credentials,
                )
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
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> JobResponse:
        kernel_ref = str(payload.model_extra.get("kernel_ref") or "").strip() if payload.model_extra else ""
        if not kernel_ref:
            raise HTTPException(status_code=400, detail="kernel_ref is required")

        principal = auth_store.authenticate_authorization(authorization)
        key_filter = None
        if principal and not principal.allow_all_keys:
            key_filter = set(auth_store.allowed_key_ids(principal))
        owner_filter = None if not principal or auth_store.legacy else principal.id
        include_unowned = bool(owner_filter and principal.allow_all_keys)

        candidates = db.get_jobs_by_kernel_ref(
            kernel_ref,
            kaggle_key_ids=key_filter,
            relay_token_id=owner_filter,
            include_unowned=include_unowned,
            limit=50,
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="job not found")

        job = next(
            (candidate for candidate in candidates if authorize_job_callback(candidate, authorization, auth_store)),
            None,
        )
        if not job:
            raise HTTPException(status_code=401, detail="unauthorized")

        apply_progress_callback(db, job, payload)
        return job_response(db, job["job_id"])

    @app.post("/v1/jobs/{job_id}/progress", response_model=JobResponse)
    def update_job_progress(
        job_id: str,
        payload: JobProgressRequest,
        authorization: str = Header(default=""),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
    ) -> JobResponse:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if not authorize_job_callback(job, authorization, auth_store):
            raise HTTPException(status_code=401, detail="unauthorized")

        apply_progress_callback(db, job, payload)
        return job_response(db, job_id)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: str,
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> JobResponse:
        get_authorized_job(db, job_id, principal, auth_store)
        return job_response(db, job_id)

    @app.get("/v1/jobs/{job_id}/artifacts.zip")
    def download_artifacts(
        job_id: str,
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> FileResponse:
        job = get_authorized_job(db, job_id, principal, auth_store)
        download = artifact_download_metadata(job)
        if not download["can_download"]:
            status_code = 404 if job["status"] == "complete" and job.get("artifact_path") else 409
            raise HTTPException(status_code=status_code, detail=download["download_unavailable_reason"])
        artifact_path = Path(job["artifact_path"])
        return FileResponse(
            artifact_path,
            media_type="application/zip",
            filename=str(download["artifact_filename"]),
        )

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(
        job_id: str,
        settings: Settings = Depends(get_settings),
        db: RelayDb = Depends(get_db),
        auth_store: AuthStore = Depends(get_auth_store),
        principal: RelayPrincipal = Depends(require_auth),
    ) -> Response:
        get_authorized_job(db, job_id, principal, auth_store)
        shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
        shutil.rmtree(settings.artifacts_dir / job_id, ignore_errors=True)
        db.update_job(job_id, status="failed", error="deleted")
        return Response(status_code=204)

    return app


app = create_app()

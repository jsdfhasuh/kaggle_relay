import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import Request, Response

from app.auth_config import AuthStore, RelayPrincipal
from app.config import Settings
from app.security import register_secret


UI_COOKIE_NAME = "relay_ui_session"
DEFAULT_UI_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60


def ui_session_max_age_seconds() -> int:
    raw = os.environ.get("RELAY_UI_SESSION_MAX_AGE_SECONDS", "").strip()
    if not raw:
        return DEFAULT_UI_SESSION_MAX_AGE_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_UI_SESSION_MAX_AGE_SECONDS
    return max(300, min(value, 30 * 24 * 60 * 60))


def ui_cookie_secure() -> bool:
    return os.environ.get("RELAY_UI_COOKIE_SECURE", "").strip().lower() == "true"


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _session_secret(settings: Settings, auth_store: AuthStore) -> bytes:
    configured = os.environ.get("RELAY_UI_SESSION_SECRET", "").strip()
    if configured:
        register_secret(configured)
        return configured.encode("utf-8")

    api_token = str(getattr(settings, "api_token", "") or "").strip()
    if api_token:
        register_secret(api_token)
        return api_token.encode("utf-8")

    token_values = [token for token, _principal in getattr(auth_store, "_tokens", []) if token]
    if token_values:
        material = "\0".join(sorted(token_values))
        return hashlib.sha256(material.encode("utf-8")).digest()

    raise RuntimeError("RELAY_UI_SESSION_SECRET is required when no relay token is configured")


def _sign(payload_b64: str, settings: Settings, auth_store: AuthStore) -> str:
    digest = hmac.new(
        _session_secret(settings, auth_store),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64_encode(digest)


def _principal_by_id(auth_store: AuthStore, principal_id: str) -> RelayPrincipal | None:
    expected_id = str(principal_id or "").strip()
    if not expected_id:
        return None
    for _token, principal in getattr(auth_store, "_tokens", []):
        if hmac.compare_digest(principal.id, expected_id):
            return principal
    return None


def create_ui_session_cookie(
    settings: Settings,
    auth_store: AuthStore,
    principal: RelayPrincipal,
    max_age_seconds: int | None = None,
) -> str:
    now = int(time.time())
    max_age = int(max_age_seconds or ui_session_max_age_seconds())
    payload: dict[str, Any] = {
        "principal_id": principal.id,
        "issued_at": now,
        "expires_at": now + max_age,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64_encode(payload_json)
    signature_b64 = _sign(payload_b64, settings, auth_store)
    return f"{payload_b64}.{signature_b64}"


def authenticate_ui_session(
    request: Request,
    settings: Settings,
    auth_store: AuthStore,
) -> RelayPrincipal | None:
    cookie_value = request.cookies.get(UI_COOKIE_NAME, "")
    if not cookie_value or "." not in cookie_value:
        return None

    payload_b64, signature_b64 = cookie_value.rsplit(".", 1)
    expected_signature = _sign(payload_b64, settings, auth_store)
    if not hmac.compare_digest(signature_b64, expected_signature):
        return None

    try:
        payload = json.loads(_b64_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    try:
        issued_at = int(payload.get("issued_at", 0))
        expires_at = int(payload.get("expires_at", 0))
    except (TypeError, ValueError):
        return None
    if issued_at <= 0 or expires_at <= int(time.time()):
        return None

    return _principal_by_id(auth_store, str(payload.get("principal_id", "")))


def set_ui_session_cookie(
    response: Response,
    value: str,
    max_age_seconds: int,
) -> None:
    response.set_cookie(
        UI_COOKIE_NAME,
        value,
        max_age=max_age_seconds,
        httponly=True,
        secure=ui_cookie_secure(),
        samesite="lax",
        path="/",
    )


def delete_ui_session_cookie(response: Response) -> None:
    response.delete_cookie(UI_COOKIE_NAME, path="/")

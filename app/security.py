import ipaddress
import re
import threading
import time
from collections import deque
from math import ceil
from urllib.parse import urlsplit

from fastapi import Request


_EXTRA_SECRETS: set[str] = set()

TOKEN_PATTERNS = [
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"KGAT_[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)(KAGGLE_(?:API_TOKEN|KEY|USERNAME)=)[^\s]+"),
]


class AuthFailureLimiter:
    def __init__(
        self,
        failure_limit: int,
        window_seconds: int,
        lockout_seconds: int,
    ):
        self.failure_limit = failure_limit
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, deque[float]] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            blocked_until = self._blocked_until.get(key, 0)
            if blocked_until <= now:
                self._blocked_until.pop(key, None)
                return 0
            return max(1, ceil(blocked_until - now))

    def record_failure(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            blocked_until = self._blocked_until.get(key, 0)
            if blocked_until > now:
                return max(1, ceil(blocked_until - now))

            failures = self._failures.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while failures and failures[0] <= cutoff:
                failures.popleft()
            failures.append(now)
            if len(failures) < self.failure_limit:
                return 0

            self._failures.pop(key, None)
            self._blocked_until[key] = now + self.lockout_seconds
            return self.lockout_seconds

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._blocked_until.pop(key, None)


def auth_source(request: Request, trusted_proxy_ips: frozenset[str] | None = None) -> str:
    peer = request.client.host if request.client and request.client.host else "unknown"
    try:
        normalized_peer = ipaddress.ip_address(peer).compressed
    except ValueError:
        normalized_peer = peer
    if normalized_peer not in (trusted_proxy_ips or frozenset()):
        return normalized_peer

    forwarded = request.headers.get("x-forwarded-for", "").rsplit(",", 1)[-1].strip()
    try:
        return ipaddress.ip_address(forwarded).compressed
    except ValueError:
        return normalized_peer


def _origin_key(value: str, require_origin_shape: bool) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if scheme not in {"http", "https"} or not hostname:
        return None
    if require_origin_shape and (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, hostname, port


def is_same_origin_request(request: Request, public_origin: str = "") -> bool:
    origin = request.headers.get("origin", "").strip()
    if not origin:
        return False
    expected_origin = str(public_origin or "").strip() or str(request.url)
    return _origin_key(origin, True) == _origin_key(expected_origin, False)


def register_secret(value: str) -> None:
    secret = str(value or "").strip()
    if secret:
        _EXTRA_SECRETS.add(secret)


def redact_secrets(text: str) -> str:
    value = str(text or "")
    for secret in sorted(_EXTRA_SECRETS, key=len, reverse=True):
        value = value.replace(secret, "***")
    for pattern in TOKEN_PATTERNS:
        value = pattern.sub(lambda match: f"{match.group(1)}***" if match.groups() else "***", value)
    return value

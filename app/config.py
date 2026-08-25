import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024
DEFAULT_WORKER_COUNT = 2
MIN_ADMIN_TOKEN_LENGTH = 8
DEFAULT_AUTH_FAILURE_LIMIT = 10
DEFAULT_AUTH_FAILURE_WINDOW_SECONDS = 60
DEFAULT_AUTH_LOCKOUT_SECONDS = 300


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _read_trusted_proxy_ips() -> frozenset[str]:
    values = []
    for raw_value in os.environ.get("RELAY_TRUSTED_PROXY_IPS", "").split(","):
        value = raw_value.strip()
        if not value:
            continue
        try:
            values.append(ipaddress.ip_address(value).compressed)
        except ValueError as exc:
            raise RuntimeError(
                f"RELAY_TRUSTED_PROXY_IPS contains an invalid IP: {value}"
            ) from exc
    return frozenset(values)


def _normalize_public_origin(value: str) -> str:
    origin = str(value or "").strip().rstrip("/")
    if not origin:
        return ""
    try:
        parsed = urlsplit(origin)
        parsed.port
    except ValueError as exc:
        raise ValueError("public_origin must be an absolute HTTP(S) origin") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("public_origin must be an absolute HTTP(S) origin")
    return origin


@dataclass
class Settings:
    api_token: str
    storage_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    retention_hours: int = 72
    dataset_poll_seconds: int = 20
    dataset_status_permission_grace_seconds: int = 300
    kernel_poll_seconds: int = 60
    kernel_max_wait_seconds: int = 12 * 60 * 60
    worker_count: int = DEFAULT_WORKER_COUNT
    auth_failure_limit: int = DEFAULT_AUTH_FAILURE_LIMIT
    auth_failure_window_seconds: int = DEFAULT_AUTH_FAILURE_WINDOW_SECONDS
    auth_lockout_seconds: int = DEFAULT_AUTH_LOCKOUT_SECONDS
    public_origin: str = ""
    trusted_proxy_ips: frozenset[str] = field(default_factory=frozenset)
    kaggle_cmd: str = "kaggle"
    auth_config_path: Path | None = None
    admin_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if type(self.worker_count) is not int or self.worker_count < 1:
            raise ValueError("worker_count must be a positive integer")
        for name in (
            "auth_failure_limit",
            "auth_failure_window_seconds",
            "auth_lockout_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        admin_token = str(self.admin_token or "").strip()
        if admin_token and not self.auth_config_path:
            raise ValueError("admin_token requires auth_config_path")
        if admin_token and len(admin_token) < MIN_ADMIN_TOKEN_LENGTH:
            raise ValueError(
                f"admin_token must be at least {MIN_ADMIN_TOKEN_LENGTH} characters"
            )
        self.admin_token = admin_token
        self.public_origin = _normalize_public_origin(self.public_origin)

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("RELAY_API_TOKEN", "").strip()
        auth_config = os.environ.get("RELAY_AUTH_CONFIG", "").strip()
        admin_token = os.environ.get("RELAY_ADMIN_TOKEN", "").strip()
        if admin_token and not auth_config:
            raise RuntimeError("RELAY_ADMIN_TOKEN requires RELAY_AUTH_CONFIG")
        if admin_token and len(admin_token) < MIN_ADMIN_TOKEN_LENGTH:
            raise RuntimeError(
                f"RELAY_ADMIN_TOKEN must be at least {MIN_ADMIN_TOKEN_LENGTH} characters"
            )
        if not token and not auth_config:
            raise RuntimeError("RELAY_API_TOKEN or RELAY_AUTH_CONFIG is required")
        storage_dir = Path(os.environ.get("RELAY_STORAGE_DIR", "/data")).expanduser()
        return cls(
            api_token=token,
            storage_dir=storage_dir,
            chunk_size=int(os.environ.get("RELAY_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            retention_hours=int(os.environ.get("RELAY_RETENTION_HOURS", "72")),
            dataset_poll_seconds=int(os.environ.get("RELAY_DATASET_POLL_SECONDS", "20")),
            dataset_status_permission_grace_seconds=int(
                os.environ.get("RELAY_DATASET_STATUS_PERMISSION_GRACE_SECONDS", "300")
            ),
            kernel_poll_seconds=int(os.environ.get("RELAY_KERNEL_POLL_SECONDS", "60")),
            kernel_max_wait_seconds=int(
                os.environ.get("RELAY_KERNEL_MAX_WAIT_SECONDS", str(12 * 60 * 60))
            ),
            worker_count=_read_positive_int(
                "RELAY_WORKER_COUNT",
                DEFAULT_WORKER_COUNT,
            ),
            auth_failure_limit=_read_positive_int(
                "RELAY_AUTH_FAILURE_LIMIT",
                DEFAULT_AUTH_FAILURE_LIMIT,
            ),
            auth_failure_window_seconds=_read_positive_int(
                "RELAY_AUTH_FAILURE_WINDOW_SECONDS",
                DEFAULT_AUTH_FAILURE_WINDOW_SECONDS,
            ),
            auth_lockout_seconds=_read_positive_int(
                "RELAY_AUTH_LOCKOUT_SECONDS",
                DEFAULT_AUTH_LOCKOUT_SECONDS,
            ),
            public_origin=os.environ.get("RELAY_PUBLIC_ORIGIN", ""),
            trusted_proxy_ips=_read_trusted_proxy_ips(),
            kaggle_cmd=os.environ.get("KAGGLE_CMD", "kaggle"),
            auth_config_path=Path(auth_config).expanduser() if auth_config else None,
            admin_token=admin_token,
        )

    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "relay.db"

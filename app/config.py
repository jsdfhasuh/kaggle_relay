import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024


@dataclass
class Settings:
    api_token: str
    storage_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    retention_hours: int = 72
    dataset_poll_seconds: int = 20
    kernel_poll_seconds: int = 60
    kernel_max_wait_seconds: int = 12 * 60 * 60
    kaggle_cmd: str = "kaggle"
    auth_config_path: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("RELAY_API_TOKEN", "").strip()
        auth_config = os.environ.get("RELAY_AUTH_CONFIG", "").strip()
        if not token and not auth_config:
            raise RuntimeError("RELAY_API_TOKEN or RELAY_AUTH_CONFIG is required")
        storage_dir = Path(os.environ.get("RELAY_STORAGE_DIR", "/data")).expanduser()
        return cls(
            api_token=token,
            storage_dir=storage_dir,
            chunk_size=int(os.environ.get("RELAY_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            retention_hours=int(os.environ.get("RELAY_RETENTION_HOURS", "72")),
            dataset_poll_seconds=int(os.environ.get("RELAY_DATASET_POLL_SECONDS", "20")),
            kernel_poll_seconds=int(os.environ.get("RELAY_KERNEL_POLL_SECONDS", "60")),
            kernel_max_wait_seconds=int(
                os.environ.get("RELAY_KERNEL_MAX_WAIT_SECONDS", str(12 * 60 * 60))
            ),
            kaggle_cmd=os.environ.get("KAGGLE_CMD", "kaggle"),
            auth_config_path=Path(auth_config).expanduser() if auth_config else None,
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

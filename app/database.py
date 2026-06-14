import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def now_ts() -> float:
    return time.time()


class RelayDb:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    dataset_ref TEXT NOT NULL,
                    kernel_ref TEXT NOT NULL,
                    dataset_archive_sha256 TEXT NOT NULL,
                    kernel_archive_sha256 TEXT NOT NULL,
                    dataset_size INTEGER NOT NULL,
                    kernel_size INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    dataset_status TEXT NOT NULL DEFAULT '',
                    kernel_status TEXT NOT NULL DEFAULT '',
                    kaggle_output TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    payload_hash TEXT NOT NULL DEFAULT '',
                    callback_token_sha256 TEXT NOT NULL DEFAULT '',
                    relay_token_id TEXT NOT NULL DEFAULT '',
                    kaggle_key_id TEXT NOT NULL DEFAULT '',
                    artifact_path TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    job_id TEXT NOT NULL,
                    archive_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (job_id, archive_type, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    message TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kaggle_dataset_cache (
                    kaggle_key_id TEXT NOT NULL DEFAULT '',
                    dataset_ref TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset_status TEXT NOT NULL DEFAULT '',
                    source_job_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (kaggle_key_id, dataset_ref, payload_hash)
                );
                CREATE TABLE IF NOT EXISTS kaggle_last_job (
                    kaggle_key_id TEXT NOT NULL DEFAULT '',
                    dataset_ref TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    dataset_status TEXT NOT NULL DEFAULT '',
                    job_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (kaggle_key_id, dataset_ref, payload_hash)
                );
                """
            )
            self._ensure_column(conn, "jobs", "callback_token_sha256", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "jobs", "relay_token_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "jobs", "kaggle_key_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_cache_key_dimension(conn, "kaggle_dataset_cache")
            self._ensure_cache_key_dimension(conn, "kaggle_last_job")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _ensure_cache_key_dimension(self, conn: sqlite3.Connection, table: str) -> None:
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
        column_names = {row["name"] for row in columns}
        pk_columns = [row["name"] for row in sorted(columns, key=lambda row: row["pk"]) if row["pk"]]
        desired_pk = ["kaggle_key_id", "dataset_ref", "payload_hash"]
        if "kaggle_key_id" in column_names and pk_columns == desired_pk:
            return

        backup = f"{table}_legacy"
        key_expr = "COALESCE(kaggle_key_id, '')" if "kaggle_key_id" in column_names else "''"
        conn.execute(f"ALTER TABLE {table} RENAME TO {backup}")
        if table == "kaggle_dataset_cache":
            conn.execute(
                """
                CREATE TABLE kaggle_dataset_cache (
                    kaggle_key_id TEXT NOT NULL DEFAULT '',
                    dataset_ref TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset_status TEXT NOT NULL DEFAULT '',
                    source_job_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (kaggle_key_id, dataset_ref, payload_hash)
                )
                """
            )
            conn.execute(
                f"""
                INSERT OR REPLACE INTO kaggle_dataset_cache (
                    kaggle_key_id, dataset_ref, payload_hash, status, dataset_status,
                    source_job_id, created_at, updated_at
                )
                SELECT
                    {key_expr}, dataset_ref, payload_hash, status,
                    dataset_status, source_job_id, created_at, updated_at
                FROM {backup}
                """
            )
        else:
            conn.execute(
                """
                CREATE TABLE kaggle_last_job (
                    kaggle_key_id TEXT NOT NULL DEFAULT '',
                    dataset_ref TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    dataset_status TEXT NOT NULL DEFAULT '',
                    job_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (kaggle_key_id, dataset_ref, payload_hash)
                )
                """
            )
            conn.execute(
                f"""
                INSERT OR REPLACE INTO kaggle_last_job (
                    kaggle_key_id, dataset_ref, payload_hash, dataset_status,
                    job_id, created_at, updated_at
                )
                SELECT
                    {key_expr}, dataset_ref, payload_hash,
                    dataset_status, job_id, created_at, updated_at
                FROM {backup}
                """
            )
        conn.execute(f"DROP TABLE {backup}")

    def create_job(self, values: dict[str, Any]) -> None:
        stamp = now_ts()
        payload = {
            **values,
            "status": "receiving",
            "progress": 0,
            "dataset_status": "",
            "kernel_status": "",
            "kaggle_output": "",
            "error": "",
            "payload_hash": values.get("payload_hash", ""),
            "callback_token_sha256": values.get("callback_token_sha256", ""),
            "relay_token_id": values.get("relay_token_id", ""),
            "kaggle_key_id": values.get("kaggle_key_id", ""),
            "artifact_path": "",
            "created_at": stamp,
            "updated_at": stamp,
            "completed_at": None,
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, dataset_ref, kernel_ref,
                    dataset_archive_sha256, kernel_archive_sha256,
                    dataset_size, kernel_size, chunk_size,
                    status, progress, dataset_status, kernel_status,
                    kaggle_output, error, payload_hash, callback_token_sha256,
                    relay_token_id, kaggle_key_id, artifact_path,
                    created_at, updated_at, completed_at
                ) VALUES (
                    :job_id, :dataset_ref, :kernel_ref,
                    :dataset_archive_sha256, :kernel_archive_sha256,
                    :dataset_size, :kernel_size, :chunk_size,
                    :status, :progress, :dataset_status, :kernel_status,
                    :kaggle_output, :error, :payload_hash, :callback_token_sha256,
                    :relay_token_id, :kaggle_key_id, :artifact_path,
                    :created_at, :updated_at, :completed_at
                )
                """,
                payload,
            )

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(
        self,
        kaggle_key_ids: Optional[set[str]] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        with self.connect() as conn:
            if kaggle_key_ids is None:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            elif not kaggle_key_ids:
                rows = []
            else:
                placeholders = ", ".join("?" for _ in kaggle_key_ids)
                rows = conn.execute(
                    f"""
                    SELECT * FROM jobs
                    WHERE kaggle_key_id IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (*sorted(kaggle_key_ids), safe_limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_job_by_kernel_ref(
        self,
        kernel_ref: str,
        kaggle_key_ids: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
        jobs = self.get_jobs_by_kernel_ref(kernel_ref, kaggle_key_ids=kaggle_key_ids, limit=1)
        return jobs[0] if jobs else None

    def get_jobs_by_kernel_ref(
        self,
        kernel_ref: str,
        kaggle_key_ids: Optional[set[str]] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if kaggle_key_ids is None:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE kernel_ref = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (kernel_ref, limit),
                ).fetchall()
            elif not kaggle_key_ids:
                rows = []
            else:
                placeholders = ", ".join("?" for _ in kaggle_key_ids)
                rows = conn.execute(
                    f"""
                    SELECT * FROM jobs
                    WHERE kernel_ref = ? AND kaggle_key_id IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (kernel_ref, *sorted(kaggle_key_ids), limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_job(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = now_ts()
        if values.get("status") in {"complete", "failed"}:
            values.setdefault("completed_at", now_ts())
        assignments = ", ".join(f"{key} = :{key}" for key in values)
        payload = {"job_id": job_id, **values}
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = :job_id", payload)

    def add_chunk(
        self,
        job_id: str,
        archive_type: str,
        chunk_index: int,
        size: int,
        sha256: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    job_id, archive_type, chunk_index, size, sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, archive_type, chunk_index, size, sha256, now_ts()),
            )

    def get_chunk(
        self,
        job_id: str,
        archive_type: str,
        chunk_index: int,
    ) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM chunks
                WHERE job_id = ? AND archive_type = ? AND chunk_index = ?
                """,
                (job_id, archive_type, chunk_index),
            ).fetchone()
        return dict(row) if row else None

    def chunks_for(self, job_id: str, archive_type: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE job_id = ? AND archive_type = ?
                ORDER BY chunk_index
                """,
                (job_id, archive_type),
            ).fetchall()
        return [dict(row) for row in rows]

    def accepted_chunks(self, job_id: str) -> dict[str, list[int]]:
        return {
            "dataset": [row["chunk_index"] for row in self.chunks_for(job_id, "dataset")],
            "kernel": [row["chunk_index"] for row in self.chunks_for(job_id, "kernel")],
        }

    def append_log(self, job_id: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO logs (job_id, created_at, message) VALUES (?, ?, ?)",
                (job_id, now_ts(), message),
            )

    def recent_logs(self, job_id: str, limit: int = 30) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT message FROM logs
                WHERE job_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (job_id, limit),
            ).fetchall()
        return [row["message"] for row in reversed(rows)]

    def get_dataset_cache(
        self,
        dataset_ref: str,
        payload_hash: str,
        kaggle_key_id: str = "",
    ) -> Optional[dict[str, Any]]:
        if not payload_hash:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM kaggle_dataset_cache
                WHERE kaggle_key_id = ? AND dataset_ref = ? AND payload_hash = ?
                """,
                (kaggle_key_id, dataset_ref, payload_hash),
            ).fetchone()
        return dict(row) if row else None

    def upsert_dataset_cache(
        self,
        dataset_ref: str,
        payload_hash: str,
        status: str,
        dataset_status: str,
        source_job_id: str,
        kaggle_key_id: str = "",
    ) -> None:
        if not payload_hash:
            return
        stamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO kaggle_dataset_cache (
                    kaggle_key_id, dataset_ref, payload_hash, status, dataset_status,
                    source_job_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kaggle_key_id, dataset_ref, payload_hash) DO UPDATE SET
                    status = excluded.status,
                    dataset_status = excluded.dataset_status,
                    source_job_id = excluded.source_job_id,
                    updated_at = excluded.updated_at
                """,
                (
                    kaggle_key_id,
                    dataset_ref,
                    payload_hash,
                    status,
                    dataset_status,
                    source_job_id,
                    stamp,
                    stamp,
                ),
            )

    def get_last_dataset_job(
        self,
        dataset_ref: str,
        payload_hash: str,
        kaggle_key_id: str = "",
    ) -> Optional[dict[str, Any]]:
        if not payload_hash:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM kaggle_last_job
                WHERE kaggle_key_id = ? AND dataset_ref = ? AND payload_hash = ?
                """,
                (kaggle_key_id, dataset_ref, payload_hash),
            ).fetchone()
        return dict(row) if row else None

    def upsert_last_dataset_job(
        self,
        dataset_ref: str,
        payload_hash: str,
        dataset_status: str,
        job_id: str,
        kaggle_key_id: str = "",
    ) -> None:
        if not payload_hash:
            return
        stamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO kaggle_last_job (
                    kaggle_key_id, dataset_ref, payload_hash, dataset_status,
                    job_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kaggle_key_id, dataset_ref, payload_hash) DO UPDATE SET
                    dataset_status = excluded.dataset_status,
                    job_id = excluded.job_id,
                    updated_at = excluded.updated_at
                """,
                (kaggle_key_id, dataset_ref, payload_hash, dataset_status, job_id, stamp, stamp),
            )

    def completed_before(self, cutoff: float) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id FROM jobs
                WHERE completed_at IS NOT NULL AND completed_at < ?
                """,
                (cutoff,),
            ).fetchall()
        return [row["job_id"] for row in rows]

    @staticmethod
    def to_response(job: dict[str, Any], accepted_chunks: dict[str, list[int]], logs: list[str]) -> dict[str, Any]:
        response = dict(job)
        response["accepted_chunks"] = accepted_chunks
        response["recent_logs"] = logs
        return response

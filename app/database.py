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
                """
            )

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
                    kaggle_output, error, payload_hash, artifact_path,
                    created_at, updated_at, completed_at
                ) VALUES (
                    :job_id, :dataset_ref, :kernel_ref,
                    :dataset_archive_sha256, :kernel_archive_sha256,
                    :dataset_size, :kernel_size, :chunk_size,
                    :status, :progress, :dataset_status, :kernel_status,
                    :kaggle_output, :error, :payload_hash, :artifact_path,
                    :created_at, :updated_at, :completed_at
                )
                """,
                payload,
            )

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

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

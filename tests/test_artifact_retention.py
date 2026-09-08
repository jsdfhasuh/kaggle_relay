import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RELAY_API_TOKEN", "test-token")
os.environ.setdefault("RELAY_STORAGE_DIR", str(Path(__file__).resolve().parents[1] / ".test-relay-data"))

from app.config import Settings
from app.main import artifact_download_metadata, cleanup_expired_jobs, create_app


def test_retention_defaults_and_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAY_API_TOKEN", "test-token")
    monkeypatch.setenv("RELAY_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("RELAY_RETENTION_HOURS", raising=False)
    assert Settings.from_env().retention_hours == 168
    monkeypatch.setenv("RELAY_RETENTION_HOURS", "24")
    assert Settings.from_env().retention_hours == 24
    monkeypatch.setenv("RELAY_RETENTION_HOURS", "0")
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_metadata_requires_cleanup_evidence(tmp_path):
    job = {"job_id": "job", "status": "complete", "completed_at": 1}
    assert artifact_download_metadata(job)["download_unavailable_code"] == "missing"
    job["kaggle_output"] = "expired by relay retention cleanup"
    expired = artifact_download_metadata(job)
    assert expired["download_unavailable_code"] == "expired"
    assert expired["artifact_expires_at"] is None
    assert not expired["can_download"]
    artifact = tmp_path / "artifacts.zip"
    artifact.write_bytes(b"test")
    job["artifact_path"] = str(artifact)
    available = artifact_download_metadata(job, 24)
    assert available["can_download"]
    assert available["artifact_expires_at"] == 1 + 24 * 3600
    assert available["download_unavailable_code"] == ""
    job["status"] = "running"
    assert artifact_download_metadata(job)["download_unavailable_code"] == "not_ready"


def test_metadata_permission_error(monkeypatch, tmp_path):
    job = {"job_id": "job", "status": "complete", "artifact_path": str(tmp_path / "a.zip")}
    def denied(_self):
        raise PermissionError("denied")
    monkeypatch.setattr("pathlib.Path.stat", denied)
    assert artifact_download_metadata(job)["download_unavailable_code"] == "inaccessible"


def test_api_retention_boundary_and_configured_expiry(tmp_path):
    settings = Settings(api_token="test-token", storage_dir=tmp_path, retention_hours=168)
    app = create_app(settings)
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(app) as client:
        for job_id, age in (("kept", 167), ("expired", 169)):
            app.state.db.create_job({
                "job_id": job_id, "dataset_ref": "demo/data", "kernel_ref": "demo/kernel",
                "dataset_archive_sha256": "a" * 64, "kernel_archive_sha256": "b" * 64,
                "dataset_size": 1, "kernel_size": 1, "chunk_size": 1,
            })
            artifact = settings.artifacts_dir / job_id / "artifacts.zip"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"package")
            app.state.db.update_job(job_id, status="complete", artifact_path=str(artifact))
            with app.state.db.connect() as conn:
                conn.execute("UPDATE jobs SET completed_at=? WHERE job_id=?", (time.time() - age * 3600, job_id))
        client.portal.call(cleanup_expired_jobs, app)
        kept = client.get("/v1/jobs/kept", headers=headers).json()
        expired = client.get("/v1/jobs/expired", headers=headers).json()
        assert kept["can_download"]
        assert kept["artifact_expires_at"] == kept["completed_at"] + 168 * 3600
        assert expired["status"] == "complete"
        assert expired["download_unavailable_code"] == "expired"
        assert not expired["can_download"]
        app.state.settings.retention_hours = 200
        listed = client.get("/v1/jobs", headers=headers).json()
        kept = next(job for job in listed if job["job_id"] == "kept")
        assert kept["artifact_expires_at"] == kept["completed_at"] + 200 * 3600

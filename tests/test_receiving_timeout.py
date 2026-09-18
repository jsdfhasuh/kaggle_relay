import asyncio
import hashlib
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from test_relay_api import auth_headers, job_request_body, seed_job
from app.config import Settings
from app.main import cleanup_expired_jobs, create_app


def make_upload(tmp_path):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path))
    client = TestClient(app)
    job = client.post("/v1/jobs", headers=auth_headers(), json=job_request_body(b"abcdefgh", b"x")).json()
    return app, client, job


def chunk_headers():
    return auth_headers({"X-Chunk-Sha256": hashlib.sha256(b"abcdefgh").hexdigest(), "X-Chunk-Size": "8"})


def test_three_hour_deadline_is_not_renewed_by_chunks_or_retries(tmp_path, monkeypatch):
    app, client, job = make_upload(tmp_path)
    path = f"/v1/jobs/{job['job_id']}"
    deadline = job["created_at"] + 3 * 3600
    assert job["upload_expires_at"] == deadline
    monkeypatch.setattr("app.main.time.time", lambda: deadline - 1)
    for _ in range(2):
        assert client.put(path + "/archives/dataset/chunks/0", headers=chunk_headers(), content=b"abcdefgh").status_code == 200
    assert client.get(path, headers=auth_headers()).json()["upload_expires_at"] == deadline
    asyncio.run(cleanup_expired_jobs(app))
    assert app.state.db.get_job(job["job_id"])["status"] == "receiving"
    monkeypatch.setattr("app.main.time.time", lambda: deadline)
    asyncio.run(cleanup_expired_jobs(app))
    failed = client.get(path, headers=auth_headers()).json()
    assert failed["status"] == "failed"
    assert "3 hours from job creation" in failed["error"]
    assert failed["upload_expires_at"] is None
    assert not (app.state.settings.jobs_dir / job["job_id"]).exists()


@pytest.mark.parametrize("operation", ["chunk", "complete"])
def test_requests_reject_expired_upload_without_waiting_for_cleanup(tmp_path, monkeypatch, operation):
    app, client, job = make_upload(tmp_path)
    monkeypatch.setattr("app.main.time.time", lambda: job["created_at"] + 10800)
    path = f"/v1/jobs/{job['job_id']}"
    if operation == "chunk":
        response = client.put(path + "/archives/dataset/chunks/0", headers=chunk_headers(), content=b"abcdefgh")
    else:
        response = client.post(path + "/complete", headers=auth_headers())
    assert response.status_code == 409
    assert "upload timed out" in response.json()["detail"]
    assert app.state.db.get_job(job["job_id"])["status"] == "failed"
    assert app.state.db.accepted_chunks(job["job_id"])["dataset"] == []
    asyncio.run(cleanup_expired_jobs(app))
    assert app.state.db.get_job(job["job_id"])["cleaned_at"] is not None


def test_expiry_marks_active_upload_failed_but_defers_file_cleanup(tmp_path):
    app, client, job = make_upload(tmp_path)
    job_id = job["job_id"]
    app.state.db.update_job(job_id, created_at=time.time() - 10801)
    partial = app.state.settings.jobs_dir / job_id / "chunks" / "dataset" / "in-flight.tmp"
    partial.write_bytes(b"partial")
    app.state.active_uploads[job_id] = 1
    for status in ("queued", "waiting_kernel", "complete"):
        other = seed_job(app, status, job_id=status)
        app.state.db.update_job(other, created_at=time.time() - 10801)
    asyncio.run(cleanup_expired_jobs(app))
    assert app.state.db.get_job(job_id)["status"] == "failed"
    assert partial.exists()
    app.state.active_uploads.clear()
    asyncio.run(cleanup_expired_jobs(app))
    assert not partial.exists()
    for status in ("queued", "waiting_kernel", "complete"):
        assert app.state.db.get_job(status)["status"] == status


def test_body_crossing_deadline_fails_and_releases_upload_slots(tmp_path, monkeypatch):
    app, client, job = make_upload(tmp_path)
    clock = [job["created_at"] + 10799]
    monkeypatch.setattr("app.main.time.time", lambda: clock[0])

    async def scenario():
        async def body():
            yield b"abcd"
            clock[0] += 1
            yield b"efgh"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.put(f"/v1/jobs/{job['job_id']}/archives/dataset/chunks/0",
                                      headers=chunk_headers(), content=body())
            assert response.status_code == 409
    asyncio.run(scenario())
    assert app.state.db.get_job(job["job_id"])["status"] == "failed"
    assert app.state.upload_count == 0 and not app.state.active_uploads
    assert not list(tmp_path.rglob("*.tmp"))
    assert app.state.db.accepted_chunks(job["job_id"])["dataset"] == []


def test_receiving_deadline_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAY_API_TOKEN", "secret")
    monkeypatch.delenv("RELAY_RECEIVING_TIMEOUT_HOURS", raising=False)
    assert Settings.from_env().receiving_timeout_hours == 3
    monkeypatch.setenv("RELAY_RECEIVING_TIMEOUT_HOURS", "4")
    assert Settings.from_env().receiving_timeout_hours == 4
    with pytest.raises(ValueError, match="receiving_timeout_hours"):
        Settings(api_token="secret", storage_dir=tmp_path, receiving_timeout_hours=0)

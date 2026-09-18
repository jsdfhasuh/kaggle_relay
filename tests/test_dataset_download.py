import asyncio
import hashlib
import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from test_relay_api import (
    auth_headers, build_zip, make_settings, make_auth_config_settings,
    multi_key_auth_config,
)
from app.main import JobLockedFileResponse, cleanup_expired_jobs, create_app


def seed(app, status="complete", owner="", key=""):
    payload = build_zip({"images/train/a.jpg": b"image", "labels/train/a.txt": b"0 0.5 0.5 1 1"})
    data = build_zip({"dataset-metadata.json": b"{}", "payload.zip": payload})
    app.state.db.create_job({
        "job_id": "sample", "dataset_ref": "demo/data", "kernel_ref": "demo/kernel",
        "dataset_archive_sha256": hashlib.sha256(data).hexdigest(),
        "kernel_archive_sha256": "b" * 64,
        "dataset_size": len(data), "kernel_size": 1, "chunk_size": 8,
        "relay_token_id": owner, "kaggle_key_id": key,
    })
    app.state.db.update_job("sample", status=status)
    path = app.state.settings.jobs_dir / "sample" / "archives" / "dataset.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path, data


@pytest.mark.parametrize("status", ["waiting_kernel", "complete", "failed", "canceled"])
def test_original_dataset_and_labels_download_before_or_after_training(tmp_path, status):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        _, data = seed(app, status)
        for endpoint in ("/v1/jobs/sample", "/v1/jobs"):
            response = client.get(endpoint, headers=auth_headers()).json()
            job = response[0] if isinstance(response, list) else response
            assert job["can_download_dataset"]
        result = client.get("/v1/jobs/sample/dataset.zip", headers=auth_headers())
        assert result.status_code == 200
        assert result.content == data
        assert 'filename="sample-dataset.zip"' in result.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(result.content)) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read("payload.zip"))) as inner:
                assert inner.read("labels/train/a.txt") == b"0 0.5 0.5 1 1"


@pytest.mark.parametrize("case,expected,code", [
    ("receiving", 409, "not_ready"), ("assembling", 409, "not_ready"),
    ("missing", 404, "missing"), ("expired", 410, "expired"),
    ("truncated", 409, "invalid"), ("corrupt", 409, "invalid"),
])
def test_unavailable_or_invalid_archives(tmp_path, case, expected, code):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        path, data = seed(app, case if case in {"receiving", "assembling"} else "failed")
        if case == "missing":
            path.unlink()
        elif case == "expired":
            app.state.db.update_job("sample", cleaned_at=time.time())
        elif case == "truncated":
            path.write_bytes(data[:10])
        elif case == "corrupt":
            path.write_bytes(b"x" * len(data))
        if case != "corrupt":
            job = client.get("/v1/jobs/sample", headers=auth_headers()).json()
            assert not job["can_download_dataset"]
            assert job["dataset_download_unavailable_code"] == code
        result = client.get("/v1/jobs/sample/dataset.zip", headers=auth_headers())
        assert result.status_code == expected
        assert result.json()["detail"].endswith(code)
        # Errors must release the job lock.
        assert client.get("/v1/jobs/sample/dataset.zip", headers=auth_headers()).status_code == expected


def test_dataset_download_enforces_owner_and_supports_browser_session(tmp_path):
    config = multi_key_auth_config()
    config["relay_tokens"].append({"id": "peer", "token": "peer-token", "allowed_kaggle_key_ids": ["ka"]})
    app = create_app(make_auth_config_settings(tmp_path, config))
    with TestClient(app) as client:
        seed(app, owner="user-a", key="ka")
        url = "/v1/jobs/sample/dataset.zip"
        assert client.get(url).status_code == 401
        assert client.get(url, headers=auth_headers(token="peer-token")).status_code == 404
        assert client.get(url, headers=auth_headers(token="user-b-token")).status_code == 404
        assert client.get(url, headers=auth_headers(token="user-a-token")).status_code == 200
        assert client.post(
            "/v1/ui/login", json={"token": "user-a-token"},
            headers={"Origin": "http://testserver"},
        ).status_code == 200
        assert client.get(url).status_code == 200


@pytest.mark.parametrize("operation", ["delete", "cleanup"])
def test_dataset_transfer_blocks_deletion_and_cleanup(tmp_path, monkeypatch, operation):
    app = create_app(make_settings(tmp_path))
    started, release = threading.Event(), threading.Event()
    original = JobLockedFileResponse.__call__

    async def slow_response(self, scope, receive, send):
        started.set()
        await asyncio.to_thread(release.wait, 5)
        await original(self, scope, receive, send)

    monkeypatch.setattr(JobLockedFileResponse, "__call__", slow_response)
    with TestClient(app) as client:
        path, data = seed(app)
        with app.state.db.connect() as conn:
            conn.execute("UPDATE jobs SET completed_at=0 WHERE job_id='sample'")
        with ThreadPoolExecutor(2) as pool:
            download = pool.submit(client.get, "/v1/jobs/sample/dataset.zip", headers=auth_headers())
            try:
                assert started.wait(2)
                if operation == "delete":
                    cleanup = pool.submit(client.delete, "/v1/jobs/sample", headers=auth_headers())
                else:
                    cleanup = pool.submit(client.portal.call, cleanup_expired_jobs, app)
                time.sleep(0.1)
                assert not cleanup.done()
                assert path.exists()
            finally:
                release.set()
            assert download.result(timeout=5).content == data
            cleanup.result(timeout=5)
            assert not path.exists()

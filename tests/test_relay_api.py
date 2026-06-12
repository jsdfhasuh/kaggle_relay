import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RELAY_API_TOKEN", "secret")
os.environ.setdefault("RELAY_STORAGE_DIR", str(ROOT / ".test-relay-data"))

from app.config import Settings
from app.kaggle_adapter import KaggleAdapter, KaggleAdapterError
from app.main import create_app
from app.worker import process_job


def make_settings(tmp_path: Path) -> Settings:
    return Settings(api_token="secret", storage_dir=tmp_path, chunk_size=8)


def auth_headers(extra=None):
    headers = {"Authorization": "Bearer secret"}
    if extra:
        headers.update(extra)
    return headers


def build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buf.getvalue()


def create_job(
    client: TestClient,
    dataset_zip: bytes,
    kernel_zip: bytes,
    payload_hash: str = "",
    callback_token_sha256: str = "",
):
    response = client.post(
        "/v1/jobs",
        headers=auth_headers(),
        json={
            "dataset_ref": "demo/data",
            "kernel_ref": "demo/kernel",
            "dataset_archive_sha256": hashlib.sha256(dataset_zip).hexdigest(),
            "kernel_archive_sha256": hashlib.sha256(kernel_zip).hexdigest(),
            "dataset_size": len(dataset_zip),
            "kernel_size": len(kernel_zip),
            "chunk_size": 8,
            "payload_hash": payload_hash,
            "callback_token_sha256": callback_token_sha256,
        },
    )
    assert response.status_code == 200
    return response.json()["job_id"]


def upload_all(client: TestClient, job_id: str, archive_type: str, data: bytes):
    for index, start in enumerate(range(0, len(data), 8)):
        chunk = data[start : start + 8]
        response = client.put(
            f"/v1/jobs/{job_id}/archives/{archive_type}/chunks/{index}",
            headers=auth_headers(
                {
                    "X-Chunk-Sha256": hashlib.sha256(chunk).hexdigest(),
                    "X-Chunk-Size": str(len(chunk)),
                }
            ),
            content=chunk,
        )
        assert response.status_code == 200


def test_auth_required(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/v1/health")
    assert response.status_code == 401


def test_chunk_upload_duplicate_and_bad_sha(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = create_job(client, dataset_zip, kernel_zip)
        chunk = dataset_zip[:8]
        headers = auth_headers(
            {
                "X-Chunk-Sha256": hashlib.sha256(chunk).hexdigest(),
                "X-Chunk-Size": str(len(chunk)),
            }
        )
        first = client.put(
            f"/v1/jobs/{job_id}/archives/dataset/chunks/0",
            headers=headers,
            content=chunk,
        )
        duplicate = client.put(
            f"/v1/jobs/{job_id}/archives/dataset/chunks/0",
            headers=headers,
            content=chunk,
        )
        bad = client.put(
            f"/v1/jobs/{job_id}/archives/kernel/chunks/0",
            headers=auth_headers({"X-Chunk-Sha256": "0" * 64, "X-Chunk-Size": "3"}),
            content=b"bad",
        )
    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert bad.status_code == 400


def test_complete_rejects_zip_path_traversal(tmp_path):
    dataset_zip = build_zip({"../dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = create_job(client, dataset_zip, kernel_zip)
        upload_all(client, job_id, "dataset", dataset_zip)
        upload_all(client, job_id, "kernel", kernel_zip)
        response = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers())
    assert response.status_code == 400
    assert "unsafe zip path" in response.json()["detail"]


def test_complete_runs_mock_worker_and_downloads_artifacts(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})

    def fake_process_job(settings, db, job_id):
        artifact_path = settings.artifacts_dir / job_id / "artifacts.zip"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact_path, "w") as archive:
            archive.writestr("best.pt", b"pt")
        db.append_log(job_id, "mock worker complete")
        db.update_job(
            job_id,
            status="complete",
            progress=100,
            artifact_path=str(artifact_path),
            dataset_status="ready",
            kernel_status="complete",
        )

    monkeypatch.setattr("app.main.process_job", fake_process_job)
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = create_job(client, dataset_zip, kernel_zip)
        upload_all(client, job_id, "dataset", dataset_zip)
        upload_all(client, job_id, "kernel", kernel_zip)
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers())
        assert complete.status_code == 200

        status = {}
        for _ in range(20):
            status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()
            if status["status"] == "complete":
                break
            time.sleep(0.05)

        download = client.get(
            f"/v1/jobs/{job_id}/artifacts.zip",
            headers=auth_headers(),
        )

    assert status["status"] == "complete"
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert archive.read("best.pt") == b"pt"


def test_create_job_reports_dataset_cache_hit(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        app.state.db.upsert_dataset_cache(
            dataset_ref="demo/data",
            payload_hash="payload-1",
            status="ready",
            dataset_status="ready",
            source_job_id="previous",
        )
        job_id = create_job(client, dataset_zip, kernel_zip, payload_hash="payload-1")
        response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers())

    assert response.status_code == 200
    assert response.json()["dataset_cache_hit"] is True
    assert response.json()["dataset_upload_required"] is False


def test_create_job_backfills_cache_from_last_ready_job(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        app.state.db.upsert_last_dataset_job(
            dataset_ref="demo/data",
            payload_hash="payload-1",
            dataset_status="ready",
            job_id="previous",
        )
        job_id = create_job(client, dataset_zip, kernel_zip, payload_hash="payload-1")
        response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers())
        cache = app.state.db.get_dataset_cache("demo/data", "payload-1")

    assert response.status_code == 200
    assert response.json()["dataset_cache_hit"] is True
    assert response.json()["dataset_upload_required"] is False
    assert cache["status"] == "ready"
    assert cache["source_job_id"] == "previous"


def test_complete_allows_kernel_only_when_dataset_cache_hit(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})

    def fake_process_job(settings, db, job_id):
        db.update_job(job_id, status="complete", progress=100, dataset_status="ready")

    monkeypatch.setattr("app.main.process_job", fake_process_job)
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        app.state.db.upsert_dataset_cache(
            dataset_ref="demo/data",
            payload_hash="payload-1",
            status="ready",
            dataset_status="ready",
            source_job_id="previous",
        )
        job_id = create_job(client, dataset_zip, kernel_zip, payload_hash="payload-1")
        upload_all(client, job_id, "kernel", kernel_zip)
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers())
        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()

    assert complete.status_code == 200
    assert status["dataset_cache_hit"] is True
    assert status["accepted_chunks"]["dataset"] == []


def test_complete_requires_dataset_when_cache_miss(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = create_job(client, dataset_zip, kernel_zip, payload_hash="payload-1")
        upload_all(client, job_id, "kernel", kernel_zip)
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers())

    assert complete.status_code == 400


def test_worker_reuses_dataset_cache_without_upload(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    settings = make_settings(tmp_path)
    monkeypatch.setattr("app.main.process_job", lambda *_args, **_kwargs: None)
    app = create_app(settings)

    with TestClient(app) as client:
        app.state.db.upsert_dataset_cache(
            dataset_ref="demo/data",
            payload_hash="payload-1",
            status="ready",
            dataset_status="ready",
            source_job_id="previous",
        )
        job_id = create_job(client, dataset_zip, kernel_zip, payload_hash="payload-1")
        upload_all(client, job_id, "kernel", kernel_zip)
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers())
        assert complete.status_code == 200

        calls = {"upload_dataset": 0, "wait_dataset": 0}

        class FakeAdapter:
            def __init__(self, _settings, _log):
                pass

            def upload_dataset(self, *_args, **_kwargs):
                calls["upload_dataset"] += 1

            def wait_dataset(self, *_args, **_kwargs):
                calls["wait_dataset"] += 1
                return "ready"

            def push_kernel(self, _kernel_dir):
                return "pushed"

            def wait_kernel(self, _kernel_ref, _progress_callback):
                return "complete"

            def download_output(self, _kernel_ref, output_dir):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "best.pt").write_bytes(b"pt")
                return "downloaded"

            def package_artifacts(self, output_dir, artifact_zip):
                artifact_zip.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(artifact_zip, "w") as archive:
                    archive.write(output_dir / "best.pt", "best.pt")

        monkeypatch.setattr("app.worker.KaggleAdapter", FakeAdapter)
        process_job(settings, app.state.db, job_id)
        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()

    assert calls == {"upload_dataset": 0, "wait_dataset": 0}
    assert status["status"] == "complete"
    assert status["dataset_status"] == "ready"


def test_callback_token_updates_job_progress_and_logs(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    callback_token = "callback-secret"
    callback_hash = hashlib.sha256(callback_token.encode("utf-8")).hexdigest()
    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            callback_token_sha256=callback_hash,
        )
        denied = client.post(
            f"/v1/jobs/{job_id}/progress",
            headers={"Authorization": "Bearer wrong"},
            json={"epoch": 1, "epochs": 300, "message": "bad"},
        )
        accepted = client.post(
            f"/v1/jobs/{job_id}/progress",
            headers={"Authorization": f"Bearer {callback_token}"},
            json={
                "epoch": 3,
                "epochs": 300,
                "message": "[Epoch 3/300] ok",
                "mAP50": 0.99,
            },
        )
        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert status["callback_enabled"] is True
    assert status["status"] == "waiting_kernel"
    assert status["progress"] > 60
    assert "[Epoch 3/300] ok" in status["recent_logs"][-1]
    assert '"mAP50": 0.99' in status["kernel_status"]


def test_relay_token_can_update_progress_for_debugging(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        job_id = create_job(client, dataset_zip, kernel_zip)
        response = client.post(
            f"/v1/jobs/{job_id}/progress",
            headers=auth_headers(),
            json={"remote_progress": 50, "message": "halfway"},
        )
        status = response.json()

    assert response.status_code == 200
    assert status["progress"] == 70
    assert status["recent_logs"][-1] == "halfway"


def test_callback_can_update_progress_by_kernel_ref(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    callback_token = "callback-secret"
    callback_hash = hashlib.sha256(callback_token.encode("utf-8")).hexdigest()
    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            callback_token_sha256=callback_hash,
        )
        missing_ref = client.post(
            "/v1/jobs/by-kernel/progress",
            headers={"Authorization": f"Bearer {callback_token}"},
            json={"epoch": 1, "epochs": 300},
        )
        denied = client.post(
            "/v1/jobs/by-kernel/progress",
            headers={"Authorization": "Bearer wrong"},
            json={
                "kernel_ref": "demo/kernel",
                "epoch": 5,
                "epochs": 300,
                "message": "wrong token",
            },
        )
        accepted = client.post(
            "/v1/jobs/by-kernel/progress",
            headers={"Authorization": f"Bearer {callback_token}"},
            json={
                "kernel_ref": "demo/kernel",
                "epoch": 5,
                "epochs": 300,
                "message": "[Epoch 5/300] ok",
            },
        )
        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()

    assert missing_ref.status_code == 400
    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["job_id"] == job_id
    assert status["progress"] > 60
    assert "[Epoch 5/300] ok" in status["recent_logs"][-1]


def test_wait_dataset_fails_fast_on_forbidden_status(tmp_path):
    class Result:
        returncode = 1
        stdout = "403 Client Error: Forbidden for url: https://api.kaggle.com/..."

    adapter = KaggleAdapter(make_settings(tmp_path), lambda _message: None)
    adapter._run = lambda *_args, **_kwargs: Result()

    with pytest.raises(KaggleAdapterError, match="Dataset status failed"):
        adapter.wait_dataset("demo/private-dataset")

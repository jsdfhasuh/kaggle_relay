import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RELAY_API_TOKEN", "secret")
os.environ.setdefault("RELAY_STORAGE_DIR", str(ROOT / ".test-relay-data"))

from app.config import Settings
from app.main import create_app


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


def create_job(client: TestClient, dataset_zip: bytes, kernel_zip: bytes):
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

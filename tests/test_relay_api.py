import hashlib
import io
import json
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


def make_auth_config_settings(tmp_path: Path, config: dict) -> Settings:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(config), encoding="utf-8")
    return Settings(api_token="", storage_dir=tmp_path, chunk_size=8, auth_config_path=auth_path)


def auth_headers(extra=None, token: str = "secret"):
    headers = {"Authorization": f"Bearer {token}"}
    if extra:
        headers.update(extra)
    return headers


def multi_key_auth_config() -> dict:
    return {
        "relay_tokens": [
            {"id": "admin", "token": "admin-token", "allowed_kaggle_key_ids": "*"},
            {"id": "user-a", "token": "user-a-token", "allowed_kaggle_key_ids": ["ka"]},
            {"id": "user-b", "token": "user-b-token", "allowed_kaggle_key_ids": ["kb"]},
        ],
        "kaggle_keys": [
            {"id": "ka", "username": "alice", "key": "alice-key"},
            {"id": "kb", "username": "bob", "key": "bob-key"},
        ],
    }


def build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, content in files.items():
            if name in {"dataset-metadata.json", "kernel-metadata.json"}:
                try:
                    metadata = json.loads(content.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    metadata = None
                if isinstance(metadata, dict) and not metadata.get("id"):
                    metadata["id"] = "demo/data" if name == "dataset-metadata.json" else "demo/kernel"
                    content = json.dumps(metadata).encode("utf-8")
            archive.writestr(name, content)
    return buf.getvalue()


def job_request_body(
    dataset_zip: bytes,
    kernel_zip: bytes,
    payload_hash: str = "",
    callback_token_sha256: str = "",
    kaggle_key_id: str | None = None,
    dataset_ref: str = "demo/data",
    kernel_ref: str = "demo/kernel",
) -> dict:
    body = {
        "dataset_ref": dataset_ref,
        "kernel_ref": kernel_ref,
        "dataset_archive_sha256": hashlib.sha256(dataset_zip).hexdigest(),
        "kernel_archive_sha256": hashlib.sha256(kernel_zip).hexdigest(),
        "dataset_size": len(dataset_zip),
        "kernel_size": len(kernel_zip),
        "chunk_size": 8,
        "payload_hash": payload_hash,
        "callback_token_sha256": callback_token_sha256,
    }
    if kaggle_key_id is not None:
        body["kaggle_key_id"] = kaggle_key_id
    return body


def create_job(
    client: TestClient,
    dataset_zip: bytes,
    kernel_zip: bytes,
    payload_hash: str = "",
    callback_token_sha256: str = "",
    kaggle_key_id: str | None = None,
    dataset_ref: str = "demo/data",
    kernel_ref: str = "demo/kernel",
    headers: dict | None = None,
):
    response = client.post(
        "/v1/jobs",
        headers=headers or auth_headers(),
        json=job_request_body(
            dataset_zip,
            kernel_zip,
            payload_hash=payload_hash,
            callback_token_sha256=callback_token_sha256,
            kaggle_key_id=kaggle_key_id,
            dataset_ref=dataset_ref,
            kernel_ref=kernel_ref,
        ),
    )
    assert response.status_code == 200
    return response.json()["job_id"]


def upload_all(
    client: TestClient,
    job_id: str,
    archive_type: str,
    data: bytes,
    token: str = "secret",
):
    for index, start in enumerate(range(0, len(data), 8)):
        chunk = data[start : start + 8]
        response = client.put(
            f"/v1/jobs/{job_id}/archives/{archive_type}/chunks/{index}",
            headers=auth_headers(
                {
                    "X-Chunk-Sha256": hashlib.sha256(chunk).hexdigest(),
                    "X-Chunk-Size": str(len(chunk)),
                },
                token=token,
            ),
            content=chunk,
        )
        assert response.status_code == 200


def test_auth_required(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/v1/health")
    assert response.status_code == 401


def test_single_key_token_auto_binds_job_to_kaggle_key(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            headers=auth_headers(token="user-a-token"),
        )
        response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers(token="user-a-token"))
        stored = app.state.db.get_job(job_id)

    assert response.status_code == 200
    assert response.json()["kaggle_key_id"] == "ka"
    assert stored["kaggle_key_id"] == "ka"
    assert stored["relay_token_id"] == "user-a"


def test_multi_key_token_auto_selects_key_and_enforces_job_access(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"alice/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"alice/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def quota(self):
            remaining = 10.0 if self.credentials.username == "alice" else 5.0
            return {
                "available": True,
                "refresh_at": "2026-06-20T00:00:00",
                "accelerators": [
                    {"resource": "GPU", "used_hours": 30.0 - remaining, "remaining_hours": remaining, "total_hours": 30.0},
                ],
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        auto_key = client.post(
            "/v1/jobs",
            headers=auth_headers(token="admin-token"),
            json=job_request_body(
                dataset_zip,
                kernel_zip,
                dataset_ref="alice/data",
                kernel_ref="alice/kernel",
            ),
        )
        forbidden_key = client.post(
            "/v1/jobs",
            headers=auth_headers(token="user-a-token"),
            json=job_request_body(dataset_zip, kernel_zip, kaggle_key_id="kb"),
        )
        admin_job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            kaggle_key_id="ka",
            headers=auth_headers(token="admin-token"),
        )
        user_a_job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            kaggle_key_id="ka",
            headers=auth_headers(token="user-a-token"),
        )
        admin_get = client.get(f"/v1/jobs/{admin_job_id}", headers=auth_headers(token="admin-token"))
        same_key_non_owner_get = client.get(f"/v1/jobs/{admin_job_id}", headers=auth_headers(token="user-a-token"))
        admin_non_owner_get = client.get(f"/v1/jobs/{user_a_job_id}", headers=auth_headers(token="admin-token"))
        other_get = client.get(f"/v1/jobs/{admin_job_id}", headers=auth_headers(token="user-b-token"))
        admin_jobs = client.get("/v1/jobs", headers=auth_headers(token="admin-token")).json()
        user_a_jobs = client.get("/v1/jobs", headers=auth_headers(token="user-a-token")).json()

    assert auto_key.status_code == 200
    assert auto_key.json()["kaggle_key_id"] == "ka"
    assert forbidden_key.status_code == 403
    assert admin_get.status_code == 200
    assert same_key_non_owner_get.status_code == 404
    assert admin_non_owner_get.status_code == 404
    assert other_get.status_code == 404
    assert {job["job_id"] for job in admin_jobs} == {auto_key.json()["job_id"], admin_job_id}
    assert {job["job_id"] for job in user_a_jobs} == {user_a_job_id}


def test_relay_tokens_sharing_a_kaggle_key_cannot_resume_each_others_jobs(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    config = multi_key_auth_config()
    config["relay_tokens"].append(
        {"id": "user-a-peer", "token": "user-a-peer-token", "allowed_kaggle_key_ids": ["ka"]}
    )
    app = create_app(make_auth_config_settings(tmp_path, config))

    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            headers=auth_headers(token="user-a-token"),
        )
        peer_get = client.get(f"/v1/jobs/{job_id}", headers=auth_headers(token="user-a-peer-token"))
        chunk = dataset_zip[:8]
        peer_upload = client.put(
            f"/v1/jobs/{job_id}/archives/dataset/chunks/0",
            headers=auth_headers(
                {
                    "X-Chunk-Sha256": hashlib.sha256(chunk).hexdigest(),
                    "X-Chunk-Size": str(len(chunk)),
                },
                token="user-a-peer-token",
            ),
            content=chunk,
        )
        peer_complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers(token="user-a-peer-token"))

    assert peer_get.status_code == 404
    assert peer_upload.status_code == 404
    assert peer_complete.status_code == 404


def test_dataset_cache_is_scoped_by_kaggle_key(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        app.state.db.upsert_dataset_cache(
            dataset_ref="demo/data",
            payload_hash="payload-1",
            status="ready",
            dataset_status="ready",
            source_job_id="previous",
            kaggle_key_id="ka",
        )
        ka_job = create_job(
            client,
            dataset_zip,
            kernel_zip,
            payload_hash="payload-1",
            headers=auth_headers(token="user-a-token"),
        )
        kb_job = create_job(
            client,
            dataset_zip,
            kernel_zip,
            payload_hash="payload-1",
            headers=auth_headers(token="user-b-token"),
        )
        ka_response = client.get(f"/v1/jobs/{ka_job}", headers=auth_headers(token="user-a-token"))
        kb_response = client.get(f"/v1/jobs/{kb_job}", headers=auth_headers(token="user-b-token"))

    assert ka_response.json()["dataset_cache_hit"] is True
    assert ka_response.json()["dataset_upload_required"] is False
    assert kb_response.json()["dataset_cache_hit"] is False
    assert kb_response.json()["dataset_upload_required"] is True


def test_create_job_auto_selects_available_kaggle_key_by_quota(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"bob/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"bob/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def quota(self):
            remaining = 30.0 if self.credentials.username == "alice" else 12.0
            return {
                "available": True,
                "refresh_at": "2026-06-20T00:00:00",
                "accelerators": [
                    {"resource": "GPU", "used_hours": 30.0 - remaining, "remaining_hours": remaining, "total_hours": 30.0},
                ],
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            headers=auth_headers(token="admin-token"),
            json=job_request_body(
                dataset_zip,
                kernel_zip,
                dataset_ref="bob/data",
                kernel_ref="bob/kernel",
            ),
        )

    assert response.status_code == 200
    assert response.json()["kaggle_key_id"] == "kb"
    assert app.state.db.get_job(response.json()["job_id"])["kaggle_key_id"] == "kb"


def test_create_job_auto_select_rejects_owner_without_allowed_key(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"carol/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"carol/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def quota(self):
            return {
                "available": True,
                "refresh_at": "2026-06-20T00:00:00",
                "accelerators": [
                    {"resource": "GPU", "used_hours": 0.0, "remaining_hours": 30.0, "total_hours": 30.0},
                ],
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            headers=auth_headers(token="admin-token"),
            json=job_request_body(
                dataset_zip,
                kernel_zip,
                dataset_ref="carol/data",
                kernel_ref="carol/kernel",
            ),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "no allowed kaggle key matches requested owner carol"


def test_create_job_returns_conflict_when_all_allowed_key_quotas_are_exhausted(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"bob/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"bob/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def quota(self):
            return {
                "available": True,
                "refresh_at": "2026-06-20T00:00:00",
                "accelerators": [
                    {"resource": "GPU", "used_hours": 30.0, "remaining_hours": 0.0, "total_hours": 30.0},
                ],
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            headers=auth_headers(token="admin-token"),
            json=job_request_body(
                dataset_zip,
                kernel_zip,
                dataset_ref="bob/data",
                kernel_ref="bob/kernel",
            ),
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "no allowed kaggle key has remaining GPU quota"


def test_kaggle_account_respects_token_key_permissions(tmp_path, monkeypatch):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def account(self):
            return {"username": self.credentials.username, "authenticated": True}

        def quota(self):
            remaining_hours = 29.0 if self.credentials.id == "kb" else 3.0
            return {
                "available": True,
                "refresh_at": "2026-06-20T00:00:00",
                "accelerators": [
                    {
                        "resource": "GPU",
                        "used_hours": 30.0 - remaining_hours,
                        "remaining_hours": remaining_hours,
                        "total_hours": 30.0,
                    },
                ],
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        single_key = client.get("/v1/kaggle/account", headers=auth_headers(token="user-a-token"))
        forbidden_key = client.get(
            "/v1/kaggle/account?kaggle_key_id=kb",
            headers=auth_headers(token="user-a-token"),
        )
        admin_auto_key = client.get("/v1/kaggle/account", headers=auth_headers(token="admin-token"))
        admin_key = client.get(
            "/v1/kaggle/account?kaggle_key_id=kb",
            headers=auth_headers(token="admin-token"),
        )

    assert single_key.status_code == 200
    assert single_key.json()["kaggle_key_id"] == "ka"
    assert single_key.json()["username"] == "alice"
    assert forbidden_key.status_code == 403
    assert admin_auto_key.status_code == 200
    assert admin_auto_key.json()["kaggle_key_id"] == "kb"
    assert admin_auto_key.json()["username"] == "bob"
    assert admin_key.status_code == 200
    assert admin_key.json()["kaggle_key_id"] == "kb"
    assert admin_key.json()["quota"]["accelerators"][0]["resource"] == "GPU"


def test_kaggle_accounts_lists_only_accessible_keys(tmp_path, monkeypatch):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def account(self):
            return {"username": self.credentials.username, "authenticated": True}

        def quota(self):
            return {
                "available": True,
                "refresh_at": "2026-06-20T00:00:00",
                "accelerators": [
                    {"resource": "GPU", "used_hours": 2.5, "remaining_hours": 27.5, "total_hours": 30.0},
                    {"resource": "TPU", "used_hours": 0.0, "remaining_hours": 20.0, "total_hours": 20.0},
                ],
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        user_response = client.get("/v1/kaggle/accounts", headers=auth_headers(token="user-a-token"))
        admin_response = client.get("/v1/kaggle/accounts", headers=auth_headers(token="admin-token"))

    assert user_response.status_code == 200
    assert [item["kaggle_key_id"] for item in user_response.json()["accounts"]] == ["ka"]
    assert user_response.json()["accounts"][0]["quota"]["available"] is True
    assert admin_response.status_code == 200
    assert [item["kaggle_key_id"] for item in admin_response.json()["accounts"]] == ["ka", "kb"]


def test_kaggle_account_probe_respects_token_key_permissions(tmp_path, monkeypatch):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            self.credentials = credentials

        def probe_username_write_access(self):
            return {
                "ok": True,
                "username": self.credentials.username,
                "dataset_ref": f"{self.credentials.username}/relay-probe-test",
                "created": True,
                "cleanup_ok": True,
                "cleanup_error": "",
                "error": "",
            }

    monkeypatch.setattr("app.main.KaggleAdapter", FakeAdapter)

    with TestClient(app) as client:
        allowed = client.post(
            "/v1/kaggle/account/probe?kaggle_key_id=ka",
            headers=auth_headers(token="user-a-token"),
        )
        forbidden = client.post(
            "/v1/kaggle/account/probe?kaggle_key_id=kb",
            headers=auth_headers(token="user-a-token"),
        )
        admin_missing_key = client.post(
            "/v1/kaggle/account/probe",
            headers=auth_headers(token="admin-token"),
        )

    assert allowed.status_code == 200
    assert allowed.json()["ok"] is True
    assert allowed.json()["username"] == "alice"
    assert allowed.json()["dataset_ref"] == "alice/relay-probe-test"
    assert forbidden.status_code == 403
    assert admin_missing_key.status_code == 400


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

    def fake_process_job(settings, db, job_id, auth_store=None):
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
    assert status["can_download"] is True
    assert status["artifact_size"] > 0
    assert status["artifact_filename"] == f"{job_id}-artifacts.zip"
    assert status["download_unavailable_reason"] == ""
    assert download.status_code == 200
    assert f'filename="{job_id}-artifacts.zip"' in download.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert archive.read("best.pt") == b"pt"


def test_complete_job_reports_missing_artifact_as_not_downloadable(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = create_job(client, dataset_zip, kernel_zip)
        missing_path = tmp_path / "artifacts" / job_id / "artifacts.zip"
        app.state.db.update_job(
            job_id,
            status="complete",
            progress=100,
            artifact_path=str(missing_path),
        )

        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers())
        download = client.get(f"/v1/jobs/{job_id}/artifacts.zip", headers=auth_headers())

    assert status.status_code == 200
    payload = status.json()
    assert payload["can_download"] is False
    assert payload["artifact_size"] is None
    assert payload["artifact_filename"] == f"{job_id}-artifacts.zip"
    assert payload["download_unavailable_reason"] == "artifact file is missing"
    assert download.status_code == 404
    assert download.json()["detail"] == "artifact file is missing"


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

    def fake_process_job(settings, db, job_id, auth_store=None):
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


def test_complete_rejects_metadata_owner_mismatch_for_selected_kaggle_key(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"bob/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"bob/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))
    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            kaggle_key_id="ka",
            dataset_ref="bob/data",
            kernel_ref="bob/kernel",
            headers=auth_headers(token="admin-token"),
        )
        upload_all(client, job_id, "dataset", dataset_zip, token="admin-token")
        upload_all(client, job_id, "kernel", kernel_zip, token="admin-token")
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers(token="admin-token"))

    assert complete.status_code == 400
    assert complete.json()["detail"] == "dataset-metadata.json owner bob does not match Kaggle key username alice"


def test_complete_rejects_kernel_metadata_owner_mismatch_when_dataset_is_cached(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"bob/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"bob/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))
    with TestClient(app) as client:
        app.state.db.upsert_dataset_cache(
            dataset_ref="bob/data",
            payload_hash="payload-1",
            status="ready",
            dataset_status="ready",
            source_job_id="previous",
            kaggle_key_id="ka",
        )
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            payload_hash="payload-1",
            kaggle_key_id="ka",
            dataset_ref="bob/data",
            kernel_ref="bob/kernel",
            headers=auth_headers(token="admin-token"),
        )
        upload_all(client, job_id, "kernel", kernel_zip, token="admin-token")
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers(token="admin-token"))

    assert complete.status_code == 400
    assert complete.json()["detail"] == "kernel-metadata.json owner bob does not match Kaggle key username alice"


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
            def __init__(self, _settings, _log, credentials=None):
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


def test_worker_uses_job_bound_kaggle_credentials(tmp_path, monkeypatch):
    dataset_zip = build_zip({"dataset-metadata.json": b'{"id":"bob/data"}'})
    kernel_zip = build_zip({
        "kernel-metadata.json": b'{"id":"bob/kernel","code_file":"train.py"}',
        "train.py": b"print(1)",
    })
    settings = make_auth_config_settings(tmp_path, multi_key_auth_config())
    monkeypatch.setattr("app.main.process_job", lambda *_args, **_kwargs: None)
    app = create_app(settings)
    captured = {}

    class FakeAdapter:
        def __init__(self, _settings, _log, credentials=None):
            captured["id"] = credentials.id
            captured["username"] = credentials.username
            captured["key"] = credentials.key

        def upload_dataset(self, *_args, **_kwargs):
            pass

        def wait_dataset(self, *_args, **kwargs):
            captured["wait_dataset_kwargs"] = kwargs
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

    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            dataset_ref="bob/data",
            kernel_ref="bob/kernel",
            headers=auth_headers(token="user-b-token"),
        )
        upload_all(client, job_id, "dataset", dataset_zip, token="user-b-token")
        upload_all(client, job_id, "kernel", kernel_zip, token="user-b-token")
        complete = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers(token="user-b-token"))
        assert complete.status_code == 200

        process_job(settings, app.state.db, job_id, app.state.auth_store)
        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers(token="user-b-token")).json()

    assert captured == {
        "id": "kb",
        "username": "bob",
        "key": "bob-key",
        "wait_dataset_kwargs": {"permission_grace_seconds": 300},
    }
    assert status["status"] == "complete"


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


def test_relay_token_progress_update_requires_job_owner(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    config = multi_key_auth_config()
    config["relay_tokens"].append(
        {"id": "user-a-peer", "token": "user-a-peer-token", "allowed_kaggle_key_ids": ["ka"]}
    )
    app = create_app(make_auth_config_settings(tmp_path, config))

    with TestClient(app) as client:
        job_id = create_job(
            client,
            dataset_zip,
            kernel_zip,
            headers=auth_headers(token="user-a-token"),
        )
        peer_response = client.post(
            f"/v1/jobs/{job_id}/progress",
            headers=auth_headers(token="user-a-peer-token"),
            json={"remote_progress": 50, "message": "wrong owner"},
        )
        owner_response = client.post(
            f"/v1/jobs/{job_id}/progress",
            headers=auth_headers(token="user-a-token"),
            json={"remote_progress": 50, "message": "right owner"},
        )

    assert peer_response.status_code == 401
    assert owner_response.status_code == 200
    assert owner_response.json()["recent_logs"][-1] == "right owner"


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


def test_wait_dataset_retries_transient_forbidden_with_permission_grace(tmp_path, monkeypatch):
    class Result:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    settings = make_settings(tmp_path)
    settings.dataset_poll_seconds = 1
    adapter = KaggleAdapter(settings, lambda _message: None)
    results = [
        Result(1, "403 Client Error: Forbidden for url: https://api.kaggle.com/..."),
        Result(0, "ready"),
    ]
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        return results.pop(0)

    monkeypatch.setattr("app.kaggle_adapter.time.sleep", lambda _seconds: None)
    adapter._run = fake_run

    assert adapter.wait_dataset("demo/private-dataset", permission_grace_seconds=60) == "ready"
    assert len(calls) == 2

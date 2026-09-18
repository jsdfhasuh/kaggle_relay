import pytest
from fastapi.testclient import TestClient

from test_relay_api import (
    auth_headers, make_settings, make_auth_config_settings, multi_key_auth_config, seed_job,
)
from app.main import create_app, delete_job_files_and_record


@pytest.mark.parametrize("status", ["failed", "canceled", "complete", "receiving"])
def test_delete_removes_history_and_files_but_preserves_remote_cache(tmp_path, status):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = seed_job(app, status)
        other = seed_job(app, "failed")
        db = app.state.db
        db.append_log(job_id, "failure details")
        db.add_chunk(job_id, "dataset", 0, 8, "a" * 64)
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO kaggle_dataset_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("", "demo/data", "hash", "ready", "ready", job_id, 1, 1),
            )
        for root in (app.state.settings.jobs_dir, app.state.settings.artifacts_dir):
            (root / job_id).mkdir(parents=True, exist_ok=True)
            (root / job_id / "file.zip").write_bytes(b"data")
            (root / other).mkdir(parents=True, exist_ok=True)
        assert client.delete(f"/v1/jobs/{job_id}", headers=auth_headers()).status_code == 204
        assert client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).status_code == 404
        assert [j["job_id"] for j in client.get("/v1/jobs", headers=auth_headers()).json()] == [other]
        with db.connect() as conn:
            for table in ("logs", "chunks"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE job_id=?", (job_id,)).fetchone()[0] == 0
            assert conn.execute("SELECT source_job_id FROM kaggle_dataset_cache").fetchone()[0] == ""
        for root in (app.state.settings.jobs_dir, app.state.settings.artifacts_dir):
            assert not (root / job_id).exists()
            assert (root / other).exists()
        assert client.delete(f"/v1/jobs/{job_id}", headers=auth_headers()).status_code == 404


@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError])
def test_delete_cleanup_failure_retains_record_and_retries(tmp_path, monkeypatch, failure):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = seed_job(app, "failed")
        (app.state.settings.jobs_dir / job_id).mkdir(parents=True, exist_ok=True)
        with monkeypatch.context() as patch:
            def denied(_path):
                raise failure("cleanup incomplete")
            patch.setattr("app.main.shutil.rmtree", denied)
            assert client.delete(f"/v1/jobs/{job_id}", headers=auth_headers()).status_code == 503
        assert app.state.db.get_job(job_id)["status"] == "failed"
        assert client.delete(f"/v1/jobs/{job_id}", headers=auth_headers()).status_code == 204


@pytest.mark.parametrize("busy", ["active_job_ids", "active_uploads"])
def test_terminal_job_cannot_be_deleted_while_io_is_active(tmp_path, busy):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        job_id = seed_job(app, "failed")
        collection = getattr(app.state, busy)
        if isinstance(collection, set):
            collection.add(job_id)
        else:
            collection[job_id] = 1
        try:
            assert client.delete(f"/v1/jobs/{job_id}", headers=auth_headers()).status_code == 409
            assert app.state.db.get_job(job_id)
        finally:
            collection.clear()


def test_delete_validates_storage_boundary_before_cleanup(tmp_path):
    app = create_app(make_settings(tmp_path))
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(OSError, match="storage root"):
        delete_job_files_and_record(app.state.settings, app.state.db, "..")
    assert sentinel.read_text() == "keep"


def test_search_before_limit_and_delete_enforce_owner_and_key_permissions(tmp_path):
    config = multi_key_auth_config()
    config["relay_tokens"].append({"id": "peer", "token": "peer-token", "allowed_kaggle_key_ids": ["ka"]})
    app = create_app(make_auth_config_settings(tmp_path, config))
    with TestClient(app) as client:
        target = seed_job(app, "failed", kernel_ref="alice/Needle-Training")
        app.state.db.update_job(target, relay_token_id="user-a", kaggle_key_id="ka", error="CUDA memory")
        for _ in range(3):
            newer = seed_job(app, "complete")
            app.state.db.update_job(newer, relay_token_id="user-a", kaggle_key_id="ka")
        for term in ("needle", "CUDA", target):
            results = client.get("/v1/jobs", params={"q": term, "limit": 1, "status": "failed"},
                                 headers=auth_headers(token="user-a-token")).json()
            assert [j["job_id"] for j in results] == [target]
        assert client.get("/v1/jobs", params={"q": "%"}, headers=auth_headers(token="user-a-token")).json() == []
        for token in ("peer-token", "user-b-token"):
            assert client.get("/v1/jobs", params={"q": "needle"}, headers=auth_headers(token=token)).json() == []
            assert client.delete(f"/v1/jobs/{target}", headers=auth_headers(token=token)).status_code == 404
        assert client.delete(f"/v1/jobs/{target}").status_code == 401
        assert client.delete(f"/v1/jobs/{target}", headers=auth_headers(token="user-a-token")).status_code == 204


def test_summary_counts_all_jobs_beyond_list_limit_and_separates_queue(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        for _ in range(205):
            seed_job(app, "failed")
        for status in ("receiving", "assembling", "uploading_dataset", "waiting_dataset", "pushing_kernel",
                       "waiting_kernel", "cancel_requested", "downloading_output", "queued", "complete", "canceled"):
            seed_job(app, status)
        assert len(client.get("/v1/jobs?limit=200", headers=auth_headers()).json()) == 200
        response = client.get("/v1/jobs/summary", headers=auth_headers())
        assert response.status_code == 200
        assert response.json() == {"total": 216, "in_progress": 8, "queued": 1, "failed": 205, "complete": 1, "canceled": 1}
        assert client.get("/v1/jobs/summary?q=absent&status=failed&limit=1", headers=auth_headers()).json() == response.json()
        assert client.get("/v1/jobs/summary").status_code == 401


def test_summary_uses_same_visibility_as_list_including_dynamic_jobs(tmp_path):
    config = multi_key_auth_config()
    config["relay_tokens"].append({"id": "peer", "token": "peer-token", "allowed_kaggle_key_ids": ["ka"]})
    app = create_app(make_auth_config_settings(tmp_path, config))
    with TestClient(app) as client:
        for owner, key in (("user-a", "ka"), ("user-b", "kb"), ("peer", "ka"), ("", "ka"), ("user-a", "kb")):
            job_id = seed_job(app, "failed")
            app.state.db.update_job(job_id, relay_token_id=owner, kaggle_key_id=key)
        pending = seed_job(app, "receiving")
        app.state.db.update_job(pending, relay_token_id="user-a", kaggle_key_id="kb", scheduling_mode="dynamic",
                                assignment_state="pending", eligible_accounts='{"ka":"alice","kb":"bob"}')
        for token, total in (("user-a-token", 2), ("user-b-token", 1), ("peer-token", 1), ("admin-token", 6)):
            counts = client.get("/v1/jobs/summary", headers=auth_headers(token=token)).json()
            jobs = client.get("/v1/jobs", headers=auth_headers(token=token)).json()
            assert counts["total"] == len(jobs) == total
        assert app.state.db.job_status_counts(set()) == {}
        app.state.db.update_job(pending, status="canceled")
        assert client.delete(f"/v1/jobs/{pending}", headers=auth_headers(token="user-a-token")).status_code == 204
        counts = client.get("/v1/jobs/summary", headers=auth_headers(token="user-a-token")).json()
        assert counts["total"] == counts["failed"] == 1
        assert counts["canceled"] == counts["in_progress"] == 0

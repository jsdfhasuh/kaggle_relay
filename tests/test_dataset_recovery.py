import asyncio
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from test_relay_api import make_settings, seed_job, auth_headers
from test_a3_content_identity import fixture_adapter

from app.dataset_recovery import (
    RECHECK_WINDOW_SECONDS, schedule_dataset_rechecks, schedule_recheck,
)
from app.dataset_verification_process import DatasetVerificationError
from app.main import create_app, recover_job_after_restart, run_worker_item, job_response
from app.upload_intent import content_digest, intent_path, write_intent
from app.worker import job_paths, process_job, recheck_dataset_job


def retained_job(tmp_path):
    app = create_app(make_settings(tmp_path))
    job_id = seed_job(app, "waiting_dataset")
    dataset_dir = job_paths(app.state.settings, job_id)["dataset_dir"]
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "file.txt").write_bytes(b"original")
    path = intent_path(app.state.settings.storage_dir, dataset_dir, "demo/data")
    write_intent(path, {"schema_version": 1, "state": "accepted", "dataset_ref": "demo/data",
                       "dataset_dir": str(dataset_dir.absolute()), "version_number": 2,
                       "content_sha256": content_digest(dataset_dir)})
    app.state.db.update_job(job_id, error="dataset_upload_outcome_unknown: original candidate retained; "
                                        "dataset_verification_retry_exhausted: HTTP 429")
    return app, job_id, path


def scan(app):
    asyncio.run(schedule_dataset_rechecks(app))


def test_legacy_job_is_scheduled_and_claimed_once_without_client(tmp_path, monkeypatch):
    app, job_id, _ = retained_job(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr("app.dataset_recovery.time.time", lambda: clock[0])
    scan(app)
    row = app.state.db.get_job(job_id)
    assert row["dataset_recheck_at"] == 1300 and row["dataset_recheck_state"] == "scheduled"
    assert app.state.queue.empty()
    scan(app)
    assert app.state.db.get_job(job_id)["dataset_recheck_at"] == 1300
    clock[0] = 1300
    async def concurrent_scans():
        await asyncio.gather(schedule_dataset_rechecks(app), schedule_dataset_rechecks(app))
    asyncio.run(concurrent_scans())
    assert app.state.queue.qsize() == 1
    assert app.state.queue.get_nowait() == {"action": "recheck_dataset", "job_id": job_id}
    row = app.state.db.get_job(job_id)
    assert row["status"] == "queued" and row["dataset_recheck_count"] == 1
    assert job_response(app.state.db, job_id).dataset_recheck_state == "checking"


@pytest.mark.parametrize("case", ["active", "cancel", "complete", "failed", "blocked", "exhausted"])
def test_scan_never_revives_ineligible_jobs(tmp_path, case):
    app, job_id, _ = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id)
    app.state.db.update_job(job_id, dataset_recheck_at=1)
    if case == "active":
        app.state.active_job_ids.add(job_id)
    elif case == "cancel":
        app.state.db.update_job(job_id, cancel_requested_at=1)
    elif case in {"complete", "failed"}:
        app.state.db.update_job(job_id, status=case)
    else:
        app.state.db.update_job(job_id, dataset_recheck_state=case)
    scan(app)
    assert app.state.queue.empty()


@pytest.mark.parametrize("case", ["missing", "malformed", "rejected", "wrong_scope"])
def test_missing_or_changed_candidate_is_not_reuploaded(tmp_path, case):
    app, job_id, path = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id)
    app.state.db.update_job(job_id, dataset_recheck_at=1)
    if case == "missing":
        path.unlink()
    elif case == "malformed":
        path.write_text("null")
    else:
        value = json.loads(path.read_text())
        value.update({"state": "rejected"} if case == "rejected" else {"dataset_ref": "other/data"})
        write_intent(path, value)
    scan(app)
    assert app.state.queue.empty()
    assert app.state.db.get_job(job_id)["dataset_recheck_state"] == "blocked"


def test_restart_preserves_deadline_and_resumes_claimed_action(tmp_path):
    app, job_id, _ = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id)
    row = app.state.db.get_job(job_id)
    assert recover_job_after_restart(app.state.settings, app.state.db, app.state.auth_store, row) is None
    reopened = create_app(app.state.settings)
    assert reopened.state.db.get_job(job_id)["dataset_recheck_at"] == row["dataset_recheck_at"]
    app.state.db.update_job(job_id, dataset_recheck_at=1)
    scan(app)
    row = app.state.db.get_job(job_id)
    assert recover_job_after_restart(app.state.settings, app.state.db, app.state.auth_store, row) == {
        "action": "recheck_dataset", "job_id": job_id}


@pytest.mark.parametrize("case", ["deleted_owner", "disabled_key", "revoked_permission"])
def test_recheck_cannot_bypass_changed_account_permissions(tmp_path, case):
    app, job_id, _ = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id)
    app.state.db.update_job(job_id, dataset_recheck_at=1, relay_token_id="user", kaggle_key_id="key")
    app.state.auth_store = SimpleNamespace(
        legacy=False,
        _tokens=[] if case == "deleted_owner" else [("fixture", SimpleNamespace(id="user"))],
        _kaggle_keys={"key": SimpleNamespace(enabled=case != "disabled_key")},
        can_access_key=lambda *args: case != "revoked_permission",
    )
    scan(app)
    assert app.state.queue.empty()
    assert app.state.db.get_job(job_id)["dataset_recheck_state"] == "blocked"


def test_backoff_and_retry_after_never_retry_early(tmp_path):
    app, job_id, _ = retained_job(tmp_path)
    for count, delay in [(0, 300), (1, 600), (2, 1200), (3, 1800), (100, 1800)]:
        app.state.db.update_job(job_id, dataset_recheck_count=count)
        schedule_recheck(app.state.db, job_id, now=1000)
        assert app.state.db.get_job(job_id)["dataset_recheck_at"] == 1000 + delay
    schedule_recheck(app.state.db, job_id, retry_after=3600, now=1000)
    assert app.state.db.get_job(job_id)["dataset_recheck_at"] == 4600
    schedule_recheck(app.state.db, job_id, retry_after=RECHECK_WINDOW_SECONDS, now=1000)
    row = app.state.db.get_job(job_id)
    assert row["dataset_recheck_state"] == "exhausted" and row["dataset_recheck_at"] is None
    assert row["status"] == "waiting_dataset"


def test_elapsed_window_and_unknown_legacy_failure_need_manual_inspection(tmp_path, monkeypatch):
    app, job_id, _ = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id, now=1000)
    monkeypatch.setattr("app.dataset_recovery.time.time", lambda: 1000 + RECHECK_WINDOW_SECONDS)
    scan(app)
    assert app.state.db.get_job(job_id)["dataset_recheck_state"] == "exhausted"
    app.state.db.update_job(job_id, dataset_recheck_state="", error="dataset_upload_outcome_unknown: unauthorized")
    scan(app)
    assert app.state.db.get_job(job_id)["dataset_recheck_state"] == "blocked"
    assert app.state.queue.empty()


def test_manual_recovery_racing_scan_only_claims_once(tmp_path):
    from fastapi.testclient import TestClient
    app, job_id, _ = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id)
    app.state.db.update_job(job_id, dataset_recheck_at=1)
    scan(app)
    response = TestClient(app).post(f"/v1/jobs/{job_id}/complete", headers=auth_headers())
    assert response.status_code == 200
    assert app.state.queue.qsize() == 1


@pytest.mark.parametrize("category", ["transport", "publication"])
def test_background_worker_checks_same_version_and_pushes_only_once(tmp_path, monkeypatch, category):
    api, adapter, _ = fixture_adapter(tmp_path, monkeypatch)
    app = create_app(adapter.settings)
    job_id = seed_job(app, "queued", dataset_ref="owner/data", kernel_ref="owner/kernel")
    paths = job_paths(adapter.settings, job_id)
    paths["dataset_dir"].mkdir(parents=True)
    paths["kernel_dir"].mkdir(parents=True)
    (paths["dataset_dir"] / "file.txt").write_bytes(b"original")
    (paths["dataset_dir"] / "dataset-metadata.json").write_text('{"id":"owner/data"}')
    metadata = paths["kernel_dir"] / "kernel-metadata.json"
    metadata.write_text('{"id":"owner/kernel","code_file":"train.py","dataset_sources":["owner/data"]}')
    (paths["kernel_dir"] / "train.py").write_text("print('fixture')")
    pushes = Mock(return_value="pushed")
    adapter.push_kernel = pushes
    monkeypatch.setattr("app.worker.KaggleAdapter", lambda *a, **kw: adapter)
    monkeypatch.setattr("app.worker.finish_kernel_job", lambda *a, **kw: None)
    wait = adapter.wait_dataset
    adapter.wait_dataset = Mock(side_effect=DatasetVerificationError(
        "retry exhausted" if category == "transport" else "payload_publication_timeout: not ready",
        category, 429 if category == "transport" else None, 900))
    process_job(adapter.settings, app.state.db, job_id)
    assert api.uploads == 1 and pushes.call_count == 0
    assert app.state.db.get_job(job_id)["dataset_recheck_state"] == "scheduled"
    app.state.db.update_job(job_id, dataset_recheck_at=1)
    scan(app)
    adapter.wait_dataset = wait
    adapter.upload_dataset = Mock(side_effect=AssertionError("automatic recheck must never upload"))
    api.latest = 8
    asyncio.run(run_worker_item(app, app.state.queue.get_nowait()))
    assert api.downloads == ["owner/data/7"] and api.uploads == 1 and pushes.call_count == 1
    assert json.loads(metadata.read_text())["dataset_sources"] == ["owner/data/7"]
    assert app.state.db.get_job(job_id)["dataset_recheck_state"] == ""
    scan(app)
    assert app.state.queue.empty()


@pytest.mark.parametrize("change", ["missing", "content"])
def test_candidate_is_revalidated_in_worker_after_queue_claim(tmp_path, monkeypatch, change):
    app, job_id, path = retained_job(tmp_path)
    schedule_recheck(app.state.db, job_id)
    app.state.db.update_job(job_id, dataset_recheck_at=1)
    scan(app)
    adapter = Mock()
    monkeypatch.setattr("app.worker.KaggleAdapter", lambda *a, **kw: adapter)
    kernel_dir = job_paths(app.state.settings, job_id)["kernel_dir"]
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        '{"id":"demo/kernel","code_file":"train.py","dataset_sources":["demo/data"]}')
    (kernel_dir / "train.py").write_text("print('fixture')")
    if change == "missing":
        path.unlink()
    else:
        (job_paths(app.state.settings, job_id)["dataset_dir"] / "file.txt").write_bytes(b"tampered")
    recheck_dataset_job(app.state.settings, app.state.db, job_id)
    adapter.upload_dataset.assert_not_called()
    adapter.push_kernel.assert_not_called()
    assert app.state.db.get_job(job_id)["status"] == "failed"
    assert "upload_intent_scope_or_content_mismatch" in app.state.db.get_job(job_id)["error"]

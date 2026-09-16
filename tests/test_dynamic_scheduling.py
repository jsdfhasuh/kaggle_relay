import asyncio
import hashlib
import json
import threading
import time

from fastapi.testclient import TestClient

from test_concurrency import pool_settings, until
from test_relay_api import auth_headers, build_zip, job_request_body, seed_job, upload_all
from app.auth_config import KaggleCredentials, RelayPrincipal
from app.main import create_app
from app.scheduler import schedule_pending_jobs, scheduler_loop
from app.worker import validate_payloads


def pending_job(app, job_id, *, owner="user0"):
    seed_job(app, "queued", job_id=job_id, dataset_ref="user0/data", kernel_ref="user0/kernel")
    app.state.db.update_job(job_id, relay_token_id=owner, kaggle_key_id="key0",
                           scheduling_mode="dynamic", assignment_state="pending",
                           eligible_accounts=json.dumps({"key0": "user0", "key1": "user1"}))
    return job_id


def occupy(app, key):
    job_id = seed_job(app, "waiting_kernel", job_id="busy-" + key)
    app.state.db.update_job(job_id, kaggle_key_id=key)
    return job_id


def enable_quota(monkeypatch, hours=None):
    hours = hours or {"key0": 30, "key1": 30}
    monkeypatch.setattr("app.kaggle_adapter.KaggleAdapter.quota", lambda adapter: {
        "available": True, "accelerators": [{"resource": "GPU", "remaining_hours": hours[adapter.credentials.id]}],
    })


def test_upload_then_dispatch_to_idle_account_preserves_frozen_identity(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    enable_quota(monkeypatch)
    occupy(app, "key0")
    client = TestClient(app)
    dataset = build_zip({"dataset-metadata.json": b'{"id":"user0/data"}', "data.bin": b"data"})
    kernel = build_zip({"kernel-metadata.json": b'{"id":"user0/kernel","code_file":"train.py","dataset_sources":["user0/data"]}',
                        "train.py": b"print(1)"})
    identity = dict(dataset_id="dataset", identity_sha256="a" * 64, run_id="run", run_identity_sha256="b" * 64)
    payload = job_request_body(dataset, kernel, dataset_ref="user0/data", kernel_ref="user0/kernel", identity=identity)
    payload["scheduling_mode"] = "dynamic"
    payload["callback_token_sha256"] = hashlib.sha256(b"test-callback").hexdigest()
    response = client.post("/v1/jobs", headers=auth_headers(token="fake-relay-token-0"), json=payload)
    assert response.status_code == 200
    job = response.json()
    assert job["assignment_state"] == "pending"
    assert job["eligible_accounts"] == {"key0": "user0", "key1": "user1"}
    for kind, content in (("dataset", dataset), ("kernel", kernel)):
        upload_all(client, job["job_id"], kind, content, token="fake-relay-token-0")
    response = client.post(f"/v1/jobs/{job['job_id']}/complete", headers=auth_headers(token="fake-relay-token-0"))
    assert response.status_code == 200
    asyncio.run(schedule_pending_jobs(app))
    current = client.get(f"/v1/jobs/{job['job_id']}", headers=auth_headers(token="fake-relay-token-0")).json()
    assert current["assignment_state"] == "bound" and current["kaggle_key_id"] == "key1"
    assert client.get(f"/v1/jobs/{job['job_id']}", headers=auth_headers(token="fake-relay-token-1")).status_code == 404
    assert current["dataset_ref"] == "user1/data" and current["kernel_ref"] == "user1/kernel"
    for field in [*identity, "dataset_archive_sha256", "kernel_archive_sha256"]:
        assert current[field] == payload[field]
    extracted = app.state.settings.jobs_dir / job["job_id"] / "extracted"
    validate_payloads(extracted / "dataset", extracted / "kernel", current["dataset_ref"], current["kernel_ref"],
                      app.state.auth_store.credentials_for("key1"))
    metadata = json.loads((extracted / "kernel/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["user1/data"]
    assert metadata["id"] == "user1/kernel"
    assert app.state.queue.get_nowait()["job_id"] == job["job_id"]
    callback = {"kernel_ref": "user0/kernel", "epoch": 1, "epochs": 10}
    assert client.post("/v1/jobs/by-kernel/progress", headers=auth_headers(token="test-callback"), json=callback).status_code == 409
    app.state.db.update_job(job["job_id"], status="pushing_kernel")
    assert client.post("/v1/jobs/by-kernel/progress", headers=auth_headers(token="wrong-callback"), json=callback).status_code == 401
    result = client.post("/v1/jobs/by-kernel/progress", headers=auth_headers(token="test-callback"), json=callback)
    assert result.status_code == 200 and result.json()["job_id"] == job["job_id"]
    assert result.json()["kernel_ref"] == "user1/kernel"
    assert client.post(f"/v1/jobs/{job['job_id']}/cancel", headers=auth_headers(token="fake-relay-token-0")).status_code == 200
    result = client.post("/v1/jobs/by-kernel/progress", headers=auth_headers(token="test-callback"), json=callback)
    assert result.status_code == 200 and result.json()["cancel_requested"]


def test_all_busy_wait_then_oldest_job_uses_first_freed_account(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    enable_quota(monkeypatch)
    busy_a, busy_b = occupy(app, "key0"), occupy(app, "key1")
    first, second = pending_job(app, "first"), pending_job(app, "second")
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.queue.empty()
    assert app.state.db.get_job(first)["assignment_state"] == "pending"
    app.state.db.finalize_job(busy_b, "complete")
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.db.get_job(first)["kaggle_key_id"] == "key1"
    assert app.state.db.get_job(second)["assignment_state"] == "pending"
    app.state.db.finalize_job(busy_a, "complete")
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.db.get_job(second)["kaggle_key_id"] == "key0"
    assert app.state.queue.qsize() == 2


def test_concurrent_dispatch_claims_each_job_and_account_once(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    enable_quota(monkeypatch)
    for i in range(5):
        pending_job(app, str(i))
    async def scenario():
        await asyncio.gather(schedule_pending_jobs(app), schedule_pending_jobs(app))
    asyncio.run(scenario())
    bound = [j for j in app.state.db.list_jobs_by_status({"queued"}) if j["assignment_state"] == "bound"]
    assert len(bound) == 2 and {j["kaggle_key_id"] for j in bound} == {"key0", "key1"}
    assert app.state.queue.qsize() == 2


def test_zero_quota_and_removed_permissions_are_not_eligible(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    enable_quota(monkeypatch, {"key0": 0, "key1": 20})
    job_id = pending_job(app, "wait")
    app.state.auth_store._tokens = [("fake", RelayPrincipal("user0", frozenset({"key0"})))]
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.db.get_job(job_id)["assignment_state"] == "pending"
    assert app.state.queue.empty()
    app.state.auth_store._tokens = []
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.db.get_job(job_id)["status"] == "failed"


def test_canceled_pending_jobs_never_submit(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    enable_quota(monkeypatch)
    job_id = pending_job(app, "canceled")
    response = TestClient(app).post(f"/v1/jobs/{job_id}/cancel", headers=auth_headers(token="fake-relay-token-0"))
    assert response.status_code == 200
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.queue.empty()
    assert app.state.db.get_job(job_id)["status"] == "canceled"


def test_pending_jobs_survive_restart_and_dispatch_once(tmp_path, monkeypatch):
    settings = pool_settings(tmp_path, count=2, shared=True)
    before = create_app(settings)
    job_id = pending_job(before, "restart")
    enable_quota(monkeypatch)
    started = []
    def process(settings, db, job_id, auth_store=None):
        started.append(db.get_job(job_id))
        db.finalize_job(job_id, "complete")
    monkeypatch.setattr("app.main.process_job", process)
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    with TestClient(app) as client:
        client.portal.call(until, lambda: app.state.db.get_job(job_id)["status"] == "complete")
    assert len(started) == 1 and started[0]["assignment_state"] == "bound"


def test_pending_dynamic_job_cannot_skip_dataset_upload_based_on_provisional_cache(tmp_path):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    job_id = pending_job(app, "cache")
    db = app.state.db
    db.update_job(job_id, status="receiving")
    db.upsert_dataset_cache("user0/data", "payload-1", "ready", '{"status":"ready","current_version_number":1}', "old", kaggle_key_id="key0")
    db.upsert_last_dataset_job("user0/data", "payload-1", "ready", "old", kaggle_key_id="key0")
    client = TestClient(app)
    headers = auth_headers(token="fake-relay-token-0")
    status = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
    assert status["dataset_upload_required"] and not status["dataset_cache_hit"]
    assert client.post(f"/v1/jobs/{job_id}/complete", headers=headers).status_code == 409


def test_explicit_key_keeps_fixed_binding_and_cannot_escape_permissions(tmp_path):
    app = create_app(pool_settings(tmp_path, count=2, shared=False))
    client = TestClient(app)
    payload = {**job_request_body(b"data", b"kernel"), "scheduling_mode": "dynamic", "kaggle_key_id": "key0"}
    result = client.post("/v1/jobs", headers=auth_headers(token="fake-relay-token-0"), json=payload)
    assert result.status_code == 200
    assert result.json()["scheduling_mode"] == "fixed" and result.json()["assignment_state"] == "bound"
    payload["kaggle_key_id"] = "key1"
    assert client.post("/v1/jobs", headers=auth_headers(token="fake-relay-token-0"), json=payload).status_code == 403


def test_slow_quota_lookup_does_not_block_other_idle_accounts(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    release = threading.Event()
    def quota(adapter):
        if adapter.credentials.id == "key0":
            assert release.wait(5)
        return {"available": True, "accelerators": [{"resource": "GPU", "remaining_hours": 20}]}
    monkeypatch.setattr("app.kaggle_adapter.KaggleAdapter.quota", quota)
    job_id = pending_job(app, "slow-quota")
    try:
        started = time.monotonic()
        asyncio.run(schedule_pending_jobs(app))
        assert time.monotonic() - started < 2
        assert app.state.db.get_job(job_id)["kaggle_key_id"] == "key1"
        assert app.state.db.get_job(job_id)["assignment_state"] == "bound"
    finally:
        release.set()
        app.state.settings._quota_cache.shutdown()


def test_credential_aliases_share_one_account_slot(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    app.state.auth_store._kaggle_keys["key1"] = KaggleCredentials("key1", username="user0", key="fake")
    enable_quota(monkeypatch)
    for job_id in ("first", "second"):
        pending_job(app, job_id)
        app.state.db.update_job(job_id, eligible_accounts=json.dumps({"key0": "user0", "key1": "user0"}))
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.queue.qsize() == 1
    assert app.state.db.get_job("second")["assignment_state"] == "pending"


def test_permission_change_during_quota_lookup_is_rechecked(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    job_id = pending_job(app, "revoked")
    def quota(adapter):
        app.state.auth_store._tokens = []
        return {"available": True, "accelerators": [{"resource": "GPU", "remaining_hours": 20}]}
    monkeypatch.setattr("app.kaggle_adapter.KaggleAdapter.quota", quota)
    asyncio.run(schedule_pending_jobs(app))
    assert app.state.queue.empty()
    assert app.state.db.get_job(job_id)["status"] == "failed"


def test_pending_upload_remains_accessible_when_only_provisional_key_is_revoked(tmp_path):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    job_id = pending_job(app, "provisional")
    app.state.db.update_job(job_id, status="receiving")
    app.state.auth_store._tokens = [("token", RelayPrincipal("user0", frozenset({"key1"})))]
    client = TestClient(app)
    headers = auth_headers(token="token")
    assert client.get(f"/v1/jobs/{job_id}", headers=headers).status_code == 200
    result = client.get("/v1/jobs", headers=headers)
    assert result.status_code == 200 and [j["job_id"] for j in result.json()] == [job_id]
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=headers).status_code == 200


def test_scheduler_wakeup_racing_shutdown_cannot_keep_the_loop_alive(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    async def scenario():
        started = asyncio.Event()
        async def tick(app):
            started.set()
        monkeypatch.setattr("app.scheduler.schedule_pending_jobs", tick)
        for _ in range(30):
            app.state.shutdown_event.clear()
            started.clear()
            task = asyncio.create_task(scheduler_loop(app))
            await started.wait()
            await asyncio.sleep(0)
            app.state.scheduler_event.set()
            app.state.shutdown_event.set()
            task.cancel()
            try:
                await asyncio.wait_for(task, 1)
            except asyncio.CancelledError:
                pass
            assert task.done()
    asyncio.run(scenario())

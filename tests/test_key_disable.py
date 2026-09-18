import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from test_concurrency import pool_settings
from test_dynamic_scheduling import enable_quota, pending_job
from test_relay_api import auth_headers, job_request_body, seed_job
from app.auth_config import AuthConfigError, AuthStore
from app.main import create_app
from app.scheduler import schedule_pending_jobs


ADMIN = "key-disable-management-token-123456789"


def setup(tmp_path):
    settings = pool_settings(tmp_path, count=2, shared=True, admin_token=ADMIN)
    app = create_app(settings)
    return settings, app, TestClient(app)


def toggle(client, key, enabled):
    return client.patch(f"/v1/auth/kaggle-keys/{key}", headers=auth_headers(token=ADMIN),
                        json={"enabled": enabled, "disabled_reason": "GPU probe returned no CUDA"})


def test_disable_preserves_credentials_permissions_and_history_but_blocks_new_work(tmp_path):
    settings, app, client = setup(tmp_path)
    original = json.loads(settings.auth_config_path.read_text())
    old = seed_job(app, "complete")
    app.state.db.update_job(old, relay_token_id="user0", kaggle_key_id="key0")
    headers = auth_headers(token="fake-relay-token-0")
    assert client.patch("/v1/auth/kaggle-keys/key0", headers=headers, json={"enabled": False}).status_code == 403
    response = toggle(client, "key0", False)
    assert response.status_code == 200
    disabled = next(k for k in response.json()["kaggle_keys"] if k["id"] == "key0")
    assert disabled["enabled"] is False
    assert "fake-account-key" not in response.text
    saved = json.loads(settings.auth_config_path.read_text())
    assert saved["relay_tokens"] == original["relay_tokens"]
    for field in ("id", "username", "key"):
        assert saved["kaggle_keys"][0][field] == original["kaggle_keys"][0][field]
    assert client.get(f"/v1/jobs/{old}", headers=headers).status_code == 200
    assert app.state.auth_store.credentials_for("key0").key == "fake-account-key-0"
    body = {**job_request_body(b"dataset", b"kernel"), "scheduling_mode": "dynamic"}
    result = client.post("/v1/jobs", headers=headers, json=body)
    assert result.status_code == 200
    assert result.json()["eligible_accounts"] == {"key1": "user1"}
    body["kaggle_key_id"] = "key0"
    assert client.post("/v1/jobs", headers=headers, json=body).status_code == 409
    assert not AuthStore.from_settings(settings).credentials_for("key0").enabled


def test_pending_snapshot_waits_when_disabled_and_uses_reenabled_key(tmp_path, monkeypatch):
    settings, app, client = setup(tmp_path)
    enable_quota(monkeypatch)
    job_id = pending_job(app, "pending")
    assert toggle(client, "key0", False).status_code == 200
    assert toggle(client, "key1", False).status_code == 200
    asyncio.run(schedule_pending_jobs(app))
    job = app.state.db.get_job(job_id)
    assert job["status"] == "queued" and job["assignment_state"] == "pending"
    assert job["queue_reason"] == "waiting for an enabled authorized account"
    assert app.state.queue.empty()
    assert toggle(client, "key1", True).status_code == 200
    assert app.state.auth_store.credentials_for("key1").disabled_reason == ""
    asyncio.run(schedule_pending_jobs(app))
    job = app.state.db.get_job(job_id)
    assert job["assignment_state"] == "bound" and job["kaggle_key_id"] == "key1"


def test_disabled_config_requires_actual_boolean(tmp_path):
    settings, _, _ = setup(tmp_path)
    data = json.loads(settings.auth_config_path.read_text())
    data["kaggle_keys"][0]["enabled"] = "false"
    settings.auth_config_path.write_text(json.dumps(data))
    with pytest.raises(AuthConfigError, match="boolean"):
        AuthStore.from_settings(settings)

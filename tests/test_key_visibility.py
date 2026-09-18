import json

import pytest
from fastapi.testclient import TestClient

from test_ui_auth import auth_headers, make_auth_config_settings, multi_key_auth_config, ui_login
from test_relay_api import job_request_body
from app.auth_config import AuthConfigError, AuthStore
from app.main import create_app


def hidden_config():
    config = multi_key_auth_config()
    for token in config["relay_tokens"]:
        token.pop("can_view_keys", None)
    config["relay_tokens"][1]["allowed_kaggle_key_ids"] = ["ka", "kb"]
    return config


def fake_accounts(monkeypatch):
    monkeypatch.setattr("app.kaggle_adapter.KaggleAdapter.account", lambda adapter: {
        "username": adapter.credentials.username, "authenticated": True,
    })
    monkeypatch.setattr("app.kaggle_adapter.KaggleAdapter.quota", lambda adapter: {
        "available": True, "accelerators": [{"resource": "GPU", "remaining_hours": 0}],
    })


def test_no_view_permission_hides_config_and_lists_but_preserves_dynamic_protocol(tmp_path, monkeypatch):
    fake_accounts(monkeypatch)
    app = create_app(make_auth_config_settings(tmp_path, hidden_config()))
    client = TestClient(app)
    login = ui_login(client, "user-a-token")
    assert login.json()["allowed_kaggle_key_ids"] == []
    session = client.get("/v1/ui/session").json()
    assert session["can_view_keys"] is False and session["allowed_kaggle_key_ids"] == []
    for headers in ({}, auth_headers("user-a-token")):
        config = client.get("/v1/auth/config", headers=headers).json()
        assert config["kaggle_keys"] == config["allowed_kaggle_key_ids"] == []
        assert config["relay_tokens"][0]["allowed_kaggle_key_ids"] == []
        assert client.get("/v1/kaggle/accounts", headers=headers).status_code == 403
        assert client.get("/v1/kaggle/account?kaggle_key_id=ka", headers=headers).status_code == 403
    assert client.post("/v1/kaggle/account/probe", headers=auth_headers("user-a-token")).status_code == 403
    assert client.get("/v1/kaggle/account", headers=auth_headers("user-a-token")).status_code == 200
    payload = {**job_request_body(b"data", b"kernel"), "scheduling_mode": "dynamic"}
    job = client.post("/v1/jobs", headers=auth_headers("user-a-token"), json=payload)
    assert job.status_code == 200
    assert job.json()["eligible_accounts"] == {"ka": "alice", "kb": "bob"}
    assert job.json()["assignment_state"] == "pending"
    assert client.get('/v1/jobs/' + job.json()["job_id"], headers=auth_headers("user-b-token")).status_code == 404


def test_admin_grant_and_revoke_apply_to_existing_sessions_and_preserve_credentials(tmp_path, monkeypatch):
    fake_accounts(monkeypatch)
    settings = make_auth_config_settings(tmp_path, hidden_config())
    app = create_app(settings)
    client = TestClient(app)
    ui_login(client, "user-b-token")
    before = json.loads(settings.auth_config_path.read_text())
    path = "/v1/auth/relay-tokens/user-b"
    assert client.patch(path, headers={"Origin": "http://testserver"}, json={"can_view_keys": True}).status_code == 403
    assert client.patch(path, headers=auth_headers("admin-token"), json={"can_view_keys": True}).status_code == 200
    assert client.get("/v1/ui/session").json()["can_view_keys"] is True
    assert [k["id"] for k in client.get("/v1/auth/config").json()["kaggle_keys"]] == ["kb"]
    assert [a["kaggle_key_id"] for a in client.get("/v1/kaggle/accounts").json()["accounts"]] == ["kb"]
    assert client.get("/v1/kaggle/account?kaggle_key_id=ka").status_code == 403
    assert AuthStore.from_settings(settings).authenticate_token("user-b-token").can_view_keys
    assert client.patch(path, headers=auth_headers("admin-token"), json={"can_view_keys": False}).status_code == 200
    assert client.get("/v1/ui/session").json()["can_view_keys"] is False
    assert client.get("/v1/kaggle/accounts").status_code == 403
    after = json.loads(settings.auth_config_path.read_text())
    assert after["kaggle_keys"] == before["kaggle_keys"]
    for old, new in zip(before["relay_tokens"], after["relay_tokens"]):
        assert new["token"] == old["token"]
        assert new["allowed_kaggle_key_ids"] == old["allowed_kaggle_key_ids"]
    assert client.patch(path, headers=auth_headers("admin-token"), json={"can_view_keys": True, "allow_all_kaggle_keys": True}).status_code == 422


def test_explicit_view_grant_on_creation_and_wildcard_does_not_grant_view(tmp_path):
    settings = make_auth_config_settings(tmp_path, hidden_config(), admin_token="dedicated-admin-token")
    client = TestClient(create_app(settings))
    assert client.get("/v1/auth/config", headers=auth_headers("admin-token")).json()["can_view_keys"] is False
    assert client.get("/v1/auth/config", headers=auth_headers("dedicated-admin-token")).json()["can_view_keys"] is True
    response = client.post("/v1/auth/relay-tokens", headers=auth_headers("dedicated-admin-token"), json={
        "id": "viewer", "token": "viewer-test-token-123", "allowed_kaggle_key_ids": ["ka"], "can_view_keys": True,
    })
    assert response.status_code == 200
    config = client.get("/v1/auth/config", headers=auth_headers("viewer-test-token-123")).json()
    assert config["can_view_keys"] is True and config["can_manage_auth"] is False
    assert [k["id"] for k in config["kaggle_keys"]] == ["ka"]


def test_invalid_string_permission_does_not_accidentally_grant_access(tmp_path):
    config = hidden_config()
    config["relay_tokens"][1]["can_view_keys"] = "false"
    settings = make_auth_config_settings(tmp_path, config)
    with pytest.raises(AuthConfigError, match="can_view_keys must be a boolean"):
        AuthStore.from_settings(settings)

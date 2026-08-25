import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RELAY_API_TOKEN", "secret")
os.environ.setdefault("RELAY_STORAGE_DIR", str(ROOT / ".test-relay-data"))
os.environ["RELAY_UI_COOKIE_SECURE"] = "false"

from app.auth_config import AuthConfigError, AuthStore
from app.config import Settings
from app.main import create_app
from app.security import auth_source
from app.ui_auth import UI_COOKIE_NAME, create_ui_session_cookie


def make_settings(tmp_path: Path) -> Settings:
    return Settings(api_token="secret", storage_dir=tmp_path, chunk_size=8)


def make_auth_config_settings(tmp_path: Path, config: dict, admin_token: str = "") -> Settings:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(config), encoding="utf-8")
    return Settings(
        api_token="",
        storage_dir=tmp_path,
        chunk_size=8,
        auth_config_path=auth_path,
        admin_token=admin_token,
    )


def auth_headers(token: str = "secret") -> dict:
    return {"Authorization": f"Bearer {token}"}


def ui_login(client: TestClient, token: str, origin: str = "http://testserver"):
    return client.post(
        "/v1/ui/login",
        headers={"Origin": origin},
        json={"token": token},
    )


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
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
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
    return buffer.getvalue()


def job_request_body(dataset_zip: bytes, kernel_zip: bytes, kaggle_key_id: str | None = None) -> dict:
    body = {
        "dataset_ref": "demo/data",
        "kernel_ref": "demo/kernel",
        "dataset_archive_sha256": hashlib.sha256(dataset_zip).hexdigest(),
        "kernel_archive_sha256": hashlib.sha256(kernel_zip).hexdigest(),
        "dataset_size": len(dataset_zip),
        "kernel_size": len(kernel_zip),
        "chunk_size": 8,
    }
    if kaggle_key_id is not None:
        body["kaggle_key_id"] = kaggle_key_id
    return body


def decode_cookie_payload(cookie_value: str) -> dict:
    payload_b64 = cookie_value.split(".", 1)[0]
    padding = "=" * (-len(payload_b64) % 4)
    return json.loads(base64.urlsafe_b64decode((payload_b64 + padding).encode("ascii")))


def test_login_page_returns_200(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/login")

    assert response.status_code == 200
    assert "Kaggle Relay" in response.text
    assert "管理密钥" in response.text


def test_admin_token_requires_multi_key_auth_config(tmp_path):
    with pytest.raises(ValueError, match="admin_token requires auth_config_path"):
        Settings(
            api_token="secret",
            storage_dir=tmp_path,
            admin_token="custom-management-token",
        )


def test_admin_token_minimum_length_is_eight_characters(tmp_path):
    auth_path = tmp_path / "auth.json"

    settings = Settings(
        api_token="",
        storage_dir=tmp_path,
        auth_config_path=auth_path,
        admin_token="12345678",
    )

    assert settings.admin_token == "12345678"
    trimmed_settings = Settings(
        api_token="",
        storage_dir=tmp_path,
        auth_config_path=auth_path,
        admin_token="  12345678  ",
    )
    assert trimmed_settings.admin_token == "12345678"
    with pytest.raises(ValueError, match="at least 8 characters"):
        Settings(
            api_token="",
            storage_dir=tmp_path,
            auth_config_path=auth_path,
            admin_token="1234567",
        )
    with pytest.raises(AuthConfigError, match="at least 8 characters"):
        AuthStore(
            relay_tokens=[("admin", "admin-token", None)],
            kaggle_keys={},
            admin_token="1234567",
        )


def test_settings_reads_admin_token_without_exposing_it_in_repr(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(multi_key_auth_config()), encoding="utf-8")
    management_token = "custom-management-token"
    monkeypatch.setenv("RELAY_AUTH_CONFIG", str(auth_path))
    monkeypatch.setenv("RELAY_ADMIN_TOKEN", management_token)
    monkeypatch.setenv("RELAY_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("RELAY_PUBLIC_ORIGIN", "https://relay.example/")
    monkeypatch.setenv("RELAY_TRUSTED_PROXY_IPS", "127.0.0.1, ::1")

    settings = Settings.from_env()

    assert settings.auth_config_path == auth_path
    assert settings.admin_token == management_token
    assert settings.public_origin == "https://relay.example"
    assert settings.trusted_proxy_ips == frozenset({"127.0.0.1", "::1"})
    assert management_token not in repr(settings)


def test_ui_pages_redirect_to_login_without_session(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        responses = [client.get(path, follow_redirects=False) for path in ("/", "/ui", "/admin")]

    assert [response.status_code for response in responses] == [303, 303, 303]
    assert [response.headers["location"] for response in responses] == ["/login", "/login", "/login"]


def test_ui_login_rejects_invalid_token(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        response = ui_login(client, "wrong")

    assert response.status_code == 401


def test_ui_login_requires_same_origin(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        missing = client.post("/v1/ui/login", json={"token": "secret"})
        cross_origin = ui_login(client, "secret", "https://attacker.example")
        same_origin = ui_login(client, "secret")

    assert missing.status_code == 403
    assert cross_origin.status_code == 403
    assert same_origin.status_code == 200


def test_auth_failures_are_rate_limited_by_channel(tmp_path):
    settings = make_settings(tmp_path)
    settings.auth_failure_limit = 2
    settings.auth_lockout_seconds = 60
    app = create_app(settings)

    with TestClient(app) as client:
        first_login = ui_login(client, "wrong-1")
        blocked_login = ui_login(client, "wrong-2")
        valid_login_while_blocked = ui_login(client, "secret")

        first_bearer = client.get("/v1/health", headers=auth_headers("wrong-1"))
        blocked_bearer = client.get("/v1/health", headers=auth_headers("wrong-2"))
        valid_bearer_while_blocked = client.get(
            "/v1/health",
            headers=auth_headers("secret"),
        )

    assert first_login.status_code == 401
    assert blocked_login.status_code == 429
    assert valid_login_while_blocked.status_code == 429
    assert int(blocked_login.headers["retry-after"]) > 0
    assert first_bearer.status_code == 401
    assert blocked_bearer.status_code == 429
    assert valid_bearer_while_blocked.status_code == 429
    assert int(blocked_bearer.headers["retry-after"]) > 0


def test_auth_source_uses_unspoofable_end_of_trusted_proxy_chain():
    request = Request(
        {
            "type": "http",
            "client": ("192.168.0.1", 12345),
            "headers": [
                (b"x-forwarded-for", b"203.0.113.66, 198.51.100.8"),
            ],
        }
    )
    untrusted_request = Request(
        {
            "type": "http",
            "client": ("198.51.100.9", 12345),
            "headers": [
                (b"x-forwarded-for", b"203.0.113.66"),
            ],
        }
    )

    assert auth_source(request, frozenset({"192.168.0.1"})) == "198.51.100.8"
    assert auth_source(untrusted_request, frozenset({"192.168.0.1"})) == "198.51.100.9"


def test_ui_login_sets_http_only_session_cookie(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        response = ui_login(client, "secret")
        cookie_value = client.cookies.get(UI_COOKIE_NAME)

    set_cookie = response.headers.get("set-cookie", "")
    payload = decode_cookie_payload(cookie_value)

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["principal_id"] == "legacy"
    assert UI_COOKIE_NAME in set_cookie
    assert "HttpOnly" in set_cookie
    assert "secret" not in set_cookie
    assert payload["principal_id"] == "legacy"
    assert payload["issued_at"] < payload["expires_at"]
    assert "token" not in payload


def test_https_public_origin_enables_secure_ui_cookie(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_UI_COOKIE_SECURE", raising=False)
    settings = make_settings(tmp_path)
    settings.public_origin = "https://relay.example"
    app = create_app(settings)

    with TestClient(app) as client:
        response = ui_login(client, "secret", "https://relay.example")

    assert response.status_code == 200
    assert "Secure" in response.headers.get("set-cookie", "")


def test_logged_in_user_can_access_ui_and_health_without_authorization_header(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        denied = client.get("/v1/health")
        login = ui_login(client, "secret")
        ui_responses = [client.get(path, follow_redirects=False) for path in ("/", "/ui", "/admin")]
        health = client.get("/v1/health")

    assert denied.status_code == 401
    assert login.status_code == 200
    assert [response.status_code for response in ui_responses] == [200, 200, 200]
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_ui_session_returns_false_when_not_logged_in(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/v1/ui/session")

    assert response.status_code == 200
    assert response.json() == {"authenticated": False}


def test_ui_logout_removes_session_cookie(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        login = ui_login(client, "secret")
        assert login.status_code == 200
        assert client.get("/v1/health").status_code == 200

        logout = client.post(
            "/v1/ui/logout",
            headers={"Origin": "http://testserver"},
        )
        denied = client.get("/v1/health")

    assert logout.status_code == 200
    assert logout.json() == {"ok": True}
    assert denied.status_code == 401


def test_tampered_or_expired_ui_session_is_rejected(tmp_path):
    app = create_app(make_settings(tmp_path))
    principal = app.state.auth_store.authenticate_token("secret")
    expired_cookie = create_ui_session_cookie(app.state.settings, app.state.auth_store, principal, -1)

    with TestClient(app) as client:
        login = ui_login(client, "secret")
        assert login.status_code == 200
        cookie_value = client.cookies.get(UI_COOKIE_NAME)
        tampered_cookie = f"{cookie_value[:-1]}{'A' if cookie_value[-1] != 'A' else 'B'}"

        client.cookies.set(UI_COOKIE_NAME, tampered_cookie)
        tampered = client.get("/v1/health")

        client.cookies.clear()
        client.cookies.set(UI_COOKIE_NAME, expired_cookie)
        expired = client.get("/v1/health")

    assert tampered.status_code == 401
    assert expired.status_code == 401


def test_existing_bearer_auth_still_works_and_takes_priority(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        login = ui_login(client, "secret")
        bad_bearer = client.get("/v1/health", headers=auth_headers("wrong"))
        good_bearer = client.get("/v1/health", headers=auth_headers("secret"))

    assert login.status_code == 200
    assert bad_bearer.status_code == 401
    assert good_bearer.status_code == 200
    assert good_bearer.json()["status"] == "ok"


def test_multi_user_key_permissions_work_with_ui_cookie(tmp_path):
    dataset_zip = build_zip({"dataset-metadata.json": b"{}"})
    kernel_zip = build_zip({"kernel-metadata.json": b'{"code_file":"train.py"}', "train.py": b"print(1)"})
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        login = ui_login(client, "user-a-token")
        session = client.get("/v1/ui/session")
        created = client.post(
            "/v1/jobs",
            headers={"Origin": "http://testserver"},
            json=job_request_body(dataset_zip, kernel_zip),
        )
        forbidden = client.post(
            "/v1/jobs",
            headers={"Origin": "http://testserver"},
            json=job_request_body(dataset_zip, kernel_zip, kaggle_key_id="kb"),
        )
        jobs = client.get("/v1/jobs")

    assert login.status_code == 200
    assert login.json()["principal_id"] == "user-a"
    assert login.json()["allowed_kaggle_key_ids"] == ["ka"]
    assert session.json() == {
        "authenticated": True,
        "principal_id": "user-a",
        "allowed_kaggle_key_ids": ["ka"],
    }
    assert created.status_code == 200
    assert created.json()["kaggle_key_id"] == "ka"
    assert forbidden.status_code == 403
    assert jobs.status_code == 200
    assert [job["kaggle_key_id"] for job in jobs.json()] == ["ka"]


def test_auth_config_is_filtered_and_does_not_return_secrets(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        login = ui_login(client, "user-a-token")
        config = client.get("/v1/auth/config")

    body = config.json()
    serialized = json.dumps(body, sort_keys=True)

    assert login.status_code == 200
    assert config.status_code == 200
    assert body["principal_id"] == "user-a"
    assert [key["id"] for key in body["kaggle_keys"]] == ["ka"]
    assert "admin-token" not in serialized
    assert "user-a-token" not in serialized
    assert "user-b-token" not in serialized
    assert "alice-key" not in serialized
    assert "bob-key" not in serialized


def test_admin_can_add_kaggle_key_and_relay_token(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        add_key = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers("admin-token"),
            json={"id": "kc", "username": "carol", "key": "carol-key"},
        )
        add_token = client.post(
            "/v1/auth/relay-tokens",
            headers=auth_headers("admin-token"),
            json={
                "id": "user-c",
                "token": "user-c-token-secret",
                "allowed_kaggle_key_ids": ["kc"],
            },
        )
        new_user_health = client.get("/v1/health", headers=auth_headers("user-c-token-secret"))
        new_user_config = client.get("/v1/auth/config", headers=auth_headers("user-c-token-secret"))

    assert add_key.status_code == 200
    assert "carol-key" not in add_key.text
    assert [key["id"] for key in add_key.json()["kaggle_keys"]] == ["ka", "kb", "kc"]
    assert add_token.status_code == 200
    assert "user-c-token-secret" not in add_token.text
    assert "carol-key" not in add_token.text
    assert any(token["id"] == "user-c" for token in add_token.json()["relay_tokens"])
    assert new_user_health.status_code == 200
    assert new_user_config.status_code == 200
    assert new_user_config.json()["principal_id"] == "user-c"
    assert [key["id"] for key in new_user_config.json()["kaggle_keys"]] == ["kc"]
    assert "user-c-token-secret" not in new_user_config.text
    assert "carol-key" not in new_user_config.text


def test_cookie_auth_mutations_require_same_origin_and_session_survives_update(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))
    token_body = {
        "id": "user-c",
        "token": "user-c-token-secret",
        "allowed_kaggle_key_ids": ["ka"],
    }

    with TestClient(app) as client:
        login = ui_login(client, "admin-token")
        missing_origin = client.post("/v1/auth/relay-tokens", json=token_body)
        cross_origin = client.post(
            "/v1/auth/relay-tokens",
            headers={"Origin": "https://attacker.example"},
            json=token_body,
        )
        same_origin = client.post(
            "/v1/auth/relay-tokens",
            headers={"Origin": "http://testserver"},
            json=token_body,
        )
        session_after_update = client.get("/v1/auth/config")
        bearer_without_origin = client.post(
            "/v1/auth/relay-tokens",
            headers=auth_headers("admin-token"),
            json={
                "id": "user-d",
                "token": "user-d-token-secret",
                "allowed_kaggle_key_ids": ["kb"],
            },
        )

    assert login.status_code == 200
    assert missing_origin.status_code == 403
    assert cross_origin.status_code == 403
    assert same_origin.status_code == 200
    assert session_after_update.status_code == 200
    assert session_after_update.json()["principal_id"] == "admin"
    assert bearer_without_origin.status_code == 200


def test_dedicated_management_token_is_only_admin_and_survives_config_updates(tmp_path):
    management_token = "custom-management-token"
    app = create_app(
        make_auth_config_settings(
            tmp_path,
            multi_key_auth_config(),
            admin_token=management_token,
        )
    )

    with TestClient(app) as client:
        login = ui_login(client, management_token)
        management_config = client.get("/v1/auth/config")
        former_admin_config = client.get(
            "/v1/auth/config",
            headers=auth_headers("admin-token"),
        )
        denied = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers("admin-token"),
            json={"id": "kc", "username": "carol", "key": "carol-key"},
        )
        add_key = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers(management_token),
            json={"id": "kc", "username": "carol", "key": "carol-key"},
        )
        add_token = client.post(
            "/v1/auth/relay-tokens",
            headers=auth_headers(management_token),
            json={
                "id": "user-c",
                "token": "user-c-token-secret",
                "allowed_kaggle_key_ids": ["kc"],
            },
        )
        management_after_updates = client.get("/v1/auth/config")
        new_user_health = client.get(
            "/v1/health",
            headers=auth_headers("user-c-token-secret"),
        )

    saved_config = (tmp_path / "auth.json").read_text(encoding="utf-8")
    management_principal = next(
        token
        for token in management_config.json()["relay_tokens"]
        if token["id"] == "management-key"
    )

    assert login.status_code == 200
    assert login.json()["principal_id"] == "management-key"
    assert management_config.status_code == 200
    assert management_config.json()["can_manage_auth"] is True
    assert management_config.json()["management_token_configured"] is True
    assert management_principal["management"] is True
    assert management_token not in management_config.text
    assert former_admin_config.status_code == 200
    assert former_admin_config.json()["can_manage_auth"] is False
    assert denied.status_code == 403
    assert add_key.status_code == 200
    assert add_token.status_code == 200
    assert management_after_updates.status_code == 200
    assert management_after_updates.json()["can_manage_auth"] is True
    assert new_user_health.status_code == 200
    assert management_token not in saved_config


def test_admin_can_update_existing_kaggle_key_without_reentering_secret(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        update_key = client.patch(
            "/v1/auth/kaggle-keys/kb",
            headers=auth_headers("admin-token"),
            json={"username": "bob_slug"},
        )
        new_user_config = client.get("/v1/auth/config", headers=auth_headers("user-b-token"))

    saved_config = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
    saved_key = next(item for item in saved_config["kaggle_keys"] if item["id"] == "kb")

    assert update_key.status_code == 200
    assert "bob-key" not in update_key.text
    assert saved_key["username"] == "bob_slug"
    assert saved_key["key"] == "bob-key"
    assert new_user_config.status_code == 200
    assert new_user_config.json()["kaggle_keys"] == [
        {"id": "kb", "username": "bob_slug", "credential_source": "username_key"}
    ]


def test_admin_must_include_username_when_adding_kaggle_key(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        add_key = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers("admin-token"),
            json={"id": "kc", "api_token": "carol-token"},
        )

    assert add_key.status_code == 400
    assert add_key.json()["detail"] == "kaggle username is required"


def test_admin_must_use_kaggle_username_slug_for_kaggle_key(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        add_key = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers("admin-token"),
            json={"id": "kc", "username": "yunru zhou", "api_token": "carol-token"},
        )
        update_key = client.patch(
            "/v1/auth/kaggle-keys/kb",
            headers=auth_headers("admin-token"),
            json={"username": "yunru zhou"},
        )

    assert add_key.status_code == 400
    assert add_key.json()["detail"] == "kaggle username must be the profile slug from kaggle.com, not the display name"
    assert update_key.status_code == 400
    assert update_key.json()["detail"] == "kaggle username must be the profile slug from kaggle.com, not the display name"


def test_admin_must_put_kgat_token_in_api_token_field(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        add_key = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers("admin-token"),
            json={"id": "kc", "username": "carol", "key": "KGAT_wrong-field"},
        )

    assert add_key.status_code == 400
    assert add_key.json()["detail"] == "KGAT token must be provided as api_token, not key"


def test_non_admin_cannot_add_auth_config_entries(tmp_path):
    app = create_app(make_auth_config_settings(tmp_path, multi_key_auth_config()))

    with TestClient(app) as client:
        add_key = client.post(
            "/v1/auth/kaggle-keys",
            headers=auth_headers("user-a-token"),
            json={"id": "kc", "username": "carol", "key": "carol-key"},
        )
        update_key = client.patch(
            "/v1/auth/kaggle-keys/kb",
            headers=auth_headers("user-a-token"),
            json={"username": "bob_slug"},
        )
        add_token = client.post(
            "/v1/auth/relay-tokens",
            headers=auth_headers("user-a-token"),
            json={
                "id": "user-c",
                "token": "user-c-token-secret",
                "allowed_kaggle_key_ids": ["ka"],
            },
        )

    assert add_key.status_code == 403
    assert update_key.status_code == 403
    assert add_token.status_code == 403

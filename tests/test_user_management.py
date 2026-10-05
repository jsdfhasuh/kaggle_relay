import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.auth_config import AuthConfigError, AuthStore
from app.ui_auth import UI_COOKIE_NAME
from test_ui_auth import auth_headers, make_auth_config_settings, multi_key_auth_config, ui_login, job_request_body
from test_relay_api import seed_job
import app.main as main

ADMIN = 'dedicated-admin-for-user-tests'
PATH = '/v1/auth/relay-tokens/'


def setup(tmp_path, dedicated=True):
    config = multi_key_auth_config()
    for user in config['relay_tokens']:
        user['can_view_keys'] = False
    settings = make_auth_config_settings(tmp_path, config, admin_token=ADMIN if dedicated else '')
    app = main.create_app(settings)
    return app, TestClient(app), settings


def test_reveal_permissions_and_no_secrets_in_summary(tmp_path):
    app, client, settings = setup(tmp_path)
    url = PATH + 'user-a/reveal'
    assert client.post(url).status_code == 401
    for token in [ADMIN, 'user-a-token']:
        response = client.post(url, headers=auth_headers(token))
        assert response.status_code == 200
        assert response.json() == {'id': 'user-a', 'token': 'user-a-token'}
        assert response.headers['cache-control'] == 'no-store'
    assert client.post(url, headers=auth_headers('user-b-token')).status_code == 403
    assert client.post(PATH + 'missing/reveal', headers=auth_headers('user-a-token')).status_code == 403
    assert client.post(PATH + 'missing/reveal', headers=auth_headers(ADMIN)).status_code == 404
    assert client.post(PATH + 'management-key/reveal', headers=auth_headers(ADMIN)).status_code == 403
    assert ui_login(client, 'user-a-token').status_code == 200
    assert client.post(url).status_code == 403
    assert client.post(url, headers={'Origin': 'https://other.example'}).status_code == 403
    assert client.post(url, headers={'Origin': 'http://testserver'}).status_code == 200
    summary = client.get('/v1/auth/config').json()
    assert summary['can_view_keys'] is False
    assert summary['kaggle_keys'] == []
    assert [u['id'] for u in summary['relay_tokens']] == ['user-a']
    assert 'user-a-token' not in json.dumps(summary)


@pytest.mark.parametrize('status', sorted(main.JOB_STATUS_VALUES))
def test_delete_checks_all_job_states_and_preserves_history(tmp_path, status):
    app, client, settings = setup(tmp_path)
    job = seed_job(app, status)
    app.state.db.update_job(job, relay_token_id='user-a', kaggle_key_id='ka')
    before = settings.auth_config_path.read_bytes()
    response = client.delete(PATH + 'user-a', headers=auth_headers(ADMIN))
    if status in main.TERMINAL_JOB_STATUSES:
        assert response.status_code == 204
        assert client.get('/v1/jobs/' + job, headers=auth_headers(ADMIN)).status_code == 200
        assert app.state.db.get_job(job)['relay_token_id'] == 'user-a'
    else:
        assert response.status_code == 409
        assert response.json()['detail']['active_job_count'] == 1
        assert settings.auth_config_path.read_bytes() == before
        assert app.state.auth_store.authenticate_token('user-a-token')


def test_delete_revokes_bearer_cookie_persists_and_retires_id(tmp_path):
    app, client, settings = setup(tmp_path)
    assert ui_login(client, 'user-a-token').status_code == 200
    cookie = client.cookies.get(UI_COOKIE_NAME)
    before = json.loads(settings.auth_config_path.read_text())
    assert client.delete(PATH + 'user-a', headers=auth_headers(ADMIN)).status_code == 204
    for headers in [{}, auth_headers('user-a-token')]:
        assert client.get('/v1/health', headers=headers).status_code == 401
    after = json.loads(settings.auth_config_path.read_text())
    assert after['kaggle_keys'] == before['kaggle_keys']
    assert after['retired_relay_token_ids'] == ['user-a']
    assert 'user-a-token' not in settings.auth_config_path.read_text()
    restarted = TestClient(main.create_app(settings))
    restarted.cookies.set(UI_COOKIE_NAME, cookie)
    assert restarted.get('/v1/health').status_code == 401
    assert restarted.get('/v1/health', headers=auth_headers('user-a-token')).status_code == 401
    response = client.post(PATH.rstrip('/'), headers=auth_headers(ADMIN), json={
        'id': 'user-a', 'token': 'replacement-token-long', 'allowed_kaggle_key_ids': ['ka']})
    assert response.status_code == 409
    after['relay_tokens'].append(before['relay_tokens'][1])
    settings.auth_config_path.write_text(json.dumps(after))
    with pytest.raises(AuthConfigError, match='retired'):
        AuthStore.from_settings(settings)


def test_last_user_and_protected_management(tmp_path):
    app, client, settings = setup(tmp_path)
    assert client.delete(PATH + 'user-b', headers=auth_headers('user-a-token')).status_code == 403
    assert client.delete(PATH + 'management-key', headers=auth_headers(ADMIN)).status_code == 403
    assert client.delete(PATH + 'missing', headers=auth_headers(ADMIN)).status_code == 404
    for token_id in ['admin', 'user-a', 'user-b']:
        assert client.delete(PATH + token_id, headers=auth_headers(ADMIN)).status_code == 204
    assert client.get('/v1/health', headers=auth_headers(ADMIN)).status_code == 200
    assert AuthStore.from_settings(settings).authenticate_token(ADMIN)
    settings.admin_token = ''
    with pytest.raises(AuthConfigError):
        AuthStore.from_settings(settings)


def test_compatibility_admin_is_protected(tmp_path):
    app, client, settings = setup(tmp_path, dedicated=False)
    for suffix, method in [('', client.delete), ('/reveal', client.post)]:
        assert method(PATH + 'admin' + suffix, headers=auth_headers('admin-token')).status_code == 403
    assert client.delete(PATH + 'user-a', headers=auth_headers('admin-token')).status_code == 204


def test_failed_atomic_write_leaves_memory_and_disk_unchanged(tmp_path, monkeypatch):
    app, client, settings = setup(tmp_path)
    before = settings.auth_config_path.read_bytes()
    store = app.state.auth_store
    def fail(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr(main.os, 'replace', fail)
    with pytest.raises(OSError, match='simulated'):
        client.delete(PATH + 'user-a', headers=auth_headers(ADMIN))
    assert app.state.auth_store is store
    assert settings.auth_config_path.read_bytes() == before
    assert not list(tmp_path.glob('.auth.json.tmp-*'))


def test_deletion_during_job_admission_rechecks_identity(tmp_path, monkeypatch):
    app, client, settings = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def candidates(*args):
        entered.set()
        assert release.wait(10)
        return [(0, 'ka')]
    monkeypatch.setattr(main, 'resolve_job_kaggle_candidates', candidates)
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(client.post, '/v1/jobs', headers=auth_headers('user-a-token'), json=job_request_body(b'', b'', 'ka'))
        try:
            assert entered.wait(10)
            assert client.delete(PATH + 'user-a', headers=auth_headers(ADMIN)).status_code == 204
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 401
    assert app.state.db.list_jobs() == []
    assert not list(settings.jobs_dir.iterdir())


def test_concurrent_config_changes_publish_consistently(tmp_path):
    app, client, settings = setup(tmp_path)
    with ThreadPoolExecutor(3) as pool:
        pending = [pool.submit(client.post, PATH.rstrip('/'), headers=auth_headers(ADMIN), json={
            'id': f'new-{i}', 'token': f'long-new-test-token-{i}', 'allowed_kaggle_key_ids': ['ka']}) for i in range(8)]
        assert all(f.result(timeout=10).status_code == 200 for f in pending)
    disk = AuthStore.from_settings(settings)
    assert {p.id for _, p in disk._tokens} == {p.id for _, p in app.state.auth_store._tokens}
    assert all(disk.authenticate_token(f'long-new-test-token-{i}') for i in range(8))


def test_owner_filter_is_exact_and_admin_only(tmp_path):
    app, client, settings = setup(tmp_path)
    mine, other = seed_job(app, 'receiving'), seed_job(app, 'receiving')
    app.state.db.update_job(mine, relay_token_id='user-a', kaggle_key_id='ka')
    app.state.db.update_job(other, relay_token_id='user-b', kaggle_key_id='kb')
    response = client.get('/v1/jobs?owner=user-a', headers=auth_headers(ADMIN))
    assert [job['job_id'] for job in response.json()] == [mine]
    assert client.get('/v1/jobs?owner=user-b', headers=auth_headers('user-a-token')).status_code == 403


def test_job_admitted_first_blocks_concurrent_deletion(tmp_path, monkeypatch):
    app, client, settings = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = app.state.db.create_job
    def create(values):
        entered.set()
        assert release.wait(10)
        return original(values)
    monkeypatch.setattr(app.state.db, 'create_job', create)
    monkeypatch.setattr(main, 'resolve_job_kaggle_candidates', lambda *args: [(0, 'ka')])
    with ThreadPoolExecutor(2) as pool:
        creation = pool.submit(client.post, '/v1/jobs', headers=auth_headers('user-a-token'), json=job_request_body(b'', b'', 'ka'))
        try:
            assert entered.wait(10)
            deletion = pool.submit(client.delete, PATH + 'user-a', headers=auth_headers(ADMIN))
        finally:
            release.set()
        assert creation.result(timeout=10).status_code == 200
        assert deletion.result(timeout=10).status_code == 409
    assert app.state.auth_store.authenticate_token('user-a-token')
    assert len(app.state.db.list_jobs()) == 1


def test_cookie_delete_requires_same_origin(tmp_path):
    app, client, settings = setup(tmp_path)
    assert ui_login(client, ADMIN).status_code == 200
    for headers in [{}, {'Origin': 'https://other.example'}]:
        assert client.delete(PATH + 'user-a', headers=headers).status_code == 403
    assert client.delete(PATH + 'user-a', headers={'Origin': 'http://testserver'}).status_code == 204
    assert client.get('/v1/health').status_code == 200


@pytest.mark.parametrize('retired', ['user-a', [None], [''], [' user-a ']])
def test_invalid_or_reused_retired_ids_fail_validation(tmp_path, retired):
    app, client, settings = setup(tmp_path)
    config = json.loads(settings.auth_config_path.read_text())
    config['retired_relay_token_ids'] = retired
    settings.auth_config_path.write_text(json.dumps(config))
    with pytest.raises(AuthConfigError, match='retired'):
        AuthStore.from_settings(settings)

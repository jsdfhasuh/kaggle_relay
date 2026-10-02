import io
import asyncio
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import subprocess
import textwrap
import zipfile

import pytest
from test_relay_api import make_settings

from app.auth_config import KaggleCredentials
from app.config import Settings
from app.kaggle_adapter import DatasetUploadReceipt, KaggleAdapter, KaggleAdapterError, DatasetUploadUnknown
from app.payload_contract import RUNTIME_PATH, verify_payload_archive
from app.upload_intent import intent_path, read_intent


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return stream.getvalue()


@pytest.mark.parametrize("layout", ["outer", "inner", "recursive"])
@pytest.mark.parametrize("tamper", [False, True])
def test_exact_content_for_real_kaggle_layouts(tmp_path, layout, tamper):
    sources = {"kaggle_runtime/__init__.py": b"", "kaggle_runtime/config.py": b"config",
               "kaggle_runtime/yolo_train.py": b"original source"}
    files = {"data.yaml": b"data", RUNTIME_PATH: zip_bytes(sources)}
    payload = tmp_path / "payload.zip"
    payload.write_bytes(zip_bytes(files))
    if tamper:
        sources["kaggle_runtime/yolo_train.py"] = b"tampered source"
        assert len(sources["kaggle_runtime/yolo_train.py"]) == len(b"original source")
        files[RUNTIME_PATH] = zip_bytes(sources)
    if layout == "recursive":
        del files[RUNTIME_PATH]
        files.update({RUNTIME_PATH[:-4] + "/" + k: v for k, v in sources.items()})
    elif layout == "outer":
        files = {"payload.zip": zip_bytes(files)}
    downloaded = tmp_path / "download.zip"
    downloaded.write_bytes(zip_bytes(files))
    if tamper:
        with pytest.raises(ValueError):
            verify_payload_archive(payload, downloaded)
    else:
        verify_payload_archive(payload, downloaded)


class Api:
    def __init__(self):
        self.config_values = {"username": "owner", "auth_method": "access_token"}
        self.latest = 6
        self.uploads = 0
        self.downloads = []
        self.response = {"status": "ok"}
        self.lost_response = False
        self.remote = {}
        self.exists_error = None
        self.status_calls = 0
        self.basic_checks = 0
        self.basic_error = None
        self.owned_refs = ["owner/data"]
        self.list_pages = []

    def authenticate(self):
        pass

    def _introspect_token(self, token):
        return self.config_values["username"]

    def dataset_list(self, mine=False, page=1):
        assert mine is True
        self.list_pages.append(page)
        self.basic_checks += 1
        if self.basic_error is not None:
            raise self.basic_error
        return [SimpleNamespace(ref=ref) for ref in self.owned_refs] if page == 1 else []

    def dataset_status(self, ref, format=None):
        self.status_calls += 1
        if self.exists_error:
            raise self.exists_error
        return json.dumps({"status": "ready", "current_version_number": self.latest})

    def dataset_create_version(self, folder, message, **kwargs):
        self.uploads += 1
        self.latest += 1
        files = {p.name: p.read_bytes() for p in Path(folder).iterdir() if p.name != "dataset-metadata.json"}
        self.remote[self.latest] = zip_bytes(files)
        if self.lost_response:
            raise TimeoutError("committed before response lost")
        return self.response

    def dataset_create_new(self, folder, **kwargs):
        self.latest = 0
        return self.dataset_create_version(folder, "create", **kwargs)

    def dataset_download_files(self, ref, path=None, **kwargs):
        self.downloads.append(ref)
        (Path(path) / "data.zip").write_bytes(self.remote[int(ref.split("/")[-1])])


def fixture_adapter(tmp_path, monkeypatch):
    api = Api()
    module = ModuleType("kaggle.api.kaggle_api_extended")
    module.KaggleApi = lambda: api
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", module)
    adapter = KaggleAdapter(Settings(api_token="test", storage_dir=tmp_path / "storage"), lambda _: None)
    adapter._sdk_in_process = True
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "file.txt").write_bytes(b"original")
    return api, adapter, dataset


def test_new_dataset_uses_complete_owned_inventory_not_status_403(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.exists_error = RuntimeError("403 Forbidden for uncreated private ref")
    pages = []
    def dataset_list(mine=False, page=1):
        assert mine is True
        pages.append(page)
        return [SimpleNamespace(ref="owner/other")] if page == 1 else []
    api.dataset_list = dataset_list
    receipt = adapter.upload_dataset(dataset, "owner/data", "create")
    assert pages == [1, 2] and api.status_calls == 0 and api.uploads == 1
    assert receipt.expected_version_number == 1
    saved = read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data"))
    assert saved["operation"] == "create" and saved["state"] == "accepted"
    assert saved["existence_basis"] == "authenticated_mine_inventory"
    # A delayed readback never licenses a second create or a latest query.
    receipt = adapter.upload_dataset(dataset, "owner/data", "retry")
    assert api.uploads == 1 and pages == [1, 2]
    assert adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert api.downloads == ["owner/data/1"]


def test_created_candidate_visibility_delay_only_retries_readback(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.owned_refs = []
    receipt = adapter.upload_dataset(dataset, "owner/data", "create")
    original_download = api.dataset_download_files
    calls = []
    def download(ref, path=None, force=False, quiet=False, unzip=True):
        calls.append(ref)
        if len(calls) == 1:
            import requests
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError("not visible yet", response=response)
        return original_download(ref, path=path, force=force, quiet=quiet, unzip=unzip)
    api.dataset_download_files = download
    adapter._sleep = lambda _: None
    adapter.wait_dataset("owner/data", permission_grace_seconds=60, upload_receipt=receipt)
    assert calls == ["owner/data/1", "owner/data/1"]
    assert api.uploads == 1 and api.status_calls == 0


def test_create_intent_write_failure_never_uploads(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.owned_refs = []
    def fail(path, value):
        assert value["operation"] == "create" and value["version_number"] == 1
        raise OSError("disk full")
    monkeypatch.setattr("app.kaggle_adapter.write_intent", fail)
    with pytest.raises(OSError, match="disk full"):
        adapter.upload_dataset(dataset, "owner/data", "create")
    assert api.uploads == api.status_calls == 0


@pytest.mark.parametrize("mode", ["403", "incomplete", "malformed", "repeated"])
def test_bad_inventory_never_creates_intent_or_mutates_dataset(tmp_path, monkeypatch, mode):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    def dataset_list(mine=False, page=1):
        assert mine is True
        if mode == "403":
            raise RuntimeError("403 Forbidden")
        if mode == "incomplete":
            return None
        if mode == "malformed":
            return [SimpleNamespace()]
        return [SimpleNamespace(ref="owner/other")]
    api.dataset_list = dataset_list
    with pytest.raises((RuntimeError, KaggleAdapterError)):
        adapter.upload_dataset(dataset, "owner/data", "create")
    assert api.uploads == api.status_calls == 0
    assert read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data")) is None


def test_existing_dataset_on_later_page_uses_version_not_create(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    def dataset_list(mine=False, page=1):
        assert mine is True
        return [SimpleNamespace(ref="owner/other" if page == 1 else "owner/data")]
    api.dataset_list = dataset_list
    receipt = adapter.upload_dataset(dataset, "owner/data", "version")
    assert receipt.expected_version_number == 7 and api.status_calls == 1


@pytest.mark.parametrize("outcome", ["conflict", "lost_response"])
def test_create_race_or_lost_response_never_falls_back_to_version(tmp_path, monkeypatch, outcome):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.owned_refs = []
    api.response = {"error": "Dataset already exists"} if outcome == "conflict" else {"status": "ok"}
    api.lost_response = outcome == "lost_response"
    with pytest.raises(KaggleAdapterError):
        adapter.upload_dataset(dataset, "owner/data", "create")
    saved = read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data"))
    assert saved["version_number"] == 1 and saved["operation"] == "create"
    if outcome == "conflict":
        assert saved["state"] == "rejected"
        with pytest.raises(KaggleAdapterError, match="rejected"):
            adapter.upload_dataset(dataset, "owner/data", "retry")
    else:
        assert saved["state"] == "unknown"
        receipt = adapter.upload_dataset(dataset, "owner/data", "retry")
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
        assert api.downloads == ["owner/data/1"]
    assert api.uploads == 1 and api.status_calls == 0


@pytest.mark.parametrize("method,source", [("access_token", "token_introspection"),
                                           ("oauth", "oauth_token_introspection"),
                                           ("legacy_api_key", "validated_basic_principal")])
def test_identity_reports_authentication_source_and_rejects_wrong_owner(tmp_path, monkeypatch, method, source):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.config_values["auth_method"] = method
    assert adapter.identity("owner") == {"identity_verified": True, "identity_source": source,
                                         "auth_method": method, "identity_error": ""}
    with pytest.raises(KaggleAdapterError, match="identity"):
        adapter.upload_dataset(dataset, "wrong/data", "test")
    assert not api.uploads
    assert not list((tmp_path / "storage").rglob("*.json"))
    assert api.basic_checks == (2 if method == "legacy_api_key" else 0)


def test_legacy_key_validates_authenticated_request_before_upload(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.config_values["auth_method"] = "legacy_api_key"
    adapter.credentials = KaggleCredentials("test", username="owner", key="fake-key")
    assert adapter.require_identity("owner")["identity_source"] == "validated_basic_principal"
    assert api.basic_checks == 1 and api.uploads == 0
    receipt = adapter.upload_dataset(dataset, "owner/data", "test")
    assert api.basic_checks == 3 and api.uploads == 1
    assert receipt.expected_version_number == 7


@pytest.mark.parametrize("failure", ["invalid_key_401", "forbidden_403", "wrong_owner"])
def test_legacy_key_failures_block_worker_without_remote_mutation(tmp_path, monkeypatch, failure):
    from requests.exceptions import HTTPError
    from app.main import create_app
    from app.worker import process_job, job_paths
    from test_relay_api import seed_job
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.config_values["auth_method"] = "legacy_api_key"
    adapter.credentials = KaggleCredentials("test", username="owner", key="fake-key")
    if failure != "wrong_owner":
        status = 401 if failure == "invalid_key_401" else 403
        api.basic_error = HTTPError(str(status), response=SimpleNamespace(status_code=status))
    else:
        api.config_values["username"] = "different-owner"
    app = create_app(adapter.settings)
    job_id = seed_job(app, "queued", dataset_ref="owner/data", kernel_ref="owner/kernel")
    paths = job_paths(adapter.settings, job_id)
    paths["dataset_dir"].mkdir(parents=True)
    paths["kernel_dir"].mkdir(parents=True)
    payload = paths["dataset_dir"] / "file.txt"
    script = paths["kernel_dir"] / "train.py"
    payload.write_bytes(b"original dataset")
    script.write_bytes(b"original frozen script")
    pushes = []
    adapter.push_kernel = lambda folder: pushes.append(folder)
    monkeypatch.setattr("app.worker.KaggleAdapter", lambda *args, **kwargs: adapter)
    process_job(adapter.settings, app.state.db, job_id)
    job = app.state.db.get_job(job_id)
    assert job["status"] == "failed"
    assert ("identity rejected" if failure == "wrong_owner" else str(status)) in job["error"]
    assert api.basic_checks == 1  # A real request boundary, not authenticate() alone.
    assert api.uploads == api.status_calls == 0 and not pushes and not api.remote
    assert read_intent(intent_path(adapter.settings.storage_dir, paths["dataset_dir"], "owner/data")) is None
    assert job["dataset_ref"] == "owner/data" and job["kernel_ref"] == "owner/kernel"
    assert payload.read_bytes() == b"original dataset" and script.read_bytes() == b"original frozen script"


@pytest.mark.parametrize("response", [{"status": "ok", "error": "denied"},
                                      SimpleNamespace(status="error", error=""),
                                      {"invalidTags": ["bad"]}])
@pytest.mark.parametrize("create", [False, True])
def test_http_success_business_error_cannot_create_receipt(tmp_path, monkeypatch, response, create):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.response = response
    if create:
        api.owned_refs = []
        api.exists_error = RuntimeError("404 Not found")
        api.exists_error.response = SimpleNamespace(status_code=404)
    with pytest.raises(KaggleAdapterError, match="business response"):
        adapter.upload_dataset(dataset, "owner/data", "test")
    saved = read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data"))
    assert saved["state"] == "rejected" and saved["version_number"] == (1 if create else 7)
    with pytest.raises(KaggleAdapterError, match="rejected"):
        adapter.upload_dataset(dataset, "owner/data", "retry")
    assert api.uploads == 1


def test_response_lost_recovers_only_original_candidate_even_after_b_upload(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.lost_response = True
    with pytest.raises(DatasetUploadUnknown):
        adapter.upload_dataset(dataset, "owner/data", "test")
    api.latest = 8
    api.remote[8] = zip_bytes({"file.txt": b"otherxxx"})
    calls = api.status_calls
    receipt = adapter.upload_dataset(dataset, "owner/data", "retry")
    assert receipt.expected_version_number == 7
    assert json.loads(adapter.wait_dataset("owner/data", upload_receipt=receipt))["current_version_number"] == 7
    assert api.status_calls == calls and api.uploads == 1
    assert api.downloads == ["owner/data/7"]
    # Same-length competing candidate cannot pass name/size checks alone.
    api.remote[7] = api.remote[8]
    with pytest.raises(ValueError, match="digest_mismatch"):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)


@pytest.mark.parametrize("response", [None, {}])
def test_missing_business_response_preserves_unknown_candidate(tmp_path, monkeypatch, response):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.response = response
    with pytest.raises(DatasetUploadUnknown):
        adapter.upload_dataset(dataset, "owner/data", "test")
    saved = read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data"))
    assert saved["state"] == "unknown" and saved["version_number"] == 7
    receipt = adapter.upload_dataset(dataset, "owner/data", "retry")
    adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert api.uploads == 1 and api.downloads == ["owner/data/7"]


@pytest.mark.parametrize("mode", ["write_failure", "null", "source_changed", "forbidden"])
def test_fail_closed_before_remote_mutation(tmp_path, monkeypatch, mode):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    saved = intent_path(adapter.settings.storage_dir, dataset, "owner/data")
    if mode == "write_failure":
        def fail(*args):
            raise OSError("disk full")
        monkeypatch.setattr("app.kaggle_adapter.write_intent", fail)
    elif mode == "null":
        saved.parent.mkdir(parents=True)
        saved.write_text("null")
    elif mode == "source_changed":
        adapter.upload_dataset(dataset, "owner/data", "first")
        (dataset / "file.txt").write_bytes(b"changed!")
    else:
        api.exists_error = RuntimeError("403 Forbidden")
        api.exists_error.response = SimpleNamespace(status_code=403)
    before = api.uploads
    with pytest.raises((OSError, ValueError, KaggleAdapterError, RuntimeError)):
        adapter.upload_dataset(dataset, "owner/data", "test")
    assert api.uploads == before


def test_permission_then_missing_file_logs_both_reasons_and_throttles(tmp_path, monkeypatch):
    adapter = KaggleAdapter(Settings(api_token="test", storage_dir=tmp_path), lambda _: None)
    messages = []
    adapter.log = messages.append
    replies = iter([SimpleNamespace(returncode=1, stdout="403 Forbidden"),
                    *[SimpleNamespace(returncode=0, stdout='{"status":"ready","current_version_number":7}')]*3])
    adapter._run = lambda *a, **kw: next(replies)
    inventories = iter([{}, {}, {"runtime.zip": 4}])
    adapter._dataset_file_inventory = lambda *a, **kw: next(inventories)
    adapter._sleep = lambda _: None
    adapter.wait_dataset("owner/data", permission_grace_seconds=900,
                         upload_receipt=DatasetUploadReceipt(7, (("runtime.zip", 4),)))
    assert len(messages) == 2
    assert "temporarily unavailable" in messages[0]
    assert "runtime.zip" in messages[1]


def test_scheduler_excludes_mismatched_identity_even_with_good_quota(tmp_path, monkeypatch):
    from app.main import create_app
    from app.scheduler import schedule_pending_jobs
    from test_concurrency import pool_settings
    from test_dynamic_scheduling import pending_job
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    pending_job(app, "a3-pending")
    checked = []
    def identity(self, configured_owner=""):
        checked.append(self.credentials.id)
        return {"identity_verified": self.credentials.id == "key1", "identity_error": "owner_mismatch"}
    def quota(self):
        assert self.credentials.id == "key1", "invalid owner must be rejected before quota admission"
        return {"available": True, "accelerators": [{"resource": "GPU", "remaining_hours": 30}]}
    monkeypatch.setattr(KaggleAdapter, "identity", identity)
    monkeypatch.setattr(KaggleAdapter, "quota", quota)
    asyncio.run(schedule_pending_jobs(app))
    job = app.state.db.get_job("a3-pending")
    assert job["kaggle_key_id"] == "key1" and job["assignment_state"] == "bound"
    assert set(checked) == {"key0", "key1"}
    # The rejected account does not get silently renamed or enabled in config.
    assert app.state.auth_store.credentials_for("key0").username == "user0"
    app.state.settings._quota_cache.shutdown()


def test_scheduler_reports_all_identity_rejections_without_binding(tmp_path, monkeypatch):
    from app.main import create_app
    from app.scheduler import schedule_pending_jobs
    from test_concurrency import pool_settings
    from test_dynamic_scheduling import pending_job
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    pending_job(app, "a3-pending")
    before = app.state.db.get_job("a3-pending")
    monkeypatch.setattr(KaggleAdapter, "identity", lambda *a, **kw: {
        "identity_verified": False, "identity_error": "owner_mismatch"})
    monkeypatch.setattr(KaggleAdapter, "quota", lambda *a: pytest.fail("quota cannot grant identity"))
    asyncio.run(schedule_pending_jobs(app))
    after = app.state.db.get_job("a3-pending")
    assert after["assignment_state"] == "pending" and app.state.queue.empty()
    assert "identity blocked" in after["queue_reason"]
    assert after["dataset_ref"] == before["dataset_ref"]
    app.state.settings._quota_cache.shutdown()


def test_fixed_pool_selection_checks_identity_before_quota(tmp_path, monkeypatch):
    from app.main import create_app, quota_key_candidates
    from test_concurrency import pool_settings
    app = create_app(pool_settings(tmp_path, count=2, shared=True))
    monkeypatch.setattr(KaggleAdapter, "identity", lambda self, owner="": {
        "identity_verified": self.credentials.id == "key1", "identity_error": "owner_mismatch"})
    def quota(self):
        assert self.credentials.id == "key1"
        return {"available": True, "accelerators": [{"resource": "GPU", "remaining_hours": 30}]}
    monkeypatch.setattr(KaggleAdapter, "quota", quota)
    candidates, exhausted, unavailable = quota_key_candidates(app.state.settings, app.state.auth_store, ["key0", "key1"])
    assert candidates == [(30, "key1")] and not exhausted
    assert len(unavailable) == 1 and "identity rejected" in unavailable[0]
    app.state.settings._quota_cache.shutdown()


@pytest.mark.parametrize("recovery", ["restart", "complete", "null_cache"])
def test_real_worker_and_complete_recover_same_candidate_after_response_loss(tmp_path, monkeypatch, recovery):
    from fastapi.testclient import TestClient
    from app.main import create_app, recover_job_after_restart
    from app.worker import process_job, job_paths
    from test_relay_api import seed_job, auth_headers
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    app = create_app(adapter.settings)
    job_id = seed_job(app, "queued", job_id="a3-original-job", dataset_ref="owner/data", kernel_ref="owner/kernel")
    paths = job_paths(adapter.settings, job_id)
    paths["dataset_dir"].mkdir(parents=True)
    paths["kernel_dir"].mkdir(parents=True)
    (paths["dataset_dir"] / "file.txt").write_bytes(b"original")
    (paths["dataset_dir"] / "dataset-metadata.json").write_text('{"id":"owner/data"}')
    (paths["kernel_dir"] / "kernel-metadata.json").write_text('{"id":"owner/kernel","code_file":"train.py","dataset_sources":["owner/data"]}')
    (paths["kernel_dir"] / "train.py").write_text("print('not executed')\n")
    pushes = []
    adapter.push_kernel = lambda folder: pushes.append(folder) or "pushed"
    monkeypatch.setattr("app.worker.KaggleAdapter", lambda *a, **kw: adapter)
    monkeypatch.setattr("app.worker.finish_kernel_job", lambda *a, **kw: None)
    api.lost_response = True
    process_job(adapter.settings, app.state.db, job_id)
    pending = app.state.db.get_job(job_id)
    assert pending["status"] == "waiting_dataset" and "outcome_unknown" in pending["error"]
    assert not pushes and api.uploads == 1
    # A later shared cache must not replace the original unknown candidate.
    app.state.db.upsert_dataset_cache(dataset_ref="owner/data", payload_hash=pending["payload_hash"],
        status="ready", dataset_status='{"status":"ready","current_version_number":8}', source_job_id="other")
    api.latest = 8
    api.remote[8] = zip_bytes({"file.txt": b"otherxxx"})
    if recovery == "null_cache":
        saved_path = intent_path(adapter.settings.storage_dir, paths["dataset_dir"], "owner/data")
        saved_path.write_text("null")
        process_job(adapter.settings, app.state.db, job_id)
        assert not pushes and api.uploads == 1 and not api.downloads
        assert saved_path.read_text() == "null"
        assert "upload_intent_invalid" in app.state.db.get_job(job_id)["error"]
        return
    # Use a non-lifespan client so startup scanning cannot race this assertion.
    if recovery == "restart":
        item = recover_job_after_restart(adapter.settings, app.state.db, app.state.auth_store, pending)
        assert item == {"action": "process", "job_id": job_id}
        from app.main import queue_item_expected_statuses
        assert app.state.db.get_job(job_id)["status"] in queue_item_expected_statuses(item)
    else:
        client = TestClient(app)
        response = client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers(token="test"))
        assert response.status_code == 200
    assert app.state.db.get_job(job_id)["status"] == "queued"
    process_job(adapter.settings, app.state.db, job_id)
    assert api.uploads == 1 and api.downloads == ["owner/data/7"]
    assert len(pushes) == 1
    metadata = json.loads((paths["kernel_dir"] / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["owner/data/7"]
    assert app.state.db.get_job(job_id)["job_id"] == job_id


def test_hard_exit_keeps_candidate_and_only_reads_back_on_retry(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    script = textwrap.dedent('''
        import os, sys
        from pathlib import Path
        from types import ModuleType
        from app.config import Settings
        from app.kaggle_adapter import KaggleAdapter
        class Api:
            config_values = {"username": "owner", "auth_method": "access_token"}
            def authenticate(self): pass
            def dataset_list(self, mine=False, page=1):
                from types import SimpleNamespace
                return [SimpleNamespace(ref="owner/data")]
            def dataset_status(self, *a, **kw):
                return '{"status":"ready","current_version_number":6}'
            def dataset_download_files(self, *a, **kw): pass
            def dataset_create_version(self, folder, *a, **kw):
                Path(sys.argv[2]).write_bytes((Path(folder) / "file.txt").read_bytes())
                os._exit(71)
        module = ModuleType("kaggle.api.kaggle_api_extended")
        module.KaggleApi = Api
        sys.modules["kaggle.api.kaggle_api_extended"] = module
        adapter = KaggleAdapter(Settings(api_token="test", storage_dir=Path(sys.argv[3])), lambda _: None)
        adapter._sdk_in_process = True
        adapter.upload_dataset(Path(sys.argv[1]), "owner/data", "test")
    ''')
    remote = tmp_path / "committed.bin"
    result = subprocess.run([sys.executable, "-c", script, str(dataset), str(remote),
                             str(adapter.settings.storage_dir)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 71, result.stderr
    saved = read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data"))
    assert saved["state"] == "unknown" and saved["version_number"] == 7
    api.remote[7] = zip_bytes({"file.txt": remote.read_bytes()})
    api.latest = 8
    receipt = adapter.upload_dataset(dataset, "owner/data", "retry")
    adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert api.uploads == api.status_calls == 0 and api.downloads == ["owner/data/7"]


def test_old_download_api_fails_before_upload_intent(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.dataset_download_files = lambda ref, path=None: None
    with pytest.raises(KaggleAdapterError, match="cannot verify exact"):
        adapter.upload_dataset(dataset, "owner/data", "test")
    assert api.uploads == 0
    assert read_intent(intent_path(adapter.settings.storage_dir, dataset, "owner/data")) is None


def test_retry_rechecks_actual_owner_without_changing_candidate(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.lost_response = True
    with pytest.raises(DatasetUploadUnknown):
        adapter.upload_dataset(dataset, "owner/data", "test")
    saved_path = intent_path(adapter.settings.storage_dir, dataset, "owner/data")
    before = saved_path.read_bytes()
    api.config_values["username"] = "other"
    with pytest.raises(KaggleAdapterError, match="identity rejected"):
        adapter.upload_dataset(dataset, "owner/data", "retry")
    assert api.uploads == 1 and saved_path.read_bytes() == before


def test_upload_rejects_changed_config_owner_even_if_ref_matches_token(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    adapter.credentials = KaggleCredentials("test", username="wrong", api_token="fake")
    with pytest.raises(KaggleAdapterError, match="identity rejected"):
        adapter.upload_dataset(dataset, "owner/data", "test")
    assert api.uploads == 0


def test_oauth_cached_username_is_not_an_identity_claim(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.config_values["auth_method"] = "oauth"
    api._introspect_token = lambda token: "different"
    assert not adapter.identity("owner")["identity_verified"]


def test_exact_content_poll_logs_permission_then_missing_source(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    messages = []
    adapter.log = messages.append
    import requests
    response = requests.Response()
    response.status_code = 403
    errors = iter([requests.HTTPError("Forbidden", response=response), requests.HTTPError("Forbidden", response=response),
                   ValueError("payload_inventory_mismatch: missing=runtime/config.py")])
    def verify(*args):
        raise next(errors)
    adapter.verify_dataset_content = verify
    adapter._sleep = lambda _: None
    with pytest.raises(ValueError, match="missing=runtime"):
        adapter.wait_dataset("owner/data", permission_grace_seconds=900,
                             upload_receipt=DatasetUploadReceipt(7, dataset_dir=str(dataset), content_sha256="frozen"))
    assert len(messages) == 2
    assert "403" in messages[0] and "missing=runtime" in messages[1]


def test_cleanup_does_not_erase_unknown_submission_intent(tmp_path, monkeypatch):
    from app.main import cleanup_expired_job, create_app
    from app.worker import job_paths
    from test_relay_api import seed_job
    from app.upload_intent import write_intent
    app = create_app(make_settings(tmp_path))
    job_id = seed_job(app, "failed", dataset_ref="owner/data")
    dataset = job_paths(app.state.settings, job_id)["dataset_dir"]
    saved_path = intent_path(tmp_path, dataset, "owner/data")
    write_intent(saved_path, {"schema_version": 1, "version_number": 7, "state": "unknown"})
    before = saved_path.read_bytes()
    cleanup_expired_job(app.state.settings, app.state.db, job_id)
    assert saved_path.read_bytes() == before


def test_readback_rejects_local_source_replacement_during_download(tmp_path, monkeypatch):
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    receipt = adapter.upload_dataset(dataset, "owner/data", "test")
    def download(ref, path=None, **kwargs):
        (dataset / "file.txt").write_bytes(b"changed!")
        (Path(path) / "data.zip").write_bytes(zip_bytes({"file.txt": b"changed!"}))
    api.dataset_download_files = download
    with pytest.raises(KaggleAdapterError, match="frozen content changed"):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)

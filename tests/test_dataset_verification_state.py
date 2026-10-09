import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from app.dataset_verification_state import (
    DatasetRequestGate, REQUEST_INTERVAL_SECONDS, VerificationDeferred, VerificationStore,
)
from app.dataset_verification_process import verification_error, verification_request_timeouts
from test_dataset_file_verification import fallback_fixture, missing_archive
from app.kaggle_adapter import KaggleAdapterInterrupted
from app.upload_intent import content_digest


def test_checkpoint_survives_reopen_and_rejects_scope_changes(tmp_path):
    expected = {"file": (4, hashlib.sha256(b"data").hexdigest())}
    store = VerificationStore(tmp_path)
    scope, verified = store.load("owner/data", 1, tmp_path / "source", "frozen", expected)
    store.mark_file(scope, "file", expected["file"])
    reopened = VerificationStore(tmp_path)
    assert reopened.load("owner/data", 1, tmp_path / "source", "frozen", expected)[1] == {"file": list(expected["file"])}
    assert reopened.snapshot("owner/data", 2, tmp_path / "source", "frozen") == {}
    for version, digest, inventory in [(2, "frozen", expected), (1, "changed", expected),
                                        (1, "frozen", {"file": (4, "different")})]:
        with pytest.raises(ValueError, match="checkpoint_scope_or_inventory"):
            reopened.load("owner/data", version, tmp_path / "source", digest, inventory)
    assert store.load("owner/data", 1, tmp_path / "other-source", "frozen", expected)[1] == {}


def test_interrupted_file_is_not_committed_and_previous_files_are_resumed(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    (dataset / "second.txt").write_bytes(b"second")
    api.files["second.txt"] = b"second"
    digest = content_digest(dataset)
    def interrupt():
        if len(api.requests) == 2:
            raise KaggleAdapterInterrupted("shutdown")
    api.stream_hook = interrupt
    with pytest.raises(KaggleAdapterInterrupted):
        adapter.verify_dataset_content("owner/data", 7, dataset, digest)
    store = VerificationStore(adapter.settings.storage_dir)
    assert store.snapshot("owner/data", 7, dataset, digest)["verified_files"] == 1
    api.stream_hook = None
    assert adapter.verify_dataset_content("owner/data", 7, dataset, digest)
    assert [r["file_name"] for r in api.requests] == ["file.txt", "second.txt", "second.txt"]
    assert all(s.closed for s in api.streams)


def test_complete_checkpoint_still_checks_identity_source_and_remote_inventory(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    digest = content_digest(dataset)
    assert adapter.verify_dataset_content("owner/data", 7, dataset, digest)
    assert adapter.verify_dataset_content("owner/data", 7, dataset, digest)
    assert len(api.requests) == 1 and api.list_calls == 4
    api.config_values["username"] = "another"
    with pytest.raises(Exception, match="identity rejected"):
        adapter.verify_dataset_content("owner/data", 7, dataset, digest)
    api.config_values["username"] = "owner"
    (dataset / "file.txt").write_bytes(b"changed!")
    with pytest.raises(Exception, match="frozen content changed"):
        adapter.verify_dataset_content("owner/data", 7, dataset, digest)


def test_corrupt_checkpoint_cannot_skip_remote_content(tmp_path):
    store = VerificationStore(tmp_path)
    expected = {"file": (4, "a" * 64)}
    scope, _ = store.load("owner/data", 1, tmp_path, "frozen", expected)
    with store.connect() as conn:
        conn.execute("UPDATE candidates SET verified=? WHERE scope=?", (json.dumps({"file": [4, "b" * 64]}), scope))
    with pytest.raises(ValueError, match="checkpoint_invalid"):
        store.load("owner/data", 1, tmp_path, "frozen", expected)


def test_request_pacing_and_cooldown_are_shared_and_restart_durable(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("app.dataset_verification_state.time.time", lambda: clock[0])
    first, alias = VerificationStore(tmp_path), VerificationStore(tmp_path)
    assert first.reserve_request("Owner") == 0
    assert alias.reserve_request("owner") == REQUEST_INTERVAL_SECONDS
    assert first.reserve_request("other") == 0
    first.cooldown("owner", 900)
    alias.cooldown("owner", 60)  # A shorter retry cannot shorten the existing cooldown.
    with pytest.raises(VerificationDeferred) as caught:
        VerificationStore(tmp_path).reserve_request("OWNER")
    assert verification_error(caught.value) == {
        "category": "http", "http_status": 429, "retry_after": 900,
        "detail": "dataset_account_cooldown: original version retained"}
    clock[0] += 900
    assert alias.reserve_request("owner") == 0


def test_http_429_records_cooldown_without_pacing_gcs_transfer(tmp_path, monkeypatch):
    store = VerificationStore(tmp_path)
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "75"
    send = Mock(return_value=response)
    monkeypatch.setattr(requests.Session, "send", send)
    gate = DatasetRequestGate(store, "owner", lambda: None)
    with verification_request_timeouts(gate):
        requests.Session().send(SimpleNamespace(url="https://api.kaggle.com/v1/test"))
    assert 74 < VerificationStore(tmp_path).cooldown_remaining("owner") <= 75
    gate.before_request = Mock(side_effect=AssertionError("GCS transfer is not a Kaggle API quota request"))
    with verification_request_timeouts(gate):
        requests.Session().send(SimpleNamespace(url="https://storage.googleapis.com/file"))


def test_pacing_remains_cancellable(tmp_path, monkeypatch):
    store = VerificationStore(tmp_path)
    monkeypatch.setattr(store, "reserve_request", lambda _: 2)
    calls = [0]
    def check():
        calls[0] += 1
        if calls[0] == 2:
            raise KaggleAdapterInterrupted("cancelled")
    with pytest.raises(KaggleAdapterInterrupted):
        DatasetRequestGate(store, "owner", check).before_request()


def test_reserved_request_honors_new_cooldown_before_sending(tmp_path, monkeypatch):
    store = VerificationStore(tmp_path)
    def reserve(owner):
        store.cooldown(owner, 75)
        return 0
    monkeypatch.setattr(store, "reserve_request", reserve)
    with pytest.raises(VerificationDeferred) as caught:
        DatasetRequestGate(store, "owner", lambda: None).before_request()
    assert caught.value.cooldown is True and 74 < caught.value.retry_after <= 75


@pytest.mark.parametrize("changes", ["total_files=0", "total_bytes=0", "complete=1"])
def test_corrupt_completion_counters_cannot_authorize_checkpoint(tmp_path, changes):
    store = VerificationStore(tmp_path)
    expected = {"file": (4, "a" * 64)}
    scope, _ = store.load("owner/data", 1, tmp_path, "frozen", expected)
    with store.connect() as conn:
        conn.execute("UPDATE candidates SET " + changes + " WHERE scope=?", (scope,))
    with pytest.raises(ValueError, match="checkpoint_invalid"):
        store.load("owner/data", 1, tmp_path, "frozen", expected)


def test_incomplete_checkpoint_cannot_be_finalized(tmp_path):
    store = VerificationStore(tmp_path)
    scope, _ = store.load("owner/data", 1, tmp_path, "frozen", {"file": (4, "a" * 64)})
    with pytest.raises(ValueError, match="checkpoint_incomplete"):
        store.mark_complete(scope)


def test_background_time_budget_saves_only_complete_files(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr("app.dataset_verification_state.time.monotonic", lambda: clock[0])
    def expire():
        clock[0] = 121
    api.stream_hook = expire
    digest = content_digest(dataset)
    with pytest.raises(VerificationDeferred, match="time budget"):
        adapter.verify_dataset_content("owner/data", 7, dataset, digest, background=True)
    assert VerificationStore(adapter.settings.storage_dir).snapshot("owner/data", 7, dataset, digest)["verified_files"] == 0
    assert all(s.closed for s in api.streams)

import json
from email.utils import formatdate
from unittest.mock import Mock

import pytest
import requests

from app.config import Settings
from app.dataset_verification_process import (
    DatasetVerificationError, retry_after_seconds, verification_error,
    verification_request_timeouts,
)
from app.kaggle_adapter import DatasetUploadReceipt, KaggleAdapter


@pytest.fixture
def verification(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.kaggle_adapter.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("app.kaggle_adapter.random.uniform", lambda *_: 0)
    logs = []
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), logs.append)
    adapter._sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    receipt = DatasetUploadReceipt(10, (), str(tmp_path), "frozen")
    return adapter, receipt, logs, clock


def http_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError("opaque error", response=response)


@pytest.mark.parametrize("error", [requests.ConnectTimeout("connect timeout"),
    requests.ReadTimeout("read timeout"), requests.ConnectionError("connection reset"),
    requests.exceptions.ChunkedEncodingError("broken stream"),
    *[http_error(status) for status in (408, 409, 429, 500, 502, 503, 504)]])
def test_transient_retries_same_frozen_candidate(verification, error):
    adapter, receipt, logs, clock = verification
    adapter.verify_dataset_content = Mock(side_effect=[error, True])
    result = adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert json.loads(result)["current_version_number"] == 10
    assert adapter.verify_dataset_content.call_count == 2
    assert adapter.verify_dataset_content.call_args_list[0] == adapter.verify_dataset_content.call_args_list[1]
    assert clock[0] == 5
    assert any("retry 1/5" in message for message in logs)
    assert not any("content rejected" in message for message in logs)


def test_retry_after_is_honored(verification):
    adapter, receipt, logs, clock = verification
    record = verification_error(http_error(429, "75"))
    adapter.verify_dataset_content = Mock(side_effect=[DatasetVerificationError(
        record["detail"], record["category"], record["http_status"], record["retry_after"]), True])
    adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert clock[0] == 75


def test_background_check_preserves_retry_after_and_releases_worker(verification):
    adapter, receipt, logs, clock = verification
    adapter.verify_dataset_content = Mock(side_effect=http_error(429, "900"))
    with pytest.raises(DatasetVerificationError) as caught:
        adapter.wait_dataset("owner/data", upload_receipt=receipt, background=True)
    assert caught.value.http_status == 429 and caught.value.retry_after == 900
    assert clock[0] == 0 and adapter.verify_dataset_content.call_count == 1
    assert adapter.verify_dataset_content.call_args.kwargs == {"archive_only": True}


def test_retry_after_date_and_invalid_values(monkeypatch):
    monkeypatch.setattr("app.dataset_verification_process.time.time", lambda: 1000)
    assert retry_after_seconds(formatdate(1060, usegmt=True)) == 60
    assert retry_after_seconds("NaN") is None
    assert retry_after_seconds("inf") is None
    assert retry_after_seconds("nonsense") is None


@pytest.mark.parametrize("error", [http_error(401), http_error(403), requests.exceptions.SSLError("certificate invalid"),
                                  ValueError("payload_digest_mismatch"), RuntimeError("timeout 429 in a path")])
def test_fatal_errors_never_authorize_retry(verification, error):
    adapter, receipt, logs, clock = verification
    adapter.verify_dataset_content = Mock(side_effect=error)
    with pytest.raises(type(error)):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert adapter.verify_dataset_content.call_count == 1
    assert clock[0] == 0


def test_retries_are_bounded(verification):
    adapter, receipt, logs, clock = verification
    adapter.verify_dataset_content = Mock(side_effect=requests.ConnectTimeout("offline"))
    with pytest.raises(DatasetVerificationError, match="dataset_verification_retry_exhausted") as error:
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert error.value.category == "transport"
    assert adapter.verify_dataset_content.call_count == 6
    assert clock[0] < 600


def test_retry_after_beyond_deadline_pauses_without_early_retry(verification):
    adapter, receipt, logs, clock = verification
    adapter.verify_dataset_content = Mock(side_effect=http_error(429, "900"))
    with pytest.raises(DatasetVerificationError, match="retry_exhausted"):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert adapter.verify_dataset_content.call_count == 1 and clock[0] == 0


def test_cancel_during_backoff_never_retries(verification):
    adapter, receipt, logs, clock = verification
    adapter.verify_dataset_content = Mock(side_effect=http_error(429, "90"))
    def cancel():
        if clock[0] >= 2:
            raise RuntimeError("canceled by user")
    adapter.dataset_cancel_check = cancel
    with pytest.raises(RuntimeError, match="canceled by user"):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert adapter.verify_dataset_content.call_count == 1 and clock[0] == 2


@pytest.mark.parametrize("timeout, expected", [(None, (15, 60)), (120, (15, 60)),
                                              ((5, 10), (5, 10)), ((None, None), (15, 60))])
def test_sdk_request_timeouts_are_scoped_and_bounded(monkeypatch, timeout, expected):
    send = Mock(return_value="response")
    monkeypatch.setattr(requests.Session, "send", send)
    with verification_request_timeouts():
        assert requests.Session().send("request", timeout=timeout) == "response"
    assert send.call_args.kwargs["timeout"] == expected
    assert requests.Session.send is send


def test_transport_category_is_typed_not_text():
    assert verification_error(requests.ConnectTimeout("arbitrary text"))["category"] == "transport"
    assert verification_error(RuntimeError("ConnectTimeoutError"))["category"] == "fatal"


@pytest.mark.parametrize("integrity", [False, True])
def test_worker_retains_accepted_candidate_without_duplicate_upload(tmp_path, monkeypatch, integrity):
    from app.main import create_app
    from app.worker import process_job, job_paths
    from app.upload_intent import intent_path, read_intent
    from test_a3_content_identity import fixture_adapter
    from test_relay_api import seed_job

    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    app = create_app(adapter.settings)
    job_id = seed_job(app, "queued", dataset_ref="owner/data", kernel_ref="owner/kernel")
    paths = job_paths(adapter.settings, job_id)
    paths["dataset_dir"].mkdir(parents=True)
    paths["kernel_dir"].mkdir(parents=True)
    (paths["dataset_dir"] / "file.txt").write_bytes(b"original")
    (paths["dataset_dir"] / "dataset-metadata.json").write_text('{"id":"owner/data"}')
    metadata = paths["kernel_dir"] / "kernel-metadata.json"
    metadata.write_text('{"id":"owner/kernel","code_file":"train.py","dataset_sources":["owner/data"]}')
    (paths["kernel_dir"] / "train.py").write_text("print('not executed')\n")
    pushes = Mock(return_value="pushed")
    adapter.push_kernel = pushes
    monkeypatch.setattr("app.worker.KaggleAdapter", lambda *a, **kw: adapter)
    monkeypatch.setattr("app.worker.finish_kernel_job", lambda *a, **kw: None)
    verify = adapter.verify_dataset_content
    error = (DatasetVerificationError("payload_digest_mismatch", "integrity") if integrity else
             DatasetVerificationError("dataset_verification_retry_exhausted: /payload_path timed out", "transport"))
    # The retry loop itself is covered above. Exercise its exhausted outcome through the real worker.
    original_wait = adapter.wait_dataset
    adapter.wait_dataset = Mock(side_effect=error)
    process_job(adapter.settings, app.state.db, job_id)
    current = app.state.db.get_job(job_id)
    saved = read_intent(intent_path(adapter.settings.storage_dir, paths["dataset_dir"], "owner/data"))
    assert saved["state"] == "accepted" and saved["version_number"] == 7
    assert current["status"] == ("failed" if integrity else "waiting_dataset")
    assert api.uploads == 1 and pushes.call_count == 0
    if integrity:
        return
    assert current["error"].startswith("dataset_upload_outcome_unknown:")
    app.state.db.update_job(job_id, status="queued", error="")
    adapter.wait_dataset = original_wait
    process_job(adapter.settings, app.state.db, job_id)
    assert api.uploads == 1 and pushes.call_count == 1
    assert json.loads(metadata.read_text())["dataset_sources"] == ["owner/data/7"]

import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import requests

from app.dataset_file_verification import archive_url_missing, version_file_inventory
from app.dataset_verification_process import DatasetVerificationError, verification_error
from app.kaggle_adapter import DatasetUploadReceipt, KaggleAdapterInterrupted, KaggleAdapterError
from app.payload_contract import RUNTIME_PATH
from app.upload_intent import content_digest
from test_a3_content_identity import fixture_adapter, zip_bytes


def missing_archive(status=404, message="No gcs url found"):
    response = requests.Response()
    response.status_code = status
    response._content = ('{"error":{"message":"' + message + '"}}').encode()
    return requests.HTTPError("download failed", response=response)


def fallback_fixture(tmp_path, monkeypatch, files=None):
    import sys
    api, adapter, dataset = fixture_adapter(tmp_path, monkeypatch)
    api.files = files or {"file.txt": b"original"}
    api.requests, api.pages, api.streams = [], [], []
    api.list_calls = 0
    api.list_hook = None
    api.stream_hook = None

    def whole(ref, path=None, **kwargs):
        api.downloads.append(ref)
        raise missing_archive()

    def listing(ref, page_token=None, page_size=200):
        api.pages.append((ref, page_token))
        api.list_calls += 1
        items = [SimpleNamespace(name=k, total_bytes=len(v)) for k, v in api.files.items()]
        result = SimpleNamespace(files=items, next_page_token="")
        return api.list_hook(result) if api.list_hook else result

    class Stream:
        closed = False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            if api.stream_hook:
                api.stream_hook()
            yield self.data

        def close(self):
            self.closed = True

    def download(req):
        api.requests.append(vars(req).copy())
        stream = Stream()
        stream.data = api.files[req.file_name]
        api.streams.append(stream)
        return stream

    @contextmanager
    def client():
        yield SimpleNamespace(datasets=SimpleNamespace(dataset_api_client=SimpleNamespace(download_dataset=download)))

    api.dataset_download_files = whole
    api.dataset_list_files = listing
    api.build_kaggle_client = client
    monkeypatch.setattr(sys.modules["kaggle.api.kaggle_api_extended"], "ApiDownloadDatasetRequest", SimpleNamespace, raising=False)
    logs = []
    adapter.log = logs.append
    return api, adapter, dataset, logs


@pytest.mark.parametrize("layout", ["loose", "payload", "expanded", "runtime"])
def test_fallback_hashes_every_file_in_original_version(tmp_path, monkeypatch, layout):
    api, adapter, dataset, logs = fallback_fixture(tmp_path, monkeypatch)
    files = {"data.yaml": b"train", "images/a.jpg": b"image", "labels/a.txt": b"label"}
    if layout != "loose":
        (dataset / "file.txt").unlink()
        if layout == "runtime":
            files[RUNTIME_PATH] = zip_bytes({"kaggle_runtime/main.py": b"source"})
        payload = zip_bytes(files)
        (dataset / "payload.zip").write_bytes(payload)
        api.files = dict(files)
        if layout == "payload":
            api.files = {"payload.zip": payload}
        elif layout == "runtime":
            del api.files[RUNTIME_PATH]
            api.files[RUNTIME_PATH[:-4] + "/kaggle_runtime/main.py"] = b"source"
    api.latest = 99  # A newer version must never affect the original candidate.
    assert adapter.verify_dataset_content("owner/data", 7, dataset, content_digest(dataset))
    assert api.downloads == ["owner/data/7"]
    assert api.pages == [("owner/data/7", None)] * 2
    assert {r["file_name"] for r in api.requests} == set(api.files)
    assert all(r["dataset_version_number"] == 7 and r["owner_slug"] == "owner"
               and r["dataset_slug"] == "data" for r in api.requests)
    assert all(s.closed for s in api.streams)
    assert api.status_calls == api.uploads == 0
    assert any("verified" in msg for msg in logs)


@pytest.mark.parametrize("failure", ["missing", "extra", "size", "digest", "truncated", "oversized", "changed_list"])
def test_fallback_never_accepts_wrong_contents(tmp_path, monkeypatch, failure):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    if failure == "missing":
        api.files = {"wrong.txt": b"original"}
    elif failure == "extra":
        api.files["extra.txt"] = b"x"
    elif failure == "size":
        api.files["file.txt"] = b"short"
    elif failure == "digest":
        api.files["file.txt"] = b"tampered"
    elif failure in {"truncated", "oversized"}:
        api.stream_hook = lambda: setattr(api.streams[-1], "data", b"x" * (3 if failure == "truncated" else 20))
    else:
        def change(result):
            if api.list_calls == 2:
                result.files.append(SimpleNamespace(name="extra", total_bytes=1))
            return result
        api.list_hook = change
    with pytest.raises(ValueError, match="payload_"):
        adapter.verify_dataset_content("owner/data", 7, dataset, content_digest(dataset))
    assert api.uploads == 0
    assert all(s.closed for s in api.streams)


@pytest.mark.parametrize("kind", ["duplicate", "case_duplicate", "unsafe", "negative", "missing", "loop", "error"])
def test_listing_fails_closed(tmp_path, monkeypatch, kind):
    api, _, _, _ = fallback_fixture(tmp_path, monkeypatch)
    def alter(result):
        if kind in {"duplicate", "case_duplicate"}:
            result.files.append(SimpleNamespace(name="file.txt" if kind == "duplicate" else "FILE.TXT", total_bytes=8))
        elif kind == "unsafe":
            result.files[0].name = "../file.txt"
        elif kind == "negative":
            result.files[0].total_bytes = -1
        elif kind == "missing":
            del result.files
        elif kind == "error":
            result.error_message = "unavailable"
        else:
            result.files = []
            result.next_page_token = "same"
        return result
    api.list_hook = alter
    with pytest.raises(ValueError, match="payload_"):
        version_file_inventory(api, "owner/data/7", lambda: None)


def test_listing_visits_every_page(tmp_path, monkeypatch):
    api, _, _, _ = fallback_fixture(tmp_path, monkeypatch)
    def pages(result):
        result.files = [SimpleNamespace(name=f"file{api.list_calls}", total_bytes=api.list_calls)]
        result.next_page_token = "second" if api.list_calls == 1 else ""
        return result
    api.list_hook = pages
    assert version_file_inventory(api, "owner/data/7", lambda: None) == {"file1": 1, "file2": 2}
    assert api.pages == [("owner/data/7", None), ("owner/data/7", "second")]


@pytest.mark.parametrize("status,message", [(401, "No gcs url found"), (403, "Forbidden"),
                                            (404, "Not Found"), (503, "No gcs url found")])
def test_unrelated_errors_do_not_enable_fallback(tmp_path, monkeypatch, status, message):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    error = missing_archive(status, message)
    assert not archive_url_missing(error)
    def fail(*args, **kwargs):
        raise error
    api.dataset_download_files = fail
    with pytest.raises(requests.HTTPError):
        adapter.verify_dataset_content("owner/data", 7, dataset, content_digest(dataset))
    assert api.pages == api.requests == []


def test_cancellation_closes_stream_and_does_not_report_success(tmp_path, monkeypatch):
    api, adapter, dataset, logs = fallback_fixture(tmp_path, monkeypatch)
    cancelled = False
    def check():
        if cancelled:
            raise KaggleAdapterInterrupted("cancelled")
    def stop():
        nonlocal cancelled
        cancelled = True
    adapter.dataset_cancel_check = check
    api.stream_hook = stop
    with pytest.raises(KaggleAdapterInterrupted):
        adapter.verify_dataset_content("owner/data", 7, dataset, content_digest(dataset))
    assert all(s.closed for s in api.streams)
    assert not any("verified" in msg for msg in logs)


def test_source_change_during_readback_is_rejected(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    api.stream_hook = lambda: (dataset / "file.txt").write_bytes(b"modified")
    with pytest.raises(Exception, match="frozen content changed"):
        adapter.verify_dataset_content("owner/data", 7, dataset, content_digest(dataset))


@pytest.mark.parametrize("version", [None, 0, -1, "7", True])
def test_invalid_version_never_downloads(tmp_path, monkeypatch, version):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="positive integer"):
        adapter.verify_dataset_content("owner/data", version, dataset, content_digest(dataset))
    assert not api.downloads and not api.requests


def test_wrong_identity_never_uses_fallback(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    api.config_values["username"] = "another"
    with pytest.raises(Exception, match="identity rejected"):
        adapter.verify_dataset_content("owner/data", 7, dataset, content_digest(dataset))
    assert not api.downloads and not api.requests


def test_wait_uses_fallback_without_new_upload(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    receipt = DatasetUploadReceipt(7, (("file.txt", 8),), str(dataset), content_digest(dataset))
    assert '"current_version_number": 7' in adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert api.uploads == api.status_calls == 0


def test_background_check_waits_for_archive_without_thousands_of_file_requests(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    receipt = DatasetUploadReceipt(7, (), str(dataset), content_digest(dataset))
    adapter._sleep = lambda _: pytest.fail("background check must release its worker, not sleep")
    with pytest.raises(DatasetVerificationError, match="dataset_archive_not_ready") as caught:
        adapter.wait_dataset("owner/data", upload_receipt=receipt, background=True)
    assert caught.value.category == "publication"
    assert api.uploads == 0 and api.list_calls == 0 and not api.requests


@pytest.mark.parametrize("pending", ["missing", "extra", "size", "empty", "changed"])
@pytest.mark.parametrize("sdk_boundary", [False, True])
def test_publishing_listing_waits_then_hashes_same_version(tmp_path, monkeypatch, pending, sdk_boundary):
    api, adapter, dataset, logs = fallback_fixture(tmp_path, monkeypatch)
    if pending == "missing":
        api.files = {"other.txt": b"original"}
    elif pending == "extra":
        api.files["extra.txt"] = b"x"
    elif pending == "size":
        api.files["file.txt"] = b"short"
    else:
        def change(result):
            if pending == "empty":
                result.files = []
            elif api.list_calls == 2:
                result.files.append(SimpleNamespace(name="extra.txt", total_bytes=1))
            return result
        api.list_hook = change
    verify = adapter.verify_dataset_content
    if sdk_boundary:
        def subprocess_error(*args):
            try:
                return verify(*args)
            except ValueError as exc:
                error = verification_error(exc)
                raise DatasetVerificationError(error["detail"], error["category"], error["http_status"]) from exc
        adapter.verify_dataset_content = subprocess_error
    waits = []
    def settle(seconds):
        waits.append(seconds)
        api.files = {"file.txt": b"original"}
        api.list_hook = None
    adapter._sleep = settle
    receipt = DatasetUploadReceipt(7, (), str(dataset), content_digest(dataset))
    assert '"current_version_number": 7' in adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert len(waits) == 1
    assert api.downloads == ["owner/data/7"] * 2
    assert all(r["dataset_version_number"] == 7 for r in api.requests)
    assert api.requests[-1]["file_name"] == "file.txt"
    assert api.uploads == api.status_calls == 0
    assert any("still publishing" in msg for msg in logs)


def test_permanent_listing_mismatch_times_out_with_diagnostics(tmp_path, monkeypatch):
    api, adapter, dataset, logs = fallback_fixture(tmp_path, monkeypatch)
    api.files = {"other.txt": b"original"}
    adapter.settings.dataset_status_permission_grace_seconds = 2
    clock = [0]
    monkeypatch.setattr("app.kaggle_adapter.time.time", lambda: clock[0])
    monkeypatch.setattr("app.kaggle_adapter.time.monotonic", lambda: clock[0])
    adapter._sleep = lambda _: clock.__setitem__(0, clock[0] + 2)
    receipt = DatasetUploadReceipt(7, (), str(dataset), content_digest(dataset))
    with pytest.raises(DatasetVerificationError, match="payload_publication_timeout") as error:
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert error.value.category == "publication"
    assert "file.txt" in str(error.value) and "other.txt" in str(error.value)
    assert len(api.downloads) == 1 and not api.requests and api.uploads == 0
    assert not any("bytes verified" in msg for msg in logs)


@pytest.mark.parametrize("failure", ["digest", "unsafe", "stream_size"])
def test_hard_integrity_failure_does_not_enter_publication_wait(tmp_path, monkeypatch, failure):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    if failure == "digest":
        api.files["file.txt"] = b"tampered"
    elif failure == "unsafe":
        api.files = {"../file.txt": b"original"}
    else:
        api.stream_hook = lambda: setattr(api.streams[-1], "data", b"short")
    adapter._sleep = lambda _: pytest.fail("integrity failures must not wait")
    receipt = DatasetUploadReceipt(7, (), str(dataset), content_digest(dataset))
    with pytest.raises(ValueError, match="payload_"):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert len(api.downloads) == 1 and api.uploads == 0


def test_publication_wait_remains_cancellable(tmp_path, monkeypatch):
    api, adapter, dataset, _ = fallback_fixture(tmp_path, monkeypatch)
    api.files = {"other.txt": b"original"}
    cancelled = [False]
    def check():
        if cancelled[0]:
            raise KaggleAdapterInterrupted("cancelled")
    adapter.dataset_cancel_check = check
    adapter._sleep = lambda _: cancelled.__setitem__(0, True)
    receipt = DatasetUploadReceipt(7, (), str(dataset), content_digest(dataset))
    with pytest.raises(KaggleAdapterInterrupted):
        adapter.wait_dataset("owner/data", upload_receipt=receipt)
    assert len(api.downloads) == 1 and api.uploads == 0


def test_sdk_process_cancel_is_checked_while_waiting_for_files(tmp_path, monkeypatch):
    _, adapter, _, _ = fallback_fixture(tmp_path, monkeypatch)
    captured = []
    real_popen = subprocess.Popen
    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        captured.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", popen)
    def cancel():
        raise KaggleAdapterInterrupted("cancelled")
    adapter.dataset_cancel_check = cancel
    real_run = adapter._run_command
    def run(cmd, **kwargs):
        assert kwargs["cancel_check"] is cancel
        return real_run([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
    monkeypatch.setattr(adapter, "_run_command", run)
    with pytest.raises(KaggleAdapterInterrupted):
        adapter._sdk_call("verify_dataset_content")
    assert captured and captured[0].poll() is not None

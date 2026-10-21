import subprocess
import sys

import pytest

from app.config import Settings
from app.kaggle_adapter import DatasetUploadUnknown, KaggleAdapter, KaggleAdapterError


def sdk_child(tmp_path, monkeypatch, code):
    logs = []
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), logs.append)
    run = adapter._run_command

    def run_fixture(cmd, **kwargs):
        return run([sys.executable, "-u", "-c", code], **kwargs)

    monkeypatch.setattr(adapter, "_run_command", run_fixture)
    return adapter, logs


def test_real_upload_child_preserves_unknown_outcome_and_redacted_cause(tmp_path, monkeypatch):
    code = '''
import os
os.environ['KAGGLE_API_TOKEN'] = 'private-upload-fixture-token'
from app.kaggle_adapter import DatasetUploadUnknown, KaggleAdapter
from app.kaggle_sdk import main
def upload(self, **kwargs):
    try:
        raise TimeoutError('response lost private-upload-fixture-token')
    except TimeoutError as exc:
        raise DatasetUploadUnknown('Dataset upload outcome unknown; original candidate retained') from exc
KaggleAdapter.upload_dataset = upload
raise SystemExit(main())
'''
    adapter, logs = sdk_child(tmp_path, monkeypatch, code)
    with pytest.raises(DatasetUploadUnknown) as error:
        adapter.upload_dataset(tmp_path, "owner/data", "fixture")
    assert "TimeoutError" in str(error.value)
    assert "response lost" in str(error.value)
    assert "private-upload-fixture-token" not in str(error.value) + "".join(logs)


def test_plain_upload_error_text_does_not_authorize_recovery(tmp_path, monkeypatch):
    code = '''
from app.kaggle_adapter import KaggleAdapter
from app.kaggle_sdk import main
def upload(self, **kwargs):
    raise ValueError('Dataset upload outcome unknown; original candidate retained')
KaggleAdapter.upload_dataset = upload
raise SystemExit(main())
'''
    adapter, _ = sdk_child(tmp_path, monkeypatch, code)
    with pytest.raises(KaggleAdapterError) as error:
        adapter.upload_dataset(tmp_path, "owner/data", "fixture")
    assert not isinstance(error.value, DatasetUploadUnknown)


@pytest.mark.parametrize("operation,record", [
    ("upload_dataset", "not-json"),
    ("upload_dataset", "[]"),
    ("upload_dataset", '{"type":"other","detail":"failure"}'),
    ("identity", '{"type":"dataset_upload_unknown","detail":"failure"}'),
])
def test_invalid_or_wrong_operation_error_is_not_recoverable(tmp_path, monkeypatch, operation, record):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), lambda _: None)
    monkeypatch.setattr(adapter, "_run_command", lambda *a, **kw: subprocess.CompletedProcess(
        [], 1, stdout="RELAY_SDK_ERROR=" + record))
    with pytest.raises(KaggleAdapterError) as error:
        adapter._sdk_call(operation)
    assert not isinstance(error.value, DatasetUploadUnknown)


def test_failed_child_cannot_return_success_receipt(tmp_path, monkeypatch):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), lambda _: None)
    monkeypatch.setattr(adapter, "_run_command", lambda *a, **kw: subprocess.CompletedProcess(
        [], 1, stdout='RELAY_SDK_RESULT={"unexpected":"result"}'))
    with pytest.raises(KaggleAdapterError):
        adapter._sdk_call("upload_dataset")

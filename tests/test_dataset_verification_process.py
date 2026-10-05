import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from app.config import Settings
from app.kaggle_adapter import KaggleAdapter, DatasetUploadReceipt, KaggleAdapterError
from app.dataset_verification_process import DatasetVerificationError, verification_error
from app.security import register_secret
from test_dataset_file_verification import fallback_fixture
from app.upload_intent import content_digest


def run_child(tmp_path, code, publication=3, content=3, log=None):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), log or (lambda _: None))
    payload = json.dumps({"operation": "verify_dataset_content", "publication_timeout_seconds": publication,
                          "transfer_timeout_seconds": content})
    return adapter._run_command([sys.executable, "-u", "-c", code], input_text=payload)


@pytest.mark.parametrize("text", ["403", "404", "409", "429", "503", "dataset_version_not_ready: "])
def test_log_text_cannot_turn_integrity_error_into_retry(tmp_path, text):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), lambda _: None)
    error = DatasetVerificationError(f"Dataset URL owner/data-{text}; payload_digest_mismatch", "integrity")
    adapter.verify_dataset_content = lambda *args: (_ for _ in ()).throw(error)
    adapter._sleep = lambda _: pytest.fail("must not retry")
    with pytest.raises(DatasetVerificationError):
        adapter.wait_dataset("owner/data", permission_grace_seconds=900,
                             upload_receipt=DatasetUploadReceipt(7, (), str(tmp_path), "frozen"))


def test_actual_http_status_survives_error_encoding():
    response = requests.Response()
    response.status_code = 429
    record = verification_error(requests.HTTPError("opaque message", response=response))
    restored = DatasetVerificationError(record["detail"], record["category"], record["http_status"])
    assert restored.category == "http" and restored.http_status == 429
    assert verification_error(RuntimeError("404 in URL"))["category"] == "fatal"


def test_real_sdk_process_serializes_error_category(tmp_path):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), lambda _: None)
    payload = {"operation": "verify_dataset_content", "arguments": {}, "storage_dir": str(tmp_path),
               "kaggle_cmd": "kaggle", "command_timeout_seconds": 5, "transfer_timeout_seconds": 5,
               "publication_timeout_seconds": 5}
    code = '''
from app.kaggle_adapter import KaggleAdapter
from app.dataset_file_verification import DatasetVersionNotReady
from app.kaggle_sdk import main
def verify(self, **kwargs):
    raise DatasetVersionNotReady('payload_inventory_mismatch: sample-404')
KaggleAdapter.verify_dataset_content = verify
raise SystemExit(main())
'''
    with pytest.raises(DatasetVerificationError) as error:
        adapter._run_command([sys.executable, "-u", "-c", code], input_text=json.dumps(payload))
    assert error.value.category == "publication" and error.value.http_status is None


def test_progress_arrives_before_child_finishes_and_is_redacted(tmp_path):
    acknowledgement = tmp_path / "progress-received"
    secret = "verification-probe-private-value"
    register_secret(secret)
    logs = []
    def log(message):
        logs.append(message)
        acknowledgement.touch()
    code = f'''
import json,time
from pathlib import Path
print('RELAY_VERIFY_EVENT='+json.dumps({{'message':'verified 1/2 {secret}'}}),flush=True)
deadline=time.monotonic()+2
while not Path({str(acknowledgement)!r}).exists() and time.monotonic()<deadline: time.sleep(.02)
assert Path({str(acknowledgement)!r}).exists(), 'progress was buffered until exit'
print('RELAY_SDK_RESULT=true',flush=True)
'''
    result = run_child(tmp_path, code, log=log)
    assert result.returncode == 0 and len(logs) == 1
    assert "verified 1/2" in logs[0] and secret not in logs[0]


def test_publication_deadline_kills_blocked_child(tmp_path, monkeypatch):
    children = []
    popen = subprocess.Popen
    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, "Popen", capture)
    started = time.monotonic()
    with pytest.raises(DatasetVerificationError, match="payload_publication_timeout"):
        run_child(tmp_path, "import time; time.sleep(30)", publication=.3, content=20)
    assert time.monotonic() - started < 4 and children[0].poll() is not None


def test_content_transfer_uses_separate_budget(tmp_path):
    code = '''
import time
print('RELAY_VERIFY_EVENT={"phase":"content"}',flush=True)
time.sleep(.8)
print('RELAY_VERIFY_EVENT={"phase":"publication"}',flush=True)
print('RELAY_SDK_RESULT=true',flush=True)
'''
    assert run_child(tmp_path, code, publication=.5, content=2).returncode == 0


def test_phase_switches_do_not_reset_publication_budget(tmp_path):
    code = '''
import time
for _ in range(10):
    time.sleep(.15)
    print('RELAY_VERIFY_EVENT={"phase":"content"}',flush=True)
    time.sleep(.01)
    print('RELAY_VERIFY_EVENT={"phase":"publication"}',flush=True)
print('RELAY_SDK_RESULT=true',flush=True)
'''
    with pytest.raises(DatasetVerificationError, match="payload_publication_timeout"):
        run_child(tmp_path, code, publication=.4, content=3)


def test_content_deadline_is_also_enforced(tmp_path):
    with pytest.raises(TimeoutError, match="content verification"):
        run_child(tmp_path, 'import time; print(\'RELAY_VERIFY_EVENT={"phase":"content"}\',flush=True); time.sleep(30)',
                  publication=3, content=.3)


def test_retry_deadline_also_bounds_content_phase(tmp_path):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), lambda _: None)
    payload = json.dumps({"operation": "verify_dataset_content", "publication_timeout_seconds": 20,
                          "transfer_timeout_seconds": 20, "verification_retry_timeout_seconds": .3})
    code = 'import time; print(\'RELAY_VERIFY_EVENT={"phase":"content"}\',flush=True); time.sleep(30)'
    with pytest.raises(DatasetVerificationError, match="retry_exhausted"):
        adapter._run_command([sys.executable, "-u", "-c", code], input_text=payload)


def test_real_child_preserves_transport_and_retry_after(tmp_path):
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path), lambda _: None)
    code = '''
import requests
from app.kaggle_adapter import KaggleAdapter
from app.kaggle_sdk import main
def verify(self, **kwargs):
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = '75'
    raise requests.HTTPError('rate limited', response=response)
KaggleAdapter.verify_dataset_content = verify
raise SystemExit(main())
'''
    payload = {"operation": "verify_dataset_content", "arguments": {}, "storage_dir": str(tmp_path),
               "kaggle_cmd": "kaggle", "command_timeout_seconds": 5, "transfer_timeout_seconds": 5,
               "publication_timeout_seconds": 5}
    with pytest.raises(DatasetVerificationError) as error:
        adapter._run_command([sys.executable, "-u", "-c", code], input_text=json.dumps(payload))
    assert error.value.category == "http" and error.value.http_status == 429
    assert error.value.retry_after == 75


def test_wait_forwards_file_verification_progress(tmp_path, monkeypatch):
    api, adapter, dataset, logs = fallback_fixture(tmp_path, monkeypatch)
    adapter.wait_dataset("owner/data", upload_receipt=DatasetUploadReceipt(7, (), str(dataset), content_digest(dataset)))
    assert any("reading exact-version file inventory" in msg for msg in logs)
    assert any("verified 1/1 files" in msg for msg in logs)
    assert any("rechecking final file inventory" in msg for msg in logs)


@pytest.mark.skipif(sys.platform != "linux", reason="production Linux process-group cleanup")
def test_timeout_kills_descendant_even_after_leader_exits(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = f'''
import subprocess,sys
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path({str(pid_file)!r}).write_text(str(child.pid))
'''
    with pytest.raises(DatasetVerificationError, match="payload_publication_timeout"):
        run_child(tmp_path, code, publication=.5, content=10)
    pid = int(pid_file.read_text())
    state_file = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while state_file.exists() and state_file.read_text().split()[2] not in {"Z", "X"}:
        assert time.monotonic() < deadline, "verification descendant is still running"
        time.sleep(.02)

import subprocess
from unittest.mock import Mock

import pytest

from app.config import Settings
from app.kaggle_adapter import (
    KaggleAdapter, KaggleAdapterError, KaggleAdapterInterrupted,
    KernelStatusQueryError, KernelStatusUnavailable, transient_kernel_query_output,
)


@pytest.fixture
def polling(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.kaggle_adapter.time.monotonic", lambda: clock[0])
    adapter = KaggleAdapter(Settings(api_token="", storage_dir=tmp_path, kernel_max_wait_seconds=100), Mock())
    adapter._sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    adapter._run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=""))
    return adapter, clock


@pytest.mark.parametrize("output", [
    "HTTPSConnectionPool: ConnectTimeoutError Connection to api.kaggle.com timed out.",
    "ReadTimeoutError", "ConnectionResetError", "NameResolutionError",
    "429 Client Error: Too Many Requests", "503 Server Error: Service Unavailable",
])
def test_query_transport_failure_recovers_without_failing_run(polling, output):
    adapter, clock = polling
    adapter.kernel_status = Mock(side_effect=[KernelStatusQueryError(output, transient=True),
                                             'KernelWorkerStatus.COMPLETE'])
    assert "COMPLETE" in adapter.wait_kernel("owner/kernel", Mock())
    assert clock[0] == 5
    assert transient_kernel_query_output(output)
    assert all(call.args == ("owner/kernel",) for call in adapter.kernel_status.call_args_list)


@pytest.mark.parametrize("output", ["401 Client Error: Unauthorized", "403 Client Error: Forbidden",
    "404 Client Error: Not Found", "SSLError certificate_verify_failed", "permission denied",
    "invalid kernel identity", "timeout in a filename"])
def test_permanent_query_errors_are_not_retry_authorization(polling, output):
    adapter, clock = polling
    adapter._run.return_value = subprocess.CompletedProcess([], 1, stdout=output)
    assert not transient_kernel_query_output(output)
    with pytest.raises(KernelStatusQueryError):
        adapter.wait_kernel("owner/kernel", Mock())
    assert clock[0] == 0


def test_retry_budget_exhaustion_keeps_unknown_not_training_failure(polling):
    adapter, clock = polling
    adapter.kernel_status = Mock(side_effect=KernelStatusQueryError("timeout", transient=True))
    with pytest.raises(KernelStatusUnavailable):
        adapter.wait_kernel("owner/kernel", Mock())
    assert clock[0] == 100
    assert adapter.kernel_status.call_count == 5


def test_explicit_kernel_error_is_still_terminal(polling):
    adapter, _ = polling
    adapter.kernel_status = Mock(return_value='KernelWorkerStatus.ERROR')
    with pytest.raises(KaggleAdapterError, match="Kernel failed") as error:
        adapter.wait_kernel("owner/kernel", Mock())
    assert not isinstance(error.value, KernelStatusUnavailable)


def test_command_timeout_is_query_error(polling):
    adapter, _ = polling
    adapter._run.side_effect = TimeoutError("configured timeout")
    with pytest.raises(KernelStatusQueryError) as error:
        adapter.kernel_status("owner/kernel")
    assert error.value.transient


def test_log_timeout_cannot_erase_complete_observation(polling):
    adapter, _ = polling
    adapter.kernel_status = Mock(return_value='KernelWorkerStatus.COMPLETE')
    adapter._run.side_effect = TimeoutError("log timeout")
    assert "COMPLETE" in adapter.wait_kernel("owner/kernel", Mock())


def test_retry_is_shutdown_interruptible(polling):
    adapter, _ = polling
    adapter.kernel_status = Mock(side_effect=KernelStatusQueryError("timeout", transient=True))
    adapter._sleep = Mock(side_effect=KaggleAdapterInterrupted("shutdown"))
    with pytest.raises(KaggleAdapterInterrupted):
        adapter.wait_kernel("owner/kernel", Mock())
    assert adapter.kernel_status.call_count == 1

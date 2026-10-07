"""Structured errors, progress and phase budgets for Dataset verification only."""

import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime

import requests

from app.dataset_file_verification import DatasetVersionNotReady
from app.security import redact_secrets


class DatasetVerificationError(RuntimeError):
    def __init__(self, detail, category="fatal", http_status=None, retry_after=None):
        super().__init__(detail)
        self.category = category
        self.http_status = http_status
        self.retry_after = retry_after


def retry_after_seconds(value):
    if value is None:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


@contextmanager
def verification_request_timeouts():
    """Bound SDK authentication and downloads inside the dedicated child only."""
    original = requests.Session.send

    def send(session, request, **kwargs):
        timeout = kwargs.get("timeout")
        if isinstance(timeout, tuple):
            connect, read = timeout
        elif hasattr(timeout, "connect_timeout"):
            connect, read = timeout.connect_timeout, timeout.read_timeout
        else:
            connect = read = timeout
        kwargs["timeout"] = (min(15, connect) if connect is not None else 15,
                             min(60, read) if read is not None else 60)
        return original(session, request, **kwargs)

    requests.Session.send = send
    try:
        yield
    finally:
        requests.Session.send = original


def verification_error(exc):
    retry_after = None
    if isinstance(exc, DatasetVerificationError):
        category, status = exc.category, exc.http_status
        retry_after = exc.retry_after
    elif isinstance(exc, DatasetVersionNotReady):
        category, status = "publication", None
    else:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if type(status) is int:
            category = "http"
            retry_after = retry_after_seconds(exc.response.headers.get("Retry-After"))
        else:
            status = None
            if isinstance(exc, requests.exceptions.SSLError):
                category = "fatal"
            elif isinstance(exc, (requests.Timeout, requests.ConnectionError,
                                  requests.exceptions.ChunkedEncodingError, TimeoutError)):
                category = "transport"
            else:
                category = "integrity" if isinstance(exc, ValueError) and str(exc).startswith("payload_") else "fatal"
    return {"category": category, "http_status": status, "retry_after": retry_after,
            "detail": redact_secrets(str(exc))}


def run_verification(adapter, cmd, env, cwd, input_text, budget, cancel_check):
    payload = json.loads(input_text)
    remaining = {
        "publication": float(payload["publication_timeout_seconds"]),
        "content": float(payload["transfer_timeout_seconds"]),
    }
    process_group = not adapter._sdk_in_process and os.name == "posix"
    process = subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", start_new_session=process_group,
    )
    lines = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    phase, last_tick, result, failure = "publication", time.monotonic(), None, None
    retry_budget = payload.get("verification_retry_timeout_seconds")
    retry_deadline = last_tick + retry_budget if retry_budget is not None else None
    try:
        process.stdin.write(input_text)
        process.stdin.close()
        # communicate() must not flush a stream we have already closed.
        process.stdin = None
        while True:
            now = time.monotonic()
            if retry_deadline is not None and now >= retry_deadline:
                raise DatasetVerificationError("dataset_verification_retry_exhausted: recovery deadline exceeded", "transport")
            remaining[phase] -= now - last_tick
            last_tick = now
            adapter._check_interrupted()
            if cancel_check is not None:
                cancel_check()
            if budget:
                budget.check_free()
            if remaining[phase] <= 0:
                if phase == "publication":
                    raise DatasetVerificationError("payload_publication_timeout: Dataset publication deadline exceeded",
                                                   "publication")
                raise TimeoutError("Dataset content verification exceeded its transfer timeout")
            try:
                line = lines.get(timeout=min(0.2, remaining[phase]))
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("RELAY_VERIFY_EVENT="):
                event = json.loads(line.removeprefix("RELAY_VERIFY_EVENT="))
                if event.get("phase") in remaining:
                    # Charge the elapsed interval to the phase that just ended.
                    now = time.monotonic()
                    remaining[phase] -= now - last_tick
                    last_tick, phase = now, event["phase"]
                if event.get("message"):
                    adapter.log(redact_secrets(str(event["message"])))
            elif line.startswith("RELAY_SDK_ERROR="):
                failure = json.loads(line.removeprefix("RELAY_SDK_ERROR="))
            elif line.startswith("RELAY_SDK_RESULT="):
                result = line
        process.wait(timeout=1)
        if process.returncode != 0:
            if failure is not None:
                raise DatasetVerificationError(failure["detail"], failure["category"], failure.get("http_status"),
                                               failure.get("retry_after"))
            raise DatasetVerificationError(f"Dataset verification process exited without structured error ({process.returncode})")
        if result is None:
            raise DatasetVerificationError("Dataset verification process did not return a result")
        return subprocess.CompletedProcess(cmd, process.returncode, stdout=result)
    finally:
        remaining[phase] -= time.monotonic() - last_tick
        adapter._verification_content_seconds = max(0.0, float(payload["transfer_timeout_seconds"]) - remaining["content"])
        # Only the reader consumes stdout; do not race it with communicate().
        if process_group or process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM) if process_group else process.terminate()
            except OSError:
                pass
        reader.join(timeout=2)
        if reader.is_alive() or process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL) if process_group else process.kill()
            except OSError:
                pass
        process.wait(timeout=5)
        reader.join(timeout=2)
        process.stdout.close()

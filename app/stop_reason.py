"""Pure, conservative classification of provider outcome and training evidence."""

import json
import math
import re

from app.security import redact_secrets


_STATES = {
    "COMPLETE": "complete", "COMPLETED": "complete", "SUCCESS": "complete",
    "SUCCEEDED": "complete", "ERROR": "error", "FAILED": "error", "FAILURE": "error",
    "CANCEL_ACKNOWLEDGED": "canceled", "CANCELED": "canceled", "CANCELLED": "canceled",
    "RUNNING": "running", "QUEUED": "queued", "PENDING": "queued",
    "NEW": "queued", "INITIALIZING": "queued", "CANCEL_REQUESTED": "running",
}
_REASONS = {"early_stopping", "epochs_completed", "time_budget", "unknown"}
_NUMBERS = {"epoch", "epochs", "best_epoch", "patience", "elapsed_seconds"}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CLI_STATUS = re.compile(
    r"[\w.-]+/[\w.-]+\s+has status\s+([\"'])([^\"'\r\n]+)\1\.?", re.ASCII,
)
_EARLY_STOP = re.compile(
    r"^EarlyStopping:\s*Training stopped early as no improvement observed in last "
    r"(?P<patience>\d{1,10}) epochs\.\s*Best results observed at epoch (?P<best_epoch>\d{1,10})\b",
)
_COMPLETED = re.compile(r"^(\d{1,10}) epochs completed in (\d+(?:\.\d+)?) hours\.?$")
_FINAL_PROGRESS = re.compile(r"^(\d{1,10})/(\d{1,10})\s+.*\b100%.*$")


def _decode(text: str):
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _status_token(status: str) -> str:
    if not isinstance(status, str):
        return "UNKNOWN"
    text = status.strip()
    if text.startswith(("{", '[', '"')):
        payload = _decode(text)
        if isinstance(payload, dict):
            text = payload.get("status")
        else:
            text = payload
        if not isinstance(text, str):
            return "UNKNOWN"
        text = text.strip()
    match = _CLI_STATUS.fullmatch(text)
    if match:
        text = match.group(2).strip()
    if text.lower().startswith("kernelworkerstatus."):
        text = text[len("KernelWorkerStatus."):]
    token = text.upper()
    return token if token in _STATES else "UNKNOWN"


def provider_state(status: str) -> str:
    """Return a canonical uppercase provider state, or '' if it is unknown.

    Accept a status token, KernelWorkerStatus enum, JSON status object/string,
    or the complete ``owner/slug has status "KernelWorkerStatus.X"`` CLI line.
    Query errors and arbitrary text are not evidence of a terminal kernel.
    """
    return {"complete": "COMPLETE", "error": "ERROR", "canceled": "CANCEL_ACKNOWLEDGED",
            "running": "RUNNING", "queued": "QUEUED"}.get(
                _STATES.get(_status_token(status)), "")


def _number(name: str, value) -> bool:
    if name == "elapsed_seconds":
        return (type(value) in (int, float) and 0 <= value <= 1e12
                and math.isfinite(value))
    return type(value) is int and (0 if name == "patience" else 1) <= value <= 1_000_000_000


def _numeric_fields(payload: dict, *, allow_epoch_overrun: bool = False) -> dict | None:
    result = {}
    for name in _NUMBERS:
        if name in payload:
            if not _number(name, payload[name]):
                return None
            result[name] = payload[name]
    if not allow_epoch_overrun and result.get("epoch", 0) > result.get("epochs", 1_000_000_000):
        return None
    if result.get("best_epoch", 0) > result.get("epoch", result.get("epochs", 1_000_000_000)):
        return None
    return result


def _log_lines(logs: str) -> tuple[list[str], float | None]:
    """Unwrap Kaggle data records, including JSONL/truncated list output."""
    if not isinstance(logs, str):
        return [], None
    lines = []
    elapsed = None

    def append_record(record) -> bool:
        nonlocal elapsed
        if not isinstance(record, dict) or not isinstance(record.get("data"), str):
            return False
        lines.extend(_ANSI.sub("", record["data"]).splitlines())
        value = record.get("time")
        if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value):
            value = float(value)
        if _number("elapsed_seconds", value):
            elapsed = value
        return True

    decoded = _decode(logs)
    if isinstance(decoded, list):
        for record in decoded:
            append_record(record)
    elif not append_record(decoded):
        for line in logs.splitlines():
            candidate = line.strip().lstrip("[,").rstrip(",]").strip()
            if not append_record(_decode(candidate)):
                lines.append(_ANSI.sub("", line))
    return lines, elapsed


def _event(line: str, prefix: str) -> dict | None:
    if not line.startswith(prefix):
        return None
    suffix = line[len(prefix):]
    if not suffix.startswith((" ", "\t", ":")):
        return None
    payload = _decode(suffix.lstrip().removeprefix(":").strip())
    return payload if isinstance(payload, dict) else None


def _training_evidence(lines: list[str], complete: bool) -> tuple[dict | None, dict]:
    marker = None
    early = None
    completed = None
    progress = {}
    for raw_line in lines:
        line = raw_line.strip()
        payload = _event(line, "TRAINING_PLATFORM_STOP")
        if payload is not None:
            # Never forward arbitrary messages, metadata, or nested values from logs.
            reason = payload.get("reason")
            if (isinstance(reason, str) and reason in _REASONS
                    and payload.get("source") == "runtime"
                    and payload.get("confidence") == "confirmed"):
                numbers = _numeric_fields(payload, allow_epoch_overrun=reason == "time_budget")
                final_epoch = (numbers is not None and "epoch" in numbers
                               and numbers.get("epochs") == numbers["epoch"])
                if numbers is not None and (reason != "epochs_completed" or final_epoch):
                    marker = {"reason": reason, "source": "runtime",
                              "confidence": "confirmed", **numbers}
                elif numbers is not None:
                    progress.update(numbers)
            continue
        payload = _event(line, "TRAINING_PLATFORM_PROGRESS")
        if payload is not None:
            numbers = _numeric_fields(payload)
            if numbers is not None and "epoch" in numbers and "epochs" in numbers:
                progress = numbers
                if complete and numbers["epoch"] == numbers["epochs"]:
                    completed = {"reason": "epochs_completed", "source": "logs",
                                 "confidence": "inferred", **numbers}
            continue
        match = _EARLY_STOP.match(line)
        if match:
            numbers = {name: int(value) for name, value in match.groupdict().items()}
            if _numeric_fields(numbers) is not None:
                early = {"reason": "early_stopping", "source": "logs",
                         "confidence": "confirmed", **numbers}
            continue
        if not complete:
            continue
        match = _COMPLETED.fullmatch(line)
        if match:
            numbers = {"epoch": int(match[1]), "elapsed_seconds": float(match[2]) * 3600}
            if _numeric_fields(numbers) is not None:
                # Ultralytics also prints this summary after early/time-budget stops.
                progress.update(numbers)
        match = _FINAL_PROGRESS.fullmatch(line)
        if match and match[1] == match[2]:
            numbers = {"epoch": int(match[1]), "epochs": int(match[2])}
            if _numeric_fields(numbers) is not None:
                completed = {"reason": "epochs_completed", "source": "logs",
                             "confidence": "inferred", **numbers}
    return marker or early or completed, progress


def classify_stop_reason(provider_status: str, logs: str = '', *, cancel_requested: bool = False) -> dict:
    """Classify a snapshot without IO or claiming an unobserved terminal state.

    ``provider_status`` in the result is an allowlisted uppercase token (UNKNOWN
    otherwise), never raw provider output. ``training_reason`` is a scalar reason
    retained when training stopped but the provider did not complete successfully.
    A cancellation near 12 hours can suggest, but cannot confirm, a time limit.
    """
    token = _status_token(provider_status)
    state = _STATES.get(token, "unknown")
    lines, elapsed = _log_lines(logs)
    training, progress = _training_evidence(lines, state == "complete")
    details = dict(progress)
    if elapsed is not None:
        details["elapsed_seconds"] = elapsed
    if training:
        details.update({key: value for key, value in training.items() if key in _NUMBERS})
    result = {"reason": "unknown", "source": "provider", "confidence": "unknown",
              "provider_status": token, **details}
    message = "No authoritative terminal provider status has been observed."
    if training and state != "complete":
        result["training_reason"] = training["reason"]
    if state == "canceled":
        result.update(reason="user_canceled" if cancel_requested else "provider_canceled",
                      source="relay" if cancel_requested else "provider", confidence="confirmed")
        message = ("Kaggle confirmed cancellation requested through Relay." if cancel_requested else
                   "Kaggle confirmed cancellation; the cause was not supplied.")
        # This is only a heuristic about the last provider log, not a diagnosis.
        if not cancel_requested and elapsed is not None and 42900 <= elapsed <= 43800:
            result["suspected_reason"] = "time_limit"
            message += " Last log timing is consistent with a 12-hour limit, but does not prove it."
    elif state == "error":
        stopped = training and training["reason"] != "unknown"
        result.update(reason="provider_failed", confidence="confirmed")
        message = ("Kaggle reported failure after a training stop was logged; later processing may have failed."
                   if stopped else "Kaggle reported kernel failure; the failing stage is not established.")
    elif state == "complete":
        result["reason"] = "completed_unknown"
        message = "Kaggle completed successfully; the training stop reason is unavailable."
        if training and training["reason"] != "unknown":
            result.update(training)
            message = {
                "early_stopping": "Training stopped after no improvement within the patience window.",
                "epochs_completed": "Training epoch completion was recorded and Kaggle completed successfully.",
                "time_budget": "The runtime reported stopping at its configured training time budget.",
            }[training["reason"]]
    elif state in {"running", "queued"}:
        message = "Kaggle has not reported a terminal state; the kernel is still running or queued."
    result["message"] = redact_secrets(message)
    return result

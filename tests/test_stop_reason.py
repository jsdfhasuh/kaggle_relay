import json

import pytest

from app.stop_reason import classify_stop_reason, provider_state


EARLY_STOP = (
    "EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. "
    "Best results observed at epoch 318, best model saved as best.pt.\n"
    "To update EarlyStopping(patience=100) pass a new patience value."
)


def stop_marker(reason="early_stopping", **fields):
    return "TRAINING_PLATFORM_STOP " + json.dumps({
        "reason": reason, "source": "runtime", "confidence": "confirmed", **fields,
    })


@pytest.mark.parametrize(("status", "expected"), [
    ("COMPLETE", "complete"), ("KernelWorkerStatus.ERROR", "error"),
    ('owner/error-success has status "KernelWorkerStatus.RUNNING"', "running"),
    ('owner/success has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "canceled"),
    ('{"status": "complete", "slug": "error"}', "complete"),
    ('{"status": "RUNNING", "failureMessage": "error complete cancel"}', "running"),
    ('"KernelWorkerStatus.COMPLETE"', "complete"), ("CANCEL_REQUESTED", "running"),
    ("queued", "queued"), ("pending", "queued"), ("FAILED", "error"),
    ("success", "complete"), ("succeeded", "complete"), ("cancelled", "canceled"),
    ("Kernel status failed: ReadTimeoutError", "unknown"),
    ("owner/success-error", "unknown"), ('{"slug": "complete"}', "unknown"),
    ('{"status": {"status": "complete"}}', "unknown"),
    ('{"status": "not complete"}', "unknown"), ("{broken", "unknown"),
    ("null", "unknown"), ("[]", "unknown"), ("", "unknown"),
    ("some text KernelWorkerStatus.COMPLETE", "unknown"),
    ("KernelWorkerStatus.COMPLETE\nKernelWorkerStatus.ERROR", "unknown"),
])
def test_exact_provider_status(status, expected):
    canonical = {"complete": "COMPLETE", "error": "ERROR", "canceled": "CANCEL_ACKNOWLEDGED",
                 "running": "RUNNING", "queued": "QUEUED", "unknown": ""}
    assert provider_state(status) == canonical[expected]


@pytest.mark.parametrize(("task_id", "epoch", "epochs", "elapsed"), [
    ("28adaefafb164a948f32de1c61799789", 786, 1000, 43212.9),
    ("b760c3494bae40df86a6964ffcef2d77", 486, 500, 43213.7),
])
def test_historical_cancellations_are_not_confirmed_time_limits(task_id, epoch, epochs, elapsed):
    # Reduced synthetic fixtures from the reported historical observations, not live queries.
    logs = json.dumps([
        {"time": elapsed - 60, "data": "TRAINING_PLATFORM_PROGRESS: " + json.dumps({
            "epoch": epoch, "epochs": epochs, "elapsed_seconds": elapsed - 60,
        })},
        {"time": elapsed, "data": f"{epoch + 1}/{epochs} 12.1G 42% | partial batch"},
    ])
    result = classify_stop_reason('owner/remote has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', logs)
    assert result["reason"] == "provider_canceled", task_id
    assert result["suspected_reason"] == "time_limit"
    assert result["confidence"] == "confirmed"  # Cancellation only, not its cause.
    assert result["elapsed_seconds"] == elapsed
    assert result["epoch"] == epoch
    assert result["epochs"] == epochs
    assert "training_reason" not in result
    assert classify_stop_reason("RUNNING", logs)["reason"] == "unknown"


def test_explicit_cancellation_requires_provider_confirmation():
    assert classify_stop_reason("RUNNING", cancel_requested=True)["reason"] == "unknown"
    result = classify_stop_reason("CANCEL_ACKNOWLEDGED", '[{"time":43213,"data":"x"}]',
                                  cancel_requested=True)
    assert result["reason"] == "user_canceled"
    assert result["source"] == "relay"
    assert "suspected_reason" not in result


@pytest.mark.parametrize("reason", ["early_stopping", "epochs_completed", "time_budget"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_runtime_marker(reason, wrapped):
    epoch = 500 if reason == "epochs_completed" else 418
    logs = stop_marker(reason, epoch=epoch, epochs=500, best_epoch=318, patience=100,
                       elapsed_seconds=36600.5)
    if wrapped:
        logs = json.dumps([{"time": 36605, "data": logs + "\n"}])
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == reason
    assert result["source"] == "runtime"
    assert result["confidence"] == "confirmed"
    assert result["epoch"] == epoch
    assert result["best_epoch"] == 318
    assert result["patience"] == 100
    assert result["elapsed_seconds"] == 36600.5


@pytest.mark.parametrize("wrapped", [False, True])
def test_canonical_early_stopping_beats_epoch_summary(wrapped):
    logs = EARLY_STOP + "\n418 epochs completed in 10.17 hours."
    if wrapped:
        logs = json.dumps([{"data": logs, "time": 36620}])
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == "early_stopping"
    assert result["best_epoch"] == 318
    assert result["patience"] == 100
    assert result["source"] == "logs"


@pytest.mark.parametrize("logs", [EARLY_STOP, stop_marker(), stop_marker("time_budget")])
@pytest.mark.parametrize(("status", "expected"), [
    ("ERROR", "provider_failed"), ("CANCEL_ACKNOWLEDGED", "provider_canceled"),
    ("RUNNING", "unknown"), ("", "unknown"),
])
def test_training_stop_does_not_hide_provider_outcome(logs, status, expected):
    result = classify_stop_reason(status, logs + "\nONNX export error: conversion failed")
    assert result["reason"] == expected
    assert result["training_reason"] in {"early_stopping", "time_budget"}


@pytest.mark.parametrize("logs", [
    "500/500 12.1G 0.2 0.1 0.3 128 640: 100% | 20/20 [01:00<00:00]",
    'TRAINING_PLATFORM_PROGRESS: {"epoch":500,"epochs":500}',
])
def test_epoch_completion_requires_complete_provider(logs):
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == "epochs_completed"
    assert result["epoch"] == 500
    for status, expected in [("ERROR", "provider_failed"), ("CANCELED", "provider_canceled"),
                             ("RUNNING", "unknown"), ("", "unknown")]:
        result = classify_stop_reason(status, logs)
        assert result["reason"] == expected
        assert "training_reason" not in result


@pytest.mark.parametrize("logs", ["", "logs unavailable: timeout", "{broken", "[]", "null",
                                        "500/500 12.1G 20%", stop_marker("unknown")])
def test_complete_without_evidence_is_unknown(logs):
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == "completed_unknown"
    assert result["confidence"] == "unknown"


@pytest.mark.parametrize(("status", "reason"), [
    ("ERROR", "provider_failed"), ("CANCELED", "provider_canceled"),
    ("RUNNING", "unknown"), ("QUEUED", "unknown"), ("unavailable", "unknown"),
])
def test_no_logs(status, reason):
    result = classify_stop_reason(status)
    assert result["reason"] == reason
    assert "suspected_reason" not in result


@pytest.mark.parametrize("fields", [
    {"reason": []}, {"reason": "KGAT_secret"}, {"source": "provider"},
    {"confidence": "inferred"}, {"epoch": True}, {"epoch": -1}, {"epoch": "418"},
    {"epochs": []}, {"best_epoch": {}}, {"patience": -1}, {"patience": 1.5},
    {"elapsed_seconds": float("nan")}, {"elapsed_seconds": float("inf")},
    {"elapsed_seconds": "secret"}, {"elapsed_seconds": -1},
    {"epoch": 501, "epochs": 500}, {"best_epoch": 419, "epoch": 418},
])
def test_malformed_marker_ignored(fields):
    payload = {"reason": "early_stopping", "source": "runtime", "confidence": "confirmed"}
    payload.update(fields)
    result = classify_stop_reason("COMPLETE", "TRAINING_PLATFORM_STOP " + json.dumps(payload))
    assert result["reason"] == "completed_unknown"


def test_unknown_fields_and_secrets_never_forwarded(monkeypatch):
    seen = []
    from app.security import redact_secrets

    def redact(message):
        seen.append(message)
        return redact_secrets(message)

    monkeypatch.setattr("app.stop_reason.redact_secrets", redact)
    result = classify_stop_reason('{"status":"ERROR","failureMessage":"Bearer private-token"}',
        stop_marker(message="KGAT_do_not_leak", metadata={"token": "private-token"},
                    training_reason={"secret": "private-token"}))
    encoded = json.dumps(result)
    assert "private-token" not in encoded
    assert "KGAT_do_not_leak" not in encoded
    assert "metadata" not in result
    assert result["provider_status"] == "ERROR"
    assert seen  # All human-readable messages pass through the shared redactor.
    assert all(not isinstance(value, (dict, list)) for value in result.values())


@pytest.mark.parametrize("logs", [
    'prefix TRAINING_PLATFORM_STOP {"reason":"early_stopping","source":"runtime","confidence":"confirmed"}',
    'TRAINING_PLATFORM_STOPPED {"reason":"early_stopping","source":"runtime","confidence":"confirmed"}',
    'TRAINING_PLATFORM_STOP {"reason":"early_stopping","source":"runtime","confidence":"confirmed"} trailing',
    'TRAINING_PLATFORM_STOP {"reason":',
    'TRAINING_PLATFORM_STOP []',
])
def test_marker_requires_whole_line_and_object(logs):
    assert classify_stop_reason("COMPLETE", logs)["reason"] == "completed_unknown"


def test_jsonl_ansi_and_truncated_log_list():
    record = {"time": "43213.7", "data": "\x1b[32m" + stop_marker("time_budget") + "\x1b[0m\n"}
    logs = "[\n" + json.dumps(record) + ",\n{broken"
    result = classify_stop_reason("CANCELED", logs)
    assert result["training_reason"] == "time_budget"
    assert result["elapsed_seconds"] == 43213.7
    assert result["suspected_reason"] == "time_limit"


@pytest.mark.parametrize("time", [True, -1, "not-a-time", 3600, 46000, None])
def test_time_limit_hint_requires_plausible_provider_log_time(time):
    logs = json.dumps([{"time": time, "data": "still training"}])
    assert "suspected_reason" not in classify_stop_reason("CANCELED", logs)


def test_last_log_time_not_an_earlier_timestamp_controls_hint():
    logs = json.dumps([{"time": 43213, "data": "x"}, {"time": 50000, "data": "y"}])
    assert "suspected_reason" not in classify_stop_reason("CANCELED", logs)


def test_runtime_marker_wins_over_legacy_text():
    logs = EARLY_STOP + "\n" + stop_marker("time_budget", epoch=418, epochs=500)
    assert classify_stop_reason("COMPLETE", logs)["reason"] == "time_budget"


@pytest.mark.parametrize("logs", [
    "EarlyStopping: Training stopped early as no improvement observed in last " + "9" * 5000
    + " epochs. Best results observed at epoch 318, best model saved as best.pt.",
    "9" * 5000 + " epochs completed in 10.0 hours.",
    "9" * 5000 + "/" + "9" * 5000 + " 12G 100%",
    "TRAINING_PLATFORM_STOP {\"reason\":\"early_stopping\",\"epoch\":" + "9" * 5000 + "}",
    "[" * 2000 + "]" * 2000,
])
def test_extreme_malformed_numbers_and_nesting_are_ignored(logs):
    assert classify_stop_reason("COMPLETE", logs)["reason"] == "completed_unknown"


def test_partial_marker_cannot_override_previous_valid_marker():
    logs = stop_marker("early_stopping", best_epoch=318, patience=100)
    logs += '\nTRAINING_PLATFORM_STOP {"reason":"time_budget"'
    result = classify_stop_reason("ERROR", logs)
    assert result["reason"] == "provider_failed"
    assert result["training_reason"] == "early_stopping"
    assert result["best_epoch"] == 318


@pytest.mark.parametrize("prefix", [
    "", 'TRAINING_PLATFORM_PROGRESS: {"epoch":417,"epochs":500}\n',
])
def test_epoch_summary_alone_does_not_prove_configured_epochs_exhausted(prefix):
    result = classify_stop_reason("COMPLETE", prefix + "418 epochs completed in 10.17 hours.")
    assert result["reason"] == "completed_unknown"
    assert result["confidence"] == "unknown"
    assert result["epoch"] == 418
    assert result["elapsed_seconds"] == pytest.approx(36612)


def test_epoch_summary_preserves_explicit_time_budget_reason():
    logs = stop_marker("time_budget", epoch=418, epochs=500)
    logs += "\n418 epochs completed in 10.17 hours."
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == "time_budget"
    assert result["epoch"] == 418
    assert result["epochs"] == 500


def test_epoch_summary_with_final_progress_confirms_epochs_completed():
    logs = 'TRAINING_PLATFORM_PROGRESS: {"epoch":500,"epochs":500}\n'
    logs += "500 epochs completed in 10.0 hours."
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == "epochs_completed"
    assert result["epoch"] == result["epochs"] == 500


@pytest.mark.parametrize("fields", [{}, {"epoch":418}, {"epoch":418, "epochs":500}])
def test_epochs_completed_marker_requires_matching_configured_epoch_count(fields):
    result = classify_stop_reason("COMPLETE", stop_marker("epochs_completed", **fields))
    assert result["reason"] == "completed_unknown"
    for name, value in fields.items():
        assert result[name] == value


@pytest.mark.parametrize("logs", [
    "", "Traceback: ONNX export failed", "418 epochs completed in 10.17 hours.",
    stop_marker("unknown"),
])
def test_provider_error_does_not_identify_failure_stage(logs):
    result = classify_stop_reason("ERROR", logs)
    assert result["reason"] == "provider_failed"
    assert result["source"] == "provider"
    assert result["confidence"] == "confirmed"


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(("status", "reason"), [
    ("COMPLETE", "time_budget"), ("ERROR", "provider_failed"),
    ("CANCEL_ACKNOWLEDGED", "provider_canceled"), ("RUNNING", "unknown"),
])
def test_confirmed_runtime_time_budget_allows_epoch_overrun(wrapped, status, reason):
    logs = stop_marker("time_budget", epoch=418, epochs=417, best_epoch=418)
    if wrapped:
        logs = json.dumps([{"data": logs}])
    result = classify_stop_reason(status, logs)
    assert result["reason"] == reason
    assert result["epoch"] == result["best_epoch"] == 418
    assert result["epochs"] == 417
    if status == "COMPLETE":
        assert result["source"] == "runtime"
        assert result["confidence"] == "confirmed"
    else:
        assert result["training_reason"] == "time_budget"


@pytest.mark.parametrize("reason", ["early_stopping", "epochs_completed", "unknown"])
def test_other_runtime_reasons_still_reject_epoch_overrun(reason):
    result = classify_stop_reason("COMPLETE", stop_marker(reason, epoch=418, epochs=417))
    assert result["reason"] == "completed_unknown"
    assert "epoch" not in result


@pytest.mark.parametrize("fields", [
    {"source": "logs"}, {"confidence": "inferred"}, {"source": None},
    {"confidence": None}, {"best_epoch": 419}, {"epoch": 418.5},
    {"epochs": -1}, {"patience": True}, {"elapsed_seconds": float("inf")},
])
def test_time_budget_overrun_does_not_relax_other_validation(fields):
    payload = {"epoch": 418, "epochs": 417, **fields}
    result = classify_stop_reason("COMPLETE", stop_marker("time_budget", **payload))
    assert result["reason"] == "completed_unknown"
    assert "epoch" not in result


def test_progress_cannot_opt_into_time_budget_epoch_overrun():
    logs = 'TRAINING_PLATFORM_PROGRESS: ' + json.dumps({
        "reason": "time_budget", "source": "runtime", "confidence": "confirmed",
        "epoch": 418, "epochs": 417,
    })
    result = classify_stop_reason("COMPLETE", logs)
    assert result["reason"] == "completed_unknown"
    assert "epoch" not in result

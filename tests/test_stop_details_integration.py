import json
import subprocess
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from test_relay_api import auth_headers, make_settings, seed_job
from app.database import RelayDb
from app.kaggle_adapter import KaggleAdapter, KaggleAdapterError, KernelStatusUnavailable
from app.main import apply_progress_callback, create_app, mark_worker_exception, request_job_cancel
from app.schemas import JobProgressRequest
from app.worker import finish_kernel_job


def test_existing_database_migrates_and_empty_details_are_api_compatible(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    job_id = seed_job(app, "failed")
    db = app.state.db
    with db.connect() as conn:
        conn.execute("ALTER TABLE jobs DROP COLUMN stop_details")
    RelayDb(db.path)
    with TestClient(app) as client:
        response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers())
        assert response.status_code == 200
        assert response.json()["stop_details"] == {}


def test_provider_cancel_persists_reason_without_claiming_time_limit(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    job_id = seed_job(app, "waiting_kernel")
    adapter = KaggleAdapter(settings, Mock())
    adapter.kernel_status = Mock(return_value='KernelWorkerStatus.CANCEL_ACKNOWLEDGED')
    logs = json.dumps([{"time": 43213.7, "data": "487/500 training"}])
    adapter._run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=logs))
    with pytest.raises(KaggleAdapterError) as caught:
        finish_kernel_job(settings, app.state.db, job_id, adapter)
    mark_worker_exception(app.state.db, job_id, caught.value)
    job = app.state.db.get_job(job_id)
    details = json.loads(job["stop_details"])
    assert job["status"] == "failed"
    assert details["reason"] == "provider_canceled"
    assert details.get("suspected_reason") == "time_limit"
    assert any("STOP_DETAILS" in message for message in app.state.db.recent_logs(job_id))
    before = job["completed_at"]
    app.state.db.update_job(job_id, cleaned_at=123)
    with TestClient(app) as client:
        body = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()
        assert body["stop_details"] == details
        assert body["completed_at"] == before


def test_monitoring_timeout_records_uncertainty_without_finalizing(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.kernel_max_wait_seconds = 1
    clock = [0.0]
    monkeypatch.setattr("app.kaggle_adapter.time.monotonic", lambda: clock[0])
    app = create_app(settings)
    job_id = seed_job(app, "waiting_kernel", progress=75)
    adapter = KaggleAdapter(settings, Mock())
    adapter.kernel_status = Mock(return_value='KernelWorkerStatus.RUNNING')
    adapter._run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=""))
    adapter._sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    with pytest.raises(KernelStatusUnavailable) as caught:
        finish_kernel_job(settings, app.state.db, job_id, adapter)
    mark_worker_exception(app.state.db, job_id, caught.value)
    job = app.state.db.get_job(job_id)
    assert job["status"] == "waiting_kernel"
    assert job["progress"] == 75
    assert job["completed_at"] is None
    assert json.loads(job["stop_details"])["reason"] == "monitoring_timeout"


def test_artifact_failure_keeps_successful_training_stop_details(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    job_id = seed_job(app, "waiting_kernel")
    details = {"reason": "early_stopping", "source": "runtime", "confidence": "confirmed",
               "epoch": 418, "best_epoch": 318, "patience": 100}
    adapter = Mock()
    adapter.last_stop_details = details
    adapter.wait_kernel.return_value = "complete"
    adapter.download_output.side_effect = OSError("output download failed")
    with pytest.raises(OSError) as caught:
        finish_kernel_job(settings, app.state.db, job_id, adapter)
    mark_worker_exception(app.state.db, job_id, caught.value)
    job = app.state.db.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "output download failed"
    assert json.loads(job["stop_details"]) == details


def test_user_cancellation_before_submission_is_recorded(tmp_path):
    app = create_app(make_settings(tmp_path))
    job_id = seed_job(app, "queued")
    request_job_cancel(app.state.db, app.state.db.get_job(job_id))
    job = app.state.db.get_job(job_id)
    assert job["status"] == "canceled"
    assert json.loads(job["stop_details"])["reason"] == "user_canceled"


def test_stop_observation_is_idempotent_and_does_not_change_terminal_status(tmp_path):
    app = create_app(make_settings(tmp_path))
    job_id = seed_job(app, "failed")
    details = {"reason": "provider_canceled", "confidence": "confirmed"}
    db = app.state.db
    completed_at = db.get_job(job_id)["completed_at"]
    db.record_stop_details(job_id, details)
    db.record_stop_details(job_id, details)
    assert len([x for x in db.recent_logs(job_id) if x.startswith("STOP_DETAILS ")]) == 1
    assert db.get_job(job_id)["status"] == "failed"
    assert db.get_job(job_id)["completed_at"] == completed_at


def test_training_stop_callback_preserves_progress_and_later_terminal_observation(tmp_path):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel", progress=76)
    progress = json.dumps({"epoch": 418, "epochs": 500})
    db.update_job(job_id, kernel_status=progress)
    details = {"reason": "early_stopping", "source": "runtime", "confidence": "confirmed",
               "epoch": 418, "epochs": 500, "best_epoch": 318, "patience": 100}
    payload = JobProgressRequest(event_type="training_stop",
                                 log="TRAINING_PLATFORM_STOP " + json.dumps(details), stop_details=details)
    apply_progress_callback(db, db.get_job(job_id), payload)
    job = db.get_job(job_id)
    assert job["status"] == "waiting_kernel"
    assert job["kernel_status"] == progress
    assert job["progress"] == 76
    assert json.loads(job["stop_details"])["training_reason"] == "early_stopping"
    final = {"reason": "provider_canceled", "confidence": "confirmed"}
    stale = db.get_job(job_id)
    db.record_stop_details(job_id, final)
    observed = json.loads(db.get_job(job_id)["stop_details"])
    apply_progress_callback(db, stale, payload)
    assert json.loads(db.get_job(job_id)["stop_details"]) == observed
    db.finalize_job(job_id, "failed", error="provider canceled")
    apply_progress_callback(db, stale, payload)
    assert json.loads(db.get_job(job_id)["stop_details"]) == observed


@pytest.mark.parametrize("reason", ["monitoring_timeout", "completed_unknown", "provider_failed"])
def test_unavailable_later_logs_do_not_erase_recorded_training_reason(tmp_path, reason):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel")
    db.record_stop_details(job_id, {"reason": "training_stopped", "training_reason": "early_stopping",
                                   "source": "runtime", "confidence": "confirmed", "best_epoch": 318})
    db.record_stop_details(job_id, {"reason": reason, "source": "provider", "confidence": "unknown"})
    details = json.loads(db.get_job(job_id)["stop_details"])
    assert details["training_reason"] == "early_stopping"
    assert details["best_epoch"] == 318
    assert details["reason"] == ("early_stopping" if reason == "completed_unknown" else reason)


@pytest.mark.parametrize("monitoring", [True, False])
def test_stop_persistence_failure_cannot_replace_original_worker_outcome(tmp_path, monkeypatch, monitoring):
    import sqlite3
    settings = make_settings(tmp_path)
    app = create_app(settings)
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel")
    adapter = Mock()
    adapter.last_stop_details = {"reason": "monitoring_timeout" if monitoring else "epochs_completed"}
    original = KernelStatusUnavailable("provider still running")
    if monitoring:
        adapter.wait_kernel.side_effect = original
    else:
        adapter.wait_kernel.return_value = "complete"
        adapter.download_output.return_value = "downloaded"
        adapter.package_artifacts.return_value = None
    monkeypatch.setattr(db, "record_stop_details", Mock(side_effect=sqlite3.OperationalError("locked")))
    if monitoring:
        with pytest.raises(KernelStatusUnavailable) as caught:
            finish_kernel_job(settings, db, job_id, adapter)
        assert caught.value is original
        mark_worker_exception(db, job_id, caught.value)
        assert db.get_job(job_id)["status"] == "waiting_kernel"
    else:
        finish_kernel_job(settings, db, job_id, adapter)
        assert adapter.download_output.called
        assert db.get_job(job_id)["status"] == "complete"


def test_late_training_evidence_enriches_provider_failure_without_replacing_it(tmp_path):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel")
    db.record_stop_details(job_id, {"reason": "provider_failed", "source": "provider", "confidence": "confirmed"})
    db.finalize_job(job_id, "failed", error="export failed")
    db.record_stop_details(job_id, {"reason": "training_stopped", "training_reason": "early_stopping",
                                   "source": "runtime", "confidence": "confirmed", "best_epoch": 318},
                           expected_statuses={"failed"})
    details = json.loads(db.get_job(job_id)["stop_details"])
    assert details["reason"] == "provider_failed"
    assert details["training_reason"] == "early_stopping"
    assert details["best_epoch"] == 318
    assert db.get_job(job_id)["error"] == "export failed"


def test_partial_terminal_training_evidence_preserves_runtime_metadata(tmp_path):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel")
    db.record_stop_details(job_id, {"reason": "training_stopped", "training_reason": "early_stopping",
                                   "source": "runtime", "confidence": "confirmed", "epochs": 500,
                                   "epoch": 418, "best_epoch": 318, "patience": 100})
    db.record_stop_details(job_id, {"reason": "early_stopping", "source": "logs", "confidence": "confirmed",
                                   "best_epoch": 318, "patience": 100})
    details = json.loads(db.get_job(job_id)["stop_details"])
    assert details["epochs"] == 500 and details["epoch"] == 418

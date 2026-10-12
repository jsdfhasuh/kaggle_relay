import json
import subprocess
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from test_relay_api import make_settings, seed_job
from app.kaggle_adapter import KaggleAdapter, parse_training_progress_logs, training_progress_key
from app.main import apply_progress_callback, create_app
from app.schemas import JobProgressRequest
from app.worker import finish_kernel_job


def startup(sequence=1, **extra):
    return dict(event_type="startup", backend="yolo", startup_id="parent",
                startup_sequence=sequence, startup_stage="dependency_install",
                startup_status="running", startup_elapsed_seconds=30,
                stage_elapsed_seconds=20, reported_at=100 + sequence,
                heartbeat=True, message="Installing dependencies", **extra)


def line(event):
    return "TRAINING_PLATFORM_PROGRESS: " + json.dumps(event)


def post(db, job_id, event):
    apply_progress_callback(db, db.get_job(job_id), JobProgressRequest(**event))


def test_callback_preserves_training_against_late_startup(tmp_path):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel", progress=60)
    post(db, job_id, startup())
    for progress in ({"epoch": 1, "epochs": 10},
                     {"phase": "feature_extraction", "backend": "patchcore", "remote_progress": 25},
                     {"phase": "environment_prepare", "backend": "patchcore", "remote_progress": 5}):
        post(db, job_id, progress)
        before = db.get_job(job_id)
        post(db, job_id, {**startup(2), "startup_status": "ready"})
        after = db.get_job(job_id)
        for key in ("kernel_status", "progress", "status", "kaggle_output"):
            assert after[key] == before[key]


def test_startup_restart_and_sequence_ordering(tmp_path):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel", progress=60)
    post(db, job_id, startup(2))
    post(db, job_id, startup(1))
    assert json.loads(db.get_job(job_id)["kernel_status"])["startup_sequence"] == 2
    child = {**startup(1), "startup_id": "child", "reported_at": 105}
    post(db, job_id, child)
    post(db, job_id, {**startup(3), "startup_status": "restarting"})
    assert json.loads(db.get_job(job_id)["kernel_status"])["startup_id"] == "child"
    assert db.get_job(job_id)["progress"] == 60


@pytest.mark.parametrize("phase", ["prepare", "input"])
def test_startup_can_coexist_with_input_preparation(tmp_path, phase):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel", progress=60)
    post(db, job_id, {"backend": "patchcore", "phase": phase})
    post(db, job_id, startup(1))
    assert json.loads(db.get_job(job_id)["kernel_status"])["event_type"] == "startup"


@pytest.mark.parametrize("status", ["complete", "failed", "canceled", "downloading_output"])
def test_late_startup_preserves_terminal_or_output_state(tmp_path, status):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, status, progress=85)
    before = db.get_job(job_id)
    post(db, job_id, startup())
    after = db.get_job(job_id)
    for key in ("status", "kernel_status", "progress", "completed_at"):
        assert after[key] == before[key]


def test_concurrent_epoch_wins_over_startup_compare_and_swap(tmp_path, monkeypatch):
    app = create_app(make_settings(tmp_path))
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel", progress=60)
    update = db.update_job_if_status
    injected = []

    def race(*args, **kwargs):
        if not injected:
            injected.append(True)
            post(db, job_id, {"epoch": 2, "epochs": 10})
        return update(*args, **kwargs)

    monkeypatch.setattr(db, "update_job_if_status", race)
    post(db, job_id, startup())
    job = db.get_job(job_id)
    assert json.loads(job["kernel_status"])["epoch"] == 2
    assert job["progress"] == 64


@pytest.mark.parametrize("bad", [{"backend": []}, {"startup_status": []},
    {"startup_sequence": 0}, {"reported_at": float("inf")}, {"message": ""}])
def test_invalid_startup_is_rejected_by_logs_and_callback(tmp_path, bad):
    app = create_app(make_settings(tmp_path))
    job_id = seed_job(app, "waiting_kernel", progress=60)
    event = {**startup(), **bad}
    assert parse_training_progress_logs(line(event)) == []
    with pytest.raises(HTTPException) as exc:
        post(app.state.db, job_id, event)
    assert exc.value.status_code == 422


def test_startup_cannot_fabricate_epoch_or_percent(tmp_path):
    app = create_app(make_settings(tmp_path))
    job_id = seed_job(app, "waiting_kernel", progress=60)
    event = startup(epoch=10, epochs=10, remote_progress=100)
    post(app.state.db, job_id, event)
    job = app.state.db.get_job(job_id)
    assert job["progress"] == 60
    stored = json.loads(job["kernel_status"])
    assert stored["epoch"] is None and stored["remote_progress"] is None
    parsed = parse_training_progress_logs(line(event))[0]
    assert not {"epoch", "epochs", "remote_progress"}.intersection(parsed)


def test_log_fallback_for_startup_and_dino_heartbeats(tmp_path):
    adapter = KaggleAdapter(make_settings(tmp_path), Mock())
    phase = dict(backend="patchcore", phase="environment_prepare", phase_current=0,
                 phase_total=1, overall_progress=5, status="running", heartbeat=True)
    events = [startup(1), startup(2), {**phase, "reported_at": 130}, {**phase, "reported_at": 160}]
    logs = [line(events[0]), line(events[0])] + ["\n".join(map(line, events[:i])) for i in range(2, 5)]
    # Exercise the provider's wrapped log format, without a direct HTTP callback.
    adapter._run = Mock(side_effect=[subprocess.CompletedProcess([], 0,
        stdout=json.dumps([{"data": log}])) for log in logs])
    adapter.kernel_status = Mock(side_effect=["RUNNING"] * 4 + ["COMPLETE"])
    adapter._sleep = Mock()
    observed = []
    assert adapter.wait_kernel("owner/kernel", observed.append) == "COMPLETE"
    assert len(observed) == 4
    assert [e["startup_sequence"] for e in observed[:2]] == [1, 2]
    assert [e["reported_at"] for e in observed[2:]] == [130, 160]
    assert training_progress_key(observed[2]) != training_progress_key(observed[3])


def test_log_fallback_cannot_overwrite_callback_epoch(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    db = app.state.db
    job_id = seed_job(app, "waiting_kernel", progress=60)
    post(db, job_id, {"epoch": 3, "epochs": 10})
    adapter = Mock()
    adapter.last_stop_details = {}

    def wait_kernel(_ref, callback):
        callback(parse_training_progress_logs(line(startup()))[0])
        assert json.loads(db.get_job(job_id)["kernel_status"])["epoch"] == 3
        raise TimeoutError("end isolated monitoring probe")

    adapter.wait_kernel.side_effect = wait_kernel
    with pytest.raises(TimeoutError, match="isolated monitoring probe"):
        finish_kernel_job(settings, db, job_id, adapter)

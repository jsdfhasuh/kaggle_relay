import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.archive import ArchiveError
from app.dino_threshold_guard import _relay_dino_lower_threshold, _RELAY_DINO_RUNTIME
from app.dino_threshold_policy import apply_dino_threshold_policy, POLICY_MESSAGE, POLICY_SHA256


def kernel(tmp_path, source):
    (tmp_path / "kernel-metadata.json").write_text(json.dumps({"code_file": "train.py"}), encoding="utf-8")
    entry = tmp_path / "train.py"
    entry.write_text(source, encoding="utf-8")
    return entry


def test_real_failure_scores_keep_ng_after_margin(capsys):
    threshold = 0.6087480783462524
    target = 0.6087471842765808
    normal, defect = [0.2071728557, 0.2345735282], [threshold, 1.0960804224]
    original = {"method": "f1_validation", "selected_confusion_matrix": {"tn": 2, "tp": 2, "fp": 0, "fn": 0}}
    calibration, lowered = _relay_dino_lower_threshold(lambda *_: (original, threshold), normal, defect, POLICY_SHA256)
    assert target < threshold and target >= lowered
    assert lowered == pytest.approx(0.6086063287305832)
    assert [x >= lowered for x in normal + defect] == [False, False, True, True]
    assert "gateway_threshold_policy" not in original
    policy = calibration["gateway_threshold_policy"]
    assert policy["source_sha256"] == POLICY_SHA256 and policy["base_runtime_sha256"] == _RELAY_DINO_RUNTIME
    assert policy["original_threshold"] == threshold and policy["effective_threshold"] == lowered
    assert "0.608" in capsys.readouterr().err


def test_narrow_gap_does_not_create_a_new_false_positive():
    normal, defect = [0.99999], [1.0]
    _, threshold = _relay_dino_lower_threshold(lambda *_: ({}, 1.0), normal, defect, POLICY_SHA256)
    assert normal[0] < threshold < defect[0]


def test_no_representable_safe_margin_is_not_silently_accepted():
    with pytest.raises(ValueError, match="representable"):
        _relay_dino_lower_threshold(lambda *_: ({}, 1.0), [math.nextafter(1.0, -math.inf)], [1.0], POLICY_SHA256)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_threshold_is_rejected(invalid):
    with pytest.raises(ValueError, match="finite"):
        _relay_dino_lower_threshold(lambda *_: ({}, invalid), [0.1], [1.0], POLICY_SHA256)


def test_script_rewrite_is_idempotent_preserves_inputs_and_runs_isolated_launcher(tmp_path, monkeypatch):
    source = '''"""Submitted training."""
from __future__ import annotations
CONFIG = {"runtime_sha256": "RUNTIME", "request_sha256": "request", "relay_callback_token": "unchanged"}
def start():
    from patchcore_dino_runtime.kaggle_bootstrap import run as train
    return train("dataset", "working", CONFIG)
'''.replace("RUNTIME", _RELAY_DINO_RUNTIME)
    entry = kernel(tmp_path, source)
    payload = tmp_path / "payload.zip"
    payload.write_bytes(b"original frozen payload")
    archive_hash = hashlib.sha256(payload.read_bytes()).hexdigest()
    metadata = (tmp_path / "kernel-metadata.json").read_bytes()
    assert apply_dino_threshold_policy(tmp_path, "patchcore_dinov2_v3") == POLICY_MESSAGE
    first = entry.read_bytes()
    apply_dino_threshold_policy(tmp_path, "patchcore_dinov2_v3")
    assert entry.read_bytes() == first
    assert (tmp_path / "kernel-metadata.json").read_bytes() == metadata
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == archive_hash

    package, bootstrap = ModuleType("patchcore_dino_runtime"), ModuleType("patchcore_dino_runtime.kaggle_bootstrap")
    package.kaggle_bootstrap = bootstrap
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, bootstrap.__name__, bootstrap)
    command = ["cloud-python", "-I", "-X", "utf8", "-m", "patchcore_dino_runtime.remote_training",
               "--dataset-root", "dataset", "--request-sha256", "request", "--output-dir", "working/artifacts"]
    captured = []
    def worker(cmd, **kw):
        captured.append((cmd, kw))
        return "finished"
    def run(root, work, config):
        assert config["relay_callback_token"] == "unchanged"
        return bootstrap.run_worker(command, request={"run_id": "frozen"}, request_sha256="request")
    bootstrap.run_worker, bootstrap.run = worker, run
    namespace = {}
    exec(compile(entry.read_text(encoding="utf-8"), str(entry), "exec"), namespace)
    assert namespace["__doc__"] == "Submitted training."
    assert namespace["start"]() == "finished"
    cmd, kwargs = captured[0]
    assert cmd[:5] == ["cloud-python", "-I", "-X", "utf8", "-c"]
    assert cmd[6:] == command[6:]
    assert kwargs == {"request": {"run_id": "frozen"}, "request_sha256": "request"}
    assert bootstrap.run_worker is worker
    assert "verify_runtime()" in cmd[5]


def test_injected_worker_applies_policy_without_disabling_integrity_or_acceptance(tmp_path):
    # Exercise the actual -I launcher, not only the parent-side AST rewrite.
    from app.dino_threshold_policy import _WORKER
    package = tmp_path / "patchcore_dino_runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "worker.py").write_text(f'def verify_runtime():\n    return {_RELAY_DINO_RUNTIME!r}\n')
    (package / "data.py").write_text('def decode_rgb(*args):\n    return None\n')
    (package / "calibration.py").write_text('''from types import SimpleNamespace
contract = SimpleNamespace(calibration_details=lambda *args: ({"method": "f1_validation"}, 0.6087480783462524),
                           make_threshold_document=lambda threshold, calibration: calibration)
def score_contract():
    return contract
''')
    (package / "remote_training.py").write_text('''import json
from .calibration import score_contract
def validate_payload(*args, **kwargs):
    return {"request": {"training": {"workers": 0}}, "samples": []}
def main():
    validate_payload()
    calibration, threshold = score_contract().calibration_details([0.23], [0.6087480783462524])
    print(json.dumps({"threshold": threshold, "calibration": calibration}))
    return 0
''')
    code = "import sys; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n" + _WORKER
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["threshold"] < 0.6087471842765808
    assert actual["calibration"]["gateway_threshold_policy"]["source_sha256"] == POLICY_SHA256
    (package / "worker.py").write_text('def verify_runtime():\n    raise ValueError("runtime bytes modified")\n')
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "runtime bytes modified" in result.stderr


@pytest.mark.parametrize("contract", ["yolo", "patchcore"])
def test_other_backends_are_unchanged(tmp_path, contract):
    entry = kernel(tmp_path, "print('original')\n")
    assert apply_dino_threshold_policy(tmp_path, contract) is None
    assert entry.read_text() == "print('original')\n"


def test_previous_deployed_overlay_upgrades_once(tmp_path):
    from app.dino_threshold_policy import _LEGACY_POLICY_SHA256
    guard = (Path(__file__).parent / "fixtures/dino_threshold_guard_v1.txt").read_text(encoding="utf-8")
    assert hashlib.sha256(guard.encode()).hexdigest() == _LEGACY_POLICY_SHA256
    header = f"_RELAY_DINO_POLICY_SHA256 = {_LEGACY_POLICY_SHA256!r}\n"
    worker = header + guard + "\nraise SystemExit(_relay_dino_worker_main())\n"
    source = (header + f"_RELAY_DINO_WORKER_SOURCE = {worker!r}\n" + guard
              + '\nfrom patchcore_dino_runtime.kaggle_bootstrap import run\nrun = _relay_dino_wrap_run(run)\n')
    entry = kernel(tmp_path, source)
    apply_dino_threshold_policy(tmp_path, "patchcore_dinov2_v3")
    first = entry.read_bytes()
    apply_dino_threshold_policy(tmp_path, "patchcore_dinov2_v3")
    assert entry.read_bytes() == first
    namespace = {}
    # Inspect only the overlay; the real bootstrap import runs on Kaggle.
    import ast
    tree = ast.parse(first)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert functions.count("_relay_dino_worker_main") == 1
    assert functions.count("_relay_dino_image_scope") == 1
    assert _LEGACY_POLICY_SHA256 not in first.decode()


@pytest.mark.parametrize("source", ["def broken(", "print('unsupported')"])
def test_unsupported_entrypoint_is_not_partially_rewritten(tmp_path, source):
    entry = kernel(tmp_path, source)
    with pytest.raises(ArchiveError):
        apply_dino_threshold_policy(tmp_path, "patchcore_dinov2_v3")
    assert entry.read_text() == source


def test_external_code_file_is_rejected(tmp_path):
    directory = tmp_path / "kernel"
    directory.mkdir()
    outside = kernel(tmp_path, "print('outside')")
    (directory / "kernel-metadata.json").write_text('{"code_file":"../train.py"}')
    with pytest.raises(ArchiveError, match="inside"):
        apply_dino_threshold_policy(directory, "patchcore_dinov2_v3")
    assert outside.read_text() == "print('outside')"


@pytest.mark.parametrize("cached", [False, True])
def test_worker_applies_policy_before_push_without_changing_frozen_upload(tmp_path, monkeypatch, cached):
    from test_relay_api import (make_settings, build_zip, create_job, auth_headers,
                               upload_all, ready_dataset_status)
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.kaggle_adapter import DatasetUploadReceipt
    from app.worker import process_job

    dataset = build_zip({"dataset-metadata.json": b"{}"})
    submitted = b'def main():\n    from patchcore_dino_runtime.kaggle_bootstrap import run\n    return run("data", "work", {})\n'
    kernel_archive = build_zip({"kernel-metadata.json": b'{"code_file":"train.py","dataset_sources":["demo/data"]}',
                                "train.py": submitted})
    identity = dict(dataset_id="data-id", identity_sha256="a" * 64, run_id="run-id", run_identity_sha256="b" * 64)
    settings = make_settings(tmp_path)
    monkeypatch.setattr("app.main.process_job", lambda *a, **kw: None)
    monkeypatch.setattr("app.worker.finish_kernel_job", lambda *a, **kw: None)
    app = create_app(settings)
    calls = []

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def upload_dataset(self, *args, **kwargs):
            calls.append("upload")
            return DatasetUploadReceipt(expected_version_number=7)

        def wait_dataset(self, *args, **kwargs):
            return ready_dataset_status(7)

        def push_kernel(self, directory):
            namespace = {}
            exec(compile((directory / "train.py").read_text(encoding="utf-8"), "train.py", "exec"), namespace)
            assert "_relay_dino_wrap_run" in namespace["main"].__code__.co_names
            assert namespace["_RELAY_DINO_POLICY_SHA256"] == POLICY_SHA256
            assert json.loads((directory / "kernel-metadata.json").read_text())["dataset_sources"] == ["demo/data/7"]
            calls.append("push")
            return "pushed"

    monkeypatch.setattr("app.worker.KaggleAdapter", Adapter)
    with TestClient(app) as client:
        if cached:
            app.state.db.upsert_dataset_cache(dataset_ref="demo/data", payload_hash="dino-policy",
                status="ready", dataset_status=ready_dataset_status(7), source_job_id="previous")
        job_id = create_job(client, dataset, kernel_archive, payload_hash="dino-policy",
                            identity=identity, artifact_contract="patchcore_dinov2_v3")
        if not cached:
            upload_all(client, job_id, "dataset", dataset)
        upload_all(client, job_id, "kernel", kernel_archive)
        assert client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers()).status_code == 200
        process_job(settings, app.state.db, job_id)
        response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()
        assert response["status"] == "pushing_kernel", response.get("error")
        assert {key: response[key] for key in identity} == identity
        assert response["kernel_archive_sha256"] == hashlib.sha256(kernel_archive).hexdigest()
        assert (settings.jobs_dir / job_id / "archives/kernel.zip").read_bytes() == kernel_archive
        assert POLICY_MESSAGE in response["recent_logs"]
    assert calls == (["push"] if cached else ["upload", "push"])

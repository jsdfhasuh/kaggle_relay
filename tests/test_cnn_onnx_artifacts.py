"""Optional CNN deployments never weaken the original training artifact requirements."""

import re
from pathlib import Path
import zipfile

import pytest

from app.cnn_onnx_artifacts import PROFILE, digest, inventory, source_binding, source_from_run, write
from app.config import Settings
from app.kaggle_adapter import KaggleAdapter, KaggleAdapterError, PATCHCORE_ARTIFACT_FILE_PATTERN


def fixture(root):
    root.mkdir()
    (root / "model.ckpt").write_bytes(b"fixture checkpoint")
    write(root / "environment.json", {"packages": {"anomalib": "2.5.1"}})
    write(root / "anomaly_metrics.json", {})
    write(root / "threshold.json", {"threshold": 1.0, "model_sha256": digest(root / "model.ckpt"),
        "score_kind": "patchcore_anomalib_raw_image_score", "decision_rule": "score_gte_threshold"})
    files = [{"path": p.name, "sha256": digest(p), "size": p.stat().st_size} for p in root.iterdir()]
    write(root / "training_artifacts.json", {"implementation": "anomalib", "anomalib_version": "2.5.1",
        "run_id": "run", "patchcore_params": {"backbone": "resnet18", "layers": ["layer2", "layer3"],
            "image_size": 32, "num_neighbors": 1, "pre_trained": False, "coreset_sampling_ratio": .1}, "artifacts": files})
    return source_from_run(root / "model.ckpt")


def deployment(root, source, status):
    name = "package-" + "c" * 32
    folder = root / "cnn_onnx" / name
    folder.mkdir(parents=True)
    for file in ("model.onnx", "model.onnx.data", "predict.py", "preprocess.py"):
        (folder / file).write_bytes(b"fixture")
    write(folder / "threshold.json", {"threshold": 1.0, "decision_rule": "score_gte_threshold"})
    write(folder / "verification.json", {"status": status, "samples": [{key: {"status": "PASS"}
        for key in ("preprocessing", "onnx_parity", "ckpt_compatibility")}]})
    write(folder / "deployment.json", {"schema": "cnn_onnx_v1", "profile": PROFILE, "status": status,
        "source": source_binding(source), "files": inventory(folder)})
    write(folder.parent / "result.json", {"status": status, "package": name})
    return folder


@pytest.mark.parametrize("status", ["PASS", "FAIL", "legacy"])
def test_training_and_optional_export_package(tmp_path, status):
    root = tmp_path / "output"
    source = fixture(root)
    if status != "legacy":
        folder = deployment(root, source, status)
        for file in folder.iterdir():
            assert re.fullmatch(PATCHCORE_ARTIFACT_FILE_PATTERN, file.relative_to(root).as_posix())
    adapter = KaggleAdapter(Settings(api_token="fixture", storage_dir=tmp_path / "data"), lambda _: None)
    target = tmp_path / "result.zip"
    adapter.package_artifacts(root, target, "patchcore", expected_identity={"run_id": "run"})
    with zipfile.ZipFile(target) as archive:
        assert "model.ckpt" in archive.namelist()
        if status != "legacy":
            assert folder.relative_to(root).as_posix() + "/model.onnx.data" in archive.namelist()


@pytest.mark.parametrize("failure", ["missing_weights", "changed_threshold", "identity", "false_success"])
def test_reject_invalid_success(tmp_path, failure):
    root = tmp_path / "output"
    source = fixture(root)
    folder = deployment(root, source, "PASS")
    identity = {"run_id": "run"}
    if failure == "missing_weights":
        (folder / "model.onnx.data").unlink()
    elif failure == "changed_threshold":
        write(folder / "threshold.json", {"threshold": 99})
    elif failure == "identity":
        identity = {"run_id": "wrong"}
    else:
        write(folder.parent / "result.json", {"status": "PASS", "package": ""})
    adapter = KaggleAdapter(Settings(api_token="fixture", storage_dir=tmp_path / "data"), lambda _: None)
    with pytest.raises(KaggleAdapterError, match="CNN ONNX"):
        adapter.package_artifacts(root, tmp_path / "bad.zip", "patchcore", expected_identity=identity)
    assert not (tmp_path / "bad.zip").exists()

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.archive import ArchiveError
from app.gpu_policy import POLICY_MESSAGE, apply_yolo_gpu_policy
from app.yolo_gpu_guard import _relay_yolo_train


def kernel(tmp_path, source, name="train.py", enable_gpu="true"):
    (tmp_path / "kernel-metadata.json").write_text(json.dumps({
        "code_file": name, "enable_gpu": enable_gpu,
        "id": "alice/train", "dataset_sources": ["alice/data/7"],
    }), encoding="utf-8")
    (tmp_path / name).write_text(source, encoding="utf-8")
    return tmp_path / name


def fake_datasets(monkeypatch, counts):
    calls = []
    dataset_module = ModuleType("ultralytics.data.dataset")
    utils_module = ModuleType("ultralytics.data.utils")

    class Dataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.count = counts[kwargs["img_path"]]
            if not self.count:
                raise RuntimeError("No valid images found")

        def __len__(self):
            return self.count

    def resolve(data, autodownload):
        assert data == "data.yaml" and autodownload is False
        return {"train": "train", "val": "val"}

    dataset_module.YOLODataset = Dataset
    utils_module.check_det_dataset = resolve
    monkeypatch.setitem(sys.modules, "ultralytics.data.dataset", dataset_module)
    monkeypatch.setitem(sys.modules, "ultralytics.data.utils", utils_module)
    return calls


@pytest.mark.parametrize("train,val,requested,expected", [
    (9, 1, "0,1", "0"), (2, 1, [0, 1], "0"),
    (1, 27, "1,2", "1"), (220, 27, "0,1", "0,1"),
    (2, 2, [0, 1], [0, 1]), (220, 3, "0,1,2,3", "0"),
])
def test_only_small_splits_limit_gpu(tmp_path, monkeypatch, capsys, train, val, requested, expected):
    calls = fake_datasets(monkeypatch, {"train": train, "val": val})
    source = '''"""Training script."""
from __future__ import annotations
DEVICE = REQUESTED
CALLBACK_TOKEN = "unchanged-callback"
class Model:
    task = "detect"
    def train(self, **kwargs):
        return kwargs
result = Model().train(data="data.yaml", device=DEVICE, epochs=50, fraction=0.5)
'''.replace("REQUESTED", repr(requested))
    entry = kernel(tmp_path, source)
    metadata = (tmp_path / "kernel-metadata.json").read_bytes()
    assert apply_yolo_gpu_policy(tmp_path) == POLICY_MESSAGE
    namespace = {}
    exec(compile(entry.read_text(encoding="utf-8"), str(entry), "exec"), namespace)
    assert namespace["result"] == dict(data="data.yaml", device=expected, epochs=50, fraction=0.5)
    assert namespace["DEVICE"] == requested
    assert namespace["CALLBACK_TOKEN"] == "unchanged-callback"
    assert namespace["__doc__"] == "Training script."
    assert calls[0]["fraction"] == 0.5 and calls[1]["fraction"] == 1.0
    assert f"train={train} val={val}" in capsys.readouterr().out
    assert (tmp_path / "kernel-metadata.json").read_bytes() == metadata
    first = entry.read_bytes()
    apply_yolo_gpu_policy(tmp_path)
    assert entry.read_bytes() == first


@pytest.mark.parametrize("device", ["cpu", "mps", "0", 1, [1], None])
def test_cpu_and_single_gpu_skip_dataset_preflight(device):
    assert _relay_yolo_train(lambda **kw: kw, data="absent.yaml", device=device)["device"] == device


def test_invalid_dataset_fails_before_training(monkeypatch):
    fake_datasets(monkeypatch, {"train": 2, "val": 0})
    def train(**kw):
        pytest.fail("training must not start for invalid data")
    with pytest.raises(RuntimeError, match="No valid"):
        _relay_yolo_train(train, data="data.yaml", device="0,1")


@pytest.mark.parametrize("contract,enabled", [("patchcore", True), ("patchcore_dinov2_v3", True), ("yolo", False)])
def test_unrelated_kernels_unchanged(tmp_path, contract, enabled):
    entry = kernel(tmp_path, "DEVICE = 'auto'\n", enable_gpu=enabled)
    before = entry.read_bytes()
    assert apply_yolo_gpu_policy(tmp_path, contract) is None
    assert entry.read_bytes() == before


def test_notebook_preserves_magics_and_is_idempotent(tmp_path):
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": "code", "id": "install", "source": ["%pip install something\n"], "metadata": {}, "outputs": [], "execution_count": None},
        {"cell_type": "code", "id": "train", "source": ["model.train(data='data.yaml', device='0,1')\n"], "metadata": {}, "outputs": [], "execution_count": None},
    ]}
    entry = kernel(tmp_path, json.dumps(notebook), name="train.ipynb")
    apply_yolo_gpu_policy(tmp_path)
    result = json.loads(entry.read_text())
    assert result["cells"][0]["id"] == "relay-small-dataset-gpu-policy"
    assert result["cells"][1]["source"] == notebook["cells"][0]["source"]
    assert "_relay_yolo_train" in "".join(result["cells"][2]["source"])
    first = entry.read_bytes()
    apply_yolo_gpu_policy(tmp_path)
    assert entry.read_bytes() == first


def test_code_path_outside_kernel_is_rejected(tmp_path):
    directory = tmp_path / "kernel"
    directory.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass")
    (directory / "kernel-metadata.json").write_text(json.dumps({"enable_gpu": True, "code_file": "../outside.py"}))
    with pytest.raises(ArchiveError, match="inside"):
        apply_yolo_gpu_policy(directory)
    assert outside.read_text() == "pass"


@pytest.mark.parametrize("source,name", [
    ("def broken(", "train.py"),
    ('{"cells": [null]}', "train.ipynb"),
])
def test_malformed_entrypoint_is_rejected_without_rewriting(tmp_path, source, name):
    entry = kernel(tmp_path, source, name=name)
    with pytest.raises(ArchiveError):
        apply_yolo_gpu_policy(tmp_path)
    assert entry.read_text(encoding="utf-8") == source

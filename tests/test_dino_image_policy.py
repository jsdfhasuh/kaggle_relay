import ast
import hashlib
import multiprocessing
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from app import dino_image_guard as guard
from app.dino_threshold_policy import IMAGE_POLICY_SHA256, _remove_legacy_prefix
from app.dino_threshold_guard import _RELAY_DINO_RUNTIME


def original_decode(path, expected_sha256=None):
    raw = path.read_bytes()
    assert expected_sha256 is None or hashlib.sha256(raw).hexdigest() == expected_sha256
    array = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None or array.dtype != np.uint8 or (array.ndim == 3 and array.shape[2] != 3):
        raise ValueError("only RGB8 and grayscale8 images are supported")
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sample(tmp_path, channels=4, alpha=255, dtype=np.uint8):
    array = np.arange(60, dtype=dtype).reshape(4, 5, 3)
    if channels == 4:
        array = np.concatenate([array, np.full((4, 5, 1), alpha, dtype=dtype)], axis=2)
    elif channels == 1:
        array = array[:, :, 0]
    path = tmp_path / f"sample-{channels}-{alpha}-{dtype.__name__}.png"
    success, raw = cv2.imencode(".png", array)
    assert success
    path.write_bytes(raw.tobytes())
    return path, hashlib.sha256(raw.tobytes()).hexdigest(), array


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_opaque_decode_preserves_bytes_and_matches_target_rgb(tmp_path, channels):
    path, digest, array = sample(tmp_path, channels)
    converted = set()
    actual = guard._relay_dino_decode_rgb(original_decode, path, digest, converted=converted)
    with Image.open(path) as image:
        assert np.array_equal(actual, np.array(image.convert("RGB")))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert converted == ({digest} if channels == 4 else set())
    if channels == 4:
        assert np.array_equal(actual, array[:, :, :3][:, :, ::-1])


@pytest.mark.parametrize("alpha", [0, 127, 254])
def test_transparent_rejected_with_filename(tmp_path, alpha):
    path, digest, _ = sample(tmp_path, alpha=alpha)
    with pytest.raises(ValueError, match="transparent.*" + path.name):
        guard._relay_dino_decode_rgb(original_decode, path, digest, converted=set())


def test_hash_failure_precedes_conversion(tmp_path):
    path, _, _ = sample(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        guard._relay_dino_decode_rgb(original_decode, path, "0" * 64, converted=set())


def test_16bit_and_corruption_rejected(tmp_path):
    path, digest, _ = sample(tmp_path, channels=1, dtype=np.uint16)
    with pytest.raises(ValueError, match="8-bit.*" + path.name):
        guard._relay_dino_decode_rgb(original_decode, path, digest, converted=set())
    for raw in (b"broken", b""):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match=path.name):
            guard._relay_dino_decode_rgb(original_decode, path, converted=set())


def install_runtime(monkeypatch, inputs):
    package = ModuleType("patchcore_dino_runtime")
    data = ModuleType("patchcore_dino_runtime.data")
    remote = ModuleType("patchcore_dino_runtime.remote_training")
    data.decode_rgb = original_decode
    remote.validate_payload = lambda *a, **kw: inputs
    package.data, package.remote_training = data, remote
    for module in (package, data, remote):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(guard, "_RELAY_DINO_IMAGE_POLICY_SHA256", IMAGE_POLICY_SHA256, raising=False)
    monkeypatch.setattr(guard, "_RELAY_DINO_RUNTIME", _RELAY_DINO_RUNTIME, raising=False)
    contract = SimpleNamespace(make_threshold_document=lambda threshold, calibration: {
        "threshold": threshold, "calibration": calibration})
    return data, remote, contract


def test_preflight_receipt_normal_only_and_restore(tmp_path, monkeypatch):
    path, digest, _ = sample(tmp_path)
    inputs = {"request": {"training": {"workers": 0}}, "samples": [
        {"sample_id": "one", "path": str(path), "sha256": digest}]}
    data, remote, contract = install_runtime(monkeypatch, inputs)
    originals = data.decode_rgb, remote.validate_payload, contract.make_threshold_document
    with guard._relay_dino_image_scope(contract):
        assert remote.validate_payload("original", "request-hash") is inputs
        doc = contract.make_threshold_document(0.5, {"method": "normal_quantile"})
        policy = doc["calibration"]["gateway_image_policy"]
        assert doc["threshold"] == 0.5
        assert policy["converted_sample_count"] == 1
        assert policy["source_sha256"] == IMAGE_POLICY_SHA256
        assert policy["original_bytes_preserved"] is True
        assert len(policy["converted_samples_sha256"]) == 64
    assert (data.decode_rgb, remote.validate_payload, contract.make_threshold_document) == originals


def test_preflight_failure_restores_and_does_not_rewrite(tmp_path, monkeypatch):
    path, digest, _ = sample(tmp_path, alpha=127)
    inputs = {"request": {"training": {"workers": 0}}, "samples": [
        {"sample_id": "one", "path": str(path), "sha256": digest}]}
    data, remote, contract = install_runtime(monkeypatch, inputs)
    original = data.decode_rgb
    with pytest.raises(ValueError, match="transparent"):
        with guard._relay_dino_image_scope(contract):
            remote.validate_payload()
    assert data.decode_rgb is original
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def _child_decode(path, digest, queue):
    from patchcore_dino_runtime.data import decode_rgb
    queue.put(decode_rgb(path, digest).tolist())


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="Kaggle Linux fork test")
def test_fork_workers_inherit_decoder(tmp_path, monkeypatch):
    path, digest, _ = sample(tmp_path)
    inputs = {"request": {"training": {"workers": 2}}, "samples": [
        {"sample_id": "one", "path": str(path), "sha256": digest}]}
    _, remote, contract = install_runtime(monkeypatch, inputs)
    monkeypatch.setattr(guard.multiprocessing, "get_start_method", lambda: "fork")
    ctx = multiprocessing.get_context("fork")
    with guard._relay_dino_image_scope(contract):
        remote.validate_payload()
        queue = ctx.Queue()
        children = [ctx.Process(target=_child_decode, args=(path, digest, queue)) for _ in range(2)]
        try:
            for child in children:
                child.start()
            results = [queue.get(timeout=15) for _ in children]
            assert results[0] == results[1]
            for child in children:
                child.join(timeout=15)
                assert child.exitcode == 0
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join()
            queue.close()


def test_spawn_is_explicitly_rejected(monkeypatch):
    inputs = {"request": {"training": {"workers": 2}}, "samples": []}
    _, remote, contract = install_runtime(monkeypatch, inputs)
    monkeypatch.setattr(guard.multiprocessing, "get_start_method", lambda: "spawn")
    with pytest.raises(RuntimeError, match="requires fork"):
        with guard._relay_dino_image_scope(contract):
            remote.validate_payload()


def test_changed_legacy_overlay_is_rejected():
    from app.dino_threshold_policy import _LEGACY_POLICY_SHA256
    from app.archive import ArchiveError
    header = f"_RELAY_DINO_POLICY_SHA256 = {_LEGACY_POLICY_SHA256!r}\n"
    worker = header + "print('modified')\n\nraise SystemExit(_relay_dino_worker_main())\n"
    tree = ast.parse(header + f"_RELAY_DINO_WORKER_SOURCE = {worker!r}\n")
    with pytest.raises(ArchiveError, match="integrity"):
        _remove_legacy_prefix(tree)

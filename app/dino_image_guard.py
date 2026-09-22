"""Read opaque RGBA inputs without modifying the frozen submission."""

import contextlib
import functools
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys


_RELAY_DINO_IMAGE_POLICY_ID = "dino_opaque_rgba_rgb_v1"


def _relay_dino_decode_rgb(original, path, expected_sha256=None, *, converted):
    import cv2
    import numpy as np

    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError(f"image exceeds encoded byte budget: {path.name}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"frozen image SHA-256 mismatch: {path.name}")
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError(f"image decoding failed: {path.name}") from exc
    if image is None or image.dtype != np.uint8:
        raise ValueError(f"image must decode to 8-bit pixels: {path.name}")
    if image.ndim == 3 and image.shape[2] == 4:
        if not np.all(image[:, :, 3] == 255):
            raise ValueError(f"transparent or semitransparent RGBA is unsupported: {path.name}")
        # Preserve the base decoder's IMREAD_COLOR orientation and RGB order.
        color = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"RGB decoding failed: {path.name}")
        converted.add(digest)
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    try:
        return original(path, expected_sha256)
    except ValueError as exc:
        raise ValueError(f"{exc}: {path.name}") from exc


@contextlib.contextmanager
def _relay_dino_image_scope(contract):
    from patchcore_dino_runtime import data, remote_training

    decode = data.decode_rgb
    validate = remote_training.validate_payload
    make_threshold = contract.make_threshold_document
    converted = set()
    receipt = {
        "id": _RELAY_DINO_IMAGE_POLICY_ID,
        "source_sha256": _RELAY_DINO_IMAGE_POLICY_SHA256,
        "base_runtime_sha256": _RELAY_DINO_RUNTIME,
        "original_bytes_preserved": True,
        "converted_sample_count": 0,
    }

    def validate_payload(*args, **kwargs):
        inputs = validate(*args, **kwargs)
        workers = inputs["request"]["training"].get("workers", 0)
        if workers and multiprocessing.get_start_method() != "fork":
            raise RuntimeError("DINO gateway image policy requires fork for DataLoader workers")
        converted.clear()
        for sample in inputs["samples"]:
            data.decode_rgb(sample["path"], sample["sha256"])
        records = sorted((sample["sample_id"], sample["sha256"]) for sample in inputs["samples"]
                         if sample["sha256"] in converted)
        receipt["converted_sample_count"] = len(records)
        receipt["converted_samples_sha256"] = hashlib.sha256(
            json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        print(f"Gateway DINO image policy {_RELAY_DINO_IMAGE_POLICY_ID}: "
              f"{len(records)} opaque RGBA samples read as RGB; original bytes preserved",
              file=sys.stderr, flush=True)
        return inputs

    def make_threshold_document(threshold, calibration=None, *args, **kwargs):
        calibration = dict(calibration or {})
        calibration["gateway_image_policy"] = dict(receipt)
        return make_threshold(threshold, calibration, *args, **kwargs)

    data.decode_rgb = functools.partial(_relay_dino_decode_rgb, decode, converted=converted)
    remote_training.validate_payload = validate_payload
    contract.make_threshold_document = make_threshold_document
    try:
        yield
    finally:
        data.decode_rgb = decode
        remote_training.validate_payload = validate
        contract.make_threshold_document = make_threshold

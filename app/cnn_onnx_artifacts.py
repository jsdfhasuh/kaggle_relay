"""CNN ONNX delivery validation. No training or inference imports."""

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import uuid


PROFILE = "cnn_fp32_features_fp64_full_l2_cpu_basic_v1"
PARAMETERS = ("backbone", "layers", "coreset_sampling_ratio", "num_neighbors", "pre_trained", "image_size")
IDENTITY = ("dataset_id", "identity_sha256", "run_id", "run_identity_sha256")


def plain(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() and (item.is_symlink() or getattr(item.stat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("unsafe_path")
    return path


def digest(path):
    value = hashlib.sha256()
    with plain(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(plain(path).read_text(encoding="utf-8"))


def write(path, value):
    path = plain(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_binding(source):
    return {"hashes": source["hashes"], "params": source["params"],
            "identity": source["identity"], "threshold": source["model_threshold"]}


def recheck_source(source):
    for key, path in source["paths"].items():
        if digest(path) != source["hashes"][key]:
            raise ValueError("source_changed")


def source_from_run(model_path, manifest_path=None):
    """Inspect the already committed training files without loading pickle."""
    model = plain(model_path)
    manifest_path = plain(manifest_path or model.parent / "training_artifacts.json")
    manifest = read(manifest_path)
    root = manifest_path.parent
    paths = {"model": model, "bundle_model": root / "model.ckpt", "manifest": manifest_path,
             "threshold": root / "threshold.json", "environment": root / "environment.json"}
    hashes = {key: digest(path) for key, path in paths.items()}
    threshold, environment = read(paths["threshold"]), read(paths["environment"])
    params = manifest.get("patchcore_params", {})
    if (manifest.get("implementation") != "anomalib" or manifest.get("anomalib_version") != "2.5.1"
            or environment.get("packages", {}).get("anomalib") != "2.5.1"
            or hashes["model"] != hashes["bundle_model"]
            or threshold.get("model_sha256") != hashes["model"]
            or threshold.get("score_kind") != "patchcore_anomalib_raw_image_score"
            or threshold.get("decision_rule") != "score_gte_threshold"
            or not all(key in params for key in PARAMETERS)):
        raise ValueError("identity_mismatch")
    entries = {item["path"]: item for item in manifest["artifacts"]}
    for name in ("model.ckpt", "threshold.json", "environment.json"):
        item = entries[name]
        if item["sha256"] != digest(root / name) or item["size"] != (root / name).stat().st_size:
            raise ValueError("source_changed")
    value = threshold.get("threshold")
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("uncalibrated")
    return {"paths": {key: str(path) for key, path in paths.items()}, "hashes": hashes,
            "params": params, "model_threshold": value, "environment": environment,
            "identity": {key: manifest[key] for key in IDENTITY if manifest.get(key)}}


def inventory(root):
    root = plain(root)
    return [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "deployment.json"]


def validate_package(root, *, source=None, require_pass=False):
    root = plain(root)
    document = read(root / "deployment.json")
    if (document.get("schema") != "cnn_onnx_v1" or document.get("profile") != PROFILE
            or document.get("status") not in {"PASS", "PENDING", "FAIL"}
            or (require_pass and document["status"] != "PASS")):
        raise ValueError("onnx_not_verified")
    if source is not None and document.get("source") != source_binding(source):
        raise ValueError("onnx_source_mismatch")
    files = document.get("files")
    if not isinstance(files, list) or not files or len(files) > 32:
        raise ValueError("onnx_inventory_invalid")
    names = set()
    for row in files:
        name = row.get("path", "")
        # All deployment files are flat, including the single external-data file.
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", "..", "deployment.json"} or name.casefold() in names:
            raise ValueError("onnx_unsafe_file")
        names.add(name.casefold())
        path = plain(root / name)
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError("onnx_content_changed")
    if not {"model.onnx", "predict.py", "preprocess.py", "verification.json", "threshold.json"}.issubset(names):
        raise ValueError("onnx_inventory_missing")
    if inventory(root) != files:
        raise ValueError("onnx_inventory_mismatch")
    report = read(root / "verification.json")
    threshold = read(root / "threshold.json")
    if (report.get("status") != document["status"] or threshold.get("threshold") != document["source"]["threshold"]
            or threshold.get("decision_rule") != "score_gte_threshold"):
        raise ValueError("onnx_report_mismatch")
    if document["status"] == "PASS" and (not report.get("samples") or any(
            row.get(key, {}).get("status") != "PASS" for row in report["samples"]
            for key in ("preprocessing", "onnx_parity", "ckpt_compatibility"))):
        raise ValueError("onnx_false_success")
    return document


def copy_package(root, target, *, source=None, cancel=None, require_pass=True):
    if source is not None:
        recheck_source(source)
    document = validate_package(root, source=source, require_pass=require_pass)
    target = plain(target)
    if target.exists():
        raise ValueError("target_conflict")
    stage = target.with_name("." + target.name + "." + uuid.uuid4().hex)
    try:
        stage.mkdir()
        for row in [*document["files"], {"path": "deployment.json"}]:
            if cancel:
                cancel.check()
            shutil.copyfile(plain(Path(root) / row["path"]), stage / row["path"])
        validate_package(stage, source=source, require_pass=require_pass)
        if source is not None:
            recheck_source(source)
        if cancel:
            cancel.publish(stage, target)
        else:
            if target.exists():
                raise ValueError("target_conflict")
            stage.rename(target)
        return document
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def receive_result(download_dir, run_dir, *, source):
    """Receive the optional, separately validated deployment without changing CKPT inventory."""
    origin = plain(Path(download_dir) / "cnn_onnx")
    if not (origin / "result.json").is_file():
        return {}
    record = read(origin / "result.json")
    if record.get("status") not in {"PASS", "FAIL", "PENDING", "CANCELLED"}:
        raise ValueError("onnx_status_invalid")
    name = record.get("package", "")
    if name and not re.fullmatch(r"package-[a-f0-9]{32}", name):
        raise ValueError("onnx_package_path_invalid")
    if record["status"] == "PASS" and not name:
        raise ValueError("onnx_package_missing")
    destination = plain(Path(run_dir) / "cnn_onnx")
    destination.mkdir(exist_ok=True)
    target = None
    if name:
        document = validate_package(origin / name, source=source)
        if document["status"] != record["status"]:
            raise ValueError("onnx_status_mismatch")
        target = destination / name
        if target.exists():
            if validate_package(target, source=source) != document:
                raise ValueError("onnx_existing_content_changed")
        else:
            copy_package(origin / name, target, source=source, require_pass=False)
    write(destination / "result.json", record)
    return {"cnn_onnx_status": record["status"], "cnn_onnx_error": record.get("error", ""),
            "cnn_onnx_dir": str(target) if target else "", "cnn_onnx_location": record.get("location", "kaggle")}

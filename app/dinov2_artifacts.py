"""Bounded DINO v3 transport validation. Never import or deserialize model code."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from app.archive import ArchiveError

ARTIFACT_CONTRACT = "patchcore_dinov2_v3"
ARTIFACT_FORMAT = "vision_workshop_dinov2_artifacts_v3"
MANIFEST_NAME = "training_artifacts.json"
ARTIFACT_SUBDIR = "artifacts"
IDENTITY_FIELDS = ("dataset_id", "identity_sha256", "run_id", "run_identity_sha256")
REQUIRED_FILES = frozenset({
    "model.ckpt", "threshold.json", "metrics.json", "runtime_result.json",
    "resolved_spec.json", "runtime.json", "environment.json", "anomaly_metrics.json",
    "com_dinov2_small.pt", "native_candidate.json", "pt_export.json",
    "pt_verification.json", "source_artifacts.json",
})
DOWNLOAD_PATTERN = r"^artifacts[/\\](?:" + "|".join(re.escape(name) for name in sorted(REQUIRED_FILES | {MANIFEST_NAME})) + r")$"


def _plain_file(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ArchiveError("DINO required regular root file missing or unsafe: " + name)
    return path


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError("DINO manifest has duplicate JSON keys")
        result[key] = value
    return result


def read_transport_manifest(output_dir, expected_identity):
    root = Path(output_dir).absolute()
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ArchiveError("unsafe DINO output directory")
    with _plain_file(root, MANIFEST_NAME).open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ArchiveError("DINO manifest exceeds 1 MiB")
    try:
        document = json.loads(raw, object_pairs_hook=_object_pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ArchiveError("invalid DINO transport manifest") from exc
    fixed = {"artifact_format": ARTIFACT_FORMAT, "schema_version": 3,
             "backend": "patchcore", "task": "anomaly", "implementation": "anomalib",
             "anomalib_version": "2.2.0", "training_status": "completed", "export_status": "PASS",
             "pt_model_path": "com_dinov2_small.pt", "pt_export_path": "pt_export.json",
             "verification_path": "pt_verification.json",
             "source_artifact_manifest_path": "source_artifacts.json"}
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or any(document.get(key) != value for key, value in fixed.items())):
        raise ArchiveError("DINO manifest format/status mismatch")
    if not isinstance(expected_identity, dict):
        raise ArchiveError("DINO transport requires frozen job identity")
    for key in IDENTITY_FIELDS:
        value = expected_identity.get(key)
        if not isinstance(value, str) or not value.strip() or document.get(key) != value:
            raise ArchiveError("DINO frozen job identity mismatch: " + key)
    rows = document.get("artifacts")
    if not isinstance(rows, list) or len(rows) != len(REQUIRED_FILES):
        raise ArchiveError("DINO artifact inventory incomplete")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}
                or not isinstance(row["path"], str) or row["path"] not in REQUIRED_FILES
                or type(row["size"]) is not int or row["size"] < 1
                or not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ArchiveError("invalid DINO artifact inventory entry")
    if {row["path"] for row in rows} != REQUIRED_FILES or rows != sorted(rows, key=lambda row: row["path"]):
        raise ArchiveError("DINO artifact inventory duplicate or unsorted")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    if digest != document.get("artifact_set_sha256"):
        raise ArchiveError("DINO artifact inventory hash mismatch")
    for row in rows:
        if _plain_file(root, row["path"]).stat().st_size != row["size"]:
            raise ArchiveError("DINO artifact size mismatch: " + row["path"])
    return root, raw, rows


def package_artifacts(output_dir, artifact_zip, *, expected_identity, storage_budget=None):
    root, manifest, rows = read_transport_manifest(output_dir, expected_identity)
    artifact_zip = Path(artifact_zip)
    artifact_zip.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".dino-artifacts-", suffix=".zip", dir=artifact_zip.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for row in rows:
                if storage_budget:
                    storage_budget.check_free(row["size"])
                digest, size = hashlib.sha256(), 0
                # Hash exactly the bytes written into the ZIP, not an earlier read of the file.
                with _plain_file(root, row["path"]).open("rb") as source:
                    with archive.open(ARTIFACT_SUBDIR + "/" + row["path"], "w", force_zip64=True) as target:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            size += len(block)
                            if size > row["size"]:
                                raise ArchiveError("DINO artifact grew during packaging: " + row["path"])
                            digest.update(block)
                            target.write(block)
                if size != row["size"] or digest.hexdigest() != row["sha256"]:
                    raise ArchiveError("DINO artifact hash/size mismatch: " + row["path"])
            if storage_budget:
                storage_budget.check_free(len(manifest))
            archive.writestr(ARTIFACT_SUBDIR + "/" + MANIFEST_NAME, manifest)
        os.replace(temporary, artifact_zip)
    finally:
        Path(temporary).unlink(missing_ok=True)

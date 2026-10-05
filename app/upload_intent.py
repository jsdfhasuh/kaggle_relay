"""Durable per-submission candidates, independent of disposable staging."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from app.payload_contract import file_digest


def intent_path(storage_dir, dataset_dir, dataset_ref):
    scope = json.dumps([dataset_ref, str(Path(dataset_dir).absolute())], separators=(",", ":"))
    return Path(storage_dir) / "dataset_upload_intents" / (hashlib.sha256(scope.encode()).hexdigest() + ".json")


def content_digest(dataset_dir):
    root = Path(dataset_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("upload_intent_source_missing_or_unsafe")
    paths = [root / "payload.zip"] if (root / "payload.zip").is_file() else sorted(root.rglob("*"))
    values = []
    for path in paths:
        if path.is_symlink():
            raise ValueError("upload_intent_unsafe_input")
        if path.is_file() and path.name != "dataset-metadata.json":
            with path.open("rb") as stream:
                values.append([path.relative_to(root).as_posix(), path.stat().st_size, file_digest(stream)])
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def read_intent(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("upload_intent_invalid")
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or type(value.get("version_number")) is not int or value["version_number"] <= 0
            or value.get("state") not in {"unknown", "accepted", "rejected"}):
        raise ValueError("upload_intent_invalid")
    return value


def write_intent(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".intent-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)

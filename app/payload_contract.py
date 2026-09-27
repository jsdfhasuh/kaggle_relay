"""Byte verification for Kaggle's preserved and expanded payload layouts.

Kept identical in the Relay repository; no Kaggle SDK or application imports.
"""
import hashlib
import io
from pathlib import Path
import stat
import zipfile


RUNTIME_PATH = "model_source/kaggle_runtime/runtime.zip"
MAX_RUNTIME_BYTES = 4 * 1024 * 1024


def file_digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def zip_members(archive):
    files = {}
    seen = set()
    for info in archive.infolist():
        name = info.filename.rstrip("/")
        parts = name.split("/")
        mode = info.external_attr >> 16
        if (not name or info.filename != info.orig_filename or "\\" in name or ":" in name
                or any(p in ("", ".", "..") for p in parts) or info.flag_bits & 1
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
                or (not info.is_dir() and stat.S_IFMT(mode) == stat.S_IFDIR)
                or name.casefold() in seen):
            raise ValueError("payload_unsafe_member")
        seen.add(name.casefold())
        if not info.is_dir():
            files[name] = info
    for name in files:
        if any("/".join(name.split("/")[:i]) in files for i in range(1, len(name.split("/")))):
            raise ValueError("payload_path_conflict")
    return files


def _archive_inventory(archive, members):
    result = {}
    for name, info in members.items():
        with archive.open(info) as stream:
            result[name] = (info.file_size, file_digest(stream))
    return result


def verify_payload_archive(payload_path, downloaded_path):
    """Accept only the original payload or its byte-identical expanded contents."""
    payload_path = Path(payload_path)
    with zipfile.ZipFile(downloaded_path) as remote:
        actual = zip_members(remote)
        if "payload.zip" in actual:
            if set(actual) != {"payload.zip"} or actual["payload.zip"].file_size != payload_path.stat().st_size:
                raise ValueError("payload_inventory_mismatch")
            with remote.open(actual["payload.zip"]) as stream, payload_path.open("rb") as original:
                if file_digest(stream) != file_digest(original):
                    raise ValueError("payload_digest_mismatch")
            return
        with zipfile.ZipFile(payload_path) as original:
            members = zip_members(original)
            expected = _archive_inventory(original, members)
            for runtime_path, max_files in ((RUNTIME_PATH, 32), ("model_source/p6_runtime/runtime.zip", 128)):
                if runtime_path in members and runtime_path not in actual:
                    runtime = members[runtime_path]
                    if runtime.file_size > MAX_RUNTIME_BYTES:
                        raise ValueError("runtime_expansion_limit")
                    with zipfile.ZipFile(io.BytesIO(original.read(runtime))) as package:
                        sources = zip_members(package)
                        if len(sources) > max_files or sum(i.file_size for i in sources.values()) > MAX_RUNTIME_BYTES:
                            raise ValueError("runtime_expansion_limit")
                        expanded = _archive_inventory(package, sources)
                    del expected[runtime_path]
                    prefix = runtime_path[:-4] + "/"
                    for name, value in expanded.items():
                        target = prefix + name
                        if target in expected:
                            raise ValueError("payload_path_conflict")
                        expected[target] = value
        _verify_inventory(remote, actual, expected)


def _verify_inventory(remote, actual, expected):
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:3]
        extra = sorted(set(actual) - set(expected))[:3]
        raise ValueError(f"payload_inventory_mismatch: missing={missing}, unexpected={extra}")
    for name, (size, digest) in expected.items():
        if actual[name].file_size != size:
            raise ValueError("payload_size_mismatch")
        with remote.open(actual[name]) as stream:
            if file_digest(stream) != digest:
                raise ValueError("payload_digest_mismatch")


def verify_upload_archive(dataset_dir, downloaded_path):
    """Legacy loose uploads retain exact names and bytes as well."""
    root = Path(dataset_dir)
    if (root / "payload.zip").is_file():
        return verify_payload_archive(root / "payload.zip", downloaded_path)
    expected = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("payload_unsafe_member")
        if path.is_file() and path.name != "dataset-metadata.json":
            with path.open("rb") as stream:
                expected[path.relative_to(root).as_posix()] = (path.stat().st_size, file_digest(stream))
    with zipfile.ZipFile(downloaded_path) as remote:
        _verify_inventory(remote, zip_members(remote), expected)

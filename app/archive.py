import hashlib
import math
import os
import shutil
import stat
import zipfile
from pathlib import Path


class ArchiveError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_chunk_count(total_size: int, chunk_size: int) -> int:
    if total_size < 0 or chunk_size <= 0:
        raise ArchiveError("invalid archive size or chunk size")
    return max(1, math.ceil(total_size / chunk_size))


def validate_chunk_index(index: int, total_size: int, chunk_size: int) -> None:
    count = expected_chunk_count(total_size, chunk_size)
    if index < 0 or index >= count:
        raise ArchiveError(f"chunk index {index} out of range 0..{count - 1}")


def assemble_archive(
    chunks_dir: Path,
    archive_path: Path,
    total_size: int,
    chunk_size: int,
    expected_sha256: str,
) -> str:
    count = expected_chunk_count(total_size, chunk_size)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as out:
        for index in range(count):
            chunk_path = chunks_dir / f"{index}.part"
            if not chunk_path.exists():
                raise ArchiveError(f"missing chunk {index}")
            with chunk_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    actual_size = archive_path.stat().st_size
    if actual_size != total_size:
        raise ArchiveError(f"archive size mismatch: expected {total_size}, got {actual_size}")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ArchiveError("archive sha256 mismatch")
    return actual_sha256


def _safe_member_target(dest_dir: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise ArchiveError(f"unsafe zip path: {member_name}")
    if normalized in {"", ".", ".."}:
        raise ArchiveError(f"unsafe zip path: {member_name}")
    target = (dest_dir / normalized).resolve()
    dest_resolved = dest_dir.resolve()
    try:
        target.relative_to(dest_resolved)
    except ValueError as exc:
        raise ArchiveError(f"unsafe zip path: {member_name}") from exc
    return target


def safe_extract_zip(zip_path: Path, dest_dir: Path, expected_max_size: int) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_uncompressed = 0
    max_uncompressed = max(
        expected_max_size * 20,
        expected_max_size + 1024 * 1024 * 1024,
    )
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            mode = (info.external_attr >> 16) & 0o777777
            if stat.S_ISLNK(mode):
                raise ArchiveError(f"zip symlink is not allowed: {info.filename}")
            target = _safe_member_target(dest_dir, info.filename)
            total_uncompressed += int(info.file_size)
            if total_uncompressed > max_uncompressed:
                raise ArchiveError("zip uncompressed size exceeds relay safety limit")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)


def require_file(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise ArchiveError(f"required file missing: {relative_path}")
    return path


def zip_directory(source_dir: Path, zip_path: Path, exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    source_dir = Path(source_dir)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            rel = path.relative_to(source_dir)
            if any(part in exclude_names for part in rel.parts):
                continue
            if path.is_file():
                archive.write(path, rel.as_posix())

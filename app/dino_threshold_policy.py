"""Add a versioned DINO calibration policy to extracted submission code."""

import ast
import hashlib
import json
from pathlib import Path

from app.archive import ArchiveError


_IMAGE_GUARD = Path(__file__).with_name("dino_image_guard.py").read_text(encoding="utf-8")
IMAGE_POLICY_SHA256 = hashlib.sha256(_IMAGE_GUARD.encode("utf-8")).hexdigest()
_GUARD = (f"_RELAY_DINO_IMAGE_POLICY_SHA256 = {IMAGE_POLICY_SHA256!r}\n" + _IMAGE_GUARD + "\n"
          + Path(__file__).with_name("dino_threshold_guard.py").read_text(encoding="utf-8"))
POLICY_SHA256 = hashlib.sha256(_GUARD.encode("utf-8")).hexdigest()
_WORKER = f"_RELAY_DINO_POLICY_SHA256 = {POLICY_SHA256!r}\n" + _GUARD + "\nraise SystemExit(_relay_dino_worker_main())\n"
_PREFIX = (f"_RELAY_DINO_POLICY_SHA256 = {POLICY_SHA256!r}\n"
           f"_RELAY_DINO_WORKER_SOURCE = {_WORKER!r}\n" + _GUARD)
_PREFIX_DUMPS = {ast.dump(node) for node in ast.parse(_PREFIX).body}
_LEGACY_POLICY_SHA256 = "832a35b6f46b1eb5c55ab5fcaa9fd72d5fcd5878846453532c85a262afbb287d"
POLICY_MESSAGE = ("Gateway DINO F1 threshold policy: lower by twice the native score tolerance, "
                  "bounded by the calibration score gap; record policy in threshold/metrics/PT receipts. "
                  "Opaque RGBA image policy: verify original bytes and read fully opaque inputs as RGB. "
                  "policy_sha256=" + POLICY_SHA256 + " image_policy_sha256=" + IMAGE_POLICY_SHA256)


def _remove_legacy_prefix(tree):
    # Authenticate the exact previous overlay before removing it on an upgrade retry.
    for index, node in enumerate(tree.body):
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
                or node.targets[0].id != "_RELAY_DINO_WORKER_SOURCE"
                or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str)):
            continue
        worker = node.value.value
        header = f"_RELAY_DINO_POLICY_SHA256 = {_LEGACY_POLICY_SHA256!r}\n"
        footer = "\nraise SystemExit(_relay_dino_worker_main())\n"
        if not worker.startswith(header) or not worker.endswith(footer):
            continue
        guard = worker[len(header):-len(footer)]
        if hashlib.sha256(guard.encode("utf-8")).hexdigest() != _LEGACY_POLICY_SHA256:
            raise ArchiveError("previous DINO gateway overlay integrity mismatch")
        prefix = ast.parse(header + f"_RELAY_DINO_WORKER_SOURCE = {worker!r}\n" + guard).body
        start = index - 1
        if start < 0 or [ast.dump(n) for n in tree.body[start:start + len(prefix)]] != [ast.dump(n) for n in prefix]:
            raise ArchiveError("previous DINO gateway overlay is incomplete or modified")
        del tree.body[start:start + len(prefix)]
        break


class _WrapBootstrap(ast.NodeTransformer):
    def __init__(self):
        self.matches = 0

    def visit_ImportFrom(self, node):
        if node.level or node.module != "patchcore_dino_runtime.kaggle_bootstrap":
            return node
        additions = []
        for alias in node.names:
            if alias.name == "run":
                self.matches += 1
                name = alias.asname or alias.name
                additions.extend(ast.parse(f"{name} = _relay_dino_wrap_run({name})").body)
        return [node, *additions]

    def visit_Assign(self, node):
        # Remove our existing wrapper before reapplying on a submission retry.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_relay_dino_wrap_run" and len(node.value.args) == 1
                and not node.value.keywords and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id == node.targets[0].id):
            return None
        return node


def apply_dino_threshold_policy(kernel_dir: Path, artifact_contract: str) -> str | None:
    """Preserve uploaded archives and payload hashes; annotate the execution overlay."""
    if artifact_contract != "patchcore_dinov2_v3":
        return None
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    entry = kernel_dir / str(metadata.get("code_file") or "train.py")
    if (not entry.resolve().is_relative_to(kernel_dir.resolve()) or entry.is_symlink()
            or not entry.is_file() or entry.suffix.lower() != ".py"):
        raise ArchiveError("DINO threshold policy requires a Python code_file inside the kernel directory")
    try:
        tree = ast.parse(entry.read_text(encoding="utf-8-sig"))
        _remove_legacy_prefix(tree)
        tree.body = [node for node in tree.body if ast.dump(node) not in _PREFIX_DUMPS]
        wrapper = _WrapBootstrap()
        tree = wrapper.visit(tree)
        if not wrapper.matches:
            raise ArchiveError("DINO threshold policy requires the supported bound-wheel bootstrap")
        insertion = 0
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str)):
            insertion = 1
        while (insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom)
               and tree.body[insertion].module == "__future__"):
            insertion += 1
        tree.body[insertion:insertion] = ast.parse(_PREFIX).body
        ast.fix_missing_locations(tree)
        updated = ast.unparse(tree) + "\n"
        compile(updated, str(entry), "exec")
    except (SyntaxError, UnicodeError, ValueError, TypeError) as exc:
        raise ArchiveError("DINO threshold policy could not prepare the submission") from exc
    entry.write_text(updated, encoding="utf-8")
    return POLICY_MESSAGE

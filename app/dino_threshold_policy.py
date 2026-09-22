"""Add a versioned DINO calibration policy to extracted submission code."""

import ast
import hashlib
import json
from pathlib import Path

from app.archive import ArchiveError


_GUARD = Path(__file__).with_name("dino_threshold_guard.py").read_text(encoding="utf-8")
POLICY_SHA256 = hashlib.sha256(_GUARD.encode("utf-8")).hexdigest()
_WORKER = f"_RELAY_DINO_POLICY_SHA256 = {POLICY_SHA256!r}\n" + _GUARD + "\nraise SystemExit(_relay_dino_worker_main())\n"
_PREFIX = (f"_RELAY_DINO_POLICY_SHA256 = {POLICY_SHA256!r}\n"
           f"_RELAY_DINO_WORKER_SOURCE = {_WORKER!r}\n" + _GUARD)
_PREFIX_DUMPS = {ast.dump(node) for node in ast.parse(_PREFIX).body}
POLICY_MESSAGE = ("Gateway DINO F1 threshold policy: lower by twice the native score tolerance, "
                  "bounded by the calibration score gap; record policy in threshold/metrics/PT receipts. "
                  "policy_sha256=" + POLICY_SHA256)


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

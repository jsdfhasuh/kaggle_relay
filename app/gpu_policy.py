"""Install a small-dataset GPU guard in submitted YOLO kernels."""
import ast
import json
from pathlib import Path

from app.archive import ArchiveError


POLICY_MESSAGE = (
    "Gateway GPU policy: check valid train/val sample counts before YOLO training; "
    "use one GPU only when a split has fewer samples than requested GPUs."
)
_DEVICE_HELPER = "_relay_yolo_train"
_PREFIX = Path(__file__).with_name("yolo_gpu_guard.py").read_text(encoding="utf-8")
_PREFIX_NODES = ast.parse(_PREFIX).body
_PREFIX_DUMPS = {ast.dump(node) for node in _PREFIX_NODES}


class _DevicePolicy(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        # Leave nn.Module.train() mode changes alone.
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "train"
                and any(keyword.arg == "data" for keyword in node.keywords)):
            return ast.copy_location(ast.Call(
                func=ast.Name(id=_DEVICE_HELPER, ctx=ast.Load()),
                args=[node.func, *node.args], keywords=node.keywords,
            ), node)
        return node


def _rewrite_python(source: str, *, prefix: bool) -> str:
    tree = ast.parse(source)
    if prefix:
        tree.body = [node for node in tree.body if ast.dump(node) not in _PREFIX_DUMPS]
    tree = _DevicePolicy().visit(tree)
    if prefix:
        insertion = 0
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            if isinstance(tree.body[0].value.value, str):
                insertion = 1
        while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom) and tree.body[insertion].module == "__future__":
            insertion += 1
        tree.body[insertion:insertion] = ast.parse(_PREFIX).body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def apply_yolo_gpu_policy(kernel_dir: Path, artifact_contract: str = "yolo") -> str | None:
    """Change only extracted submission code; original upload ZIPs stay intact."""
    if artifact_contract != "yolo":
        return None
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    if str(metadata.get("enable_gpu", False)).strip().lower() not in {"true", "1"}:
        return None
    entry = kernel_dir / str(metadata.get("code_file") or "train.py")
    if not entry.resolve().is_relative_to(kernel_dir.resolve()) or entry.is_symlink() or not entry.is_file():
        raise ArchiveError("GPU policy requires a code_file inside the kernel directory")
    try:
        if entry.suffix.lower() == ".py":
            updated = _rewrite_python(entry.read_text(encoding="utf-8-sig"), prefix=True)
        elif entry.suffix.lower() == ".ipynb":
            notebook = json.loads(entry.read_text(encoding="utf-8"))
            cells = notebook["cells"]
            if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
                raise ArchiveError("GPU policy requires a valid notebook cells list")
            # Replace our own policy cell when retrying the same submission.
            cells[:] = [cell for cell in cells if cell.get("id") != "relay-small-dataset-gpu-policy"]
            for cell in cells:
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    source = source if isinstance(source, str) else "".join(source)
                    try:
                        cell["source"] = _rewrite_python(source, prefix=False).splitlines(keepends=True)
                    except SyntaxError:
                        # Preserve magic cells. Generated training calls are in
                        # ordinary Python cells.
                        pass
            cells.insert(0, dict(cell_type="code", id="relay-small-dataset-gpu-policy", metadata={},
                                 execution_count=None, outputs=[], source=_PREFIX.splitlines(keepends=True)))
            updated = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
        else:
            raise ArchiveError("GPU YOLO jobs require a Python script or notebook for the dataset GPU policy")
    except (SyntaxError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise ArchiveError("GPU policy could not prepare the YOLO entrypoint") from exc
    entry.write_text(updated, encoding="utf-8")
    return POLICY_MESSAGE

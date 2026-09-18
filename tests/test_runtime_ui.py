import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser presentation logic tests")
def test_runtime_ui_behavior():
    result = subprocess.run(
        [shutil.which("node"), "--test", str(Path(__file__).with_suffix(".cjs"))],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from app.archive import require_file
from app.config import Settings
from app.security import redact_secrets


ARTIFACT_FILE_PATTERN = (
    r".*(artifacts[/\\].*|best\.(pt|onnx)|training_artifacts\.json|"
    r"results\.(csv|png)|args\.yaml|confusion_matrix.*\.png|"
    r"PR_curve\.png|F1_curve\.png|P_curve\.png|R_curve\.png)$"
)
TRAINING_PROGRESS_PREFIX = "TRAINING_PLATFORM_PROGRESS"


class KaggleAdapterError(RuntimeError):
    pass


def parse_training_progress_logs(output: str) -> list[dict]:
    events = []
    for line in (output or "").splitlines():
        marker_index = line.find(TRAINING_PROGRESS_PREFIX)
        if marker_index < 0:
            continue
        payload_text = line[marker_index + len(TRAINING_PROGRESS_PREFIX):].strip()
        if payload_text.startswith(":"):
            payload_text = payload_text[1:].strip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            epoch = int(payload["epoch"])
            epochs = int(payload["epochs"])
        except (KeyError, TypeError, ValueError):
            continue
        payload["epoch"] = epoch
        payload["epochs"] = epochs
        payload["remote_progress"] = round(min(100.0, max(0.0, epoch / epochs * 100)), 2)
        events.append(payload)
    return events


class KaggleAdapter:
    def __init__(self, settings: Settings, log: Callable[[str], None]):
        self.settings = settings
        self.log = log

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        token = env.get("KAGGLE_API_TOKEN", "").strip()
        if token:
            env["KAGGLE_API_TOKEN"] = token
        return env

    def _run(
        self,
        args: list[str],
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        cmd = [self.settings.kaggle_cmd] + args
        self.log("[CMD] " + redact_secrets(" ".join(cmd)))
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        output = redact_secrets(result.stdout or "")
        if output:
            self.log(output[-4000:])
        if check and result.returncode != 0:
            raise KaggleAdapterError(f"Kaggle command failed: {result.returncode}\n{output}")
        result.stdout = output
        return result

    def account(self) -> dict:
        version = self._run(["--version"], check=False).stdout.strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not username:
            kaggle_json = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
            if kaggle_json.exists():
                try:
                    data = json.loads(kaggle_json.read_text(encoding="utf-8"))
                    username = str(data.get("username", "")).strip()
                except (OSError, json.JSONDecodeError):
                    username = ""
        auth_result = self._run(["datasets", "list", "--mine", "-p", "1"], check=False)
        return {
            "version": version,
            "username": username,
            "authenticated": auth_result.returncode == 0,
            "auth_output": auth_result.stdout[-2000:],
        }

    def dataset_exists(self, dataset_ref: str) -> bool:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            result = api.dataset_status(dataset_ref)
            text = str(result).lower()
            return "not found" not in text and "404" not in text
        except Exception as exc:
            detail = str(exc).lower()
            if "not found" in detail or "404" in detail:
                return False
            return False

    def upload_dataset(self, dataset_dir: Path, dataset_ref: str, update_message: str) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        exists = self.dataset_exists(dataset_ref)
        if exists:
            self.log(f"Updating dataset {dataset_ref}")
            api.dataset_create_version(
                str(dataset_dir),
                update_message,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="tar",
            )
        else:
            self.log(f"Creating dataset {dataset_ref}")
            api.dataset_create_new(
                str(dataset_dir),
                public=False,
                quiet=False,
                convert_to_csv=False,
                dir_mode="tar",
            )

    def wait_dataset(self, dataset_ref: str) -> str:
        start = time.time()
        while True:
            result = self._run(["datasets", "status", dataset_ref], check=False)
            output = result.stdout.strip() or f"returncode={result.returncode}"
            status_text = output.lower()
            if result.returncode == 0 and status_text in {"ready", "complete", "ok"}:
                return output
            if result.returncode == 0 and any(word in status_text for word in ["failed", "error", "deleted"]):
                raise KaggleAdapterError(f"Dataset failed:\n{output}")
            if result.returncode != 0 and any(
                word in status_text
                for word in ["401", "403", "404", "forbidden", "unauthorized", "not found"]
            ):
                raise KaggleAdapterError(f"Dataset status failed:\n{output}")
            if time.time() - start > 30 * 60:
                raise TimeoutError(f"Dataset wait timed out:\n{output}")
            time.sleep(self.settings.dataset_poll_seconds)

    def push_kernel(self, kernel_dir: Path) -> str:
        return self._run(["kernels", "push", "-p", str(kernel_dir)]).stdout

    def wait_kernel(self, kernel_ref: str, progress_callback: Callable[[dict], None]) -> str:
        start = time.time()
        seen_progress = set()
        while True:
            result = self._run(["kernels", "status", kernel_ref], check=False)
            output = result.stdout.strip() or f"returncode={result.returncode}"
            logs = self._run(["kernels", "logs", kernel_ref], check=False).stdout
            for progress in parse_training_progress_logs(logs):
                key = (progress.get("epoch"), progress.get("epochs"))
                if key in seen_progress:
                    continue
                seen_progress.add(key)
                progress_callback(progress)
            status_text = output.lower()
            if result.returncode != 0:
                raise KaggleAdapterError(f"Kernel status failed:\n{output}")
            if any(word in status_text for word in ["complete", "succeeded", "success"]):
                return output
            if any(word in status_text for word in ["error", "failed", "failure", "cancel"]):
                raise KaggleAdapterError(f"Kernel failed:\n{output}")
            if time.time() - start > self.settings.kernel_max_wait_seconds:
                raise TimeoutError(f"Kernel wait timed out:\n{output}")
            time.sleep(self.settings.kernel_poll_seconds)

    def download_output(self, kernel_ref: str, output_dir: Path) -> str:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._run(
            [
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(output_dir),
                "--force",
                "--file-pattern",
                ARTIFACT_FILE_PATTERN,
            ],
            check=False,
        ).stdout

    def package_artifacts(self, output_dir: Path, artifact_zip: Path) -> None:
        require_file(output_dir, "best.pt")
        artifact_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir).as_posix())

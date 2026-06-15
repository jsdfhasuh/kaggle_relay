import json
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

from app.archive import require_file
from app.auth_config import KAGGLE_ENV_KEYS, KaggleCredentials
from app.config import Settings
from app.security import redact_secrets


ARTIFACT_FILE_PATTERN = (
    r".*(artifacts[/\\].*|best\.(pt|onnx)|training_artifacts\.json|"
    r"results\.(csv|png)|args\.yaml|confusion_matrix.*\.png|"
    r"PR_curve\.png|F1_curve\.png|P_curve\.png|R_curve\.png)$"
)
TRAINING_PROGRESS_PREFIX = "TRAINING_PLATFORM_PROGRESS"
_ENV_LOCK = threading.RLock()


class KaggleAdapterError(RuntimeError):
    pass


def parse_kaggle_duration(value: str) -> timedelta:
    text = str(value or "").strip()
    if text.endswith("s"):
        text = text[:-1]
    seconds_raw, _, nanos_raw = text.partition(".")
    seconds = int(seconds_raw or "0")
    nanos_text = re.sub(r"\D", "", nanos_raw)
    nanos = int((nanos_text + "0" * 9)[:9]) if nanos_text else 0
    return timedelta(seconds=seconds, microseconds=nanos // 1000)


def patch_kaggle_duration_parser() -> None:
    from kagglesdk.kaggle_object import TimeDeltaSerializer

    TimeDeltaSerializer._from_dict_value = staticmethod(parse_kaggle_duration)


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
    def __init__(
        self,
        settings: Settings,
        log: Callable[[str], None],
        credentials: KaggleCredentials | None = None,
    ):
        self.settings = settings
        self.log = log
        self.credentials = credentials

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        if self.credentials:
            self.credentials.apply_to_env(env)
        else:
            token = env.get("KAGGLE_API_TOKEN", "").strip()
            if token:
                env["KAGGLE_API_TOKEN"] = token
        return env

    @contextmanager
    def _temporary_kaggle_env(self):
        if not self.credentials:
            yield
            return
        with _ENV_LOCK:
            previous = {name: os.environ.get(name) for name in KAGGLE_ENV_KEYS}
            try:
                env = dict(os.environ)
                self.credentials.apply_to_env(env)
                for name in KAGGLE_ENV_KEYS:
                    os.environ.pop(name, None)
                for name in KAGGLE_ENV_KEYS:
                    if name in env:
                        os.environ[name] = env[name]
                yield
            finally:
                for name in KAGGLE_ENV_KEYS:
                    os.environ.pop(name, None)
                    if previous[name] is not None:
                        os.environ[name] = previous[name] or ""

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
        env = self._env()
        username = env.get("KAGGLE_USERNAME", "").strip()
        if not username:
            kaggle_json = Path(env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
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

    def quota(self) -> dict:
        patch_kaggle_duration_parser()
        try:
            with self._temporary_kaggle_env():
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
                response = api.quota_view()
        except SystemExit as exc:
            raise KaggleAdapterError(f"Kaggle quota authentication failed: {exc}") from exc

        accelerators = []
        for resource, quota in (("GPU", response.gpu_quota), ("TPU", response.tpu_quota)):
            if quota is None:
                continue
            used_hours = quota.time_used.total_seconds() / 3600
            total_hours = quota.total_time_allowed.total_seconds() / 3600
            accelerators.append(
                {
                    "resource": resource,
                    "used_hours": round(used_hours, 4),
                    "remaining_hours": round(max(0.0, total_hours - used_hours), 4),
                    "total_hours": round(total_hours, 4),
                }
            )

        return {
            "available": True,
            "refresh_at": response.quota_refresh_time.isoformat() if response.quota_refresh_time else "",
            "accelerators": accelerators,
            "error": "",
        }

    def probe_username_write_access(self) -> dict:
        env = self._env()
        username = env.get("KAGGLE_USERNAME", "").strip()
        if not username:
            return {
                "ok": False,
                "username": "",
                "dataset_ref": "",
                "created": False,
                "cleanup_ok": False,
                "cleanup_error": "",
                "error": "kaggle username is required for write probe",
            }

        slug = f"relay-probe-{uuid.uuid4().hex[:12]}"
        dataset_ref = f"{username}/{slug}"
        created = False
        cleanup_ok = False
        cleanup_error = ""

        try:
            with tempfile.TemporaryDirectory(prefix="kaggle-relay-probe-") as temp_dir:
                probe_dir = Path(temp_dir)
                (probe_dir / "probe.txt").write_text("kaggle relay credential probe\n", encoding="utf-8")
                (probe_dir / "dataset-metadata.json").write_text(
                    json.dumps(
                        {
                            "id": dataset_ref,
                            "title": f"Relay Probe {slug[-8:]}",
                            "licenses": [{"name": "CC0-1.0"}],
                            "resources": [
                                {
                                    "path": "probe.txt",
                                    "description": "Kaggle Relay credential probe",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                with self._temporary_kaggle_env():
                    from kaggle.api.kaggle_api_extended import KaggleApi

                    api = KaggleApi()
                    api.authenticate()
                    response = api.dataset_create_new(
                        str(probe_dir),
                        public=False,
                        quiet=True,
                        convert_to_csv=False,
                        dir_mode="skip",
                    )
                    error = str(getattr(response, "error", "") or "").strip()
                    if error:
                        return {
                            "ok": False,
                            "username": username,
                            "dataset_ref": dataset_ref,
                            "created": False,
                            "cleanup_ok": False,
                            "cleanup_error": "",
                            "error": redact_secrets(error),
                        }
                    created = True
                    try:
                        result = self._run(
                            ["datasets", "delete", dataset_ref, "-y"],
                            check=False,
                        )
                        if result.returncode != 0:
                            raise KaggleAdapterError(result.stdout.strip() or f"returncode={result.returncode}")
                        cleanup_ok = True
                    except Exception as exc:
                        cleanup_error = redact_secrets(str(exc))[-2000:]
        except SystemExit as exc:
            return {
                "ok": False,
                "username": username,
                "dataset_ref": dataset_ref,
                "created": created,
                "cleanup_ok": cleanup_ok,
                "cleanup_error": cleanup_error,
                "error": f"Kaggle probe authentication failed: {exc}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "username": username,
                "dataset_ref": dataset_ref,
                "created": created,
                "cleanup_ok": cleanup_ok,
                "cleanup_error": cleanup_error,
                "error": redact_secrets(str(exc))[-2000:],
            }

        return {
            "ok": created,
            "username": username,
            "dataset_ref": dataset_ref,
            "created": created,
            "cleanup_ok": cleanup_ok,
            "cleanup_error": cleanup_error,
            "error": "" if cleanup_ok else "probe dataset was created but cleanup failed",
        }

    def dataset_exists(self, dataset_ref: str) -> bool:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            with self._temporary_kaggle_env():
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

        with self._temporary_kaggle_env():
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

    def wait_dataset(self, dataset_ref: str, permission_grace_seconds: int = 0) -> str:
        start = time.time()
        visibility_retry_logged = False
        visibility_grace = max(0, int(permission_grace_seconds or 0))
        while True:
            result = self._run(["datasets", "status", dataset_ref], check=False)
            output = result.stdout.strip() or f"returncode={result.returncode}"
            status_text = output.lower()
            if result.returncode == 0 and status_text in {"ready", "complete", "ok"}:
                return output
            if result.returncode == 0 and any(word in status_text for word in ["failed", "error", "deleted"]):
                raise KaggleAdapterError(f"Dataset failed:\n{output}")
            elapsed = time.time() - start
            if result.returncode != 0 and any(word in status_text for word in ["401", "unauthorized"]):
                raise KaggleAdapterError(f"Dataset status failed:\n{output}")
            if result.returncode != 0 and any(
                word in status_text for word in ["403", "404", "forbidden", "not found"]
            ):
                if visibility_grace > 0 and elapsed <= visibility_grace:
                    if not visibility_retry_logged:
                        self.log(
                            "Dataset status is temporarily unavailable after upload; "
                            f"retrying for up to {visibility_grace} seconds"
                        )
                        visibility_retry_logged = True
                    time.sleep(self.settings.dataset_poll_seconds)
                    continue
                raise KaggleAdapterError(f"Dataset status failed:\n{output}")
            if elapsed > 30 * 60:
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

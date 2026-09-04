import json
import math
import os
import re
import signal
import shutil
import subprocess
import threading
import time
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

from app.archive import require_file
from app.auth_config import KAGGLE_ENV_KEYS, KaggleCredentials
from app.config import Settings
from app.security import redact_secrets


YOLO_ARTIFACT_FILE_PATTERN = (
    r".*(artifacts[/\\].*|best\.(pt|onnx)|training_artifacts\.json|"
    r"results\.(csv|png)|args\.yaml|confusion_matrix.*\.png|"
    r"PR_curve\.png|F1_curve\.png|P_curve\.png|R_curve\.png)$"
)
PATCHCORE_ARTIFACT_FILE_PATTERN = (
    r"^(model\.ckpt|threshold\.json|"
    r"anomaly_metrics\.json|environment\.json|heatmap_sample\.png|"
    r"overlay_sample\.png|training_artifacts\.json)$"
)
ARTIFACT_FILE_PATTERNS = {
    "yolo": YOLO_ARTIFACT_FILE_PATTERN,
    "patchcore": PATCHCORE_ARTIFACT_FILE_PATTERN,
}
REQUIRED_ARTIFACT_FILES = {
    "yolo": ("best.pt",),
    "patchcore": (
        "model.ckpt",
        "threshold.json",
        "anomaly_metrics.json",
        "environment.json",
        "training_artifacts.json",
    ),
}
TRAINING_PROGRESS_PREFIX = "TRAINING_PLATFORM_PROGRESS"
READY_KAGGLE_STATUSES = {"ready", "complete", "ok"}
_ENV_LOCK = threading.RLock()


class KaggleAdapterError(RuntimeError):
    pass


class KaggleAdapterInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetUploadReceipt:
    expected_version_number: int | None = None
    expected_files: tuple[tuple[str, int], ...] = ()


def kaggle_status_lines(output: str) -> list[str]:
    return [line.strip().lower() for line in (output or "").splitlines() if line.strip()]


def parse_kaggle_dataset_status(output: str) -> tuple[str, int | None]:
    text = str(output or "").strip()
    json_start = text.find("{")
    json_end = text.rfind("}")
    if 0 <= json_start < json_end:
        try:
            payload = json.loads(text[json_start:json_end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").strip().lower()
            version_raw = payload.get(
                "current_version_number",
                payload.get("currentVersionNumber"),
            )
            try:
                version_number = int(version_raw) if version_raw is not None else None
            except (TypeError, ValueError):
                version_number = None
            return status, version_number

    lines = kaggle_status_lines(text)
    known_status = next(
        (
            line
            for line in reversed(lines)
            if line in READY_KAGGLE_STATUSES | {"failed", "error", "deleted"}
        ),
        "",
    )
    return known_status or (lines[-1] if lines else ""), None


def is_ready_kaggle_status(output: str) -> bool:
    status, _version_number = parse_kaggle_dataset_status(output)
    return status in READY_KAGGLE_STATUSES


def dataset_upload_inventory(dataset_dir: Path) -> tuple[tuple[str, int], ...]:
    root = Path(dataset_dir)
    payload_zip = root / "payload.zip"
    inventory: dict[str, int] = {}

    if payload_zip.is_file():
        try:
            with zipfile.ZipFile(payload_zip) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename.replace("\\", "/").lstrip("/")
                    if name:
                        inventory[name] = int(info.file_size)
        except (OSError, zipfile.BadZipFile) as exc:
            raise KaggleAdapterError(f"invalid dataset payload.zip: {exc}") from exc
    else:
        for path in root.rglob("*"):
            if not path.is_file() or path.name == "dataset-metadata.json":
                continue
            inventory[path.relative_to(root).as_posix()] = path.stat().st_size

    return tuple(sorted(inventory.items()))


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


def _wrapped_kaggle_log_lines(output: str) -> list[str]:
    raw_output = str(output or "")
    raw_lines = raw_output.splitlines()
    extracted = []

    def append_data(value) -> bool:
        found = False
        if isinstance(value, list):
            for item in value:
                found = append_data(item) or found
        elif isinstance(value, dict) and isinstance(value.get("data"), str):
            extracted.extend(value["data"].splitlines())
            found = True
        return found

    try:
        decoded = json.loads(raw_output)
    except (TypeError, json.JSONDecodeError):
        decoded = None
    if append_data(decoded):
        return raw_lines + extracted

    for line in raw_lines:
        candidate = line.strip()
        if candidate.startswith("["):
            candidate = candidate[1:].lstrip()
        if candidate.startswith(","):
            candidate = candidate[1:].lstrip()
        if candidate.endswith("]"):
            candidate = candidate[:-1].rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1].rstrip()
        if not candidate:
            continue
        try:
            decoded_line = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        append_data(decoded_line)
    return raw_lines + extracted


def parse_training_progress_logs(output: str) -> list[dict]:
    events = []
    for line in _wrapped_kaggle_log_lines(output):
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
        if payload.get("phase"):
            if (
                payload.get("backend") != "patchcore"
                or not isinstance(payload.get("phase"), str)
                or not payload["phase"].strip()
            ):
                continue
            try:
                overall_progress = float(payload["overall_progress"])
                phase_current = int(payload["phase_current"])
                phase_total = int(payload["phase_total"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(overall_progress)
                or phase_current < 0
                or phase_total < 0
                or (phase_total > 0 and phase_current > phase_total)
                or not isinstance(payload.get("status", ""), str)
                or not isinstance(payload.get("error_code", ""), str)
            ):
                continue
            payload["phase"] = payload["phase"].strip()
            payload["phase_current"] = phase_current
            payload["phase_total"] = phase_total
            payload["remote_progress"] = round(
                min(100.0, max(0.0, overall_progress)),
                2,
            )
            events.append(payload)
            continue
        try:
            epoch = int(payload["epoch"])
            epochs = int(payload["epochs"])
        except (KeyError, TypeError, ValueError):
            continue
        if epoch <= 0 or epochs <= 0:
            continue
        payload["epoch"] = epoch
        payload["epochs"] = epochs
        payload["remote_progress"] = round(min(100.0, max(0.0, epoch / epochs * 100)), 2)
        events.append(payload)
    return events


def training_progress_key(progress: dict) -> tuple:
    if progress.get("phase"):
        return (
            "patchcore",
            progress.get("phase"),
            progress.get("phase_current"),
            progress.get("phase_total"),
            progress.get("status"),
            progress.get("error_code"),
        )
    return ("epoch", progress.get("epoch"), progress.get("epochs"))


class KaggleAdapter:
    def __init__(
        self,
        settings: Settings,
        log: Callable[[str], None],
        credentials: KaggleCredentials | None = None,
        shutdown_event: threading.Event | None = None,
    ):
        self.settings = settings
        self.log = log
        self.credentials = credentials
        self.shutdown_event = shutdown_event

    def _check_interrupted(self) -> None:
        if self.shutdown_event and self.shutdown_event.is_set():
            raise KaggleAdapterInterrupted("relay shutdown interrupted Kaggle polling")

    def _sleep(self, seconds: int | float) -> None:
        if self.shutdown_event:
            if self.shutdown_event.wait(max(0, seconds)):
                self._check_interrupted()
            return
        time.sleep(seconds)

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
        self._check_interrupted()
        cmd = [self.settings.kaggle_cmd] + args
        self.log("[CMD] " + redact_secrets(" ".join(cmd)))
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name == "posix",
        )
        try:
            while True:
                try:
                    stdout, _stderr = process.communicate(
                        timeout=0.2 if self.shutdown_event else None,
                    )
                    break
                except subprocess.TimeoutExpired:
                    self._check_interrupted()
        except BaseException:
            stdout = self._stop_process(process)
            output = redact_secrets(stdout)
            if output:
                self.log(output[-4000:])
            raise

        output = redact_secrets(stdout or "")
        if output:
            self.log(output[-4000:])
        if check and process.returncode != 0:
            raise KaggleAdapterError(f"Kaggle command failed: {process.returncode}\n{output}")
        return subprocess.CompletedProcess(cmd, process.returncode, stdout=output)

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> str:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except OSError:
                pass
        try:
            stdout, _stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            stdout, _stderr = process.communicate()
        return stdout or ""

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

    @staticmethod
    def _dataset_version_number(api, dataset_ref: str) -> int:
        try:
            output = api.dataset_status(dataset_ref, format="json")
        except Exception as exc:
            raise KaggleAdapterError(
                f"Unable to read current Dataset version for {dataset_ref}: {exc}"
            ) from exc
        _status, version_number = parse_kaggle_dataset_status(str(output))
        if version_number is None or version_number <= 0:
            raise KaggleAdapterError(
                f"Kaggle did not return a valid current Dataset version for {dataset_ref}"
            )
        return version_number

    def _dataset_file_inventory(
        self,
        dataset_ref: str,
        version_number: int | None = None,
    ) -> dict[str, int]:
        from kaggle.api.kaggle_api_extended import KaggleApi

        target_ref = (
            f"{dataset_ref}/{version_number}"
            if version_number is not None
            else dataset_ref
        )
        inventory: dict[str, int] = {}
        page_token = None
        with self._temporary_kaggle_env():
            api = KaggleApi()
            api.authenticate()
            while True:
                response = api.dataset_list_files(
                    target_ref,
                    page_token=page_token,
                    page_size=200,
                )
                if isinstance(response, tuple):
                    files = response[0] or []
                    page_token = response[1] if len(response) > 1 else None
                else:
                    error = str(
                        getattr(response, "error_message", "")
                        or getattr(response, "errorMessage", "")
                        or ""
                    ).strip()
                    if error:
                        raise KaggleAdapterError(
                            f"Dataset file listing failed: {error}"
                        )
                    files = (
                        getattr(response, "dataset_files", None)
                        or getattr(response, "datasetFiles", None)
                        or []
                    )
                    page_token = (
                        getattr(response, "next_page_token", None)
                        or getattr(response, "nextPageToken", None)
                    )

                for item in files:
                    name = str(getattr(item, "name", "") or "").replace("\\", "/")
                    if not name:
                        continue
                    size = getattr(item, "total_bytes", None)
                    if size is None:
                        size = getattr(item, "totalBytes", None)
                    if size is None:
                        size = getattr(item, "size", None)
                    try:
                        inventory[name] = int(size)
                    except (TypeError, ValueError):
                        continue

                if not page_token:
                    return inventory

    def upload_dataset(
        self,
        dataset_dir: Path,
        dataset_ref: str,
        update_message: str,
    ) -> DatasetUploadReceipt:
        self._check_interrupted()
        from kaggle.api.kaggle_api_extended import KaggleApi

        expected_files = dataset_upload_inventory(dataset_dir)
        with self._temporary_kaggle_env():
            api = KaggleApi()
            api.authenticate()
            exists = self.dataset_exists(dataset_ref)
            if exists:
                current_version_number = self._dataset_version_number(api, dataset_ref)
                expected_version_number = current_version_number + 1
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
                expected_version_number = 1
                self.log(f"Creating dataset {dataset_ref}")
                api.dataset_create_new(
                    str(dataset_dir),
                    public=False,
                    quiet=False,
                    convert_to_csv=False,
                    dir_mode="tar",
                )
        return DatasetUploadReceipt(
            expected_version_number=expected_version_number,
            expected_files=expected_files,
        )

    def wait_dataset(
        self,
        dataset_ref: str,
        permission_grace_seconds: int = 0,
        upload_receipt: DatasetUploadReceipt | None = None,
    ) -> str:
        start = time.time()
        visibility_retry_logged = False
        visibility_grace = max(0, int(permission_grace_seconds or 0))
        expected_version_number = getattr(
            upload_receipt,
            "expected_version_number",
            None,
        )
        expected_files = dict(getattr(upload_receipt, "expected_files", ()) or ())
        if upload_receipt is not None and expected_version_number is None:
            raise KaggleAdapterError(
                "Dataset upload receipt is missing the expected version number"
            )
        while True:
            status_args = ["datasets", "status", dataset_ref]
            if expected_version_number is not None:
                status_args.extend(["--format", "json"])
            result = self._run(status_args, check=False)
            output = result.stdout.strip() or f"returncode={result.returncode}"
            status_text = output.lower()
            status, current_version_number = parse_kaggle_dataset_status(output)
            elapsed = time.time() - start
            if result.returncode == 0 and status in READY_KAGGLE_STATUSES:
                if (
                    expected_version_number is not None
                    and current_version_number is not None
                    and current_version_number > expected_version_number
                ):
                    raise KaggleAdapterError(
                        f"Dataset {dataset_ref} advanced past uploaded version "
                        f"{expected_version_number} to {current_version_number}"
                    )
                version_visible = (
                    expected_version_number is None
                    or (
                        current_version_number is not None
                        and current_version_number == expected_version_number
                    )
                )
                missing_files = []
                size_mismatches = []
                inventory_error = ""
                if version_visible and expected_files:
                    try:
                        remote_files = self._dataset_file_inventory(
                            dataset_ref,
                            version_number=expected_version_number,
                        )
                    except Exception as exc:
                        inventory_error = redact_secrets(str(exc))
                        detail = inventory_error.lower()
                        if any(word in detail for word in ["401", "unauthorized"]):
                            raise KaggleAdapterError(
                                f"Dataset file visibility check failed: {inventory_error}"
                            ) from exc
                        if any(
                            word in detail
                            for word in ["403", "404", "forbidden", "not found"]
                        ) and (visibility_grace <= 0 or elapsed > visibility_grace):
                            raise KaggleAdapterError(
                                f"Dataset file visibility check failed: {inventory_error}"
                            ) from exc
                    else:
                        missing_files = sorted(set(expected_files) - set(remote_files))
                        size_mismatches = sorted(
                            name
                            for name, expected_size in expected_files.items()
                            if name in remote_files
                            and remote_files[name] != expected_size
                        )

                files_visible = (
                    not expected_files
                    or (
                        not inventory_error
                        and not missing_files
                        and not size_mismatches
                    )
                )
                if version_visible and files_visible:
                    return output
                if not visibility_retry_logged:
                    details = []
                    if not version_visible:
                        details.append(
                            "expected version "
                            f"{expected_version_number}, current version "
                            f"{current_version_number or 'unknown'}"
                        )
                    if inventory_error:
                        details.append(f"file listing unavailable: {inventory_error}")
                    if missing_files:
                        details.append(
                            f"{len(missing_files)} expected files missing "
                            f"(for example {missing_files[0]})"
                        )
                    if size_mismatches:
                        details.append(
                            f"{len(size_mismatches)} expected file sizes differ "
                            f"(for example {size_mismatches[0]})"
                        )
                    self.log(
                        "Dataset reports ready before the uploaded version is fully visible; "
                        + "; ".join(details)
                    )
                    visibility_retry_logged = True
            if result.returncode == 0 and any(word in status_text for word in ["failed", "error", "deleted"]):
                raise KaggleAdapterError(f"Dataset failed:\n{output}")
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
                    self._sleep(self.settings.dataset_poll_seconds)
                    continue
                raise KaggleAdapterError(f"Dataset status failed:\n{output}")
            if elapsed > 30 * 60:
                raise TimeoutError(
                    "Dataset wait timed out before the uploaded version became visible:\n"
                    f"{output}"
                )
            self._sleep(self.settings.dataset_poll_seconds)

    def dataset_status(self, dataset_ref: str) -> str:
        result = self._run(["datasets", "status", dataset_ref], check=False)
        output = result.stdout.strip() or f"returncode={result.returncode}"
        if result.returncode != 0:
            raise KaggleAdapterError(f"Dataset status failed:\n{output}")
        return output

    def push_kernel(self, kernel_dir: Path) -> str:
        return self._run(["kernels", "push", "-p", str(kernel_dir)]).stdout

    def kernel_status(self, kernel_ref: str) -> str:
        result = self._run(["kernels", "status", kernel_ref], check=False)
        output = result.stdout.strip() or f"returncode={result.returncode}"
        if result.returncode != 0:
            raise KaggleAdapterError(f"Kernel status failed:\n{output}")
        return output

    def wait_kernel(self, kernel_ref: str, progress_callback: Callable[[dict], None]) -> str:
        start = time.time()
        last_progress_key = None
        while True:
            output = self.kernel_status(kernel_ref)
            logs = self._run(["kernels", "logs", kernel_ref], check=False).stdout
            progress_events = parse_training_progress_logs(logs)
            if progress_events:
                progress = progress_events[-1]
                key = training_progress_key(progress)
                if key != last_progress_key:
                    last_progress_key = key
                    progress_callback(progress)
            status_text = output.lower()
            if any(word in status_text for word in ["complete", "succeeded", "success"]):
                return output
            if any(word in status_text for word in ["error", "failed", "failure", "cancel"]):
                raise KaggleAdapterError(f"Kernel failed:\n{output}")
            if time.time() - start > self.settings.kernel_max_wait_seconds:
                raise TimeoutError(f"Kernel wait timed out:\n{output}")
            self._sleep(self.settings.kernel_poll_seconds)

    def download_output(
        self,
        kernel_ref: str,
        output_dir: Path,
        artifact_contract: str = "yolo",
    ) -> str:
        if artifact_contract not in ARTIFACT_FILE_PATTERNS:
            raise KaggleAdapterError(
                f"unknown artifact contract: {artifact_contract}"
            )
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
                ARTIFACT_FILE_PATTERNS[artifact_contract],
            ],
            check=False,
        ).stdout

    def package_artifacts(
        self,
        output_dir: Path,
        artifact_zip: Path,
        artifact_contract: str = "yolo",
    ) -> None:
        required_files = REQUIRED_ARTIFACT_FILES.get(artifact_contract)
        if required_files is None:
            raise KaggleAdapterError(
                f"unknown artifact contract: {artifact_contract}"
            )
        for relative_path in required_files:
            require_file(output_dir, relative_path)
        artifact_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir).as_posix())

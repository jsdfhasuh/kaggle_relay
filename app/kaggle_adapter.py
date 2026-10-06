import json
import random
import inspect
import math
import os
import re
import signal
import shutil
import subprocess
import sys
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
from app.dinov2_artifacts import (ARTIFACT_CONTRACT as DINO_CONTRACT, ARTIFACT_SUBDIR as DINO_SUBDIR,
                                DOWNLOAD_PATTERN as DINO_PATTERN)
from app.dinov2_artifacts import package_artifacts as package_dinov2_artifacts
from app.dinov2_251_artifacts import CONTRACT as DINO251_CONTRACT, CONTRACT_V2 as DINO251_V2, CONTRACTS as DINO251_CONTRACTS, DOWNLOAD_PATTERN as DINO251_PATTERN, package_result
from app.auth_config import KAGGLE_ENV_KEYS, KaggleCredentials
from app.config import Settings
from app.security import redact_secrets, register_secret
from app.stop_reason import classify_stop_reason, provider_state
from app.payload_contract import verify_upload_archive
from app.dataset_file_verification import archive_url_missing, verify_version_files
from app.dataset_verification_process import DatasetVerificationError, run_verification, verification_error
from app.upload_intent import content_digest, intent_path, read_intent, write_intent


YOLO_ARTIFACT_FILE_PATTERN = (
    r".*(artifacts[/\\].*|best\.(pt|onnx)|training_artifacts\.json|"
    r"results\.(csv|png)|args\.yaml|confusion_matrix.*\.png|"
    r"PR_curve\.png|F1_curve\.png|P_curve\.png|R_curve\.png)$"
)
PATCHCORE_ARTIFACT_FILE_PATTERN = (
    r"^(model\.ckpt|deployment_model\.pt|pt_export\.json|threshold\.json|"
    r"anomaly_metrics\.json|environment\.json|heatmap_sample\.png|"
    r"overlay_sample\.png|training_artifacts\.json|cnn_onnx/result\.json|"
    r"cnn_onnx/package-[a-f0-9]{32}/(?:model\.onnx(?:\.data)?|deployment\.json|"
    r"verification\.json|threshold\.json|predict\.py|preprocess\.py|README\.txt))$"
)
ARTIFACT_FILE_PATTERNS = {
    DINO251_CONTRACT: DINO251_PATTERN,
    DINO251_V2: DINO251_PATTERN,
    DINO_CONTRACT: DINO_PATTERN,
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


class KernelStatusUnavailable(KaggleAdapterError):
    """The submitted Kernel has no authoritative terminal observation yet."""


class KernelStatusQueryError(KernelStatusUnavailable):
    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def transient_kernel_query_output(output: str) -> bool:
    text = output.lower()
    if any(word in text for word in ("unauthorized", "forbidden", "not found", "sslerror",
                                     "certificate_verify_failed", "permission denied")):
        return False
    if re.search(r"\b(?:401|403|404)\b", text):
        return False
    return any(word in text for word in (
        "connecttimeouterror", "readtimeouterror", "connectionerror", "newconnectionerror",
        "connectionreseterror", "remotedisconnected", "nameresolutionerror",
        "connection to api.kaggle.com timed out", "temporary failure in name resolution",
    )) or bool(re.search(r"\b(?:408|429|500|502|503|504)\s+(?:client|server)\s+error\b", text))


class DatasetUploadUnknown(KaggleAdapterError):
    """The durable candidate must be reconciled, never uploaded again."""


@dataclass(frozen=True)
class DatasetUploadReceipt:
    expected_version_number: int | None = None
    expected_files: tuple[tuple[str, int], ...] = ()
    dataset_dir: str = ""
    content_sha256: str = ""


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
        self._sdk_in_process = False
        self.dataset_cancel_check: Callable[[], None] | None = None
        self.verification_phase = lambda phase: None
        self._publication_remaining = None
        self._verification_retry_remaining = None

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
        cmd = [self.settings.kaggle_cmd] + args
        self.log("[CMD] " + redact_secrets(" ".join(cmd)))
        transfer = len(args) > 1 and args[0] == "kernels" and args[1] in {"push", "output"}
        return self._run_command(
            cmd, cwd=cwd, check=check,
            timeout=self.settings.transfer_timeout_seconds if transfer else self.settings.command_timeout_seconds,
            check_space=transfer,
        )

    def _run_command(self, cmd, *, cwd=None, check=True, timeout=None, input_text=None, check_space=False,
                     cancel_check=None):
        self._check_interrupted()
        env = self._env()
        budget = getattr(self.settings, "_storage_budget", None) if check_space else None
        if budget:
            budget.check_free()
        for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            register_secret(env.get(name, ""))
        # Explicit credentials must not fall back to another account's cached login.
        with tempfile.TemporaryDirectory(prefix="relay-kaggle-config-") as config_dir:
            if self.credentials and not self.credentials.config_dir:
                env["KAGGLE_CONFIG_DIR"] = config_dir
            return self._communicate(cmd, env, cwd, check, timeout, input_text, budget, cancel_check)

    def _communicate(self, cmd, env, cwd, check, timeout, input_text, budget=None, cancel_check=None):
        if input_text is not None and json.loads(input_text).get("operation") == "verify_dataset_content":
            return run_verification(self, cmd, env, cwd, input_text, budget, cancel_check)
        sdk_result = input_text is not None
        process_group = os.name == "posix" and not self._sdk_in_process
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            start_new_session=process_group,
        )
        deadline = time.monotonic() + (timeout or self.settings.command_timeout_seconds)
        try:
            while True:
                try:
                    stdout, _stderr = process.communicate(
                        input=input_text,
                        timeout=0.2,
                    )
                    break
                except subprocess.TimeoutExpired:
                    input_text = None
                    self._check_interrupted()
                    if cancel_check is not None:
                        cancel_check()
                    if budget:
                        budget.check_free()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Kaggle operation exceeded its configured timeout")
        except BaseException:
            stdout = self._stop_process(process, process_group=process_group)
            output = redact_secrets(stdout)
            if output:
                self.log(output[-4000:])
            raise

        output = redact_secrets(stdout or "")
        log_output = "\n".join(line for line in output.splitlines() if not line.startswith("RELAY_SDK_RESULT="))
        if log_output:
            self.log(log_output[-4000:])
        if check and process.returncode != 0:
            raise KaggleAdapterError(f"Kaggle command failed: {process.returncode}\n{output}")
        return subprocess.CompletedProcess(cmd, process.returncode, stdout=(stdout if sdk_result else output))

    def _sdk_call(self, operation: str, **arguments):
        payload = {
            "operation": operation,
            "arguments": arguments,
            "storage_dir": str(self.settings.storage_dir),
            "kaggle_cmd": self.settings.kaggle_cmd,
            "command_timeout_seconds": self.settings.command_timeout_seconds,
            "transfer_timeout_seconds": self.settings.transfer_timeout_seconds,
            "publication_timeout_seconds": (
                self._publication_remaining if self._publication_remaining is not None
                else self.settings.dataset_status_permission_grace_seconds
            ),
            "verification_retry_timeout_seconds": self._verification_retry_remaining,
        }
        result = self._run_command(
            [sys.executable, "-m", "app.kaggle_sdk"],
            cwd=Path(__file__).resolve().parent.parent,
            timeout=(self.settings.transfer_timeout_seconds if operation in {"p6_download", "upload_dataset", "verify_dataset_content", "probe_username_write_access"}
                     else self.settings.command_timeout_seconds),
            input_text=json.dumps(payload),
            check_space=operation in {"p6_download", "upload_dataset", "verify_dataset_content", "probe_username_write_access"},
            cancel_check=self.dataset_cancel_check if operation == "verify_dataset_content" else None,
        )
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("RELAY_SDK_RESULT="):
                return json.loads(line.removeprefix("RELAY_SDK_RESULT="))
        raise KaggleAdapterError("Kaggle SDK process did not return a result")

    @staticmethod
    def _stop_process(process: subprocess.Popen, *, process_group: bool = False) -> str:
        # Remember group ownership: the leader may exit while descendants keep pipes open.
        if process_group or process.poll() is None:
            try:
                if process_group:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except OSError:
                pass
        try:
            stdout, _stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if process_group:
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

    def identity(self, configured_owner: str = "") -> dict:
        configured_owner = configured_owner or (self.credentials.username if self.credentials else
                                                 self._env().get("KAGGLE_USERNAME", ""))
        if not self._sdk_in_process:
            return self._sdk_call("identity", configured_owner=configured_owner)
        with self._temporary_kaggle_env():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return self._authenticated_identity(api, configured_owner)

    def _authenticated_identity(self, api, configured_owner):
        config = api.config_values
        method = str(config.get("auth_method", "")).split(".")[-1].lower()
        sources = {"access_token": "token_introspection", "oauth": "oauth_token_introspection",
                   "legacy_api_key": "validated_basic_principal"}
        source = sources.get(method, "unverified")
        actual = str(config.get("username", "") or "").strip()
        if method == "oauth":
            # OAuth's cached username is not itself an authenticated claim.
            actual = str(api._introspect_token(config.get("token", "")) or "").strip()
        credential_owner = (self.credentials.username if self.credentials else
                            self._env().get("KAGGLE_USERNAME", "")) or ""
        # Basic auth's username is a credential input, not a token introspection
        # result. Prove that exact principal/key pair against an authenticated API.
        if method == "legacy_api_key":
            api.dataset_list(mine=True, page=1)
        error = ""
        if source == "unverified":
            error = "unsupported_auth_method"
        elif not actual:
            error = "authenticated_owner_missing"
        elif not configured_owner:
            error = "configured_owner_missing"
        elif (actual.casefold() != configured_owner.strip().casefold() or
              (credential_owner and actual.casefold() != credential_owner.strip().casefold())):
            error = "configured_owner_identity_mismatch"
        return {"identity_verified": not error, "auth_method": method,
                "identity_source": source, "identity_error": error}

    def require_identity(self, owner):
        identity = self.identity(owner)
        if not identity.get("identity_verified"):
            raise KaggleAdapterError("Kaggle owner identity rejected: " + identity.get("identity_error", "unverified"))
        return identity

    def scheduling_status(self):
        try:
            identity = self.identity()
        except Exception as exc:
            return {"available": False, "identity_verified": False,
                    "identity_error": "identity_lookup_failed: " + redact_secrets(str(exc))[-300:]}
        if not identity.get("identity_verified"):
            return {"available": False, **identity, "error": identity.get("identity_error", "identity_unverified")}
        return {**self.quota(), **identity}

    def quota(self) -> dict:
        if not self._sdk_in_process:
            return self._sdk_call("quota")
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
        if not self._sdk_in_process:
            return self._sdk_call("probe_username_write_access")
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
        if not self._sdk_in_process:
            return self._sdk_call("dataset_exists", dataset_ref=dataset_ref)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            with self._temporary_kaggle_env():
                api = KaggleApi()
                api.authenticate()
                result = api.dataset_status(dataset_ref)
            return True
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                return False
            raise

    @staticmethod
    def _owned_dataset_exists(api, dataset_ref: str) -> bool:
        """Select create/version from authenticated inventory, never from a 403."""
        seen = set()
        for page in range(1, 1001):
            rows = api.dataset_list(mine=True, page=page)
            if not isinstance(rows, list):
                raise KaggleAdapterError("Dataset owner inventory is incomplete")
            if not rows:
                return False
            refs = [getattr(row, "ref", None) for row in rows]
            if any(not isinstance(ref, str) or len(ref.split("/")) != 2
                   or not all(ref.split("/")) for ref in refs):
                raise KaggleAdapterError("Dataset owner inventory contains invalid refs")
            if len(set(refs)) != len(refs) or seen.intersection(refs):
                raise KaggleAdapterError("Dataset owner inventory pagination repeated")
            if dataset_ref in refs:
                return True
            seen.update(refs)
        raise KaggleAdapterError("Dataset owner inventory pagination limit reached")

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
        if not self._sdk_in_process:
            return self._sdk_call("_dataset_file_inventory", dataset_ref=dataset_ref, version_number=version_number)
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
        if not self._sdk_in_process:
            result = self._sdk_call("upload_dataset", dataset_dir=str(dataset_dir), dataset_ref=dataset_ref,
                                    update_message=update_message)
            return DatasetUploadReceipt(
                expected_version_number=result["expected_version_number"],
                expected_files=tuple(tuple(item) for item in result["expected_files"]),
                dataset_dir=result["dataset_dir"], content_sha256=result["content_sha256"],
            )
        self._check_interrupted()
        from kaggle.api.kaggle_api_extended import KaggleApi

        expected_files = dataset_upload_inventory(dataset_dir)
        source_digest = content_digest(dataset_dir)
        saved_path = intent_path(self.settings.storage_dir, dataset_dir, dataset_ref)
        intent = read_intent(saved_path)
        if intent is not None:
            if (intent.get("dataset_ref") != dataset_ref or intent.get("content_sha256") != source_digest
                    or intent.get("dataset_dir") != str(Path(dataset_dir).absolute())):
                raise KaggleAdapterError("upload_intent_scope_or_content_mismatch")
            if intent["state"] == "rejected":
                raise KaggleAdapterError("Dataset upload was rejected; retained original intent")
            self.require_identity(dataset_ref.split("/")[0])
            return DatasetUploadReceipt(intent["version_number"], expected_files,
                                        str(Path(dataset_dir).absolute()), source_digest)
        with self._temporary_kaggle_env():
            api = KaggleApi()
            api.authenticate()
            identity = self._authenticated_identity(api, dataset_ref.split("/")[0])
            if not identity["identity_verified"]:
                raise KaggleAdapterError("Kaggle owner identity rejected: " + identity["identity_error"])
            try:
                inspect.signature(api.dataset_download_files).bind(
                    dataset_ref + "/1", path="unused", force=True, quiet=True, unzip=False)
            except (AttributeError, TypeError, ValueError) as exc:
                raise KaggleAdapterError("Kaggle API cannot verify exact Dataset content; upgrade explicitly") from exc
            metadata_path = Path(dataset_dir) / "dataset-metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("id") != dataset_ref:
                    raise KaggleAdapterError("Dataset metadata owner/ref binding mismatch")
            # A new private ref can return 403 from status before it exists.
            # A complete authenticated inventory permits only a create attempt;
            # the create response remains authoritative for conflicts/races.
            exists = self._owned_dataset_exists(api, dataset_ref)
            if exists:
                current_version_number = self._dataset_version_number(api, dataset_ref)
                expected_version_number = current_version_number + 1
            else:
                expected_version_number = 1
            intent = {"schema_version": 1, "dataset_ref": dataset_ref,
                      "dataset_dir": str(Path(dataset_dir).absolute()), "content_sha256": source_digest,
                      "version_number": expected_version_number, "operation": "version" if exists else "create",
                      "state": "unknown", "existence_basis": "authenticated_mine_inventory", **identity}
            # Durably record BEFORE remote mutation. Failures here never upload.
            write_intent(saved_path, intent)
            self.log(f"{'Updating' if exists else 'Creating'} dataset {dataset_ref}")
            try:
                if exists:
                    response = api.dataset_create_version(str(dataset_dir), update_message, quiet=False,
                                                          convert_to_csv=False, delete_old_versions=False, dir_mode="tar")
                else:
                    response = api.dataset_create_new(str(dataset_dir), public=False, quiet=False,
                                                      convert_to_csv=False, dir_mode="tar")
            except BaseException as exc:
                raise DatasetUploadUnknown("Dataset upload outcome unknown; original candidate retained") from exc
            response_fields = ("status", "error", "error_message", "errorMessage", "invalid_tags", "invalidTags")
            if not any(name in response if isinstance(response, dict) else hasattr(response, name)
                       for name in response_fields):
                raise DatasetUploadUnknown("Dataset response missing; original candidate retained")
            get = response.get if isinstance(response, dict) else lambda k, d=None: getattr(response, k, d)
            error = get("error") or get("error_message") or get("errorMessage") or get("invalid_tags") or get("invalidTags")
            status = str(get("status", "") or "").lower()
            if error or (status and status not in {"ok", "ready", "complete"}):
                write_intent(saved_path, {**intent, "state": "rejected"})
                raise KaggleAdapterError("Dataset business response rejected: " + redact_secrets(str(error or status)))
            if content_digest(dataset_dir) != source_digest:
                raise DatasetUploadUnknown("Dataset source changed after upload; original candidate retained")
            write_intent(saved_path, {**intent, "state": "accepted"})
        return DatasetUploadReceipt(
            expected_version_number=expected_version_number,
            expected_files=expected_files,
            dataset_dir=str(Path(dataset_dir).absolute()), content_sha256=source_digest,
        )

    def verify_dataset_content(self, dataset_ref, version_number, dataset_dir, content_sha256):
        if not self._sdk_in_process:
            return self._sdk_call("verify_dataset_content", dataset_ref=dataset_ref, version_number=version_number,
                                  dataset_dir=str(dataset_dir), content_sha256=content_sha256)
        if type(version_number) is not int or version_number <= 0:
            raise KaggleAdapterError("Dataset exact version must be a positive integer")
        self.verification_phase("content")
        if content_digest(dataset_dir) != content_sha256:
            raise KaggleAdapterError("Dataset frozen content changed")
        self.verification_phase("publication")
        self.log(f"Dataset exact version {version_number}: checking publication and archive availability")
        with self._temporary_kaggle_env():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            identity = self._authenticated_identity(api, dataset_ref.split("/")[0])
            if not identity["identity_verified"]:
                raise KaggleAdapterError("Kaggle owner identity rejected")
            with tempfile.TemporaryDirectory(prefix="relay-version-check-") as directory:
                # The SDK calls download_file only after obtaining the archive response.
                # Retain its download behavior while separating discovery from transfer.
                original_download = getattr(api, "download_file", None)
                if original_download is not None:
                    def download_content(*args, **kwargs):
                        self.verification_phase("content")
                        self.log(f"Dataset exact version {version_number}: downloading and verifying archive")
                        return original_download(*args, **kwargs)
                    api.download_file = download_content
                try:
                    api.dataset_download_files(f"{dataset_ref}/{version_number}", path=directory,
                                               force=True, quiet=True, unzip=False)
                except Exception as exc:
                    if not archive_url_missing(exc):
                        raise
                    self.log(f"Dataset version {version_number}: archive URL unavailable; "
                             "checking exact-version file contents")
                    def check():
                        self._check_interrupted()
                        if self.dataset_cancel_check is not None:
                            self.dataset_cancel_check()
                    verify_version_files(api, dataset_ref, version_number, dataset_dir, check, self.log,
                                         phase=self.verification_phase)
                else:
                    self.verification_phase("content")
                    paths = list(Path(directory).iterdir())
                    if len(paths) != 1 or not paths[0].is_file() or paths[0].is_symlink():
                        raise KaggleAdapterError("Dataset version archive missing")
                    verify_upload_archive(dataset_dir, paths[0])
                finally:
                    if original_download is not None:
                        api.download_file = original_download
                self.verification_phase("content")
                if content_digest(dataset_dir) != content_sha256:
                    raise KaggleAdapterError("Dataset frozen content changed during verification")
        return True

    def wait_dataset(
        self,
        dataset_ref: str,
        permission_grace_seconds: int = 0,
        upload_receipt: DatasetUploadReceipt | None = None,
    ) -> str:
        start = time.time()
        last_reason, last_reason_at = None, 0.0
        def report(reason, message):
            nonlocal last_reason, last_reason_at
            now = time.monotonic()
            if reason != last_reason or now - last_reason_at >= 60:
                self.log(message)
                last_reason, last_reason_at = reason, now

        @contextmanager
        def quiet_poll():
            original_log = self.log
            self.log = lambda message: None
            try:
                yield
            finally:
                self.log = original_log
        visibility_grace = max(0, int(permission_grace_seconds or 0))
        publication_grace = max(visibility_grace, self.settings.dataset_status_permission_grace_seconds)
        publication_start = time.monotonic()
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
        source_dir = getattr(upload_receipt, "dataset_dir", "")
        source_digest = getattr(upload_receipt, "content_sha256", "")
        publication_detail = ""
        transient_attempts = 0
        retry_deadline = None
        while True:
            self._check_interrupted()
            status_args = ["datasets", "status", dataset_ref]
            if self.dataset_cancel_check is not None:
                self.dataset_cancel_check()
            if source_dir:
                if retry_deadline is not None:
                    self._verification_retry_remaining = max(0, retry_deadline - time.monotonic())
                    if self._verification_retry_remaining <= 0:
                        self._verification_retry_remaining = None
                        raise DatasetVerificationError(
                            "dataset_verification_retry_exhausted: original candidate retained; " + publication_detail,
                            "transport")
                self._publication_remaining = max(0, publication_grace - (time.monotonic() - publication_start))
                if self._publication_remaining <= 0:
                    self._publication_remaining = None
                    raise KaggleAdapterError("payload_publication_timeout: Dataset publication deadline exceeded; " + publication_detail)
                self._verification_content_seconds = 0
                try:
                    try:
                        self.verify_dataset_content(dataset_ref, expected_version_number, source_dir, source_digest)
                    finally:
                        publication_start += self._verification_content_seconds
                        self._publication_remaining = None
                        self._verification_retry_remaining = None
                except Exception as exc:
                    detail = redact_secrets(str(exc))
                    publication_detail = detail[-1200:]
                    elapsed = time.monotonic() - publication_start
                    error = verification_error(exc)
                    if error["category"] == "publication":
                        report("publishing", f"Dataset exact version {expected_version_number} is still publishing; "
                               "waiting before full verification: " + detail[-1200:])
                        if elapsed >= publication_grace:
                            raise KaggleAdapterError(
                                f"Dataset exact version {expected_version_number} publication verification timed out "
                                f"after {publication_grace}s; payload_publication_timeout: " + detail[-1200:]
                            ) from exc
                    elif error["category"] == "http" and error["http_status"] in {403, 404}:
                        report("permission", f"Dataset exact candidate is not yet accessible (HTTP {error['http_status']}): " + detail[-500:])
                        if visibility_grace <= 0 or elapsed > visibility_grace:
                            raise
                    elif (error["category"] == "transport" or
                          error["category"] == "http" and error["http_status"] in {408, 409, 429, 500, 502, 503, 504}):
                        if retry_deadline is None:
                            retry_deadline = time.monotonic() + 600
                        transient_attempts += 1
                        remaining = min(retry_deadline - time.monotonic(), publication_grace - elapsed)
                        delay = max(error.get("retry_after") or 0,
                                    min(60, 5 * 2 ** min(transient_attempts - 1, 4)) + random.uniform(0, 3))
                        if transient_attempts > 5 or delay >= remaining:
                            raise DatasetVerificationError(
                                "dataset_verification_retry_exhausted: original candidate retained; " + detail[-1200:],
                                "transport") from exc
                        reason = f"HTTP {error['http_status']}" if error["http_status"] else "network"
                        self.log(f"Dataset verification retry {transient_attempts}/5 in {delay:.0f}s ({reason}); "
                                 f"original version {expected_version_number} retained; " + detail[-500:])
                        # Keep cancellation responsive during Retry-After and backoff.
                        wake_at = time.monotonic() + delay
                        while time.monotonic() < wake_at:
                            self._check_interrupted()
                            if self.dataset_cancel_check is not None:
                                self.dataset_cancel_check()
                            self._sleep(min(1, max(0, wake_at - time.monotonic())))
                        continue
                    else:
                        if error["category"] == "integrity":
                            report("content_mismatch", "Dataset exact candidate content rejected: " + detail[-500:])
                        else:
                            report("verification_stopped", "Dataset exact candidate verification stopped: " + detail[-500:])
                        raise
                    self._sleep(min(self.settings.dataset_poll_seconds, max(0, publication_grace - elapsed)))
                    continue
                report("verified", f"Dataset exact version {expected_version_number} bytes verified")
                return json.dumps({"status": "ready", "current_version_number": expected_version_number})
            if expected_version_number is not None:
                status_args.extend(["--format", "json"])
            with quiet_poll():
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
                        with quiet_poll():
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
                report(tuple(details),
                    "Dataset reports ready before the uploaded version is fully visible; "
                    + "; ".join(details)
                )
            if result.returncode == 0 and any(word in status_text for word in ["failed", "error", "deleted"]):
                raise KaggleAdapterError(f"Dataset failed:\n{output}")
            if result.returncode != 0 and any(word in status_text for word in ["401", "unauthorized"]):
                raise KaggleAdapterError(f"Dataset status failed:\n{output}")
            if result.returncode != 0 and any(
                word in status_text for word in ["403", "404", "forbidden", "not found"]
            ):
                if visibility_grace > 0 and elapsed <= visibility_grace:
                    report("permission", "Dataset status is temporarily unavailable after upload; "
                           f"retrying for up to {visibility_grace} seconds")
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
        try:
            result = self._run(["kernels", "status", kernel_ref], check=False)
        except TimeoutError as exc:
            raise KernelStatusQueryError("Kernel status query timed out", transient=True) from exc
        output = result.stdout.strip() or f"returncode={result.returncode}"
        if result.returncode != 0:
            raise KernelStatusQueryError(
                f"Kernel status failed:\n{output}", transient=transient_kernel_query_output(output),
            )
        return output

    def wait_kernel(self, kernel_ref: str, progress_callback: Callable[[dict], None]) -> str:
        deadline = time.monotonic() + self.settings.kernel_max_wait_seconds
        self.last_stop_details = {}
        last_progress_key = None
        query_failures = 0
        output = "Kernel terminal status has not been observed"
        while True:
            self._check_interrupted()
            if time.monotonic() >= deadline:
                self.last_stop_details = {
                    "reason": "monitoring_timeout", "source": "relay", "confidence": "unknown",
                    "provider_status": provider_state(output),
                    "message": "Relay monitoring deadline exceeded; the original run is retained and its terminal state is unknown.",
                }
                raise KernelStatusUnavailable(f"Kernel monitoring deadline exceeded; retained original run:\n{output}")
            try:
                output = self.kernel_status(kernel_ref)
            except KernelStatusQueryError as exc:
                if not exc.transient:
                    self.last_stop_details = {
                        "reason": "status_unavailable", "source": "relay", "confidence": "unknown",
                        "message": redact_secrets(str(exc))[-2000:],
                    }
                    raise
                query_failures += 1
                remaining = max(0, deadline - time.monotonic())
                if not remaining:
                    self.last_stop_details = {
                        "reason": "monitoring_timeout", "source": "relay", "confidence": "unknown",
                        "message": "Relay status-query retry budget exhausted; training termination is unconfirmed.",
                    }
                    raise KernelStatusUnavailable(str(exc)) from exc
                delay = min(60, 5 * 2 ** min(query_failures - 1, 4), remaining)
                self.log(f"Kernel status temporarily unavailable; retry {query_failures} in {delay:.0f}s: "
                         + redact_secrets(str(exc)))
                self._sleep(delay)
                continue
            if query_failures:
                self.log("Kernel status query recovered; continuing original run")
                query_failures = 0
            # Logs are supplementary; a log-fetch error is not a training failure.
            try:
                log_result = self._run(["kernels", "logs", kernel_ref], check=False)
                logs = log_result.stdout if log_result.returncode == 0 else ""
            except TimeoutError:
                self.log("Kernel log query timed out; preserving last training progress")
                logs = ""
            progress_events = parse_training_progress_logs(logs)
            if progress_events:
                progress = progress_events[-1]
                key = training_progress_key(progress)
                if key != last_progress_key:
                    last_progress_key = key
                    progress_callback(progress)
            state = provider_state(output)
            if state in {"COMPLETE", "COMPLETED", "SUCCEEDED", "SUCCESS"}:
                self.last_stop_details = classify_stop_reason(output, logs)
                return output
            if state in {"ERROR", "FAILED", "FAILURE", "CANCELED", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
                self.last_stop_details = classify_stop_reason(output, logs)
                raise KaggleAdapterError(f"Kernel failed:\n{output}")
            self._sleep(min(self.settings.kernel_poll_seconds, max(0, deadline - time.monotonic())))

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

    def p6_download(self, kernel_ref, output_dir, kernel_dir):
        if not self._sdk_in_process:
            return self._sdk_call('p6_download', kernel_ref=kernel_ref, output_dir=str(output_dir), kernel_dir=str(kernel_dir))
        from app.dinov2_kaggle_output import observe_kernel, download_version
        from app.dinov2_251_artifacts import read_json, plain
        from kaggle.api.kaggle_api_extended import KaggleApi
        root = Path(kernel_dir)
        task = read_json(plain(root, 'p6_task.json'))
        metadata = read_json(plain(root, 'kernel-metadata.json'))
        source = plain(root, 'train.py').read_text(encoding='utf-8')
        observed_path = root/'p6_output_observation.json'
        saved = read_json(observed_path) if observed_path.exists() else None
        with self._temporary_kaggle_env():
            api = KaggleApi()
            api.authenticate()
            observation = observe_kernel(api, kernel_ref, source, metadata['dataset_sources'],
                                         version=saved['kernel_version'] if saved else 1)
            if saved is not None and observation != saved:
                raise ValueError('P6 original Kernel candidate changed')
            write_intent(observed_path, observation)
            destination = Path(output_dir)
            if destination.exists():
                shutil.rmtree(destination)
            return download_version(api, observation, destination, pattern=DINO251_PATTERN,
                                    max_bytes=task['budget']['disk_bytes'])

    def package_artifacts(
        self,
        output_dir: Path,
        artifact_zip: Path,
        artifact_contract: str = "yolo",
        *,
        expected_identity: dict | None = None,
        expected_task_sha256: str = '',
    ) -> None:
        if artifact_contract in DINO251_CONTRACTS:
            return package_result(output_dir / 'p6_result', artifact_zip,
                                  expected_identity=expected_identity, expected_task_sha256=expected_task_sha256,
                                  storage_budget=getattr(self.settings, '_storage_budget', None), expected_contract=artifact_contract)
        if artifact_contract == DINO_CONTRACT:
            package_dinov2_artifacts(output_dir / DINO_SUBDIR, artifact_zip, expected_identity=expected_identity,
                                    storage_budget=getattr(self.settings, "_storage_budget", None))
            return
        required_files = REQUIRED_ARTIFACT_FILES.get(artifact_contract)
        if required_files is None:
            raise KaggleAdapterError(
                f"unknown artifact contract: {artifact_contract}"
            )
        for relative_path in required_files:
            require_file(output_dir, relative_path)
        if artifact_contract == "patchcore" and (output_dir / "cnn_onnx" / "result.json").is_file():
            from app.cnn_onnx_artifacts import read, source_from_run, validate_package
            try:
                source = source_from_run(output_dir / "model.ckpt")
                if expected_identity and any(source["identity"].get(key) != value for key, value in expected_identity.items()):
                    raise ValueError("onnx_identity_mismatch")
                report = read(output_dir / "cnn_onnx" / "result.json")
                if report.get("status") not in {"PASS", "FAIL", "CANCELLED", "PENDING"}:
                    raise ValueError("onnx_status_invalid")
                name = report.get("package", "")
                if name and not re.fullmatch(r"package-[a-f0-9]{32}", name):
                    raise ValueError("onnx_package_path_invalid")
                if report["status"] == "PASS" and not name:
                    raise ValueError("onnx_package_missing")
                if name:
                    document = validate_package(output_dir / "cnn_onnx" / name, source=source)
                    if document["status"] != report["status"]:
                        raise ValueError("onnx_status_mismatch")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise KaggleAdapterError("CNN ONNX artifact validation failed: " + str(exc)) from exc
        artifact_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    budget = getattr(self.settings, "_storage_budget", None)
                    if budget:
                        budget.check_free(path.stat().st_size)
                    archive.write(path, path.relative_to(output_dir).as_posix())

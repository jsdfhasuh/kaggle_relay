"""Explicit live private GPU probes. Run inside the configured Relay container.

No credentials are written to reports or command-line arguments. This checks
actual CUDA computation, not authentication or reported GPU quota. It does not
change key permissions or enable/disable keys.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time

from app.auth_config import AuthStore
from app.config import Settings
from app.kaggle_adapter import KaggleAdapter
from app.security import redact_secrets


PROBE = '''import json, time, socket, subprocess
from pathlib import Path
report = {"case_id": CASE_ID, "utc_timestamp": time.time(), "computation_passed": False}
try:
    import torch
    report.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
    if report["cuda_available"] and report["device_count"] > 0:
        x = torch.ones((128, 128), device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        assert y.is_cuda and bool(torch.all(y == 128).item())
        report.update(computation_passed=True, device=str(y.device),
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except Exception as exc:
    report["runtime_error"] = type(exc).__name__ + ": " + str(exc)
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=10)
    report["nvidia_smi"] = {"returncode": result.returncode, "gpu_names": result.stdout.strip()}
except Exception as exc:
    report["nvidia_smi"] = {"error": type(exc).__name__}
try:
    socket.getaddrinfo("pypi.org", 443)
    report["pypi_dns_available"] = True
except OSError:
    report["pypi_dns_available"] = False
Path("/kaggle/working/gpu-probe.json").write_text(json.dumps(report, indent=2))
print("RELAY_GPU_PROBE " + json.dumps(report), flush=True)
'''


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def probe(settings, credentials, aliases, root, run_id, timeout):
    folder = root / credentials.id
    folder.mkdir(exist_ok=True)
    case_id = f"relay-gpu-check-{run_id}-{credentials.id}"
    kernel_ref = f"{credentials.username}/{case_id}"
    adapter = KaggleAdapter(settings, lambda _: None, credentials=credentials)
    result = dict(key=credentials.id, aliases=aliases, account=credentials.username,
                  kernel_ref=kernel_ref, requested_gpu=True, verdict="unverified")
    try:
        state_file = folder / "state.json"
        if not state_file.exists():
            kernel = folder / "kernel"
            kernel.mkdir(exist_ok=True)
            source = PROBE.replace("CASE_ID", repr(case_id))
            (kernel / "probe.py").write_text(source, encoding="utf-8")
            write(kernel / "kernel-metadata.json", dict(id=kernel_ref, title=case_id, code_file="probe.py",
                  language="python", kernel_type="script", is_private=True, enable_gpu=True,
                  enable_internet=True, machine_shape="NvidiaTeslaT4", dataset_sources=[],
                  competition_sources=[], kernel_sources=[]))
            state = dict(kernel_ref=kernel_ref, started_at=time.time(), deadline=time.time() + timeout,
                         source_sha256=hashlib.sha256(source.encode()).hexdigest(), push_attempted=True)
            write(state_file, state)
            adapter.push_kernel(kernel)
        state = json.loads(state_file.read_text())
        assert state["kernel_ref"] == kernel_ref
        while True:
            status = adapter.kernel_status(kernel_ref)
            result["official_status"] = status
            write(folder / "status.json", result)
            if any(word in status.lower() for word in ["complete", "error", "cancel", "fail"]):
                break
            if time.time() > state["deadline"]:
                raise TimeoutError("Probe observation timed out; remote termination is not confirmed")
            time.sleep(10)
        destination = folder / "output"
        destination.mkdir(exist_ok=True)
        download = adapter._run(["kernels", "output", kernel_ref, "-p", str(destination),
                                 "--force", "--file-pattern", "^gpu-probe[.]json$"], check=False)
        assert download.returncode == 0, "Probe result download failed"
        runtime = json.loads((destination / "gpu-probe.json").read_text())
        assert runtime["case_id"] == case_id, "Probe output identity mismatch"
        result["runtime"] = runtime
        if runtime.get("computation_passed") is True and runtime.get("cuda_available") is True:
            result["verdict"] = "gpu_passed"
        elif runtime.get("cuda_available") is False and runtime.get("device_count") == 0:
            result["verdict"] = "no_cuda_device"
    except Exception as exc:
        result["error"] = redact_secrets(str(exc))[-1200:]
    result["finished_at"] = time.time()
    write(folder / "report.json", result)
    print(json.dumps({key: result[key] for key in ["key", "account", "verdict"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9-]{1,20}", args.run_id)
    settings = Settings.from_env()
    auth = AuthStore.from_settings(settings)
    root = Path(args.output)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    accounts = {}
    for credentials in auth._kaggle_keys.values():
        if credentials.enabled:
            assert credentials.username
            accounts.setdefault(credentials.username.lower(), []).append(credentials)
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(accounts)))) as pool:
        futures = [pool.submit(probe, settings, group[0], [c.id for c in group], root,
                               args.run_id, args.timeout) for group in accounts.values()]
        results = [future.result() for future in futures]
    write(root / "report.json", {"run_id": args.run_id, "accounts": results,
          "passed": sum(row["verdict"] == "gpu_passed" for row in results),
          "no_cuda_device": sum(row["verdict"] == "no_cuda_device" for row in results),
          "unverified": sum(row["verdict"] == "unverified" for row in results)})


if __name__ == "__main__":
    main()

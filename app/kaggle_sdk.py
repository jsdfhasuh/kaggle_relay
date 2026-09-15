"""Run one SDK operation with credentials supplied only in this process's environment."""

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from app.config import Settings
from app.kaggle_adapter import KaggleAdapter
from app.security import redact_secrets, register_secret


def main() -> int:
    for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        register_secret(os.environ.get(name, ""))
    try:
        payload = json.load(sys.stdin)
        operation = payload["operation"]
        if operation not in {"quota", "upload_dataset", "dataset_exists", "_dataset_file_inventory",
                             "probe_username_write_access"}:
            raise ValueError("unsupported SDK operation")
        settings = Settings(
            api_token="", storage_dir=Path(payload["storage_dir"]),
            kaggle_cmd=payload["kaggle_cmd"],
            command_timeout_seconds=payload["command_timeout_seconds"],
            transfer_timeout_seconds=payload["transfer_timeout_seconds"],
        )
        adapter = KaggleAdapter(settings, lambda message: print(redact_secrets(message), flush=True))
        adapter._sdk_in_process = True
        arguments = payload["arguments"]
        if operation == "upload_dataset":
            arguments["dataset_dir"] = Path(arguments["dataset_dir"])
        result = getattr(adapter, operation)(**arguments)
        if is_dataclass(result):
            result = asdict(result)
        print("RELAY_SDK_RESULT=" + json.dumps(result), flush=True)
        return 0
    except (Exception, SystemExit) as exc:
        print(redact_secrets(f"Kaggle SDK operation failed: {exc}"), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

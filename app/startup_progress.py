"""Startup observations and handoff ordering shared by callbacks and log polling."""
import json
import math


def parse_startup_progress(payload: dict) -> dict:
    if payload.get("event_type") != "startup" or payload.get("backend") not in ("yolo", "patchcore"):
        return {}
    if payload.get("startup_status") not in ("running", "ready", "restarting", "failed"):
        return {}
    if not all(isinstance(payload.get(key), str) and payload[key].strip()
               for key in ("startup_id", "startup_stage", "message")):
        return {}
    try:
        sequence = int(payload["startup_sequence"])
        numbers = {key: float(payload[key]) for key in (
            "startup_elapsed_seconds", "stage_elapsed_seconds", "reported_at")}
    except (KeyError, TypeError, ValueError, OverflowError):
        return {}
    if sequence < 1 or any(not math.isfinite(value) or value < 0 for value in numbers.values()):
        return {}
    keys = ("event_type", "backend", "startup_id", "startup_stage", "startup_status", "message", "error_code")
    return {**{key: payload[key] for key in keys if key in payload}, **numbers,
            "startup_sequence": sequence, "heartbeat": payload.get("heartbeat") is True,
            "progress_source": "startup"}


def ignore_startup_progress(job: dict, incoming: dict) -> bool:
    if incoming.get("event_type") != "startup":
        return False
    event = parse_startup_progress(incoming)
    if not event or job.get("status") not in {"pushing_kernel", "waiting_kernel", "cancel_requested"}:
        return True
    previous = job.get("kernel_status") or {}
    if isinstance(previous, str):
        try:
            previous = json.loads(previous)
        except ValueError:
            previous = {}
    if not isinstance(previous, dict):
        return False
    old = parse_startup_progress(previous)
    if old:
        if old["startup_id"] == event["startup_id"]:
            return event["startup_sequence"] <= old["startup_sequence"]
        # Events are timestamped before I/O, so a delayed restart parent loses
        # to the child's newer event even when their HTTP requests arrive reversed.
        return event["reported_at"] <= old["reported_at"]
    if previous.get("epoch") is not None:
        return True
    # These preparation phases coexist with startup diagnostics until handoff.
    return bool(previous.get("phase") and previous["phase"] not in {"prepare", "input"})

"""Gateway calibration overlay, executed only in the submitted Kaggle worker.

The frozen wheel remains the base runtime. Its integrity checks still run;
the separate policy and its digest are recorded in the calibration receipt.
"""

import functools
import math
import sys


_RELAY_DINO_POLICY_ID = "dino_f1_lower_margin_v1"
_RELAY_DINO_RUNTIME = "d31e1303ff94dd205ea5a9438cfb3f5247aa4002a0af876b2bc06c6f5fe5a266"


def _relay_dino_lower_threshold(original, normal_scores, defect_scores, policy_sha256):
    calibration, threshold = original(normal_scores, defect_scores)
    threshold = float(threshold)
    scores = [float(value) for group in (normal_scores, defect_scores) for value in group]
    if not scores or not all(math.isfinite(value) for value in [threshold, *scores]):
        raise ValueError("DINO gateway threshold policy requires finite calibration scores")
    # Twice the native verifier's absolute + relative score tolerance.
    requested_margin = 2 * (1e-5 + 1e-4 * abs(threshold))
    lower_scores = [value for value in scores if value < threshold]
    lower_bound = max(lower_scores) if lower_scores else None
    effective = threshold - requested_margin
    if lower_bound is not None:
        effective = max(effective, lower_bound / 2 + threshold / 2)
    if (not math.isfinite(effective) or not effective < threshold
            or (lower_bound is not None and not lower_bound < effective)):
        raise ValueError("DINO threshold has no representable lower margin preserving calibration labels")
    if any((value >= threshold) != (value >= effective) for value in scores):
        raise ValueError("DINO gateway threshold policy would change calibration labels")
    calibration = dict(calibration)
    calibration["gateway_threshold_policy"] = {
        "id": _RELAY_DINO_POLICY_ID,
        "source_sha256": policy_sha256,
        "base_runtime_sha256": _RELAY_DINO_RUNTIME,
        "original_threshold": threshold,
        "effective_threshold": effective,
        "requested_margin": requested_margin,
        "applied_margin": threshold - effective,
        "lower_calibration_score": lower_bound,
        "calibration_labels_preserved": True,
    }
    print(f"Gateway DINO threshold policy {_RELAY_DINO_POLICY_ID}: "
          f"{threshold:.17g} -> {effective:.17g}", file=sys.stderr, flush=True)
    return calibration, effective


def _relay_dino_worker_main():
    from patchcore_dino_runtime.worker import verify_runtime
    from patchcore_dino_runtime.calibration import score_contract
    from patchcore_dino_runtime.remote_training import main

    if verify_runtime() != _RELAY_DINO_RUNTIME:
        raise RuntimeError("DINO gateway threshold policy runtime mismatch")
    # score_contract verifies the original calibration source before loading it.
    contract = score_contract()
    original = contract.calibration_details
    contract.calibration_details = functools.partial(
        _relay_dino_lower_threshold, original, policy_sha256=_RELAY_DINO_POLICY_SHA256,
    )
    try:
        return main()
    finally:
        contract.calibration_details = original


def _relay_dino_wrap_run(original):
    @functools.wraps(original)
    def run(dataset_root, working_root, config):
        import patchcore_dino_runtime.kaggle_bootstrap as bootstrap

        if config.get("runtime_sha256") != _RELAY_DINO_RUNTIME:
            print("Gateway DINO threshold policy skipped: unrecognized frozen runtime", flush=True)
            return original(dataset_root, working_root, config)
        original_worker = bootstrap.run_worker

        def run_worker(command, **kwargs):
            # Keep the bootstrap's isolated interpreter, environment, request and
            # output arguments. Only add the explicit calibration policy launcher.
            if command[1:6] != ["-I", "-X", "utf8", "-m", "patchcore_dino_runtime.remote_training"]:
                raise RuntimeError("DINO threshold policy requires the isolated cloud worker entrypoint")
            command = [command[0], "-I", "-X", "utf8", "-c", _RELAY_DINO_WORKER_SOURCE, *command[6:]]
            return original_worker(command, **kwargs)

        bootstrap.run_worker = run_worker
        try:
            return original(dataset_root, working_root, config)
        finally:
            bootstrap.run_worker = original_worker
    return run

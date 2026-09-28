# P6 v2 deployment calibration transport preparation

Baseline: `66e76912adc7fabf3c9ef234573c0bde8330eedc`, branch `codex/p6-dino-cloud-onnx`. Local and origin matched and the feature worktree was clean before changes. Original repository README dirty change is preserved.

Add independent `patchcore_dinov2_251_onnx_v2` / `dino_cloud_result_p6_v2` dispatch through schema, DB preservation, exact-Kernel download, packaging and authenticated receipt. V1/old DINO/CNN/YOLO remain supported by their existing rules. V2 transport requires corresponding task/deployment versions, frozen ONNX deployment calibration bound to actual graph/threshold and Linux producer scope. Training-side candidate threshold and all cloud reference files are separately inventoried. The shared artifact validator is byte-identical to application's `anomaly_func/dinov2_transport.py`.

Validation: `.venv/Scripts/python.exe -m pytest tests/test_dinov2_p6.py tests/test_dinov2_artifacts.py tests/test_relay_api.py -q` -> **161 PASS**, exit 0. Includes v2 schema/DB reopen and version-mismatched partial-result rejection. These are protocol fixtures, not real cloud/model qualification.

`scripts/p6_calibration_sandbox.py` prepares the explicit existing-sandbox update path. Default is read-only; `--apply` requires fresh user authorization. It requires the exact reviewed SHA/branch, clean source, a stopped sandbox and no account/sandbox active tasks; backs up restricted state/private config; keeps old image/container; builds with network disabled against the existing image; and checks running file hashes and actual image. It never updates production. `--help` was run (exit 0); deployment was **NOT_RUN**.

Read-only Oracle observation: test container stopped, actual source `52bdd1db952aa863af7d171b72e173e6f74c4d47`, image `sha256:ab344299bda2c6a073170b3173613b9acdf9bf09dcdc096d52f1c74ef1c3da07`. Production stayed running at `e8255b5665c113f8be74359ef3cbad8369d98049`, image `sha256:07bfc8f35deb099802edacde3751bbf1cb8551032d4a703ece27fef02f194fe8`, clean. The prior test account had one unrelated waiting Kernel in production; no task was canceled or changed.

The app's two retained 129×768 model local replays passed new deployment labels and numeric checks, while retaining old-label FAIL. This does **not** establish Linux-cloud v2 calibration or successful Relay-v2 cloud reception. Both require the newly authorized two-route serial batch. No new cloud task, test-service restart, dependency installation or production operation has been performed in this preparation.

After authorization, prefer one Relay and one direct task, same original-role 4 train/3 val/2 test sample set and existing Small weights. Reuse the runs for recovery, receipt, registered target validation and fresh-process prediction. Preserve original v1 failed runs and all original artifacts. P6 overall qualification remains pending.

Implementation revisions prepared for review (not deployed): application `cefb7d79f8be5d28821f367c373d446a6c496feb`; Relay `3fe98a1e73185a572cd41a761349f195b3ca238f`. The final app local build preflight used this committed application version.

# DINO GPU contract wiring for user acceptance

Baseline `65a933c`, branch `codex/p6-dino-cloud-onnx`. Production observed read-only at detached `e8255b5665c113f8be74359ef3cbad8369d98049`, container `kaggle_relay-kaggle-relay-1`, image `sha256:07bfc8f35deb099802edacde3751bbf1cb8551032d4a703ece27fef02f194fe8`. Production was not updated by this change. No task was started. Its current health lacks the P6 v2 contract; pushing this branch alone does not change that.

This branch already implements `patchcore_dinov2_251_onnx_v1/v2`, version-bound result recovery, complete artifacts and authenticated receipts. This change adds the explicit training capability `dino_cuda_features_cpu_coreset_v1`. Before remote account or Dataset submission, the worker checks the frozen task's CUDA policy, FP32/cu126 selection, CPU deployment execution and Kernel GPU flag. It accepts Kaggle's legitimate boolean or lowercase string boolean metadata. It does not rewrite source, thresholds, task bytes or GPU selection. Old DINO v3/CNN/YOLO do not enter this new policy.

Application freezes the optional, independently versioned `training_execution` into new P6 v2 tasks. Existing tasks with no such field keep CPU semantics. GPU numerical/model validation is **PENDING user acceptance**, not inherited from P6 CPU evidence. Coreset remains CPU; no full-GPU speed claim.

Validation with existing interpreter:

```powershell
& 'C:/Users/jsdfhasuh/my_scripts/kaggle_relay/.venv/Scripts/python.exe' -X utf8 -m pytest tests/test_dinov2_training_policy.py tests/test_dinov2_p6.py tests/test_dinov2_artifacts.py tests/test_relay_api.py -q --disable-warnings
```

**169 PASS, exit 0**. First attempted use of the application interpreter failed collection because FastAPI is absent; switched to the existing Relay environment, no installation. Fixtures cover policy rejection, old-contract compatibility, metadata preservation, health capabilities, API, delivery and partial failure; they are not live GPU acceptance.

Manual acceptance: use a service actually deployed from this branch, verify its health lists both v2 contract and GPU profile; choose GPU in application, submit one user-authorized task, record real GPU name and execution device, then separately check CPU deployment calibration/validation, reception, registered inference and reopen. The application repository contains the detailed Chinese acceptance table.

Deployment requires explicit authorization, not implied by this report. The production worktree was clean and contained only terminal tasks during inspection. Recheck before changing state, back up configuration/SQLite/state and retain the old image; current requirements, Dockerfile and compose have no difference from the deployed source. Reuse pinned existing image dependencies, replace only reviewed source, use the existing compose service and mounts. Verify running file hashes and health without submitting tasks. Roll back image/state before user activity if startup fails; never restore an older database over newly created user jobs. Do not merge main or overwrite the unrelated dirty README in the original local checkout.

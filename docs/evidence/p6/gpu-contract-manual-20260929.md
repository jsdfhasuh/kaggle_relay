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

## Authorized production deployment, 2026-09-29

The paragraphs above describe the pre-deployment observation. The user subsequently explicitly authorized production deployment to reviewed code `3975d1595c7d34c62e20fb2030145eb59763cb9f`. No live training/model acceptance was authorized or performed by the agent in this rollout.

- Production `/docker_volume/kaggle_relay`: clean detached HEAD at that exact code SHA. Evidence-only commits after this point are not runtime versions.
- Existing service/container: `kaggle_relay-kaggle-relay-1`, original Compose ports, environment and `/data` mount retained.
- New actual image: `sha256:0bd0e945d7b14b4f76c8e091d0e0179c0cbdc0021636912e41927eb5f074d1b1`, tag `kaggle-relay-reviewed:3975d15-20260929t011624z`.
- Retained rollback image: `kaggle-relay-rollback:20260929t011624z`, resolving to original `sha256:07bfc8f35deb099802edacde3751bbf1cb8551032d4a703ece27fef02f194fe8`.
- Image built from that existing image using only `COPY app /app/app` and `docker build --pull=false`. No dependency installation. All installed distributions remained identical, including Kaggle 2.2.4, FastAPI 0.141.1, Uvicorn 0.53.0, HTTPX 0.28.1 and Pydantic 2.13.5.
- All 28 running application files matched reviewed Git source bytes. Local authenticated health and public HTTPS health passed; public HTTP 200. Health lists old contracts plus P6 v1/v2 and `dino_cuda_features_cpu_coreset_v1`. Container running, restart count 0.
- No active jobs before or after; canceled 33, complete 135, failed 103. Full jobs rows ordered by `job_id` unchanged. Configuration/auth/upload-intent bytes unchanged. No Dataset/Kernel submission.

Restricted backup (0700): `/docker_volume/kaggle_relay-deploy-backups/gpu-contract-20260929T011624Z`. Contains configuration, auth configuration, upload intents, consistent SQLite snapshots (final `relay-final.db`), private container metadata, build log, per-file hashes, `deployment.json`, image overrides and executable `rollback.sh`. Existing 58 GB task/artifact storage remains in place. No sensitive material is in this repository.

Deployment command, after the final idle check and consistent backup:

```sh
cd /docker_volume/kaggle_relay
git checkout --detach 3975d1595c7d34c62e20fb2030145eb59763cb9f
docker compose -f docker-compose.yml -f /docker_volume/kaggle_relay-deploy-backups/gpu-contract-20260929T011624Z/image.override.yml up -d --no-build --no-deps kaggle-relay
```

The first pre-switch verification attempt exited 1 because its SQL assumed `id` rather than actual `job_id`. The old service had been stopped; it was immediately restarted without a source/image switch. After inspecting the real schema, the corrected rollout and verification exited 0. This interruption is retained as evidence rather than omitted.

Selected verified runtime SHA256:

| File | SHA256 |
|---|---|
| `app/main.py` | `5b8f9b5731889cdbd1e13884e7d889087d2b1a048b4651075314d63df85763a4` |
| `app/worker.py` | `51e7aca565312bf97469315dcb316647f4d7429d26ee5e272afac1a9f1792bb2` |
| `app/dinov2_artifacts.py` | `d6c6ab97f03856eb8e0e92305361e3d9364e19076c2d98c11fa2070c19796227` |
| `app/dinov2_training_policy.py` | `04fc76e393899ae0fc2b1b3b0eddd2ecbb2cd8cd6173d3ebd895f02b8254bc22` |

For authorized rollback, execute the backup directory's `rollback.sh`; it refuses a dirty checkout, restores the old code/image using the original Compose config and rollback image override, and deliberately does not restore the DB over potentially new user jobs. For subsequent container recreation retain the explicit image override; a default build would not preserve this dependency provenance.

Status: **production startup/capability/source verification PASS; real GPU numerical, training, delivery and desktop acceptance PENDING user**. No EXE build or release.

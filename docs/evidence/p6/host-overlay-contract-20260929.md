# Host-overlay DINO GPU contract

Baseline `f5e7c18`, runtime previously `3975d15`. Adds explicit capability `dino_cuda_host_torch210_cu128_cpu_coreset_v2` with `torch_index=host_cu128` and `environment=verified_host_overlay_v1`. Preserves the original cu126 GPU policy and old CPU/YOLO/CNN/DINO contracts. Strictly requires P6 v2, Kernel GPU enabled and CPU deployment execution; no uploaded snapshot is rewritten.

Application uses a verified host GPU stack with private supplementary packages, rather than installing another Torch/CUDA stack. Host Torch 2.10.0+cu128 / Torchvision 0.25.0+cu128 / NumPy 2.0.2 and real GPU operations must pass before installation. Independent ORT remains free of host training packages. New capability cannot inherit the old GPU qualification.

Tests: existing Relay interpreter, `pytest tests/test_dinov2_training_policy.py tests/test_dinov2_p6.py tests/test_dinov2_artifacts.py tests/test_relay_api.py -q --disable-warnings`. Policy/health tests cover both profiles and immutable request preservation. Real GPU training remains user acceptance.

Result: **176 passed, 2 warnings, exit 0, 31.79 seconds**. No cloud job or production mutation performed. The application also rejects a service advertising only the older cu126 GPU capability before preparing uploads.

Production update is pending explicit authorization for this new commit. Concrete rollout: recheck clean `/docker_volume/kaggle_relay`, no active jobs, back up configuration/auth/consistent SQLite/intents; retain actual old image; build source-only image from current image with no dependency installation; use existing Compose config/mounts with a fixed image override; verify all running source hashes, dependency inventory, authenticated public health and both GPU profiles. Retain rollback script/image and never restore an old DB over new jobs. No training submission, business data upload or EXE build.

## Actual authorized deployment, 2026-09-29

The user subsequently authorized deployment of **`770db9c87e5441b0d2959efcf30a5309e6eb1974`**. Production now runs that exact detached commit, upgraded from `3975d1595c7d34c62e20fb2030145eb59763cb9f`. Later evidence commits are not runtime code versions.

- Container: `kaggle_relay-kaggle-relay-1`.
- Actual image: `sha256:4ba6492a07176a9e1bc1e167bdf17545f128889f2da2416ea8e5f4fde4ee3b61`; tag `kaggle-relay-reviewed:770db9c-20260929t022527z`.
- Source-only image from previous image `sha256:0bd0e945d7b14b4f76c8e091d0e0179c0cbdc0021636912e41927eb5f074d1b1`, `COPY app /app/app`, `docker build --pull=false`. No package install. All dependency distributions unchanged.
- All 28 running source files match reviewed Git bytes. Authenticated local health and public HTTPS health PASS (HTTP 200), advertising both GPU profiles and unchanged legacy contracts.
- No active jobs before/after; all 273 historical jobs rows unchanged (canceled 33, complete 135, failed 105). Config/auth/intents unchanged. No cloud task submitted.
- Verified SHA256: `app/dinov2_training_policy.py` = `7ac36621a9696cf52926484095424bc4a98621b734ba6692fd8b829195dc2408`; `app/schemas.py` = `0547790960105c45eabfb5744a9e6a636b139cd997baa047d658ea1a0fd76cbf`.

Restricted backup (0700): `/docker_volume/kaggle_relay-deploy-backups/host-overlay-20260929T022527Z`. Includes final consistent `relay-final.db`, config/auth/intents, container metadata, build log, dependency/file verification, fixed image override and executable rollback script. Old image retained as `kaggle-relay-rollback:20260929t022527z`.

Actual switch command, exit 0:

```sh
cd /docker_volume/kaggle_relay
git checkout --detach 770db9c87e5441b0d2959efcf30a5309e6eb1974
docker compose -f docker-compose.yml -f /docker_volume/kaggle_relay-deploy-backups/host-overlay-20260929T022527Z/image.override.yml up -d --no-build --no-deps kaggle-relay
```

Rollback uses the backup directory's `rollback.sh`, refuses a dirty tree, switches old code/image, and deliberately does not overwrite the current DB with a backup. Keep the fixed image override for subsequent recreation. Deployment verification PASS; actual GPU model acceptance remains with the user.

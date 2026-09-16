# PatchCore dual output transport

2026-09-16. Branch: codex/patchcore-dual-artifacts.

Allow the two additional canonical root outputs deployment_model.pt and pt_export.json
in the PatchCore Kaggle download whitelist. Preserve required model.ckpt and existing
metadata; old CKPT-only jobs remain valid. Do not download duplicate nested checkpoints
or unrelated PT files. The desktop continues strict manifest/RunPlan/hash validation.

Verification: full pytest tests suite, 128 passed, 1 warning. The archive tests use
fixture bytes to verify transport only, never native model or EXE compatibility.

The initial branch was not deployed. Following explicit user authorization, production
was updated on 2026-09-16 at 07:27:37 UTC (15:27:37 Asia/Shanghai).

## Production Rollout

- Base: main 49f0c8caac1108b0858a9c03a9b39572f7326330. Preserved its latest idle-account
  dispatch and ten-worker concurrency updates. No unrelated commits were removed.
- Deployed runtime commit: 13908c60b135486279a576fbc0dd9ae7199cf198, fast-forwarded to main.
  This carries the original 2bc986f dual-output change on the current main base.
- Updated-base tests: 156 passed, 1 skipped, 1 warning, 40.48 seconds. The skip is the
  POSIX process-group test on Windows, not a model compatibility test.
- Oracle path and working tree verified: /docker_volume/kaggle_relay, main, clean.
  Used the existing docker-compose.yml and local build, not an unverified GHCR image.
- Deployment: docker compose up -d --build --force-recreate kaggle-relay.
- Container: kaggle_relay-kaggle-relay-1, running, restart count 0 after rollout.
- Image: sha256:467011c015ebf9aef564706a36c4c362a3387872bb125e69ca079948428a3c91.
- Running adapter SHA-256 matches checked-out source:
  ca8c685af0511da6a0f56bb672c7b28985b8c7beec4f62334d326015e1380433.
- Authenticated /v1/health returned 200/ok internally and through the production HTTPS
  endpoint. Unauthenticated health remained 401. Startup logs contained no error lines.
- Actual container filter and temporary fixture ZIP packaging accepted model.ckpt,
  deployment_model.pt and pt_export.json; unrelated and duplicate nested files remained
  excluded. This is a transport smoke test, not real model or Kaggle-download acceptance.
- Existing /data bind mount and environment/auth configuration preserved. Database
  backup: /data/backups/relay-pre-dual-20260916T072639Z.db, integrity_check=ok, 98 jobs.
  Post-deploy database integrity remained ok, with unchanged counts: 36 complete,
  54 failed, 5 canceled, 3 old receiving records. No active/queued training was present.
- Runtime limits preserved: workers=10, assembly_workers=2, account_concurrency=1,
  max_active_jobs=40, max_parallel_uploads=40. No Kaggle task was submitted.

Real Kaggle training/output download and competition EXE compatibility remain PENDING.

# DINO v3 Artifact Transport

Date: 2026-09-18. Branch: `codex/dinov2-artifacts-v3`.

## Protocol

- Authenticated `GET /v1/health` advertises `patchcore_dinov2_v3` in
  `artifact_contracts`. Missing capability is not support.
- Create requests explicitly select that contract with all four frozen identity
  fields. Create/get/list responses preserve the contract and identity.
- Database initialization preserves the new contract across restarts. Legacy
  inference from missing contract fields remains unchanged.
- Only `artifacts/<canonical filename>` is downloaded from Kaggle. Root-level
  files, nested checkpoint copies, diagnostics and other PTs are excluded.
- ZIP entries retain the `artifacts/` prefix. Its manifest inventories relative
  filenames inside that directory, exactly as produced by the cloud bootstrap.

Required files:

```text
artifacts/
  model.ckpt
  com_dinov2_small.pt
  threshold.json
  metrics.json
  runtime_result.json
  resolved_spec.json
  runtime.json
  environment.json
  anomaly_metrics.json
  native_candidate.json
  pt_export.json
  pt_verification.json
  source_artifacts.json
  training_artifacts.json
```

## Validation Boundary

The server enforces the v3 envelope, Anomalib 2.2.0, completed/PASS statuses,
canonical PT/metadata paths, job identity, complete sorted inventory and its
digest. Every archived file must match its declared size and SHA-256. Linked
files and malformed/oversized manifest JSON are rejected. Data is hashed while
streaming into a ZIP64-capable temporary ZIP in the destination directory; only
a complete verified archive replaces the target. Failures remove that temporary
file and do not finalize a successful job or overwrite a prior complete archive.

This is transport validation, not inference validation. The server does not
unpickle PT/CKPT files, install Anomalib/Torch, execute model code or reinterpret
the client score policy. The desktop still validates runtime/spec/threshold/PT
evidence against its frozen RunPlan after downloading. CNN PatchCore and YOLO
contracts keep their existing behavior.

## Verification

Run `python -m pytest tests -q` in this repository. New tests cover the exact
download filter, complete ZIP bytes, every missing file, mismatched hashes,
wrong identity/version/status, traversal/linked paths, malformed manifests,
disk/write/rename failures, API capability discovery, restart persistence and
worker completion/download behavior.

The desktop repository has an opt-in cross-repository fixture test. Set
`PYTHONPATH` to its `tests` directory and `KAGGLE_RELAY_SOURCE` to this repository,
then run `tests/integration/test_relay_dinov2_transport.py`. It generates the
actual native v3 envelope with mock model/verification fixtures, uses this
adapter download filter and ZIP packager, extracts through the real desktop
downloader helper and performs strict, idempotent receipt without loading PT.
It does not submit Kaggle training or prove competition EXE compatibility.

Local results on 2026-09-18 using the desktop `training_platform` interpreter:

- Relay full suite: 201 passed, 1 skipped (POSIX-only process-group test on
  Windows), 1 warning.
- Desktop targeted regression: 553 passed, 82 subtests passed. This covers
  submission, upload/resume/download, DINO inputs/kernel/receipt/native delivery,
  settings and the training entrypoint.
- Final rerun of client negotiation plus cross-repository transport: 19 passed.
  The test used the actual `artifacts/` output prefix and server download filter.

No live Kaggle training, server deployment, container restart or packaged EXE
acceptance was performed.

## Rollout Status

Not deployed. A read-only production health check on 2026-09-18 returned
`status=ok`, `version=0.1.0`, without `artifact_contracts`. That deployment
must not accept DINO v3 submissions from the updated client.

Deployment/restart requires explicit authorization. Before rollout: commit and
push the tested change, inspect `/docker_volume/kaggle_relay` branch/worktree,
container identity and deployment configuration, check active jobs and back up
the database. Preserve local changes and use the existing Compose deployment.
After rollout, verify authenticated health capabilities, running source hashes
and a temporary artifact fixture; do not submit training as a health check.

Do not roll back to an older database initializer while v3 jobs are present:
older code rewrites unknown contract values to the legacy PatchCore contract.
Drain/retain v3 jobs and plan database recovery before any such rollback.

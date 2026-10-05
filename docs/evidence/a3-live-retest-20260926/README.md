# A3 live retest — 2026-09-26

**All six serial Relay cloud cases passed. A3 as a whole remains incomplete.**
The previous Dataset403 preflight blocker and application diagnostic overwrite were resolved in real runs. This report adds evidence only; no main merge, release, submodule change, dependency upgrade or later-phase work.

## Running versions and preservation

- Application `feat/kaggle-job-build-foundation`: `58e33d9ae69486a953cc5bc23664edb37db7dbc2`.
- Deployed Relay `codex/a3-transport-identity-fix`: `e8255b5665c113f8be74359ef3cbad8369d98049`.
- Source-only image `sha256:07bfc8f35deb099802edacde3751bbf1cb8551032d4a703ece27fef02f194fe8`.
- Production `/docker_volume/kaggle_relay`, existing Compose service `kaggle-relay`, container `kaggle_relay-kaggle-relay-1`. Clean checkout and zero active jobs verified before replacement.
- Old version33e361c/image retained. Backup `/docker_volume/kaggle-relay-backups/a3-retest-20260926` (0700) contains configuration, SQLite, job bindings, image details and executable rollback.sh. Check no unrelated active tasks before rollback; rollback restores old source/image without overwriting DB/auth.
- Only app/kaggle_adapter.py changed in deployed business code. Build used existing image, `COPY app /app/app`, `docker build --network=none --pull=false`. pip freeze unchanged.
- Existing deployment command `docker compose up -d --no-build --pull never --force-recreate kaggle-relay` exited0. Verified hashes of **25 actual running container files**, not only host HEAD; see [deployment.json](deployment.json).
- Python3.11.16, Kaggle SDK2.2.4. First SSH fetch connection closed; retry succeeded before deployment. No package install/upgrade.
- Real test identities rechecked: first/jsdfhasuh and mhb/mhbggzh match their configured owners. Actual jobs use existing first principal, limited to first key. Access Token + token_introspection. No account configuration or credential changes, no authorization expansion.
- Final protected auth bytes and old task bindings/statuses are unchanged. Original local Relay README dirty change remains SHA256 `50f3903e06f216f396e043ee0f1552a71ba3c0d50201ec42ff33fa3a15e1623a`.

## Real jobs

| Case | Relay job | Exact Dataset version | Kernel version | Final |
|---|---|---:|---:|---|
| YOLO workers0 | 6f56c95c859146e9b9c66573bd05bada | 2, trusted cache | 1 | COMPLETE |
| CNN quantile workers0 | 55ced6caa73845b290191d4bfeb276e1 | 1, create | 1 | COMPLETE |
| YOLO workers2 | 7885fd31467d4f96885ede8cca287572 | 1, create | 1 | COMPLETE |
| CNN quantile workers2 | fcb66e776c5646f88ce681b1cca9bef6 | 1 | 1 | COMPLETE |
| CNN F1 workers0 | ef7b680ee4374eaf91b8fbc5ae693654 | 1, create | 1 | COMPLETE |
| CNN F1 workers2 | 5070d13eeb064916b8469087db6d7abd | 1, create | 1 | COMPLETE |

See [remote-summary.json](remote-summary.json) for full refs, actual SDK Kernel terminal states, authenticated owner, original intents, exact-version dataset_sources, source hashes and one-push counts. Six new runs total, at most one active test job; no old task rewritten. Each Kernel stayed private.

YOLOw0 revalidated cached Dataset2 bytes, without creating another version. Other candidate operations and accepted intents are recorded explicitly. New Dataset creation now reached successful business response and exact-version content verification. No latest substitution or repeat upload/Kernel submission was used.

Real recursive runtime layout verified: YOLO338 members; CNN quantile31/F1 35 members. Runtime/support bytes passed the frozen content gate before training. CNN entry script fetched through SDK has CRLF→LF normalization: raw SHA differs, normalized text matches exactly. Both are recorded; raw-byte equality is not falsely claimed.

Application completed real download, registration and fresh-process reopen for all six runs. Independent artifact checks found no mismatch: YOLO23 files each, CNN6 core files each. CNN environment confirms Anomalib2.5.1, calibrated_normal_quantile or calibrated_validation_f1 with actual workers0/2. YOLO's existing PT/ONNX outputs are preserved; no PatchCore ONNX work.
YOLOw0 detached its monitor only after confirmed submission, exited, and resumed the same run/job through the actual platform resume entry in a new process. No direct sync substitute or duplicate training.

Data are deterministic synthetic RGB; CNN uses the already agreed random initialized resnet18, layer2/layer3, 64px, coreset0.1, neighbors3, batch2, CKPT. These verify execution, not business detection quality/pretrained backbone effectiveness.

## Verification, cleanup and remaining scope

Same deployed Linux image, network disabled, read-only test mount and isolated storage:

```sh
docker run --rm --network none \
  -v /docker_volume/kaggle_relay/tests:/app/tests:ro \
  -v /docker_volume/kaggle-relay-backups/a3-retest-20260926:/evidence \
  -e RELAY_API_TOKEN=test -e RELAY_STORAGE_DIR=/tmp/a3-tests \
  kaggle_relay-kaggle-relay:a3-e8255b5 python -m pytest \
  tests/test_a3_content_identity.py \
  tests/test_concurrency.py::test_timeout_kills_descendants_after_group_leader_exits \
  -q --junitxml=/evidence/linux-retest.xml
```

Exit0, **52 passed**,2 deprecation warnings. Existing fork test needs cv2; actual container find_spec confirms absent, so NOT_RUN. No dependency installation. Previous Windows346-pass repair evidence is not counted as tests newly run in this batch.

All application prepare/run/audit/reopen commands exited0; YOLOw0 additional resume exited0. Exact arguments, run identities, model/threshold hashes, local receipt evidence and V4 mapping are in the application branch `docs/evidence/kaggle-build-a3/live-retest-20260926/README.md` and local-evidence.json.
Read-only evidence collection initially exited1 for job-list parsing and assuming every job had a new intent; corrected diagnostics use the recorded exact cache version when appropriate. No production business edits or remote mutation occurred during that correction.

Final state: six Relay jobs complete and six actual Kernels COMPLETE; zero active jobs, no cancellation needed. GPU0.146h used/29.854h remaining, TPU20h remaining; free storage55,628,595,200bytes. See [verification-summary.json](verification-summary.json) and [state-check.json](state-check.json).
Application staging cleaned only after successful receipt/archive; all six recorded-submission audits remain verifiable and frozen hashes unchanged. Original payload digests/accepted intents retained; no unknown intent erased. Server material follows existing retention; no historical Dataset deleted.
Raw evidence is protected under `relay-data/diagnostics/a3-retest-20260926` (0700) and local `D:/kaggle-a3-retest-20260926` (user/SYSTEM). Temporary exported first credentials removed, original protected backups retained; repository contains only whitelisted evidence.

**Still NOT_RUN/PENDING:** local no-development-directory `/kaggle`, real-library offline/spawn matrix, frozen EXE, direct-cloud/interleaving, and cv2-dependent Linux fork. Passing this Relay matrix cannot replace them. Stop at A3 review; do not infer overall A3 acceptance or authority for later phases.

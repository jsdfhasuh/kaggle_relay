# A3 live acceptance — 2026-09-26

**Not released for A3.** Two new runs were executed serially. YOLO trained and the real application resumed/reopened the same run, but an independent receiver audit found three overwritten diagnostic images. CNN stopped before Dataset mutation on HTTP 403. No further runs, production business hotfix, main merge, force push or desktop release.

## Reviewed deployment

- Application `feat/kaggle-job-build-foundation`: `3bbb1006eb1fc58b117bb7056ac531823bd1c5b6`.
- Relay `codex/a3-transport-identity-fix`: `33e361c5e0b2d5c8e04bd84261506dbc5f055a41`.
- Production `/docker_volume/kaggle_relay`, container `kaggle_relay-kaggle-relay-1`, existing `docker-compose.yml` service `kaggle-relay`.
- Old HEAD `4d7723373d5d49482a054e480f8f0db986374581`; old/new image IDs in [deployment.json](deployment.json).
- Clean production tree and zero active jobs checked before updating. Host detached at the exact reviewed commit; main untouched.
- Protected backup `/docker_volume/kaggle-relay-backups/a3-20260926` (0700) holds configuration, SQLite backup, old job bindings, image/container details and executable rollback script.
- Source-only overlay of existing image: `FROM kaggle_relay-kaggle-relay:rollback-a3-20260926`, `COPY app /app/app`. Built with `docker build --network=none --pull=false`; no apt/pip install or upgrade. Before/after pip freeze equal.
- First attempt using `FROM sha256:<image-id>` failed image resolution without changing the service. Verified local rollback tag corrected the deployment command only.
- Existing Compose update: `docker compose up -d --no-build --pull never --force-recreate kaggle-relay`, exit0.
- Verified **25 actual container app files**, not only host HEAD: [running-files.json](running-files.json). SDK 2.2.4, health HTTP200/status ok.
- Rollback: first ensure no unrelated active tasks, then execute the protected `rollback.sh`; it restores old auth/image and main checkout, recreates Compose, and does **not** overwrite the current DB. Old image tag retained.
- This evidence-only commit is not deployed. Original local Relay README modification preserved (SHA256 `50f3903e06f216f396e043ee0f1552a71ba3c0d50201ec42ff33fa3a15e1623a`). No Relay AGENTS found.

## Authenticated owner correction

Real Access Token introspection, not quota/authenticate success:

| Test key | Original configured owner | Authenticated subject | Result |
|---|---|---|---|
| first | jsdfhasuh | jsdfhasuh | unchanged; verified |
| mhb | mhb | mhbggzh | backed up; only username corrected; reverified |

No credential replacement or authorization expansion. Historical task bindings/statuses unchanged. The two acceptance jobs used the existing first principal, limited to key first; Dataset and Kernel owner both jsdfhasuh. OAuth and legacy-key live jobs were not exercised in this batch; their regressions are separate evidence.

## Exact jobs and remote results

Full safe fields: [remote-summary.json](remote-summary.json).

| | YOLOw0 | CNN quantilew0 |
|---|---|---|
| Application run | 61742b06-cd65-4d7b-ae3b-4e939136d472 | 9b01485d-f914-4c44-97ca-1fc0bc69833d |
| Relay job | 0c4dae77ba3043b08e5582e0583a08c4 | 59ebab0d18db4cd9843a644286367550 |
| Dataset | jsdfhasuh/yolo-w0-detect-730716a64536 | jsdfhasuh/cnn-quantile-w0-anomaly-e63e9fc61242 (intended) |
| Version | 2, accepted version intent | none; intent not reached |
| Kernel | jsdfhasuh/yolo-w0-yolo-w0-61742b06-tr-20260926-081229-dd0850 | jsdfhasuh/cnn-quantile-w0-cnn-quantil-20260926-081812-bc1142 (not submitted) |
| Kernel version/pushes | 1 / 1 | none / 0 |
| Terminal | complete; actual Kernel COMPLETE | failed; pre-upload403 |

YOLO exact Dataset2 bytes verified; 338-member recursive runtime layout observed, no runtime.zip remaining remotely. Kernel metadata binds Dataset `/2`. Actual execution source and submitted source both SHA256 `34837a79e9c381974ecc82df40e656716d763bffb5d0372744f758e99258c0db`.
Numbered `kernels_pull(ref + '/1')` returned403. Read-only SDK ApiGetKernelRequest instead returned current_version_number=1, matching the only logged version and exact submitted source bytes. No Dataset latest substitution.
Synthetic RGB, existing yolo11n.pt, epoch1, batch2, imgsz128, workers0; real Tesla T4 execution. Existing YOLO best.pt/best.onnx output preserved; no PatchCore ONNX development.
Application detached after waiting_kernel and exited, then a new process used the real resume entry, another reopened history. No new Dataset version, Kernel or duplicate training.

CNN uses the frozen Anomalib2.5.1 contract, synthetic data, random initialized resnet18, quantile/workers0. It never reached runtime or calibration, so no quality/pretrained acceptance claim.
`dataset_status(exact_ref)` repeatedly returned403 under the verified owner. Authenticated mine searches returned no exact Dataset or Kernel. No upload intent, mutation log or push exists. **Do not interpret403 as absence.** Review an explicit create-intent/existence reconciliation design preserving original candidates and unknown outcomes; no change made here.

## Additional receiver finding

The original server `kaggle_output/artifacts` BoxF1/BoxPR/BoxP files match the cloud manifest. Application received copies match the different `runs/..._test` files instead. [final-check.json](final-check.json) records original/test SHA256 pairs.
Application `_copy_result_files` copies canonical artifacts, then copies all diagnostics by basename, overwriting three canonical files. Its AST matches main `ae23165d1fddc220bb269400137f60c93104b8ee`: existing receiver behavior, not this deployment's transport fix. Models match, but full artifact acceptance **FAILS**, despite successful registration/reopen. No onsite fix.
Reproduction/checker and full V4 matrix are in application `docs/evidence/kaggle-build-a3/live-20260926/README.md` and `scripts/verify_kaggle_a3_received_artifacts.py` on its feature branch.

## Tests, resource and evidence disposition

Linux tests used the exact deployed image, `--network none`, read-only reviewed tests and isolated temporary storage; commands invoked `python -m pytest`:

| Selection | Exit/result |
|---|---|
| tests/test_concurrency.py::test_timeout_kills_descendants_after_group_leader_exits | 0; 1 PASS, 6.91s |
| tests/test_a3_content_identity.py | 0; 41 PASS, 2.45s |
| Initial combined selection with tests/test_dino_image_policy.py::test_fork_workers_inherit_decoder | outer command1; collection FAIL missing cv2; fork itself NOT_RUN |

No dependency installation. This is an existing platform fork regression, not DINO migration. App audit/resume/monitor/Relay-contract regressions: 37 PASS + 5 subtests, exit0. [test-results.json](test-results.json) preserves JUnit aggregate data. Previous full-suite numbers are not presented as this batch's execution.

YOLO preview/recorded-submission archive survives successful staging cleanup; 12 remaining frozen inputs match pre-cleanup hashes. CNN 16 frozen entries and failed staging retained. No unknown intent removed (YOLO accepted, CNN no attempted mutation).
Final read-only check: zero active Relay jobs, old bindings/statuses unchanged, only authorized owner config difference, same deployed image. GPU used0.019h/remains29.981h; TPU remains20h; free storage56,002,523,136bytes. YOLO complete, CNN not submitted: no cancellation required. No historical Dataset deleted.
Raw sensitive evidence is protected locally and in server `relay-data/diagnostics/a3-20260926`; no credentials/raw logs committed. Temporary exported client credentials removed after inspection; original protected backups remain.

Remaining YOLOw2, CNN quantilew2/F1w0/F1w2: NOT_RUN after blocker. Direct cloud and interleaving: PENDING. Local isolated `/kaggle`, real-library offline/spawn and frozen EXE: NOT_RUN in this batch. These cannot be replaced by a successful cloud run. Stop at A3 review.

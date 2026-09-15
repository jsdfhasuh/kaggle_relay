# Ten-user concurrency validation — 2026-09-16

## Scope and result

The change separates training waits from file I/O, isolates Kaggle SDK credentials
in subprocesses, schedules work per account, and adds storage admission, upload
backpressure, inactive-upload expiry and bounded logs. The desktop client honors
integer `Retry-After` delays while remaining immediately pauseable.

Validation against the current source:

| Environment | Result |
| --- | --- |
| Windows, isolated Python 3.12.14 environment, full Relay suite | 142 passed; 1 POSIX-only test skipped |
| Oracle, disposable Python 3.11.16 Linux container, full Relay suite | 143 passed |
| Desktop client, `tests/test_relay_upload.py` | 12 passed |

The server suite reports two dependency deprecation warnings. The intended local
`training_platform` conda interpreter was absent; an isolated environment with
Relay requirements was used instead. No desktop GUI or packaged executable
acceptance was performed.

The ten-user scenario exercises real API creation, four-way chunk uploads,
SHA-256 checks and ZIP assembly. Ten mocked long-running training jobs occupy
all job workers, while an additional upload and assembly still finish with the
default I/O executor restricted to two threads. Other tests cover account
balancing and queue cancellation, isolated SDK processes with fake Kaggle APIs,
timeouts including orphaned POSIX descendants, mid-copy/transfer disk pressure,
upload inactivity, cleanup retry/idempotence, schema migration and shutdown.

Linux tests used the existing production image with a read-only source mount,
no network, no published ports, no production data or credentials, two CPUs and
1 GiB of memory. Tests used sparse temporary storage, not multi-GB datasets.
These results establish functional concurrency; they do not measure production
throughput or prove that ten real Kaggle GPU runs will be accepted simultaneously.

## Pending Oracle rollout

Production remains at `cd39e44bc69936b33e0d91e91db04f152d1df703` on clean `main`
under `/docker_volume/kaggle_relay`. Service `kaggle-relay` is managed by the local
`docker-compose.yml`; container `kaggle_relay-kaggle-relay-1` was not restarted.
Authenticated `/v1/health` returned HTTP 200 after validation. Root storage is
174 GiB with approximately 90 GiB available after the authorized 180G expansion.

After explicit deployment authorization:

1. Commit and push the reviewed Relay source, then verify the remote branch,
   worktree and deployment method again. Preserve all unrelated desktop changes.
2. Back up SQLite with its online backup API, and preserve the private environment
   and auth configuration locally on the host with restricted permissions. Keep
   the previous image available for rollback; never print credentials.
3. Update only the relevant environment values: `RELAY_WORKER_COUNT=10`,
   `RELAY_ASSEMBLY_WORKERS=2`, `RELAY_ACCOUNT_CONCURRENCY=1`,
   `RELAY_MAX_ACTIVE_JOBS=40`, `RELAY_MAX_ACTIVE_JOBS_PER_USER=4`,
   `RELAY_MAX_PARALLEL_UPLOADS=40`, `RELAY_MIN_FREE_BYTES=5368709120`,
   `RELAY_MAX_ARCHIVE_BYTES=8589934592`, `RELAY_RECEIVING_RETENTION_HOURS=168`,
   `RELAY_MAX_LOGS_PER_JOB=2000`. Preserve existing credentials, proxy settings
   and completed-result retention of 168 hours.
4. Synchronize the reviewed commit and rebuild with the existing command:
   `docker compose up -d --build --force-recreate kaggle-relay`.
5. Check authenticated health, effective limits, recovered-job state and logs.
   Coordinate a small real-client upload/training check before a full ten-user
   production run. The client retry change also needs distribution to clients.

Each user needs an individual Relay token. Ten workers can run ten jobs only
when their permitted account assignments and Kaggle quotas allow it; jobs sharing
one username queue by default. The current server has eight Relay tokens and ten
credential entries, so user provisioning must match the intended ten users.

The 8 GiB archive limit is a per-request ceiling, not a ten-user capacity promise.
Admission reserves three times the combined input ZIP size and revises extraction
space from ZIP entries. With about 90 GiB free, ten simultaneous 8 GiB uploads
will be rejected by admission; actual dataset sizes and output growth determine
usable concurrency.

On first startup, old incomplete uploads receive a fresh seven-day inactivity
lease. Existing terminal results remain subject to the seven-day retention
policy. Maintenance truncates each job's historical logs to its newest 2,000
entries; expired file deletion cannot be undone by rolling back the image.
Rollback should stop the service, restore the previous image/configuration and
use the consistent database backup if needed, accounting for jobs created since
that backup. No production rollout or cleanup was performed during validation.

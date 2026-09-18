# Kaggle Relay

FastAPI relay for routing Training Platform Kaggle traffic through one Linux server.

## Run

Completed job files are retained for 7 days by default (`RELAY_RETENTION_HOURS=168`).
Job responses include `artifact_expires_at` for available packages and a
`download_unavailable_code` (`not_ready`, `expired`, `missing`, or `inaccessible`).
Only retention cleanup evidence identifies an expired package. Extending retention
does not restore packages already removed. The hourly cleanup waits for active downloads.

```bash
cp .env.example .env
# Set RELAY_API_TOKEN, or configure RELAY_AUTH_CONFIG before starting.
docker compose up --build
```

The Compose files publish Relay only on `127.0.0.1:8000`. Access it locally or
put an HTTPS reverse proxy on the same host in front of that address. The sample
environment uses `RELAY_UI_COOKIE_SECURE=false` for plain-HTTP local access;
production HTTPS deployments must set it to `true`.

Live servers should deploy the current checkout with a forced local rebuild:

```bash
docker compose up -d --build --force-recreate kaggle-relay
```

Do not use `docker-compose.ghcr.yml` with an implicit `latest` image for
recovery-sensitive deploys. If GHCR is required, set `KAGGLE_RELAY_IMAGE` to a
verified tag or digest; see [docs/container-image.md](docs/container-image.md).

Hosts intentionally managed by Watchtower can use the mutable GHCR deployment:

```bash
docker compose -f docker-compose.watchtower.yml up -d --pull always --force-recreate kaggle-relay
```

Only switch after the intended local commits are pushed and the matching
GitHub Action run succeeds. This Compose file keeps `./relay-data:/data` and
uses the same Compose project/service as the local build deployment.

Relay runs a single process with an in-process job queue. Set
`RELAY_WORKER_COUNT` to the number of Kaggle jobs that may execute at the same
time; the default is `10`. Keep Uvicorn at one process because multiple Uvicorn
processes would create separate queues without a shared worker coordinator.
Jobs using different Kaggle datasets can run concurrently. Jobs targeting the
same Kaggle account and dataset are serialized until the Kernel push completes,
so one worker cannot replace the dataset version while another worker submits
its Kernel.

Training waits use a dedicated executor; archive assembly uses a separate pool
(`RELAY_ASSEMBLY_WORKERS=2`), leaving the default executor available for upload
file I/O. SDK requests execute in isolated subprocesses with explicit credential
environments. General commands time out after 300 seconds and transfers after
7200 seconds; shutdown terminates in-flight child processes. No SDK network call
holds the server's environment lock.

`RELAY_ACCOUNT_CONCURRENCY=1` limits running jobs per Kaggle username, including
aliases with different key IDs. Jobs waiting for the same account leave workers
available for other accounts. Recovered remote runs are monitored even if their
count already exceeds the new limit. Account selection balances receiving,
queued and running jobs across eligible accounts, then considers remaining GPU
hours. Existing owner preference and key permissions still apply. Quota lookups
are coalesced per credential for 30 seconds. Remote Kaggle limits and runs started
outside this Relay are not reservations managed by this scheduler.

### Dynamic account scheduling

New clients can send `scheduling_mode: "dynamic"` when creating a job without
an explicit `kaggle_key_id`. Relay accepts and verifies both input archives,
then keeps the job in a shared queue until an authorized account is idle and
has GPU quota. The original owner is not preferred over an idle account.
Busy or temporarily unavailable accounts leave the job queued; completions
wake the scheduler immediately, with a 15-second periodic retry for quota and
configuration changes. Slow quota requests do not block other idle accounts.

The create response includes `assignment_state: "pending"` and an immutable
`eligible_accounts` map of key IDs to usernames. The initial key and owner are
provisional. Dispatch intersects that snapshot with the token's current rights,
atomically claims an account, rewrites only the owners of the dataset/kernel
references, and persists `assignment_state: "bound"` before submission. The
job ID, input archive digests, slugs and frozen RunPlan identity stay unchanged.
The account is never changed again; queued jobs and committed bindings survive
restart. Revoked permissions or a changed account username cannot expand the
original account pool. No Kaggle credentials are returned in these fields.
Progress callbacks keep an internal alias for the script's original kernel
reference, with the same per-job callback token verification after assignment.

Dynamic jobs always upload the dataset ZIP to Relay so any selected account can
receive it; the worker can still reuse a verified cache on the selected account.
One Kaggle username has one slot by default even with multiple credential IDs.
"Idle" refers to Relay-managed queued/running work, including input transfer
and output collection; jobs started separately in the Kaggle website are not
counted. Account permissions remain unchanged by enabling dynamic scheduling.

Older clients/jobs retain `fixed` scheduling and immutable create-time bindings.
An explicit key or legacy single-account authentication also stays fixed. Updated
desktop clients request dynamic scheduling and verify the one permitted binding
transition against the original eligible accounts, archive hashes, slugs and
frozen identity. They then persist the actual references for observation,
resumption, artifact downloads and direct recovery. Distribute the updated
desktop client to enable this protocol; old clients are not silently reassigned.

The default limits are 40 active jobs globally, 4 per Relay token, 8 GiB per
archive, and 40 incoming chunk streams globally (4 per token). A chunk must not
exceed `RELAY_CHUNK_SIZE`, and an archive must not exceed 65,536 chunks. Each
person should have a separate token. Busy upload slots return HTTP 429; resource
admission failures return HTTP 503 with `Retry-After`. Clients must retain the
same job and retry/resume its missing chunks, not create a replacement job after
an uncertain submission response.

Input storage reservations are persisted in SQLite. Creation initially reserves
three times the declared input size for chunks, merged archives and estimated
extraction; assembly revises this using ZIP member sizes and filesystem overhead
before extraction. Materialized data is counted by the filesystem, while future
writes remain reserved. `RELAY_MIN_FREE_BYTES` defaults to 5 GiB. Insufficient
assembly space leaves the task `receiving` with confirmed chunks intact. Training
outputs, SDK temporary files and other services also consume disk, so this input
budget does not replace disk monitoring and capacity planning.
Archive copies check free space between 1 MiB blocks; transfer subprocesses are
stopped if free space falls below the reserve. These checks are not a filesystem
quota and concurrent external writes can still exhaust the disk.

Incomplete uploads have their own inactivity lease
(`RELAY_RECEIVING_RETENTION_HOURS=168`). Accepted or retried chunks renew it;
active body streams are excluded from expiry, with a 60-second idle-body timeout.
Ordinary status polling does not renew the lease. Existing jobs receive a full
lease during the schema upgrade. `upload_expires_at` and `queue_reason` are
additive response fields. Manual pause/resume remains available within the lease.
Expired uploads become failed and their partial files are removed. Terminal
results retain the independent `RELAY_RETENTION_HOURS` policy.

Cleanup records completion once and retries filesystem failures; each job keeps
at most `RELAY_MAX_LOGS_PER_JOB=2000` recent logs, including historical jobs after
the next maintenance pass. Logs have an index for per-job reads. Before upgrading,
back up the SQLite database and confirm the desired retention settings. Existing
environment variables override these defaults: a deployment with
`RELAY_WORKER_COUNT=2` stays at two until its configuration is explicitly changed.
See [the validation and rollout record](docs/concurrency-acceptance-2026-09-16.md)
for the ten-user test scope and the Oracle rollout configuration.

For legacy single-user mode, set `RELAY_API_TOKEN` to a long random value and
provide Kaggle credentials with `KAGGLE_API_TOKEN`,
`KAGGLE_USERNAME`/`KAGGLE_KEY`, or by mounting `/root/.kaggle`.

For multi-user/multi-key mode, set `RELAY_AUTH_CONFIG` to a JSON file path. Each
job is bound to one `kaggle_key_id`; relay tokens can be limited to one key, a
list of keys, or all keys. Set `RELAY_ADMIN_TOKEN` to a custom value of at
least 8 characters to create one dedicated management key:

```text
RELAY_AUTH_CONFIG=/data/auth.json
RELAY_ADMIN_TOKEN=replace-with-a-long-custom-management-key
```

The management key is stored only in the environment. It can log into the web
UI, access all jobs and Kaggle keys, and modify the auth configuration. When it
is configured, ordinary relay tokens cannot modify auth configuration, even if
they can access all Kaggle keys. Without `RELAY_ADMIN_TOKEN`, an all-key relay
token keeps the previous administrator behavior for backward compatibility.
Eight characters is the enforced minimum, not the recommended production
length. Use a longer random value. Relay limits failed UI login, Bearer, and
callback authentication attempts by source; tune `RELAY_AUTH_FAILURE_LIMIT`,
`RELAY_AUTH_FAILURE_WINDOW_SECONDS`, and `RELAY_AUTH_LOCKOUT_SECONDS` if needed.

Mutating requests authenticated by a UI cookie, including UI login itself, must
carry a same-origin `Origin` header. HTTPS deployments behind a proxy should set
`RELAY_PUBLIC_ORIGIN` to the browser-visible origin and keep
`RELAY_UI_COOKIE_SECURE=true`. If Relay must read `X-Forwarded-For` for per-source
login limits, list only immediate trusted proxy addresses in
`RELAY_TRUSTED_PROXY_IPS`; forwarded headers from other peers are ignored. An
optional `RELAY_UI_SESSION_SECRET` can provide an independent stable
cookie-signing key; otherwise Relay uses the dedicated management key or a
compatibility admin key. Plain-HTTP development can explicitly set
`RELAY_UI_COOKIE_SECURE=false`.

Example auth configuration:

```json
{
  "relay_tokens": [
    {"id": "user-a", "token": "user-a-token", "allowed_kaggle_key_ids": ["ka"]}
  ],
  "kaggle_keys": [
    {"id": "ka", "username": "alice", "key": "alice-kaggle-key"}
  ]
}
```

New Kaggle key entries added through the admin API/UI must include `username`
along with `key`, `api_token`, or `config_dir`. Existing entries can be edited
with `PATCH /v1/auth/kaggle-keys/{id}` or the admin UI. Relay uses that username
to validate uploaded Kaggle metadata before submitting jobs, so it must be the
Kaggle profile URL slug, not the display name. Tokens beginning with `KGAT_`
should be stored as `api_token`; the `key` field is only for the legacy
username/key credential shape.
Use `POST /v1/kaggle/account/probe?kaggle_key_id=<id>` or the admin UI
"强校验" button to verify the token can create a private dataset under the
configured username. This creates a tiny probe dataset and then deletes it.

## API

All `/v1/*` requests require:

```text
Authorization: Bearer <relay token or management key>
```

Main endpoints:

- `GET /v1/health`
- `GET /v1/kaggle/account`
- `POST /v1/kaggle/account/probe`
- `GET /v1/kaggle/accounts`
- `PATCH /v1/auth/kaggle-keys/{id}`
- `POST /v1/jobs`
- `PUT /v1/jobs/{job_id}/archives/{dataset|kernel}/chunks/{index}`
- `POST /v1/jobs/{job_id}/complete`
- `POST /v1/jobs/{job_id}/cancel`
- `POST /v1/jobs/{job_id}/progress`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/artifacts.zip`
- `GET /v1/jobs/{job_id}/dataset.zip`
- `DELETE /v1/jobs/{job_id}`

### Resumable parallel uploads

The job list has a separate **下载提交包** (download submitted dataset) link.
It downloads the original `dataset.zip`, including `dataset-metadata.json` and
`payload.zip`; uploaded images and annotations remain inside `payload.zip`.
This is distinct from the training results ZIP and is not a full annotation
project backup. The browser streams the download without buffering it in JavaScript.
The endpoint uses the same job authorization as status/results, checks the
original SHA-256, and holds the job lock through transfer so deletion and retention
cleanup wait. It works before training completes, once assembly has finished.
Job responses expose `can_download_dataset` and `dataset_download_unavailable_code`
(`not_ready`, `expired`, `missing`, `invalid`, or `inaccessible`). The endpoint returns
409 for incomplete/invalid archives, 410 after retention cleanup, 404 for missing
archives, and 503 for inaccessible files. A job reusing a Kaggle dataset may have
no local input archive; Relay does not retrieve another job's dataset automatically.
Submitted packages follow the existing job retention policy (7 days by default).

Job creation/status responses include `chunk_size`, archive sizes and SHA-256
digests, `accepted_chunks`, and `max_parallel_uploads` (currently 4). Clients
must preserve the original archives and job ID, query status again after an
interruption, and send only missing chunks using the job's original chunk size.
Clients talking to a gateway without `max_parallel_uploads` should use one
upload connection. Pausing an upload does not cancel the remote job.

Different chunks can be received concurrently. Committing a chunk rechecks
authorization and job status under the submission lock; duplicate chunks with
the same size/digest are idempotent and conflicting digests return 409. Only
complete, verified chunks are recorded. Incomplete uploads return 409 from
`complete` and remain resumable in `receiving`, rather than becoming failed.
If a `complete` response is lost, query the original job before taking another
action; never create a replacement job merely because a request timed out.

When `POST /v1/jobs` omits `kaggle_key_id`, Relay binds the job to the only
allowed key, or for multi-key tokens first prefers an allowed key whose username
matches the requested owner and has remaining GPU quota. If that owner has no
remaining quota, Relay may choose another allowed key with remaining quota and
rewrite `dataset_ref`, `kernel_ref`, and uploaded Kaggle metadata to that key's
username. Supplying `kaggle_key_id` still forces that specific key when the token
is allowed to use it, with the same owner rewrite if needed.

PatchCore clients freeze four identity fields before creating a job:
`dataset_id`, `identity_sha256`, `run_id`, and `run_identity_sha256`. They must
be sent together or all omitted. Relay persists and returns the exact values from
create, get, and list responses. Clients must stop before archive upload if any
returned identity value is missing or differs from the frozen request. Legacy
YOLO requests that omit all four fields remain supported.

Clients also send `artifact_contract` as `yolo`, `patchcore`, or
`patchcore_dinov2_v3`. Relay persists
and returns it, then uses it for both Kaggle output filtering and required-file
validation. PatchCore jobs require `model.ckpt`, `threshold.json`,
`anomaly_metrics.json`, `environment.json`, and `training_artifacts.json`;
YOLO jobs continue to require `best.pt`. For pre-contract jobs, Relay derives
`patchcore` only when all four frozen identity fields are present, otherwise it
uses `yolo`.

DINO v3 clients first check the authenticated `/v1/health` response
`artifact_contracts` list. They explicitly request `patchcore_dinov2_v3` and
send all four frozen identity fields. This contract downloads only the
14 canonical files in `artifacts/`, including `com_dinov2_small.pt`, its
verification records and the training checkpoint. Relay verifies the manifest
format/status, exact inventory, sizes, SHA-256 hashes and job identity before
atomically publishing the ZIP. It never loads PT/CKPT files or imports the
training runtime. The desktop still performs full frozen-contract validation.
See [DINO v3 transport](docs/dinov2-v3-artifacts.md) for compatibility and rollout.

## Kernel Progress Callback

`POST /v1/jobs` may include `callback_token_sha256`. Store only the SHA-256
hash in Relay, then put the raw callback token in the generated Kaggle script.

The Kaggle script can report progress with:

```text
POST /v1/jobs/{job_id}/progress
Authorization: Bearer <raw-callback-token>
```

If the generated Kaggle script does not know the Relay `job_id`, report by
`kernel_ref` instead:

```text
POST /v1/jobs/by-kernel/progress
Authorization: Bearer <raw-callback-token>
```

Example body:

```json
{
  "kernel_ref": "owner/kernel-slug",
  "epoch": 4,
  "epochs": 300,
  "message": "[Epoch 4/300] Loss: 2.667",
  "mAP50": 0.992
}
```

Relay maps `epoch / epochs` into the existing kernel progress range and stores
the payload in `kernel_status` plus `recent_logs`.

To stop a running Kaggle job, call `POST /v1/jobs/{job_id}/cancel`. Kaggle does
not expose a public hard-stop API for a running kernel, so cancellation is
cooperative: generated `train.py` should read `cancel_requested` from the next
progress callback response, save useful outputs under `/kaggle/working`, and
exit cleanly. Relay then downloads available artifacts and marks the job
`canceled`.

## Reverse Proxy

Use HTTPS, allow large request bodies, and set upload/proxy timeouts to at least
one hour for multi-GB payloads.

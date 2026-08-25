# Kaggle Relay

FastAPI relay for routing Training Platform Kaggle traffic through one Linux server.

## Run

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
time; the default is `2`. Keep Uvicorn at one process because multiple Uvicorn
processes would create separate queues without a shared worker coordinator.
Jobs using different Kaggle datasets can run concurrently. Jobs targeting the
same Kaggle account and dataset are serialized until the Kernel push completes,
so one worker cannot replace the dataset version while another worker submits
its Kernel.

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
- `DELETE /v1/jobs/{job_id}`

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

Clients also send `artifact_contract` as `yolo` or `patchcore`. Relay persists
and returns it, then uses it for both Kaggle output filtering and required-file
validation. PatchCore jobs require `model.ckpt`, `threshold.json`,
`anomaly_metrics.json`, `environment.json`, and `training_artifacts.json`;
YOLO jobs continue to require `best.pt`. For pre-contract jobs, Relay derives
`patchcore` only when all four frozen identity fields are present, otherwise it
uses `yolo`.

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

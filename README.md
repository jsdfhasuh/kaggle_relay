# Kaggle Relay

FastAPI relay for routing Training Platform Kaggle traffic through one Linux server.

## Run

```bash
cp .env.example .env
docker compose up --build
```

For legacy single-user mode, set `RELAY_API_TOKEN` to a long random value and
provide Kaggle credentials with `KAGGLE_API_TOKEN`,
`KAGGLE_USERNAME`/`KAGGLE_KEY`, or by mounting `/root/.kaggle`.

For multi-user/multi-key mode, set `RELAY_AUTH_CONFIG` to a JSON file path. Each
job is bound to one `kaggle_key_id`; relay tokens can be limited to one key, a
list of keys, or all keys:

```json
{
  "relay_tokens": [
    {"id": "admin", "token": "admin-token", "allowed_kaggle_key_ids": "*"},
    {"id": "user-a", "token": "user-a-token", "allowed_kaggle_key_ids": ["ka"]}
  ],
  "kaggle_keys": [
    {"id": "ka", "username": "alice", "key": "alice-kaggle-key"}
  ]
}
```

## API

All `/v1/*` requests require:

```text
Authorization: Bearer <RELAY_API_TOKEN>
```

Main endpoints:

- `GET /v1/health`
- `GET /v1/kaggle/account`
- `POST /v1/jobs`
- `PUT /v1/jobs/{job_id}/archives/{dataset|kernel}/chunks/{index}`
- `POST /v1/jobs/{job_id}/complete`
- `POST /v1/jobs/{job_id}/progress`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/artifacts.zip`
- `DELETE /v1/jobs/{job_id}`

When a token can access exactly one Kaggle key, `POST /v1/jobs` may omit
`kaggle_key_id` and Relay will bind the job to that key. When a token can access
multiple keys or `"*"`, the create request must include `kaggle_key_id`.

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

## Reverse Proxy

Use HTTPS, allow large request bodies, and set upload/proxy timeouts to at least
one hour for multi-GB payloads.

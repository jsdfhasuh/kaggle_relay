# Kaggle Relay

FastAPI relay for routing Training Platform Kaggle traffic through one Linux server.

## Run

```bash
cp .env.example .env
docker compose up --build
```

Set `RELAY_API_TOKEN` to a long random value. Provide Kaggle credentials with
`KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`/`KAGGLE_KEY`, or by mounting
`/root/.kaggle`.

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

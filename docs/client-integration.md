# Relay Client Integration

This document describes how a training client should submit Kaggle jobs through
Relay, let Relay decide whether the dataset archive must be uploaded, and report
live progress from the Kaggle notebook/script back to Relay.

## Overview

The client is responsible for packaging the dataset and kernel locally, then
submitting both archives through Relay. Relay remains a transport and execution
gateway: it stores uploaded archives under the job directory, pushes the dataset
and kernel to Kaggle, waits for completion, and packages output artifacts.

Relay decides whether dataset chunks must be uploaded for each job. The client
must not use local dataset cache state to skip uploads. The only upload decision
source is the `dataset_upload_required` field returned by `POST /v1/jobs`.

Kernel/notebook chunks are always uploaded because each run uses a fresh kernel.

## Client Flow

1. Build the dataset payload and compute a stable `payload_hash`.
2. Generate a per-job callback token:

   ```python
   import hashlib
   import secrets

   callback_token = secrets.token_urlsafe(32)
   callback_token_sha256 = hashlib.sha256(callback_token.encode("utf-8")).hexdigest()
   ```

3. Generate the Kaggle kernel/notebook with these values embedded:

   ```python
   RELAY_CALLBACK_URL = "https://kaggle.oracle.19970219.xyz"
   RELAY_KERNEL_REF = "<owner/kernel-slug>"
   RELAY_CALLBACK_TOKEN = "<raw callback_token>"
   ```

4. Create `dataset.zip` and `kernel.zip`.
5. Call `POST /v1/jobs` with archive sizes, archive SHA-256 hashes, `payload_hash`,
   and `callback_token_sha256`.
6. If the response has `dataset_upload_required=true`, upload dataset chunks.
   If the field is absent, default to `true` for old Relay compatibility.
7. Always upload kernel chunks.
8. Call `POST /v1/jobs/{job_id}/complete`.
9. Poll `GET /v1/jobs/{job_id}` until `status` is `complete` or `failed`.
10. Download `GET /v1/jobs/{job_id}/artifacts.zip` after completion.

Reference client pseudocode:

```python
job = relay.create_job(
    dataset_ref=dataset_ref,
    kernel_ref=kernel_ref,
    dataset_archive_sha256=sha256_file(dataset_zip),
    dataset_size=dataset_zip.stat().st_size,
    kernel_archive_sha256=sha256_file(kernel_zip),
    kernel_size=kernel_zip.stat().st_size,
    chunk_size=chunk_size,
    payload_hash=payload_hash,
    callback_token_sha256=callback_token_sha256,
)

if job.get("dataset_upload_required", True):
    relay.upload_archive_chunks(job["job_id"], "dataset", dataset_zip)

relay.upload_archive_chunks(job["job_id"], "kernel", kernel_zip)
relay.complete_job(job["job_id"])
```

## Create Job Request

`POST /v1/jobs`

Required auth:

```text
Authorization: Bearer <RELAY_API_TOKEN>
```

Request body:

```json
{
  "dataset_ref": "owner/dataset-slug",
  "kernel_ref": "owner/kernel-slug",
  "dataset_archive_sha256": "<64 hex chars>",
  "kernel_archive_sha256": "<64 hex chars>",
  "dataset_size": 889214867,
  "kernel_size": 7396,
  "chunk_size": 67108864,
  "payload_hash": "<stable dataset payload hash>",
  "callback_token_sha256": "<sha256(raw callback token)>"
}
```

Important field rules:

- `payload_hash` represents dataset content only. Do not include kernel/notebook
  content in this hash.
- `dataset_archive_sha256` and `dataset_size` must be computed from the real
  local `dataset.zip`, even if Relay later says dataset upload can be skipped.
- `callback_token_sha256` is optional but recommended. Relay stores only this
  hash. The raw callback token is embedded in the generated Kaggle script.
- Do not send a client-side `reuse_dataset` field. Relay makes that decision.

Relevant response fields:

```json
{
  "job_id": "7f198a0951434e81ad73f71ef8a57fb1",
  "dataset_cache_hit": true,
  "dataset_upload_required": false,
  "callback_enabled": true,
  "accepted_chunks": {
    "dataset": [],
    "kernel": []
  }
}
```

Client behavior:

- `dataset_upload_required=true`: upload dataset chunks and kernel chunks.
- `dataset_upload_required=false`: skip dataset chunks and upload kernel chunks.
- field missing: treat as `true`.

## Kernel Progress Callback

The generated Kaggle script should report progress directly to Relay. This avoids
depending on `kaggle kernels logs`, which may lag behind or return empty output
while the Kaggle web UI already shows live logs.

Recommended endpoint for generated kernels:

```text
POST /v1/jobs/by-kernel/progress
Authorization: Bearer <raw callback token>
Content-Type: application/json
```

Example body:

```json
{
  "kernel_ref": "owner/kernel-slug",
  "epoch": 5,
  "epochs": 300,
  "message": "[Epoch 5/300] Loss: 2.66",
  "loss": 2.66,
  "mAP50": 0.992,
  "mAP50_95": 0.770,
  "precision": 0.991,
  "recall": 1.0
}
```

Relay finds the latest job for `kernel_ref`, verifies the callback token against
`callback_token_sha256`, then updates:

- `progress`
- `kernel_status`
- `kaggle_output`
- `recent_logs`

Relay maps `epoch / epochs` into the existing kernel progress range from `60` to
`80`. If `remote_progress` is provided directly, Relay uses that value instead.

## Notebook Code Template

Embed this helper in the generated Kaggle script or notebook:

```python
import json
import urllib.request

RELAY_CALLBACK_URL = "https://kaggle.oracle.19970219.xyz"
RELAY_KERNEL_REF = "owner/kernel-slug"
RELAY_CALLBACK_TOKEN = "raw-client-generated-callback-token"


def relay_progress(payload):
    payload = dict(payload)
    payload["kernel_ref"] = RELAY_KERNEL_REF

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{RELAY_CALLBACK_URL}/v1/jobs/by-kernel/progress",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {RELAY_CALLBACK_TOKEN}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except Exception as exc:
        print(f"[WARN] relay callback failed: {exc}", flush=True)
```

Call it wherever the training code emits structured progress:

```python
payload = {
    "epoch": epoch,
    "epochs": EPOCHS,
    "message": message,
    "loss": total_loss,
    "mAP50": metric_data.get("mAP50"),
    "mAP50_95": metric_data.get("mAP50_95"),
    "precision": metric_data.get("precision"),
    "recall": metric_data.get("recall"),
}

print("TRAINING_PLATFORM_PROGRESS " + json.dumps(payload, ensure_ascii=False), flush=True)
relay_progress(payload)
```

Keep the existing `TRAINING_PLATFORM_PROGRESS` print. Relay still supports
Kaggle CLI log polling as a fallback.

## Security Notes

- Do not embed the main `RELAY_API_TOKEN` in Kaggle code.
- Use one callback token per job.
- Store only `sha256(callback_token)` in Relay via `callback_token_sha256`.
- Treat the raw callback token as write-only for progress on the matching job.
- Use HTTPS for `RELAY_CALLBACK_URL`.

## Testing Checklist

Client unit tests should cover:

- Relay returns `dataset_upload_required=true`: client uploads dataset and kernel.
- Relay returns `dataset_upload_required=false`: client uploads only kernel.
- Relay omits `dataset_upload_required`: client defaults to uploading dataset.
- Local cache says ready but Relay returns `true`: client uploads dataset.
- Local cache is empty but Relay returns `false`: client skips dataset.
- Kernel chunks are uploaded on every run.
- `POST /v1/jobs` includes `callback_token_sha256`.
- Generated notebook contains `RELAY_CALLBACK_URL`, `RELAY_KERNEL_REF`, and
  `RELAY_CALLBACK_TOKEN`.
- Callback body includes `kernel_ref`, `epoch`, `epochs`, and `message`.


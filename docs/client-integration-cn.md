# Kaggle Relay 客户端接入指南

这份文档给训练客户端用，覆盖一次训练任务从本地打包、上传到 Relay、
Kaggle 执行、进度回调、协作式停止、下载产物的完整接入方式。

## 接入结论

- 客户端负责生成 `dataset.zip` 和 `kernel.zip`，并通过 Relay 分块上传。
- Relay 返回的 `dataset_upload_required` 是是否上传 dataset 的唯一依据。
- Kernel 每次都要上传，因为每次训练都应该提交新的 Kaggle kernel。
- `train.py` 必须接入 progress callback；如果要支持停止训练，`train.py`
  还必须读取 callback 返回的 `cancel_requested`。
- 不要把主 `RELAY_API_TOKEN` 放进 `train.py`。Kaggle 代码里只放每个 job
  独立生成的 callback token。

## 推荐流程

1. 客户端先确定使用哪个 Kaggle key。
   - 最稳妥：明确传 `kaggle_key_id`。
   - 如果客户端要自动选择，可以先调 `GET /v1/kaggle/accounts` 看每个 key
     的 username 和 quota，然后用选中的 username 生成 refs。
2. 用最终 owner 生成：
   - `dataset_ref = "<owner>/<dataset-slug>"`
   - `kernel_ref = "<owner>/<kernel-slug>"`
3. 生成一次性 callback token：

   ```python
   import hashlib
   import secrets

   callback_token = secrets.token_urlsafe(32)
   callback_token_sha256 = hashlib.sha256(callback_token.encode("utf-8")).hexdigest()
   ```

4. 生成 `train.py`，把这些值写进去：

   ```python
   RELAY_CALLBACK_URL = "https://kaggle.oracle.19970219.xyz"
   RELAY_KERNEL_REF = "<owner>/<kernel-slug>"
   RELAY_CALLBACK_TOKEN = "<raw callback_token>"
   ```

5. 创建 `dataset.zip` 和 `kernel.zip`。
6. 调 `POST /v1/jobs` 创建 job。
7. 检查创建响应：
   - 后续所有流程都以响应里的 `job_id`、`dataset_ref`、`kernel_ref`、
     `kaggle_key_id` 为准。
   - 如果响应里的 `kernel_ref` 和 `train.py` 内嵌的 `RELAY_KERNEL_REF`
     不一致，说明 Relay 做了 owner fallback。此时不要继续上传旧
     `kernel.zip`；应该取消这个 job，按响应里的 owner 重新生成
     `train.py` / `kernel.zip`，再创建新 job。
8. 如果 `dataset_upload_required=true`，上传 dataset chunks。
   如果字段缺失，按 `true` 处理。
9. 总是上传 kernel chunks。
10. 调 `POST /v1/jobs/{job_id}/complete` 开始 Relay 组装并提交到 Kaggle。
11. 轮询 `GET /v1/jobs/{job_id}`，直到 `status` 为 `complete`、`canceled`
    或 `failed`。
12. 如果 `can_download=true`，下载
    `GET /v1/jobs/{job_id}/artifacts.zip`。
13. 客户端重启或断线恢复时，继续使用同一个 Relay Token 调
    `GET /v1/jobs?active=true`，找回这个 token 绑定的未结束 job，再根据
    `accepted_chunks`、`status` 和 `can_download` 决定继续上传、继续轮询、
    取消，或下载产物。

## 创建 Job

请求：

```http
POST /v1/jobs
Authorization: Bearer <RELAY_API_TOKEN>
Content-Type: application/json
```

Body:

```json
{
  "kaggle_key_id": "ka",
  "dataset_ref": "owner/dataset-slug",
  "kernel_ref": "owner/kernel-slug",
  "dataset_archive_sha256": "<dataset.zip sha256>",
  "kernel_archive_sha256": "<kernel.zip sha256>",
  "dataset_size": 889214867,
  "kernel_size": 7396,
  "chunk_size": 67108864,
  "payload_hash": "<stable dataset payload hash>",
  "callback_token_sha256": "<sha256(raw callback token)>"
}
```

关键规则：

- `payload_hash` 只代表 dataset 内容，不要包含 `train.py` 或 kernel 内容。
- `dataset_archive_sha256` / `kernel_archive_sha256` 必须来自真实 zip 文件。
- `callback_token_sha256` 强烈建议传；否则 Kaggle 侧无法用安全的
  callback token 上报进度和读取停止信号。
- 如果传了 `kaggle_key_id`，Relay 会绑定这个 key；如果不传，Relay 会按
  owner 和 GPU quota 自动选择可用 key。

典型响应：

```json
{
  "job_id": "7f198a0951434e81ad73f71ef8a57fb1",
  "kaggle_key_id": "ka",
  "dataset_ref": "owner/dataset-slug",
  "kernel_ref": "owner/kernel-slug",
  "status": "receiving",
  "dataset_upload_required": true,
  "callback_enabled": true,
  "accepted_chunks": {
    "dataset": [],
    "kernel": []
  }
}
```

## 查询和恢复已有 Job

客户端可以按当前 Relay Token 和状态查询 job：

```http
GET /v1/jobs?active=true
Authorization: Bearer <RELAY_API_TOKEN>
```

规则：

- 普通 relay token 只能查询和操作自己创建的 job。不要把 raw token 放到
  URL 查询参数里，只放在 `Authorization: Bearer ...` 里。
- admin / `allowed_kaggle_key_ids="*"` token 按当前权限返回所有可见 job。
- 如果要以某个普通 token 的身份恢复，就直接用那个普通 token 作为
  `Authorization`。
- `active=true` 表示只返回未终止 job，即不是 `complete`、`canceled`、
  `failed` 的 job。
- 也可以用 `status` 做精确过滤，例如
  `GET /v1/jobs?status=receiving&status=waiting_kernel`。`status` 也支持
  逗号写法：`status=receiving,waiting_kernel`。

恢复建议：

- `receiving`：读取响应里的 `accepted_chunks`，只补传缺失 chunk。
- `assembling` / `queued` / `uploading_dataset` / `waiting_dataset` /
  `pushing_kernel` / `waiting_kernel` / `downloading_output`：不要新建重复 job；
  继续轮询 `GET /v1/jobs/{job_id}`。
- `complete` 或 `canceled` 且 `can_download=true`：下载 artifacts。
- `failed`：按 `error` 判断是否需要重新创建 job。

## 上传 Chunks

每个 chunk 使用：

```http
PUT /v1/jobs/{job_id}/archives/{dataset|kernel}/chunks/{index}
Authorization: Bearer <RELAY_API_TOKEN>
X-Chunk-Sha256: <chunk sha256>
X-Chunk-Size: <chunk bytes>
Content-Type: application/octet-stream
```

客户端行为：

- `dataset_upload_required=true`：上传 dataset chunks 和 kernel chunks。
- `dataset_upload_required=false`：跳过 dataset chunks，只上传 kernel chunks。
- kernel chunks 永远上传。
- chunk 可以断点续传；重复上传同 index 且 checksum 一致会被当作 duplicate。

上传完成后调用：

```http
POST /v1/jobs/{job_id}/complete
Authorization: Bearer <RELAY_API_TOKEN>
```

## train.py 进度回调

推荐使用 by-kernel callback，因为 Kaggle 代码通常不知道 Relay `job_id`：

```http
POST /v1/jobs/by-kernel/progress
Authorization: Bearer <raw callback token>
Content-Type: application/json
```

`train.py` helper 示例：

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
            body = resp.read()
        return json.loads(body.decode("utf-8") or "{}")
    except Exception as exc:
        print(f"[WARN] relay callback failed: {exc}", flush=True)
        return {}


def relay_should_stop(payload):
    response = relay_progress(payload)
    return bool(response.get("cancel_requested"))
```

训练循环里至少每个 epoch 调一次：

```python
payload = {
    "epoch": epoch,
    "epochs": EPOCHS,
    "message": message,
    "loss": float(loss),
    "mAP50": metrics.get("mAP50"),
    "mAP50_95": metrics.get("mAP50_95"),
    "precision": metrics.get("precision"),
    "recall": metrics.get("recall"),
}

print("TRAINING_PLATFORM_PROGRESS " + json.dumps(payload, ensure_ascii=False), flush=True)

if relay_should_stop(payload):
    save_checkpoint_to_kaggle_working()
    print("[INFO] relay cancel requested; checkpoint saved", flush=True)
    raise SystemExit(0)
```

长 epoch 任务建议每 N step 也检查一次，避免用户点停止后要等很久才退出。

## 停止训练

客户端或 UI 调：

```http
POST /v1/jobs/{job_id}/cancel
Authorization: Bearer <RELAY_API_TOKEN>
```

Relay 不会硬杀 Kaggle kernel。Kaggle 公共接口没有可靠的运行中硬停止能力，
所以这里是协作式停止：

1. Relay 把 job 标记为 `cancel_requested`。
2. 下一次 `train.py` progress callback 会收到：

   ```json
   {
     "status": "cancel_requested",
     "cancel_requested": true,
     "cancel_reason": "cancel requested"
   }
   ```

3. `train.py` 保存 checkpoint / 重要输出到 `/kaggle/working`。
4. `train.py` 用 `SystemExit(0)` 正常退出。
5. Relay 等 Kaggle kernel 结束后下载 output，打包 artifact，并把 job 标记为
   `canceled`。

如果旧版 `train.py` 没接入这个检查，Relay 只能记录取消请求，Kaggle 上的训练
仍会继续跑到脚本自己结束。

## 轮询状态和下载产物

状态查询：

```http
GET /v1/jobs/{job_id}
Authorization: Bearer <RELAY_API_TOKEN>
```

客户端应该重点看：

- `status`
- `progress`
- `recent_logs`
- `can_download`
- `download_unavailable_reason`
- `cancel_requested`

终态：

- `complete`：正常完成。
- `canceled`：用户请求停止，脚本正常退出；如果有 artifact 仍可下载。
- `failed`：异常失败。

下载：

```http
GET /v1/jobs/{job_id}/artifacts.zip
Authorization: Bearer <RELAY_API_TOKEN>
```

只有 `can_download=true` 时才下载。

## 最小客户端伪代码

```python
job = relay.create_job(
    kaggle_key_id=kaggle_key_id,
    dataset_ref=dataset_ref,
    kernel_ref=kernel_ref,
    dataset_archive_sha256=sha256_file(dataset_zip),
    kernel_archive_sha256=sha256_file(kernel_zip),
    dataset_size=dataset_zip.stat().st_size,
    kernel_size=kernel_zip.stat().st_size,
    chunk_size=chunk_size,
    payload_hash=dataset_payload_hash,
    callback_token_sha256=callback_token_sha256,
)

if job["kernel_ref"] != kernel_ref:
    relay.cancel_job(job["job_id"])
    raise RuntimeError("kernel_ref changed; rebuild train.py with returned refs and create a new job")

if job.get("dataset_upload_required", True):
    relay.upload_archive_chunks(job["job_id"], "dataset", dataset_zip)

relay.upload_archive_chunks(job["job_id"], "kernel", kernel_zip)
relay.complete_job(job["job_id"])

while True:
    status = relay.get_job(job["job_id"])
    if status["status"] in {"complete", "canceled", "failed"}:
        break
    time.sleep(10)

if status.get("can_download"):
    relay.download_artifacts(job["job_id"], "artifacts.zip")
```

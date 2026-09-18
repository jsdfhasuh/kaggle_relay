from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


JobStatus = Literal[
    "receiving",
    "assembling",
    "queued",
    "uploading_dataset",
    "waiting_dataset",
    "pushing_kernel",
    "waiting_kernel",
    "cancel_requested",
    "downloading_output",
    "complete",
    "canceled",
    "failed",
]


class CreateJobRequest(BaseModel):
    kaggle_key_id: str = ""
    scheduling_mode: Literal["fixed", "dynamic"] = "fixed"
    dataset_ref: str
    kernel_ref: str
    dataset_archive_sha256: str = Field(min_length=64, max_length=64)
    kernel_archive_sha256: str = Field(min_length=64, max_length=64)
    dataset_size: int = Field(ge=0)
    kernel_size: int = Field(ge=0)
    chunk_size: int = Field(gt=0)
    payload_hash: str = ""
    callback_token_sha256: str = ""
    dataset_id: str = ""
    identity_sha256: str = ""
    run_id: str = ""
    run_identity_sha256: str = ""
    artifact_contract: Literal["", "yolo", "patchcore", "patchcore_dinov2_v3"] = ""

    @model_validator(mode="after")
    def validate_frozen_identity(self):
        fields = (
            "dataset_id",
            "identity_sha256",
            "run_id",
            "run_identity_sha256",
        )
        values = {
            field: str(getattr(self, field, "") or "").strip()
            for field in fields
        }
        if any(values.values()) and not all(values.values()):
            raise ValueError(
                "PatchCore job identity must contain all frozen fields"
            )
        for field, value in values.items():
            setattr(self, field, value)
        artifact_contract = str(self.artifact_contract or "").strip().lower()
        if not artifact_contract:
            artifact_contract = "patchcore" if all(values.values()) else "yolo"
        if artifact_contract in {"patchcore", "patchcore_dinov2_v3"} and not all(values.values()):
            raise ValueError(
                "PatchCore artifact contract requires all frozen identity fields"
            )
        if artifact_contract == "yolo" and any(values.values()):
            raise ValueError(
                "YOLO artifact contract cannot contain PatchCore frozen identity"
            )
        self.artifact_contract = artifact_contract
        return self


class JobResponse(BaseModel):
    job_id: str
    kaggle_key_id: str = ""
    scheduling_mode: Literal["fixed", "dynamic"] = "fixed"
    assignment_state: Literal["pending", "bound"] = "bound"
    eligible_accounts: dict[str, str] = Field(default_factory=dict)
    dataset_ref: str
    kernel_ref: str
    status: JobStatus
    queue_reason: str = ""
    upload_expires_at: Optional[float] = None
    progress: float
    dataset_status: str = ""
    kernel_status: str = ""
    kaggle_output: str = ""
    error: str = ""
    payload_hash: str = ""
    dataset_id: str = ""
    identity_sha256: str = ""
    run_id: str = ""
    run_identity_sha256: str = ""
    artifact_contract: Literal["yolo", "patchcore", "patchcore_dinov2_v3"] = "yolo"
    callback_enabled: bool = False
    created_at: float
    updated_at: float
    completed_at: Optional[float] = None
    cancel_requested: bool = False
    cancel_requested_at: Optional[float] = None
    cancel_reason: str = ""
    artifact_path: str = ""
    can_download: bool = False
    artifact_size: Optional[int] = None
    artifact_filename: str = ""
    download_unavailable_reason: str = ""
    download_unavailable_code: str = ""
    artifact_expires_at: Optional[float] = None
    can_download_dataset: bool = False
    dataset_download_unavailable_code: str = ""
    dataset_cache_hit: bool = False
    dataset_upload_required: bool = True
    accepted_chunks: dict[str, list[int]]
    chunk_size: int = 64 * 1024 * 1024
    dataset_size: int = 0
    kernel_size: int = 0
    dataset_archive_sha256: str = ""
    kernel_archive_sha256: str = ""
    max_parallel_uploads: int = 4
    recent_logs: list[str] = []


class JobProgressRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: str = "progress"
    message: str = ""
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    remote_progress: Optional[float] = Field(default=None, ge=0, le=100)
    metrics: dict[str, Any] = Field(default_factory=dict)
    log: str = ""


class ChunkResponse(BaseModel):
    job_id: str
    archive_type: Literal["dataset", "kernel"]
    index: int
    size: int
    sha256: str
    accepted: bool = True
    duplicate: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    storage_dir: str
    free_bytes: int
    artifact_contracts: list[str] = Field(default_factory=lambda: ["yolo", "patchcore", "patchcore_dinov2_v3"])


class UiLoginRequest(BaseModel):
    token: str = Field(min_length=1)


class CreateKaggleKeyRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    username: str = ""
    key: str = ""
    api_token: str = ""
    config_dir: str = ""


class UpdateKaggleKeyRequest(BaseModel):
    username: str = ""
    key: str = ""
    api_token: str = ""
    config_dir: str = ""


class CreateRelayTokenRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    token: str = Field(min_length=16)
    allowed_kaggle_key_ids: list[str] = Field(default_factory=list)
    allow_all_kaggle_keys: bool = False

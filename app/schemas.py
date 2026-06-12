from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


JobStatus = Literal[
    "receiving",
    "assembling",
    "queued",
    "uploading_dataset",
    "waiting_dataset",
    "pushing_kernel",
    "waiting_kernel",
    "downloading_output",
    "complete",
    "failed",
]


class CreateJobRequest(BaseModel):
    dataset_ref: str
    kernel_ref: str
    dataset_archive_sha256: str = Field(min_length=64, max_length=64)
    kernel_archive_sha256: str = Field(min_length=64, max_length=64)
    dataset_size: int = Field(ge=0)
    kernel_size: int = Field(ge=0)
    chunk_size: int = Field(gt=0)
    payload_hash: str = ""
    callback_token_sha256: str = ""


class JobResponse(BaseModel):
    job_id: str
    dataset_ref: str
    kernel_ref: str
    status: JobStatus
    progress: float
    dataset_status: str = ""
    kernel_status: str = ""
    kaggle_output: str = ""
    error: str = ""
    payload_hash: str = ""
    callback_enabled: bool = False
    artifact_path: str = ""
    dataset_cache_hit: bool = False
    dataset_upload_required: bool = True
    accepted_chunks: dict[str, list[int]]
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

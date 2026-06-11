from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    artifact_path: str = ""
    accepted_chunks: dict[str, list[int]]
    recent_logs: list[str] = []


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


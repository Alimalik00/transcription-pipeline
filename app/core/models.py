"""
Pydantic models used across the API and service layers.
Keeping them in one place avoids circular imports and makes
the data contract easy to find.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Segment(BaseModel):
    """A single chunk of transcribed speech with timing info."""
    id: int
    start: float     # seconds from beginning of audio
    end: float       # seconds from beginning of audio
    text: str
    confidence: Optional[float] = None  # not all engines expose this


class TranscriptResult(BaseModel):
    """Full transcript returned once a job is complete."""
    job_id: str
    source_file: str
    language: str
    duration_seconds: float
    full_text: str
    segments: List[Segment]
    backend: str


class JobResponse(BaseModel):
    """Returned immediately after a file is uploaded."""
    job_id: str
    status: JobStatus
    message: str


class StatusResponse(BaseModel):
    """Returned when the client polls for job progress."""
    job_id: str
    status: JobStatus
    error: Optional[str] = None

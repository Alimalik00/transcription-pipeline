"""
Simple in-memory job store.

Good enough for a single-process setup and easy to reason about.
In production you would replace this with Redis or a database table
so job state survives restarts and works across multiple worker processes.
"""

import uuid
from typing import Dict, Optional

from app.core.models import JobStatus, TranscriptResult


class Job:
    def __init__(self, file_path: str, original_name: str):
        self.id: str = str(uuid.uuid4())
        self.file_path: str = file_path
        self.original_name: str = original_name
        self.status: JobStatus = JobStatus.PENDING
        self.result: Optional[TranscriptResult] = None
        self.error: Optional[str] = None


# Shared dict that lives for the lifetime of the process
_store: Dict[str, Job] = {}


def create_job(file_path: str, original_name: str) -> Job:
    job = Job(file_path=file_path, original_name=original_name)
    _store[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _store.get(job_id)


def update_status(job_id: str, status: JobStatus, error: str = None):
    job = _store.get(job_id)
    if job:
        job.status = status
        if error:
            job.error = error


def save_result(job_id: str, result: TranscriptResult):
    job = _store.get(job_id)
    if job:
        job.result = result
        job.status = JobStatus.DONE

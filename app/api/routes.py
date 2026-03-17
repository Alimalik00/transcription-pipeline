"""
REST API routes.

Three endpoints cover the full lifecycle of a transcription job:
  POST /transcribe    -> upload a file, get back a job ID
  GET  /transcribe/{job_id}  -> poll job status
  GET  /transcript/{job_id}  -> fetch the finished result
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.job_store import create_job, get_job, save_result, update_status
from app.core.models import JobResponse, JobStatus, StatusResponse, TranscriptResult
from app.services.backends import get_backend
from app.services.transcription_service import run_transcription

logger = logging.getLogger(__name__)
router = APIRouter()

# Load the backend once at module level so it is ready before the first request
_backend = get_backend(model_size=settings.WHISPER_MODEL)


@router.post("/transcribe", response_model=JobResponse, status_code=202)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept an audio file and kick off a transcription job.
    Returns immediately with a job ID so the client does not have to wait.
    """
    # Validate format before saving anything to disk
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted: {settings.SUPPORTED_FORMATS}",
        )

    # Save upload to the configured directory
    dest = settings.UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Register job in the store
    job = create_job(file_path=str(dest), original_name=file.filename)
    logger.info("Job created: %s for file: %s", job.id, file.filename)

    # Run transcription in the background so this response returns fast
    background_tasks.add_task(_process_job, job.id, str(dest), file.filename)

    return JobResponse(
        job_id=job.id,
        status=JobStatus.PENDING,
        message="Job accepted. Poll /api/transcribe/{job_id} for status.",
    )


@router.get("/transcribe/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Check whether a job is pending, processing, done, or failed."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return StatusResponse(job_id=job.id, status=job.status, error=job.error)


@router.get("/transcript/{job_id}", response_model=TranscriptResult)
async def get_transcript(job_id: str):
    """Fetch the full transcript once the job status is 'done'."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Transcript not ready. Current status: {job.status}",
        )

    return job.result


def _process_job(job_id: str, file_path: str, original_name: str):
    """
    Background worker function. Called by FastAPI's BackgroundTasks.
    Handles status transitions and catches any errors so the job
    always ends in a terminal state (done or failed).
    """
    update_status(job_id, JobStatus.PROCESSING)
    try:
        result = run_transcription(
            job_id=job_id,
            file_path=file_path,
            original_name=original_name,
            backend=_backend,
        )
        save_result(job_id, result)
        logger.info("Job %s completed successfully", job_id)
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        update_status(job_id, JobStatus.FAILED, error=str(exc))

"""
Orchestrates the full transcription flow for a single job.

Steps:
  1. Normalize audio to 16kHz mono WAV
  2. Check if the file is long enough to need chunking
  3. Transcribe (single pass or chunked)
  4. Merge chunk results and correct timestamps
  5. Build and return the final TranscriptResult
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from app.core.config import settings
from app.core.models import Segment, TranscriptResult
from app.services.audio_handler import (
    get_duration,
    normalize_to_wav,
    split_into_chunks,
)
from app.services.backends import TranscriptionBackend

logger = logging.getLogger(__name__)

# Files shorter than this are transcribed in one shot
SHORT_FILE_THRESHOLD_SECS = 30


def run_transcription(
    job_id: str,
    file_path: str,
    original_name: str,
    backend: TranscriptionBackend,
) -> TranscriptResult:
    """
    Main entry point called by the background task worker.
    Returns a fully populated TranscriptResult.
    """

    src = Path(file_path)
    # Isolated temp dir per job so parallel jobs never touch each other's files
    work_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))

    try:
        # Step 1: normalize to WAV regardless of input format
        wav_path = normalize_to_wav(src, out_dir=work_dir)
        total_duration = get_duration(wav_path)

        logger.info("Job %s | duration: %.1fs", job_id, total_duration)

        if total_duration <= SHORT_FILE_THRESHOLD_SECS:
            # Short enough to transcribe in a single pass
            raw = backend.transcribe(wav_path)
            segments = _build_segments(raw["segments"], offset=0.0)
            language = raw.get("language", "unknown")
        else:
            # Split and process each chunk, then merge
            segments, language = _transcribe_chunked(wav_path, work_dir, backend)

        full_text = " ".join(s.text for s in segments)

        return TranscriptResult(
            job_id=job_id,
            source_file=original_name,
            language=language,
            duration_seconds=round(total_duration, 2),
            full_text=full_text,
            segments=segments,
            backend=backend.name,
        )

    finally:
        # Clean up temp files regardless of success or failure
        shutil.rmtree(work_dir, ignore_errors=True)


def _transcribe_chunked(
    wav_path: Path,
    work_dir: Path,
    backend: TranscriptionBackend,
) -> tuple:
    """
    Split the WAV, transcribe each chunk, merge results.
    Timestamps from each chunk are shifted by the chunk's start offset
    so the final segments reflect position in the full file.
    """
    chunks = split_into_chunks(wav_path, out_dir=work_dir / "chunks")

    all_segments: List[Segment] = []
    detected_language = "unknown"
    seg_id_counter = 0

    for chunk in chunks:
        logger.info("Processing chunk %d (%.1fs - %.1fs)", chunk.index, chunk.start_sec, chunk.end_sec)
        raw = backend.transcribe(chunk.path)

        if detected_language == "unknown":
            detected_language = raw.get("language", "unknown")

        # Shift each segment timestamp by where this chunk starts in the full file
        for raw_seg in raw["segments"]:
            all_segments.append(
                Segment(
                    id=seg_id_counter,
                    start=round(raw_seg["start"] + chunk.start_sec, 3),
                    end=round(raw_seg["end"] + chunk.start_sec, 3),
                    text=raw_seg["text"].strip(),
                    confidence=raw_seg.get("confidence"),
                )
            )
            seg_id_counter += 1

    # Remove segments that are obvious duplicates from the overlap window
    merged = _dedupe_overlap(all_segments)
    return merged, detected_language


def _build_segments(raw_segments: list, offset: float) -> List[Segment]:
    """Convert raw backend output to typed Segment objects."""
    return [
        Segment(
            id=i,
            start=round(seg["start"] + offset, 3),
            end=round(seg["end"] + offset, 3),
            text=seg["text"].strip(),
            confidence=seg.get("confidence"),
        )
        for i, seg in enumerate(raw_segments)
    ]


def _dedupe_overlap(segments: List[Segment]) -> List[Segment]:
    """
    Remove segments that are likely duplicates from chunk overlaps.
    Simple heuristic: if two consecutive segments have the same text,
    keep the one with the earlier start time and drop the other.
    """
    if not segments:
        return segments

    deduped = [segments[0]]
    for seg in segments[1:]:
        prev = deduped[-1]
        if seg.text.lower().strip() == prev.text.lower().strip():
            # Same text appeared twice due to overlap, skip the duplicate
            logger.debug("Removing duplicate segment: '%s'", seg.text)
            continue
        deduped.append(seg)

    return deduped

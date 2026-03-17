"""
Handles everything audio-format related before transcription starts.

Two responsibilities:
  1. Normalize any supported format to 16kHz mono WAV via ffmpeg
  2. Split long WAV files into overlapping chunks so the transcription
     engine never has to deal with arbitrarily large inputs
"""

import logging
import math
import os
import subprocess
import wave
from pathlib import Path
from typing import List, NamedTuple

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioChunk(NamedTuple):
    index: int
    start_sec: float
    end_sec: float
    path: Path


def validate_format(filename: str) -> bool:
    """Return True if the file extension is in our supported list."""
    ext = Path(filename).suffix.lower()
    return ext in settings.SUPPORTED_FORMATS


def get_duration(wav_path: Path) -> float:
    """
    Read duration straight from the WAV header.
    No need to decode the entire file just to get length.
    """
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def normalize_to_wav(src: Path, out_dir: Path) -> Path:
    """
    Convert any supported audio file to 16kHz mono WAV.

    ffmpeg handles the heavy lifting here. The output spec is strict:
    16000 Hz sample rate, 1 channel, 16-bit signed PCM. This is what
    Whisper expects and what gives the best transcription accuracy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_normalized.wav"

    cmd = [
        "ffmpeg", "-y",          # overwrite output if it exists
        "-i", str(src),
        "-ar", "16000",          # 16kHz sample rate
        "-ac", "1",              # mono
        "-sample_fmt", "s16",    # 16-bit signed PCM
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg error: %s", result.stderr)
        raise RuntimeError(f"Audio conversion failed: {result.stderr}")

    logger.info("Normalized %s -> %s", src.name, out_path.name)
    return out_path


def split_into_chunks(wav_path: Path, out_dir: Path) -> List[AudioChunk]:
    """
    Split a WAV file into fixed-length chunks with a small overlap.

    The overlap (default 1s) means the end of chunk N and the start of
    chunk N+1 share a second of audio. This prevents words from being
    silently dropped at the boundary. Duplicate text is removed during
    the merge step in the transcription service.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = get_duration(wav_path)
    chunks: List[AudioChunk] = []

    chunk_dur = settings.CHUNK_DURATION
    overlap = settings.CHUNK_OVERLAP
    step = chunk_dur - overlap

    starts = list(range(0, math.ceil(total), step))

    for idx, start in enumerate(starts):
        end = min(start + chunk_dur, total)
        chunk_path = out_dir / f"chunk_{idx:04d}.wav"

        # Use ffmpeg to extract the slice without re-encoding
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            str(chunk_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to extract chunk %d: %s", idx, result.stderr)
            continue

        chunks.append(AudioChunk(index=idx, start_sec=start, end_sec=end, path=chunk_path))
        logger.debug("Chunk %d: %.1fs -> %.1fs", idx, start, end)

    return chunks

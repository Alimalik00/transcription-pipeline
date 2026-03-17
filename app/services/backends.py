"""
Transcription backends.

The base class defines the interface. Whisper is the default implementation.
To swap engines (Google, AWS, Azure), just write a new class that
implements transcribe() and plug it in via get_backend().

Nothing else in the codebase knows which engine is running.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from app.core.models import Segment

logger = logging.getLogger(__name__)


class TranscriptionBackend(ABC):
    """Every engine must implement this. That is the whole contract."""

    @abstractmethod
    def transcribe(self, wav_path: Path) -> dict:
        """
        Accept a WAV file path, return a dict with:
          - segments: list of dicts with id, start, end, text
          - language: detected language code (e.g. 'en')
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class WhisperBackend(TranscriptionBackend):
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        logger.info("Loading Whisper model: %s", model_size)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"faster-whisper-{self._model_size}"

    def transcribe(self, wav_path: Path) -> dict:
        logger.info("Transcribing: %s", wav_path.name)
        segments_raw, info = self._model.transcribe(str(wav_path))
        segments = []
        for i, seg in enumerate(segments_raw):
            segments.append({
                "id": i,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "confidence": None,
            })
        return {
            "segments": segments,
            "language": info.language,
        }

class MockBackend(TranscriptionBackend):
    """
    Fake backend for local dev and testing when Whisper is not installed.
    Returns plausible-looking segments so the full pipeline can be tested
    end to end without a real model.
    """

    @property
    def name(self) -> str:
        return "mock"

    def transcribe(self, wav_path: Path) -> dict:
        logger.info("[MOCK] Transcribing: %s", wav_path.name)
        return {
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0,  "end": 3.5,  "text": "Hello and welcome to the call.", "confidence": 0.97},
                {"id": 1, "start": 3.5,  "end": 7.2,  "text": "Today we are going over the project update.", "confidence": 0.95},
                {"id": 2, "start": 7.2,  "end": 12.0, "text": "The pipeline is working as expected.", "confidence": 0.93},
            ],
        }


def get_backend(model_size: str = "base") -> TranscriptionBackend:
    """
    Return the appropriate backend.
    Falls back to the mock if Whisper is not installed.
    """
    try:
        return WhisperBackend(model_size=model_size)
    except ImportError:
        logger.warning("openai-whisper not found. Falling back to MockBackend.")
        return MockBackend()

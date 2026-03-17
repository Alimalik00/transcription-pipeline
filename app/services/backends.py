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
    """
    Uses openai-whisper (local, no API key needed).
    Install with: pip install openai-whisper
    """

    def __init__(self, model_size: str = "base"):
        # Model is loaded once and reused across requests.
        # Loading is slow (a few seconds) so we do it at startup, not per request.
        import whisper
        logger.info("Loading Whisper model: %s", model_size)
        self._model = whisper.load_model(model_size)
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"whisper-{self._model_size}"

    def transcribe(self, wav_path: Path) -> dict:
        logger.info("Transcribing with Whisper: %s", wav_path.name)

        # verbose=False keeps whisper quiet so our own logs stay clean
        result = self._model.transcribe(str(wav_path), verbose=False)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "id": seg["id"],
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
                # Whisper does not expose per-segment confidence directly
                "confidence": None,
            })

        return {
            "segments": segments,
            "language": result.get("language", "unknown"),
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

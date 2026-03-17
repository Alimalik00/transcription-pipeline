"""
Tests for the transcription pipeline.

Uses the MockBackend so no real model or audio file is needed.
Run with: pytest tests/
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from app.core.models import Segment, TranscriptResult
from app.services.backends import MockBackend
from app.services.transcription_service import _dedupe_overlap, _build_segments
from app.utils.exporters import to_srt, to_vtt


# ------------------------------------------------------------------ helpers

def make_wav(path: Path, duration_secs: float = 5.0, sample_rate: int = 16000):
    """Write a minimal valid WAV file for testing."""
    n = int(sample_rate * duration_secs)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            val = int(1000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames += struct.pack("<h", val)
        wf.writeframes(bytes(frames))


def sample_result() -> TranscriptResult:
    return TranscriptResult(
        job_id="test-123",
        source_file="test.wav",
        language="en",
        duration_seconds=12.0,
        full_text="Hello world. This is a test.",
        segments=[
            Segment(id=0, start=0.0, end=4.0, text="Hello world."),
            Segment(id=1, start=4.0, end=8.0, text="This is a test."),
        ],
        backend="mock",
    )


# ------------------------------------------------------------------ tests

def test_mock_backend_returns_segments(tmp_path):
    wav = tmp_path / "audio.wav"
    make_wav(wav)
    backend = MockBackend()
    result = backend.transcribe(wav)
    assert "segments" in result
    assert "language" in result
    assert len(result["segments"]) > 0


def test_mock_backend_segment_structure(tmp_path):
    wav = tmp_path / "audio.wav"
    make_wav(wav)
    backend = MockBackend()
    result = backend.transcribe(wav)
    for seg in result["segments"]:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert seg["end"] > seg["start"]


def test_build_segments_applies_offset():
    raw = [{"id": 0, "start": 0.0, "end": 3.0, "text": "hello", "confidence": 0.9}]
    segs = _build_segments(raw, offset=30.0)
    assert segs[0].start == 30.0
    assert segs[0].end == 33.0


def test_dedupe_removes_consecutive_duplicates():
    segs = [
        Segment(id=0, start=0.0, end=3.0, text="hello"),
        Segment(id=1, start=29.0, end=32.0, text="hello"),   # duplicate from overlap
        Segment(id=2, start=32.0, end=36.0, text="world"),
    ]
    result = _dedupe_overlap(segs)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert result[1].text == "world"


def test_srt_export_format():
    result = sample_result()
    srt = to_srt(result)
    assert "00:00:00,000 --> 00:00:04,000" in srt
    assert "Hello world." in srt


def test_vtt_export_starts_with_webvtt():
    result = sample_result()
    vtt = to_vtt(result)
    assert vtt.startswith("WEBVTT")


def test_full_text_joins_segments():
    result = sample_result()
    assert "Hello world." in result.full_text
    assert "This is a test." in result.full_text

"""
Export helpers for turning a TranscriptResult into subtitle formats.

SRT and WebVTT are the two most common formats used by video editors,
captioning tools, and search indexers so it makes sense to support both.
"""

from app.core.models import TranscriptResult


def _format_time_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def to_srt(result: TranscriptResult) -> str:
    """Return the transcript formatted as an SRT subtitle file."""
    lines = []
    for seg in result.segments:
        lines.append(str(seg.id + 1))  # SRT indices are 1-based
        lines.append(f"{_format_time_srt(seg.start)} --> {_format_time_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")   # blank line between blocks
    return "\n".join(lines)


def to_vtt(result: TranscriptResult) -> str:
    """Return the transcript formatted as a WebVTT subtitle file."""
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{_format_time_vtt(seg.start)} --> {_format_time_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)

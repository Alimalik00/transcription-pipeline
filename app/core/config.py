"""
Central config. Values are pulled from environment variables so the
app behaves differently in dev vs production without code changes.
"""

import os
from pathlib import Path


class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # Where uploaded files land temporarily before processing
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/transcription/uploads"))

    # Whisper model size: tiny, base, small, medium, large
    # Smaller = faster but less accurate. 'base' is a good starting point.
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

    # Max file size accepted (bytes). Default is 100 MB.
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))

    # How many seconds per audio chunk when splitting long files
    CHUNK_DURATION: int = int(os.getenv("CHUNK_DURATION", 30))

    # Overlap between chunks so words at boundaries are not cut off
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 1))

    # Formats the pipeline accepts
    SUPPORTED_FORMATS: list = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"]

    def __init__(self):
        # Make sure upload directory exists at startup
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()

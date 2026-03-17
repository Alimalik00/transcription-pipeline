# Transcription Pipeline

A REST API that converts audio files into timestamped transcripts using OpenAI Whisper.

---

## What it does

- Accepts audio uploads (WAV, MP3, M4A, OGG, FLAC, AAC)
- Normalizes any format to 16kHz mono WAV before processing
- Splits long files into overlapping chunks to handle recordings of any length
- Returns a timestamped transcript per segment in JSON, SRT, or WebVTT
- Processes jobs asynchronously so the API never blocks on a long file

---

## Project structure

```
transcription-pipeline/
  app/
    api/
      routes.py              # REST endpoints
    core/
      config.py              # environment-based config
      job_store.py           # in-memory job state (swap for Redis in prod)
      models.py              # shared Pydantic schemas
    services/
      audio_handler.py       # format normalization and chunking via ffmpeg
      backends.py            # Whisper backend + MockBackend for testing
      transcription_service.py  # orchestrates the full pipeline
    utils/
      exporters.py           # SRT and WebVTT export helpers
  tests/
    test_pipeline.py
  main.py
  requirements.txt
  .env.example
```

---

## Setup

**System dependency (required):**

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**Python dependencies:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment variables:**

```bash
cp .env.example .env
# Edit .env if you want to change model size or upload directory
```

**Run the server:**

```bash
python -m app.main
```

Server starts at `http://localhost:8000`. Docs available at `http://localhost:8000/docs`.

---

## API

### Upload a file

```
POST /api/transcribe
Content-Type: multipart/form-data
Body: file=<audio file>
```

Response (202):
```json
{
  "job_id": "abc-123",
  "status": "pending",
  "message": "Job accepted. Poll /api/transcribe/{job_id} for status."
}
```

### Poll job status

```
GET /api/transcribe/{job_id}
```

Response:
```json
{ "job_id": "abc-123", "status": "processing", "error": null }
```

Possible statuses: `pending`, `processing`, `done`, `failed`

### Get transcript

```
GET /api/transcript/{job_id}
```

Response (200 when done):
```json
{
  "job_id": "abc-123",
  "source_file": "interview.mp3",
  "language": "en",
  "duration_seconds": 142.3,
  "full_text": "Hello and welcome...",
  "segments": [
    { "id": 0, "start": 0.0, "end": 4.2, "text": "Hello and welcome to the call.", "confidence": null }
  ],
  "backend": "whisper-base"
}
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests use the MockBackend so no real model or audio file is needed.

---

## Key design decisions

**Why Whisper?**
It runs locally with no API cost, handles accents and background noise well, and returns segment-level timestamps out of the box. The backend is behind an interface so swapping to Google or AWS Transcribe is a one-class change.

**Why async jobs?**
Transcription takes time. A synchronous endpoint would time out on anything longer than a few seconds. The job queue pattern keeps the API responsive and makes it easy to scale workers independently from the HTTP layer.

**Why chunk at 30 seconds with 1 second overlap?**
Whisper was trained on 30-second audio windows. Feeding it more than that at once degrades accuracy. The 1-second overlap prevents words at chunk boundaries from being dropped, and duplicates from the overlap are removed during the merge step.

**Why ffmpeg for format normalization?**
It handles every codec edge case that pure-Python libraries miss. One subprocess call converts anything to the exact WAV spec the engine expects.

**Why an in-memory job store?**
Simplest thing that works for a single-process setup and easy to reason about during evaluation. In production this would be replaced with Redis so job state survives restarts and works across multiple workers.

---

## Swapping the transcription engine

Open `app/services/backends.py` and add a new class:

```python
class GoogleBackend(TranscriptionBackend):
    @property
    def name(self): return "google-stt"

    def transcribe(self, wav_path):
        # call Google Speech-to-Text here
        ...
```

Then change one line in `app/api/routes.py`:

```python
_backend = GoogleBackend()
```

Nothing else changes.

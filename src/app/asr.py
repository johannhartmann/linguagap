"""
Automatic Speech Recognition - thin shim delegating to the ASR backend.

This module provides backward-compatible functions for file-based transcription
used by the /transcribe_translate endpoint and /asr_smoke test. Streaming
transcription uses the backend directly via streaming.py.

Configuration:
    ASR_BACKEND: Backend to use (default: "whisper")
    ASR_MODEL: Model name or path (passed to backend)
    ASR_DEVICE: cuda or cpu (passed to backend)
    ASR_COMPUTE_TYPE: Compute type for inference (passed to backend)
"""

from typing import Any

from app.backends import get_asr_backend


def transcribe_wav_path(path: str) -> dict[str, Any]:
    """Transcribe a WAV file and return segments with language detection."""
    backend = get_asr_backend()
    result = backend.transcribe_file(path)

    return {
        "language": result.detected_language,
        "language_probability": result.language_probability,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text} for seg in result.segments
        ],
    }

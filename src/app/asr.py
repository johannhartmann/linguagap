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


def get_model():
    """Get the ASR backend (for backward compatibility with warmup code).

    Returns the backend object, which is no longer a raw WhisperModel but
    implements the ASRBackend interface. Code that only calls warmup/load_model
    will work transparently.
    """
    backend = get_asr_backend()
    backend.load_model()
    return backend


def transcribe_wav_path(path: str) -> dict[str, Any]:
    """Transcribe a WAV file and return segments with language detection.

    Uses the underlying faster-whisper model's ability to accept file paths
    directly. This bypasses the backend.transcribe() method to avoid
    requiring soundfile as a dependency for file-based transcription.
    """
    backend = get_asr_backend()
    backend.load_model()

    # Access the underlying model for file-path transcription
    # This is Whisper-specific but this shim is only used by HTTP endpoints
    model = backend._model  # noqa: SLF001
    segments, info = model.transcribe(path)

    result_segments = []
    for seg in segments:
        result_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
        )

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": result_segments,
    }

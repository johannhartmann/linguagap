import os
from typing import Any

from faster_whisper import WhisperModel

ASR_MODEL = os.getenv("ASR_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            ASR_MODEL,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        )
    return _model


def transcribe_wav_path(path: str) -> dict[str, Any]:
    model = get_model()
    segments, info = model.transcribe(path)

    result_segments = []
    for seg in segments:
        result_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": result_segments,
    }

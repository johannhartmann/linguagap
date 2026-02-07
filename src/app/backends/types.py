"""Shared data types for backend interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ASRSegment:
    """A single transcribed segment from ASR."""

    start: float  # seconds relative to input audio start
    end: float
    text: str
    language: str  # ISO 639-1
    confidence: float = 1.0


@dataclass(frozen=True)
class ASRResult:
    """Result from an ASR transcription call."""

    segments: list[ASRSegment] = field(default_factory=list)
    detected_language: str = "unknown"
    language_probability: float = 0.0

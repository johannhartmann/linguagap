"""Pluggable backend factory functions.

Each factory returns a singleton backend instance based on environment variables.
Only the selected backend is imported (lazy), so missing dependencies for other
backends don't cause ImportError.

Environment variables:
    ASR_BACKEND: "whisper" (default)
    MT_BACKEND: "translategemma" (default)
    SUMM_BACKEND: "" (disabled by default), "qwen3" to enable
"""

from __future__ import annotations

import os
from functools import lru_cache

from app.backends.base import ASRBackend, SummarizationBackend, TranslationBackend


@lru_cache(maxsize=1)
def get_asr_backend() -> ASRBackend:
    """Get the configured ASR backend singleton."""
    name = os.getenv("ASR_BACKEND", "whisper")
    if name == "whisper":
        from app.backends.asr.whisper import WhisperASRBackend

        return WhisperASRBackend()
    raise ValueError(f"Unknown ASR backend: {name}")


@lru_cache(maxsize=1)
def get_translation_backend() -> TranslationBackend:
    """Get the configured translation backend singleton."""
    name = os.getenv("MT_BACKEND", "translategemma")
    if name == "translategemma":
        from app.backends.translation.translategemma import TranslateGemmaBackend

        return TranslateGemmaBackend()
    raise ValueError(f"Unknown translation backend: {name}")


@lru_cache(maxsize=1)
def get_summarization_backend() -> SummarizationBackend | None:
    """Get the configured summarization backend singleton, or None if disabled."""
    name = os.getenv("SUMM_BACKEND", "")
    if not name:
        return None
    if name == "qwen3":
        from app.backends.summarization.qwen3 import Qwen3SummarizationBackend

        return Qwen3SummarizationBackend()
    raise ValueError(f"Unknown summarization backend: {name}")

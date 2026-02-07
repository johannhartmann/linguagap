"""Abstract base classes for pluggable ML backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.backends.types import ASRResult, ASRSegment


class ASRBackend(ABC):
    """Abstract interface for automatic speech recognition backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load or download the model."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy inference to prime the GPU."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        """Transcribe audio to text.

        Args:
            audio: Float32 audio samples at 16kHz.
            language: ISO 639-1 language hint (None for auto-detect).
            initial_prompt: Optional prompt to guide transcription style.

        Returns:
            ASRResult with segments and detected language.
        """

    def post_process(self, segments: list[ASRSegment]) -> list[ASRSegment]:
        """Backend-specific post-processing (e.g. hallucination filtering).

        Default implementation: no-op.
        """
        return segments

    def supports_language(self, lang_code: str) -> bool:  # noqa: ARG002
        """Check if this backend supports a given language code."""
        return True

    def get_language_fallback(self, lang_code: str) -> str | None:
        """Get a fallback language code for unsupported languages.

        Returns None if the language should use multilingual mode.
        """
        return lang_code

    def get_bilingual_prompt(self, foreign_lang: str) -> str | None:  # noqa: ARG002
        """Get an initial prompt for bilingual transcription context.

        Returns None if no bilingual prompt is available.
        """
        return None


class TranslationBackend(ABC):
    """Abstract interface for machine translation backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load or download the model."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy inference to prime the GPU."""

    @abstractmethod
    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a list of texts.

        Args:
            texts: Texts to translate.
            src_lang: Source language code.
            tgt_lang: Target language code.

        Returns:
            Translated texts in the same order.
        """

    def supports_language_pair(self, src: str, tgt: str) -> bool:  # noqa: ARG002
        """Check if a language pair is supported."""
        return True


class SummarizationBackend(ABC):
    """Abstract interface for conversation summarization backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load or download the model."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy inference to prime the GPU."""

    @abstractmethod
    def summarize_bilingual(
        self,
        segments: list[dict],
        foreign_lang: str,
    ) -> tuple[str, str]:
        """Generate dual summaries of a bilingual conversation.

        Args:
            segments: List of segment dicts with 'src', 'src_lang', 'translations'.
            foreign_lang: The non-German language code.

        Returns:
            Tuple of (foreign_summary, german_summary).
        """

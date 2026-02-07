"""Tests for WhisperASRBackend."""

from unittest.mock import MagicMock

import numpy as np

from app.backends.asr.whisper import WhisperASRBackend
from app.backends.types import ASRSegment


class TestWhisperASRBackendDeloop:
    """Tests for deloop_text static method."""

    def test_no_loops(self):
        text = "This is a normal sentence"
        assert WhisperASRBackend._deloop_text(text) == text

    def test_simple_loop(self):
        text = "hello world hello world hello world hello world"
        result = WhisperASRBackend._deloop_text(text, min_ngram=2, min_repeats=3)
        assert result == "hello world"

    def test_empty_text(self):
        assert WhisperASRBackend._deloop_text("") == ""
        assert WhisperASRBackend._deloop_text("  ") == "  "

    def test_short_text(self):
        text = "hi there"
        assert WhisperASRBackend._deloop_text(text) == text

    def test_single_word_loop(self):
        # min_ngram=2 so single word repeats are not caught by deloop
        text = "hello hello hello"
        assert WhisperASRBackend._deloop_text(text, min_ngram=2) == text

    def test_triple_repeat(self):
        text = "good morning good morning good morning"
        result = WhisperASRBackend._deloop_text(text, min_ngram=2, min_repeats=3)
        assert result == "good morning"


class TestWhisperASRBackendHallucination:
    """Tests for is_hallucination static method."""

    def test_empty_text(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("", 1.0)
        assert is_hal is True
        assert reason == "empty"

    def test_known_hallucination(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("Thank you.", 0.5)
        assert is_hal is True
        assert reason == "boh_exact"

    def test_hallucination_stripped_punct(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("Thanks for watching!", 0.5)
        assert is_hal is True

    def test_normal_text(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("Guten Tag, ich brauche Hilfe", 2.0)
        assert is_hal is False

    def test_single_word_repeat(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("hello hello hello", 1.0)
        assert is_hal is True
        assert reason == "single_word_repeat"

    def test_too_long_segment(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("Some text here", 12.0)
        assert is_hal is True
        assert reason == "too_long"

    def test_low_word_rate(self):
        is_hal, reason = WhisperASRBackend._is_hallucination("Hi", 5.0)
        assert is_hal is True


class TestWhisperASRBackendPostProcess:
    """Tests for post_process method."""

    def setup_method(self):
        self.backend = WhisperASRBackend()

    def test_filters_hallucinations(self):
        segments = [
            ASRSegment(start=0.0, end=0.5, text="Thank you.", language="en"),
            ASRSegment(start=1.0, end=2.5, text="Guten Tag, wie geht es Ihnen?", language="de"),
        ]
        result = self.backend.post_process(segments)
        assert len(result) == 1
        assert result[0].text == "Guten Tag, wie geht es Ihnen?"

    def test_deloops_text(self):
        segments = [
            ASRSegment(
                start=0.0,
                end=3.0,
                text="good morning good morning good morning good morning",
                language="en",
            ),
        ]
        result = self.backend.post_process(segments)
        # After deloop, text is "good morning" which is short_text_long_duration (3s, 12 chars)
        # Actually "good morning" is 12 chars, duration 3.0 - let's check
        # duration > 3.0 is False (3.0 is not > 3.0), so it passes
        # word_count=2, duration=3.0, rate=0.66 > 0.5, so it passes
        assert len(result) == 1
        assert result[0].text == "good morning"

    def test_empty_list(self):
        result = self.backend.post_process([])
        assert result == []


class TestWhisperASRBackendLanguage:
    """Tests for language support methods."""

    def setup_method(self):
        self.backend = WhisperASRBackend()

    def test_supports_common_languages(self):
        assert self.backend.supports_language("en") is True
        assert self.backend.supports_language("de") is True
        assert self.backend.supports_language("fr") is True
        assert self.backend.supports_language("bg") is True

    def test_unsupported_language(self):
        assert self.backend.supports_language("xx") is False

    def test_language_fallback_supported(self):
        assert self.backend.get_language_fallback("en") == "en"
        assert self.backend.get_language_fallback("de") == "de"

    def test_language_fallback_unsupported(self):
        assert self.backend.get_language_fallback("ku") is None

    def test_bilingual_prompt(self):
        prompt = self.backend.get_bilingual_prompt("bg")
        assert prompt is not None
        assert "Здравейте" in prompt

    def test_bilingual_prompt_none(self):
        assert self.backend.get_bilingual_prompt("xx") is None


class TestWhisperASRBackendTranscribe:
    """Tests for transcribe method with mocked model."""

    def test_transcribe_returns_result(self):
        backend = WhisperASRBackend()

        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.text = "Hello world"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = ([mock_seg], mock_info)
        backend._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = backend.transcribe(audio, language="en")

        assert result.detected_language == "en"
        assert result.language_probability == 0.95
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"

    def test_transcribe_filters_short_text(self):
        backend = WhisperASRBackend()

        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 0.5
        mock_seg.text = "A"  # Too short (< 2 chars)

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model.transcribe.return_value = ([mock_seg], mock_info)
        backend._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = backend.transcribe(audio)

        assert len(result.segments) == 0

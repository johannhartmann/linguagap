"""Tests for backend factory functions."""

import os
from unittest.mock import patch

import pytest

from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend


class TestGetASRBackend:
    """Tests for get_asr_backend factory."""

    def setup_method(self):
        get_asr_backend.cache_clear()

    def teardown_method(self):
        get_asr_backend.cache_clear()

    @patch.dict(os.environ, {"ASR_BACKEND": "whisper"}, clear=False)
    def test_whisper_backend(self):
        from app.backends.asr.whisper import WhisperASRBackend

        backend = get_asr_backend()
        assert isinstance(backend, WhisperASRBackend)

    @patch.dict(os.environ, {}, clear=False)
    def test_default_is_whisper(self):
        # Remove ASR_BACKEND if set
        os.environ.pop("ASR_BACKEND", None)
        from app.backends.asr.whisper import WhisperASRBackend

        backend = get_asr_backend()
        assert isinstance(backend, WhisperASRBackend)

    @patch.dict(os.environ, {"ASR_BACKEND": "nonexistent"}, clear=False)
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown ASR backend"):
            get_asr_backend()

    @patch.dict(os.environ, {"ASR_BACKEND": "whisper"}, clear=False)
    def test_singleton(self):
        backend1 = get_asr_backend()
        backend2 = get_asr_backend()
        assert backend1 is backend2


class TestGetTranslationBackend:
    """Tests for get_translation_backend factory."""

    def setup_method(self):
        get_translation_backend.cache_clear()

    def teardown_method(self):
        get_translation_backend.cache_clear()

    @patch.dict(os.environ, {"MT_BACKEND": "translategemma"}, clear=False)
    def test_translategemma_backend(self):
        from app.backends.translation.translategemma import TranslateGemmaBackend

        backend = get_translation_backend()
        assert isinstance(backend, TranslateGemmaBackend)

    @patch.dict(os.environ, {}, clear=False)
    def test_default_is_translategemma(self):
        os.environ.pop("MT_BACKEND", None)
        from app.backends.translation.translategemma import TranslateGemmaBackend

        backend = get_translation_backend()
        assert isinstance(backend, TranslateGemmaBackend)

    @patch.dict(os.environ, {"MT_BACKEND": "nonexistent"}, clear=False)
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown translation backend"):
            get_translation_backend()


class TestGetSummarizationBackend:
    """Tests for get_summarization_backend factory."""

    def setup_method(self):
        get_summarization_backend.cache_clear()

    def teardown_method(self):
        get_summarization_backend.cache_clear()

    @patch.dict(os.environ, {"SUMM_BACKEND": ""}, clear=False)
    def test_disabled_by_default(self):
        backend = get_summarization_backend()
        assert backend is None

    @patch.dict(os.environ, {}, clear=False)
    def test_empty_env_disables(self):
        os.environ.pop("SUMM_BACKEND", None)
        backend = get_summarization_backend()
        assert backend is None

    @patch.dict(os.environ, {"SUMM_BACKEND": "qwen3"}, clear=False)
    def test_qwen3_backend(self):
        from app.backends.summarization.qwen3 import Qwen3SummarizationBackend

        backend = get_summarization_backend()
        assert isinstance(backend, Qwen3SummarizationBackend)

    @patch.dict(os.environ, {"SUMM_BACKEND": "nonexistent"}, clear=False)
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown summarization backend"):
            get_summarization_backend()

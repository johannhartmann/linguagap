"""Tests for MT (machine translation) module."""

from unittest.mock import MagicMock, patch

from app.backends import get_translation_backend
from app.mt import LANG_NAMES, translate_texts


class TestLangNames:
    """Tests for language name mapping."""

    def test_common_languages(self):
        """Test common language codes are mapped."""
        assert LANG_NAMES["en"] == "English"
        assert LANG_NAMES["de"] == "German"
        assert LANG_NAMES["fr"] == "French"
        assert LANG_NAMES["es"] == "Spanish"

    def test_all_languages_have_names(self):
        """Test all language codes have non-empty names."""
        for code, name in LANG_NAMES.items():
            assert len(code) == 2
            assert len(name) > 0


class TestTranslateTexts:
    """Tests for translate_texts function via backend."""

    def setup_method(self):
        get_translation_backend.cache_clear()

    def teardown_method(self):
        get_translation_backend.cache_clear()

    @patch("app.backends.translation.translategemma.TranslateGemmaBackend.load_model")
    def test_translate_single_text(self, mock_load):  # noqa: ARG002
        backend = get_translation_backend()
        backend._llm = MagicMock()
        backend._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hallo Welt"}}]
        }

        result = translate_texts(["Hello world"], src_lang="en", tgt_lang="de")

        assert result == ["Hallo Welt"]
        backend._llm.create_chat_completion.assert_called_once()

    @patch("app.backends.translation.translategemma.TranslateGemmaBackend.load_model")
    def test_translate_multiple_texts(self, mock_load):  # noqa: ARG002
        backend = get_translation_backend()
        backend._llm = MagicMock()
        backend._llm.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": "Hallo"}}]},
            {"choices": [{"message": {"content": "Welt"}}]},
        ]

        result = translate_texts(["Hello", "World"], src_lang="en", tgt_lang="de")

        assert result == ["Hallo", "Welt"]
        assert backend._llm.create_chat_completion.call_count == 2

    @patch("app.backends.translation.translategemma.TranslateGemmaBackend.load_model")
    def test_translate_empty_text(self, mock_load):  # noqa: ARG002
        backend = get_translation_backend()
        backend._llm = MagicMock()

        result = translate_texts([""], src_lang="en", tgt_lang="de")

        assert result == [""]
        backend._llm.create_chat_completion.assert_not_called()

    @patch("app.backends.translation.translategemma.TranslateGemmaBackend.load_model")
    def test_translate_whitespace_text(self, mock_load):  # noqa: ARG002
        backend = get_translation_backend()
        backend._llm = MagicMock()

        result = translate_texts(["   "], src_lang="en", tgt_lang="de")

        assert result == [""]
        backend._llm.create_chat_completion.assert_not_called()

    @patch("app.backends.translation.translategemma.TranslateGemmaBackend.load_model")
    def test_translate_parameters(self, mock_load):  # noqa: ARG002
        backend = get_translation_backend()
        backend._llm = MagicMock()
        backend._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }

        translate_texts(["Hello"], src_lang="en", tgt_lang="de")

        call_args = backend._llm.create_chat_completion.call_args
        assert call_args[1]["max_tokens"] == 256
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["top_p"] == 0.9

"""Tests for TranslateGemmaBackend."""

from unittest.mock import MagicMock


class TestTranslateGemmaBackend:
    """Tests for TranslateGemmaBackend."""

    def _make_backend(self):
        from app.backends.translation.translategemma import TranslateGemmaBackend

        backend = TranslateGemmaBackend()
        backend._llm = MagicMock()
        return backend

    def test_translate_single_text(self):
        backend = self._make_backend()
        backend._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hallo Welt"}}]
        }

        result = backend.translate(["Hello world"], src_lang="en", tgt_lang="de")
        assert result == ["Hallo Welt"]

    def test_translate_multiple_texts(self):
        backend = self._make_backend()
        backend._llm.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": "Hallo"}}]},
            {"choices": [{"message": {"content": "Welt"}}]},
        ]

        result = backend.translate(["Hello", "World"], src_lang="en", tgt_lang="de")
        assert result == ["Hallo", "Welt"]

    def test_translate_empty_text(self):
        backend = self._make_backend()
        result = backend.translate([""], src_lang="en", tgt_lang="de")
        assert result == [""]
        backend._llm.create_chat_completion.assert_not_called()

    def test_translate_whitespace(self):
        backend = self._make_backend()
        result = backend.translate(["   "], src_lang="en", tgt_lang="de")
        assert result == [""]
        backend._llm.create_chat_completion.assert_not_called()

    def test_unsupported_source_lang_fallback(self):
        backend = self._make_backend()
        backend._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }

        backend.translate(["Hello"], src_lang="xx", tgt_lang="de")
        # Should not crash - falls back to "en"
        backend._llm.create_chat_completion.assert_called_once()

    def test_supports_language_pair(self):
        backend = self._make_backend()
        assert backend.supports_language_pair("en", "de") is True
        assert backend.supports_language_pair("xx", "de") is False
        assert backend.supports_language_pair("en", "xx") is False

    def test_translate_parameters(self):
        backend = self._make_backend()
        backend._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }

        backend.translate(["Hello"], src_lang="en", tgt_lang="de")

        call_args = backend._llm.create_chat_completion.call_args
        assert call_args[1]["max_tokens"] == 256
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["top_p"] == 0.9

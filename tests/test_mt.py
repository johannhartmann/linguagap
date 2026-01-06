"""Tests for MT (machine translation) module."""

from unittest.mock import MagicMock, patch

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
    """Tests for translate_texts function."""

    @patch("app.mt.get_llm")
    def test_translate_single_text(self, mock_get_llm):
        """Test translating a single text."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hallo Welt"}}]
        }
        mock_get_llm.return_value = mock_llm

        result = translate_texts(["Hello world"], src_lang="en", tgt_lang="de")

        assert result == ["Hallo Welt"]
        mock_llm.create_chat_completion.assert_called_once()

    @patch("app.mt.get_llm")
    def test_translate_multiple_texts(self, mock_get_llm):
        """Test translating multiple texts."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": "Hallo"}}]},
            {"choices": [{"message": {"content": "Welt"}}]},
        ]
        mock_get_llm.return_value = mock_llm

        result = translate_texts(["Hello", "World"], src_lang="en", tgt_lang="de")

        assert result == ["Hallo", "Welt"]
        assert mock_llm.create_chat_completion.call_count == 2

    @patch("app.mt.get_llm")
    def test_translate_empty_text(self, mock_get_llm):
        """Test translating empty text returns empty string."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        result = translate_texts([""], src_lang="en", tgt_lang="de")

        assert result == [""]
        mock_llm.create_chat_completion.assert_not_called()

    @patch("app.mt.get_llm")
    def test_translate_whitespace_text(self, mock_get_llm):
        """Test translating whitespace-only text returns empty string."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        result = translate_texts(["   "], src_lang="en", tgt_lang="de")

        assert result == [""]
        mock_llm.create_chat_completion.assert_not_called()

    @patch("app.mt.get_llm")
    def test_translate_strips_thinking_tags(self, mock_get_llm):
        """Test that thinking tags are stripped from output."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "<think>Let me translate...</think>Hallo Welt"}}]
        }
        mock_get_llm.return_value = mock_llm

        result = translate_texts(["Hello world"], src_lang="en", tgt_lang="de")

        assert result == ["Hallo Welt"]

    @patch("app.mt.get_llm")
    def test_translate_uses_correct_language_names(self, mock_get_llm):
        """Test that full language names are used in prompts."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Bonjour"}}]
        }
        mock_get_llm.return_value = mock_llm

        translate_texts(["Hello"], src_lang="en", tgt_lang="fr")

        call_args = mock_llm.create_chat_completion.call_args
        messages = call_args[1]["messages"]
        system_content = messages[0]["content"]
        assert "English" in system_content
        assert "French" in system_content

    @patch("app.mt.get_llm")
    def test_translate_unknown_language_code(self, mock_get_llm):
        """Test that unknown language codes are used as-is."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }
        mock_get_llm.return_value = mock_llm

        translate_texts(["Hello"], src_lang="xx", tgt_lang="yy")

        call_args = mock_llm.create_chat_completion.call_args
        messages = call_args[1]["messages"]
        system_content = messages[0]["content"]
        assert "xx" in system_content
        assert "yy" in system_content

    @patch("app.mt.get_llm")
    def test_translate_parameters(self, mock_get_llm):
        """Test that correct parameters are passed to LLM."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }
        mock_get_llm.return_value = mock_llm

        translate_texts(["Hello"], src_lang="en", tgt_lang="de")

        call_args = mock_llm.create_chat_completion.call_args
        assert call_args[1]["max_tokens"] == 512
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["top_p"] == 0.9

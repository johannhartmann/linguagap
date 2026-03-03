"""Tests for ASR (automatic speech recognition) module."""

from unittest.mock import MagicMock, patch

from app.backends import get_asr_backend
from app.backends.types import ASRResult, ASRSegment


class TestASRBackendConfig:
    """Tests for ASR backend configuration."""

    def setup_method(self):
        get_asr_backend.cache_clear()

    def teardown_method(self):
        get_asr_backend.cache_clear()

    def test_default_backend_is_whisper(self):
        from app.backends.asr.whisper import WhisperASRBackend

        backend = get_asr_backend()
        assert isinstance(backend, WhisperASRBackend)


class TestTranscribeWavPath:
    """Tests for transcribe_wav_path function."""

    def setup_method(self):
        get_asr_backend.cache_clear()

    def teardown_method(self):
        get_asr_backend.cache_clear()

    @patch("app.asr.get_asr_backend")
    def test_transcribe_returns_dict(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.transcribe_file.return_value = ASRResult(
            segments=[ASRSegment(start=0.0, end=1.5, text="Hello world", language="en")],
            detected_language="en",
            language_probability=0.95,
        )
        mock_get_backend.return_value = mock_backend

        from app.asr import transcribe_wav_path

        result = transcribe_wav_path("/fake/path.wav")

        assert "language" in result
        assert "language_probability" in result
        assert "segments" in result
        assert result["language"] == "en"
        assert result["language_probability"] == 0.95

    @patch("app.asr.get_asr_backend")
    def test_transcribe_segments_format(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.transcribe_file.return_value = ASRResult(
            segments=[
                ASRSegment(start=0.0, end=1.5, text="First segment", language="en"),
                ASRSegment(start=2.0, end=3.5, text="Second segment", language="en"),
            ],
            detected_language="en",
            language_probability=0.9,
        )
        mock_get_backend.return_value = mock_backend

        from app.asr import transcribe_wav_path

        result = transcribe_wav_path("/fake/path.wav")

        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.5
        assert result["segments"][0]["text"] == "First segment"

    @patch("app.asr.get_asr_backend")
    def test_transcribe_empty_audio(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.transcribe_file.return_value = ASRResult(
            segments=[],
            detected_language="en",
            language_probability=0.5,
        )
        mock_get_backend.return_value = mock_backend

        from app.asr import transcribe_wav_path

        result = transcribe_wav_path("/fake/silent.wav")

        assert result["segments"] == []
        assert result["language"] == "en"

    @patch("app.asr.get_asr_backend")
    def test_transcribe_calls_backend(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.transcribe_file.return_value = ASRResult(
            segments=[],
            detected_language="en",
            language_probability=0.9,
        )
        mock_get_backend.return_value = mock_backend

        from app.asr import transcribe_wav_path

        transcribe_wav_path("/test/audio.wav")

        mock_backend.transcribe_file.assert_called_once_with("/test/audio.wav")

"""Tests for ASR (automatic speech recognition) module."""

from unittest.mock import MagicMock, patch

from app.asr import ASR_COMPUTE_TYPE, ASR_DEVICE, ASR_MODEL, transcribe_wav_path


class TestASRConfig:
    """Tests for ASR configuration."""

    def test_default_model(self):
        """Test default ASR model is set."""
        assert ASR_MODEL == "deepdml/faster-whisper-large-v3-turbo-ct2"

    def test_default_device(self):
        """Test default device is CUDA."""
        assert ASR_DEVICE == "cuda"

    def test_default_compute_type(self):
        """Test default compute type."""
        assert ASR_COMPUTE_TYPE == "int8_float16"


class TestTranscribeWavPath:
    """Tests for transcribe_wav_path function."""

    @patch("app.asr.get_model")
    def test_transcribe_returns_dict(self, mock_get_model):
        """Test that transcribe returns a dictionary with expected keys."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_segment.text = "Hello world"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_get_model.return_value = mock_model

        result = transcribe_wav_path("/fake/path.wav")

        assert "language" in result
        assert "language_probability" in result
        assert "segments" in result
        assert result["language"] == "en"
        assert result["language_probability"] == 0.95

    @patch("app.asr.get_model")
    def test_transcribe_segments_format(self, mock_get_model):
        """Test that segments have correct format."""
        mock_model = MagicMock()
        mock_segment1 = MagicMock()
        mock_segment1.start = 0.0
        mock_segment1.end = 1.5
        mock_segment1.text = "First segment"

        mock_segment2 = MagicMock()
        mock_segment2.start = 2.0
        mock_segment2.end = 3.5
        mock_segment2.text = "Second segment"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        mock_get_model.return_value = mock_model

        result = transcribe_wav_path("/fake/path.wav")

        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.5
        assert result["segments"][0]["text"] == "First segment"
        assert result["segments"][1]["start"] == 2.0
        assert result["segments"][1]["end"] == 3.5
        assert result["segments"][1]["text"] == "Second segment"

    @patch("app.asr.get_model")
    def test_transcribe_empty_audio(self, mock_get_model):
        """Test transcription of empty/silent audio."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.5

        mock_model.transcribe.return_value = ([], mock_info)
        mock_get_model.return_value = mock_model

        result = transcribe_wav_path("/fake/silent.wav")

        assert result["segments"] == []
        assert result["language"] == "en"

    @patch("app.asr.get_model")
    def test_transcribe_calls_model(self, mock_get_model):
        """Test that transcribe calls the model with correct path."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model.transcribe.return_value = ([], mock_info)
        mock_get_model.return_value = mock_model

        transcribe_wav_path("/test/audio.wav")

        mock_model.transcribe.assert_called_once_with("/test/audio.wav")

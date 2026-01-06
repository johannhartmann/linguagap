"""Tests for FastAPI main application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_models():
    """Mock ASR and MT models to avoid loading them."""
    with (
        patch("app.main.get_model") as mock_asr,
        patch("app.main.get_llm") as mock_llm,
        patch("app.main.translate_texts") as mock_translate,
    ):
        # Mock ASR model
        mock_asr_model = MagicMock()
        mock_asr_model.transcribe.return_value = iter([])
        mock_asr.return_value = mock_asr_model

        # Mock LLM
        mock_llm.return_value = MagicMock()

        # Mock translate
        mock_translate.return_value = ["Translated"]

        yield {
            "asr": mock_asr,
            "llm": mock_llm,
            "translate": mock_translate,
        }


@pytest.fixture
def client(mock_models):  # noqa: ARG001
    """Create test client with mocked models."""
    # Import app after mocking to avoid model loading
    from app.main import app

    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_returns_dict(self, client):
        """Test metrics endpoint returns dictionary."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "avg_asr_time_ms" in data
        assert "avg_mt_time_ms" in data
        assert "avg_tick_time_ms" in data
        assert "sample_count" in data


class TestASRSmokeEndpoint:
    """Tests for ASR smoke test endpoint."""

    @patch("app.main.transcribe_wav_path")
    @patch("app.main.generate_silence_wav")
    def test_asr_smoke_returns_result(self, _mock_generate, mock_transcribe, client):
        """Test ASR smoke endpoint."""
        mock_transcribe.return_value = {
            "language": "en",
            "language_probability": 0.9,
            "segments": [],
        }

        response = client.get("/asr_smoke")
        assert response.status_code == 200
        data = response.json()
        assert "language" in data


class TestMTSmokeEndpoint:
    """Tests for MT smoke test endpoint."""

    def test_mt_smoke_returns_result(self, client, mock_models):
        """Test MT smoke endpoint."""
        mock_models["translate"].return_value = ["Hallo Welt!"]

        response = client.get("/mt_smoke")
        assert response.status_code == 200
        data = response.json()
        assert "input" in data
        assert "output" in data
        assert data["input"] == ["Hello world!"]


class TestTranscribeTranslateEndpoint:
    """Tests for transcribe_translate endpoint."""

    @patch("app.main.transcribe_wav_path")
    def test_transcribe_translate_success(self, mock_transcribe, client, mock_models):
        """Test transcribe_translate with valid file."""
        mock_transcribe.return_value = {
            "language": "en",
            "language_probability": 0.95,
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello world"},
            ],
        }
        mock_models["translate"].return_value = ["Hallo Welt"]

        # Create a simple WAV file content
        wav_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/transcribe_translate",
            files={"file": ("test.wav", wav_content, "audio/wav")},
            data={"src_lang": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "src_lang_detected" in data
        assert "segments" in data
        assert len(data["segments"]) == 1

    @patch("app.main.transcribe_wav_path")
    def test_transcribe_translate_auto_lang(self, mock_transcribe, client, mock_models):
        """Test transcribe_translate with auto language detection."""
        mock_transcribe.return_value = {
            "language": "fr",
            "language_probability": 0.9,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Bonjour"},
            ],
        }
        mock_models["translate"].return_value = ["Hallo"]

        wav_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/transcribe_translate",
            files={"file": ("test.wav", wav_content, "audio/wav")},
            data={"src_lang": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["src_lang_detected"] == "fr"

    @patch("app.main.transcribe_wav_path")
    def test_transcribe_translate_empty_segment(self, mock_transcribe, client, _mock_models):
        """Test transcribe_translate with empty segment text."""
        mock_transcribe.return_value = {
            "language": "en",
            "language_probability": 0.9,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "   "},
            ],
        }

        wav_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/transcribe_translate",
            files={"file": ("test.wav", wav_content, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        # Empty text should result in empty translation
        assert data["segments"][0]["de"] == ""

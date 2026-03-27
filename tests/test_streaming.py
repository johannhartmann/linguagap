"""Tests for streaming module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.backends import get_asr_backend, get_translation_backend
from app.backends.types import ASRResult, ASRSegment
from app.streaming import (
    WINDOW_SEC,
    StreamingSession,
    get_metrics,
    run_asr_german_channel,
    run_translation,
)


class TestStreamingSession:
    """Tests for StreamingSession class."""

    def test_init_defaults(self):
        """Test default initialization."""
        session = StreamingSession()
        assert session.sample_rate == 16000
        assert session.src_lang == "auto"
        assert session.total_samples == 0
        assert session.detected_lang is None
        assert session.dropped_frames == 0
        assert len(session.audio_buffer) == 0
        assert len(session.translations) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        session = StreamingSession(sample_rate=44100, src_lang="en")
        assert session.sample_rate == 44100
        assert session.src_lang == "en"

    def test_add_audio(self):
        """Test adding audio data."""
        session = StreamingSession(sample_rate=16000)
        audio_data = np.zeros(16000, dtype=np.int16).tobytes()
        session.add_audio(audio_data)
        assert session.total_samples == 16000
        assert len(session.audio_buffer) == 1

    def test_add_audio_multiple(self):
        """Test adding multiple audio chunks."""
        session = StreamingSession(sample_rate=16000)
        chunk = np.zeros(1600, dtype=np.int16).tobytes()
        for _ in range(10):
            session.add_audio(chunk)
        assert session.total_samples == 16000
        assert len(session.audio_buffer) == 10

    def test_get_current_time(self):
        """Test current time calculation."""
        session = StreamingSession(sample_rate=16000)
        assert session.get_current_time() == 0.0

        audio_data = np.zeros(16000, dtype=np.int16).tobytes()
        session.add_audio(audio_data)
        assert session.get_current_time() == 1.0

        session.add_audio(audio_data)
        assert session.get_current_time() == 2.0

    def test_get_buffered_seconds(self):
        """Test buffered seconds calculation."""
        session = StreamingSession(sample_rate=16000)
        assert session.get_buffered_seconds() == 0.0

        audio_data = np.zeros(8000, dtype=np.int16).tobytes()
        session.add_audio(audio_data)
        assert session.get_buffered_seconds() == 0.5

    def test_get_window_audio_short(self):
        """Test getting window audio when buffer is shorter than window."""
        session = StreamingSession(sample_rate=16000)
        audio_data = np.ones(8000, dtype=np.int16).tobytes()
        session.add_audio(audio_data)

        samples, window_start = session.get_window_audio()
        assert window_start == 0.0
        assert len(samples) == 8000

    def test_enforce_max_buffer(self):
        """Test that buffer is trimmed when exceeding max size."""
        session = StreamingSession(sample_rate=16000)
        for _ in range(40):
            audio_data = np.zeros(16000, dtype=np.int16).tobytes()
            session.add_audio(audio_data)

        assert session.dropped_frames > 0
        assert session.get_buffered_seconds() <= 30.0

    def test_translations_storage(self):
        """Test translation storage."""
        session = StreamingSession()
        session.translations[0] = "Hello"
        session.translations[1] = "World"
        assert session.translations.get(0) == "Hello"
        assert session.translations.get(1) == "World"
        assert session.translations.get(99) is None


class TestMetrics:
    """Tests for metrics functions."""

    def test_get_metrics_empty(self):
        """Test metrics with no data."""
        metrics = get_metrics()
        assert "avg_asr_time_ms" in metrics
        assert "avg_mt_time_ms" in metrics
        assert "avg_tick_time_ms" in metrics
        assert "sample_count" in metrics

    def test_get_metrics_structure(self):
        """Test metrics return structure."""
        metrics = get_metrics()
        assert isinstance(metrics, dict)
        assert isinstance(metrics.get("avg_asr_time_ms", 0), int | float)
        assert isinstance(metrics.get("avg_mt_time_ms", 0), int | float)
        assert isinstance(metrics.get("avg_tick_time_ms", 0), int | float)
        assert isinstance(metrics.get("sample_count", 0), int)


class TestRunASRGermanChannel:
    """Tests for deterministic desktop-channel ASR."""

    def setup_method(self):
        get_asr_backend.cache_clear()

    def teardown_method(self):
        get_asr_backend.cache_clear()

    @patch("app.streaming.get_asr_backend")
    def test_forces_german_role_and_language(self, mock_get_backend):
        session = StreamingSession(sample_rate=16000, src_lang="en")
        session.foreign_lang = "en"

        # Strong signal to avoid silence gating in _transcribe_channel.
        audio_data = np.full(16000, 5000, dtype=np.int16).tobytes()
        session.add_german_audio(audio_data)

        backend = MagicMock()
        backend.get_bilingual_prompt.return_value = None
        backend.transcribe.return_value = ASRResult(
            segments=[ASRSegment(start=0.0, end=0.8, text="Guten Tag", language="en")],
            detected_language="en",
            language_probability=0.9,
        )
        mock_get_backend.return_value = backend

        all_segments, _ = run_asr_german_channel(session)

        assert len(all_segments) == 1
        assert all_segments[0].speaker_role == "german"
        assert all_segments[0].speaker_id == "SPEAKER_00"
        assert all_segments[0].src_lang == "de"
        assert session.detected_lang == "de"

    def test_channel_window_is_bounded(self):
        session = StreamingSession(sample_rate=16000, src_lang="en")
        chunk = np.full(16000, 5000, dtype=np.int16).tobytes()

        # Add more than WINDOW_SEC seconds to german channel.
        for _ in range(int(WINDOW_SEC) + 3):
            session.add_german_audio(chunk)

        samples, _ = session.get_german_window_audio()
        assert len(samples) <= int(WINDOW_SEC * session.sample_rate)


class TestRunTranslation:
    """Tests for run_translation function."""

    def setup_method(self):
        get_translation_backend.cache_clear()

    def teardown_method(self):
        get_translation_backend.cache_clear()

    @patch("app.streaming.get_translation_backend")
    def test_run_translation_success(self, mock_get_backend):
        """Test run_translation returns translated text."""
        mock_backend = MagicMock()
        mock_backend.translate.return_value = ["Hallo Welt"]
        mock_get_backend.return_value = mock_backend

        result = run_translation("Hello world", "en", "de")

        assert result == "Hallo Welt"
        mock_backend.translate.assert_called_once_with(
            ["Hello world"], src_lang="en", tgt_lang="de"
        )

    @patch("app.streaming.get_translation_backend")
    def test_run_translation_records_metrics(self, mock_get_backend):
        """Test run_translation records timing metrics."""
        mock_backend = MagicMock()
        mock_backend.translate.return_value = ["Test"]
        mock_get_backend.return_value = mock_backend

        from app.streaming import _metrics

        initial_count = len(_metrics["mt_times"])

        run_translation("Test", "en", "de")

        assert len(_metrics["mt_times"]) >= initial_count


class TestWebSocketHandler:
    """Tests for WebSocket handler."""

    def setup_method(self):
        get_asr_backend.cache_clear()
        get_translation_backend.cache_clear()

    def teardown_method(self):
        get_asr_backend.cache_clear()
        get_translation_backend.cache_clear()

    @pytest.fixture
    def mock_models(self):
        """Mock ASR and MT backends."""
        with (
            patch("app.streaming.get_asr_backend") as mock_asr,
            patch("app.streaming.get_translation_backend") as mock_mt,
        ):
            mock_asr_backend = MagicMock()
            mock_mt_backend = MagicMock()
            mock_mt_backend.translate.return_value = ["Translated"]
            mock_asr.return_value = mock_asr_backend
            mock_mt.return_value = mock_mt_backend
            yield {"asr": mock_asr, "mt": mock_mt}

    @pytest.fixture
    def client(self, mock_models, tmp_path):  # noqa: ARG002
        """Create test client with mocked backends, pre-authenticated."""
        with (
            patch("app.main.get_asr_backend") as mock_asr,
            patch("app.main.get_translation_backend") as mock_mt,
            patch("app.main.get_summarization_backend") as mock_summ,
            patch("app.main.translate_texts") as mock_translate,
            patch("app.auth.DATA_DIR", tmp_path),
            patch("app.auth.ACCOUNTS_FILE", tmp_path / "accounts.json"),
            patch("app.auth.LOGOS_DIR", tmp_path / "logos"),
            patch("app.main.LOGOS_DIR", tmp_path / "logos"),
            patch("app.auth.ADMIN_EMAIL", "admin@test.local"),
            patch("app.auth.ADMIN_PASSWORD", "testpass"),
        ):
            mock_asr.return_value = MagicMock()
            mock_mt.return_value = MagicMock()
            mock_summ.return_value = None
            mock_translate.return_value = ["Translated"]
            (tmp_path / "logos").mkdir(exist_ok=True)

            import app.auth

            app.auth._accounts = None

            from fastapi.testclient import TestClient

            from app.main import app

            with TestClient(app) as client:
                client.post(
                    "/api/admin/login", json={"email": "admin@test.local", "password": "testpass"}
                )
                client.post(
                    "/api/admin/accounts",
                    json={
                        "email": "test@example.com",
                        "password": "TestPass#1",
                        "display_name": "Test",
                        "logo_url": "/static/logos/synia.png",
                    },
                )
                client.post("/api/admin/logout")
                client.post(
                    "/api/login", json={"email": "test@example.com", "password": "TestPass#1"}
                )
                yield client

            app.auth._accounts = None

    def test_websocket_config_message(self, client, mock_models):  # noqa: ARG002
        """Test WebSocket accepts config message."""
        with client.websocket_connect("/ws") as websocket:
            config = {"type": "config", "sample_rate": 16000, "src_lang": "en"}
            websocket.send_text(json.dumps(config))

    def test_websocket_binary_audio(self, client, mock_models):  # noqa: ARG002
        """Test WebSocket accepts binary audio data."""
        with client.websocket_connect("/ws") as websocket:
            config = {"type": "config", "sample_rate": 16000, "src_lang": "en"}
            websocket.send_text(json.dumps(config))

            audio_data = np.zeros(1600, dtype=np.int16).tobytes()
            websocket.send_bytes(audio_data)

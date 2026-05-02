"""Tests for FastAPI main application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_models():
    """Mock ASR and MT models to avoid loading them."""
    with (
        patch("app.main.get_asr_backend") as mock_asr,
        patch("app.main.get_translation_backend") as mock_mt,
        patch("app.main.get_summarization_backend") as mock_summ,
        patch("app.routes.inference.translate_texts") as mock_translate,
    ):
        mock_asr_backend = MagicMock()
        mock_asr.return_value = mock_asr_backend

        mock_mt_backend = MagicMock()
        mock_mt.return_value = mock_mt_backend

        mock_summ.return_value = None

        mock_translate.return_value = ["Translated"]

        yield {
            "asr": mock_asr,
            "mt": mock_mt,
            "summ": mock_summ,
            "translate": mock_translate,
        }


@pytest.fixture
def client(mock_models, tmp_path):  # noqa: ARG001
    """Create test client with mocked models, pre-authenticated."""
    with (
        patch("app.auth.DATA_DIR", tmp_path),
        patch("app.auth.ACCOUNTS_FILE", tmp_path / "accounts.json"),
        patch("app.auth.LOGOS_DIR", tmp_path / "logos"),
        patch("app.routes.admin.LOGOS_DIR", tmp_path / "logos"),
        patch("app.auth.ADMIN_EMAIL", "admin@test.local"),
        patch("app.auth.ADMIN_PASSWORD", "testpass"),
    ):
        import app.auth as auth_mod

        auth_mod._accounts = None
        (tmp_path / "logos").mkdir(exist_ok=True)

        from app.main import app

        with TestClient(app) as client:
            # Seed a test account via admin, then authenticate
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
            client.post("/api/login", json={"email": "test@example.com", "password": "TestPass#1"})
            yield client

        auth_mod._accounts = None


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

    @patch("app.routes.inference.transcribe_wav_path")
    @patch("app.routes.inference.generate_silence_wav")
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

    @patch("app.routes.inference.transcribe_wav_path")
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

    @patch("app.routes.inference.transcribe_wav_path")
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

    @patch("app.routes.inference.transcribe_wav_path")
    def test_transcribe_translate_empty_segment(
        self,
        mock_transcribe,
        client,
        mock_models,  # noqa: ARG002
    ):
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


class TestTranslateEndpoint:
    """Tests for the text-to-text translation API."""

    def test_translate_success(self, client, mock_models):
        mock_models["translate"].return_value = ["Hello world"]

        response = client.post(
            "/api/translate",
            json={"text": "Hallo Welt", "src_lang": "de", "tgt_lang": "en"},
        )

        assert response.status_code == 200
        assert response.json() == {"output": "Hello world"}
        mock_models["translate"].assert_called_once_with(["Hallo Welt"], "de", "en")

    def test_translate_empty_text_short_circuits(self, client, mock_models):
        response = client.post(
            "/api/translate",
            json={"text": "   ", "src_lang": "de", "tgt_lang": "en"},
        )

        assert response.status_code == 200
        assert response.json() == {"output": ""}
        mock_models["translate"].assert_not_called()

    def test_translate_same_language_returns_input(self, client, mock_models):
        response = client.post(
            "/api/translate",
            json={"text": "Hallo", "src_lang": "de", "tgt_lang": "de"},
        )

        assert response.status_code == 200
        assert response.json() == {"output": "Hallo"}
        mock_models["translate"].assert_not_called()

    def test_translate_boundary_4000_chars_passes(self, client, mock_models):
        mock_models["translate"].return_value = ["ok"]
        response = client.post(
            "/api/translate",
            json={"text": "x" * 4000, "src_lang": "de", "tgt_lang": "en"},
        )
        assert response.status_code == 200

    def test_translate_oversize_rejected(self, client, mock_models):
        response = client.post(
            "/api/translate",
            json={"text": "x" * 4001, "src_lang": "de", "tgt_lang": "en"},
        )

        assert response.status_code == 400
        mock_models["translate"].assert_not_called()

    def test_translate_dense_lang_has_lower_cap(self, client, mock_models):
        """CJK source languages have a 1500-char cap due to higher token density."""
        response = client.post(
            "/api/translate",
            json={"text": "x" * 1501, "src_lang": "zh", "tgt_lang": "en"},
        )
        assert response.status_code == 400
        mock_models["translate"].assert_not_called()

    def test_translate_backend_failure_returns_500(self, client, mock_models):
        mock_models["translate"].side_effect = RuntimeError("model exploded")
        response = client.post(
            "/api/translate",
            json={"text": "Hallo", "src_lang": "de", "tgt_lang": "en"},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Übersetzung fehlgeschlagen"

    def test_translate_requires_auth(self, mock_models, tmp_path):
        """Unauthenticated POST /api/translate must return 401 before any backend call."""
        with (
            patch("app.auth.DATA_DIR", tmp_path),
            patch("app.auth.ACCOUNTS_FILE", tmp_path / "accounts.json"),
            patch("app.auth.LOGOS_DIR", tmp_path / "logos"),
            patch("app.routes.admin.LOGOS_DIR", tmp_path / "logos"),
        ):
            (tmp_path / "logos").mkdir(exist_ok=True)
            from app.main import app

            with TestClient(app) as anon_client:
                response = anon_client.post(
                    "/api/translate",
                    json={"text": "Hi", "src_lang": "de", "tgt_lang": "en"},
                )
                assert response.status_code == 401
        mock_models["translate"].assert_not_called()

    def test_translate_page_serves_html(self, client):
        response = client.get("/translate")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestLanguagesEndpoint:
    """Tests for the /api/languages registry endpoint."""

    def test_speech_scope_is_alphabetical_within_tier(self, client):
        response = client.get("/api/languages")
        assert response.status_code == 200
        langs = response.json()
        codes_a = [lang["code"] for lang in langs if lang["tier"] == "a"]
        codes_b = [lang["code"] for lang in langs if lang["tier"] == "b"]
        # Tier A precedes tier B
        assert [lang["tier"] for lang in langs] == ["a"] * len(codes_a) + ["b"] * len(codes_b)
        # Each tier sorted alphabetically by label
        labels_a = [lang["label"] for lang in langs if lang["tier"] == "a"]
        labels_b = [lang["label"] for lang in langs if lang["tier"] == "b"]
        assert labels_a == sorted(labels_a)
        assert labels_b == sorted(labels_b)

    def test_speech_scope_excludes_german(self, client):
        langs = client.get("/api/languages").json()
        assert "de" not in {lang["code"] for lang in langs}

    def test_beta_suffix_on_tier_b(self, client):
        langs = client.get("/api/languages").json()
        for lang in langs:
            if lang["tier"] == "b":
                assert lang["label"].endswith(" (beta)")
            else:
                assert "(beta)" not in lang["label"]

    def test_translate_scope_includes_german_first(self, client):
        langs = client.get("/api/languages?scope=translate").json()
        assert langs[0] == {"code": "de", "label": "Deutsch", "tier": "host"}
        # Speech languages still present after the German entry
        assert any(lang["code"] == "en" for lang in langs[1:])

    def test_languages_is_public(self, mock_models, tmp_path):  # noqa: ARG002
        """Endpoint must be reachable without a session — viewer.html needs it."""
        with (
            patch("app.auth.DATA_DIR", tmp_path),
            patch("app.auth.ACCOUNTS_FILE", tmp_path / "accounts.json"),
            patch("app.auth.LOGOS_DIR", tmp_path / "logos"),
            patch("app.routes.admin.LOGOS_DIR", tmp_path / "logos"),
        ):
            (tmp_path / "logos").mkdir(exist_ok=True)
            from app.main import app

            with TestClient(app) as anon_client:
                response = anon_client.get("/api/languages")
                assert response.status_code == 200

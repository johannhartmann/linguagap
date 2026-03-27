"""Tests for demo authentication system."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_ACCOUNT = {
    "email": "test@example.com",
    "password": "TestPass#1",
    "display_name": "Test Account",
    "logo_url": "/static/logos/synia.png",
}


@pytest.fixture
def mock_models():
    """Mock ASR and MT models to avoid loading them."""
    with (
        patch("app.main.get_asr_backend") as mock_asr,
        patch("app.main.get_translation_backend") as mock_mt,
        patch("app.main.get_summarization_backend") as mock_summ,
    ):
        mock_asr.return_value = MagicMock()
        mock_mt.return_value = MagicMock()
        mock_summ.return_value = None
        yield


@pytest.fixture
def client(mock_models, tmp_path):  # noqa: ARG001
    with (
        patch("app.auth.DATA_DIR", tmp_path),
        patch("app.auth.ACCOUNTS_FILE", tmp_path / "accounts.json"),
        patch("app.auth.LOGOS_DIR", tmp_path / "logos"),
        patch("app.main.LOGOS_DIR", tmp_path / "logos"),
        patch("app.auth.ADMIN_EMAIL", "admin@test.local"),
        patch("app.auth.ADMIN_PASSWORD", "testpass"),
    ):
        import app.auth

        app.auth._accounts = None
        (tmp_path / "logos").mkdir(exist_ok=True)

        from app.main import app

        with TestClient(app) as client:
            # Seed a test account via admin API
            client.post(
                "/api/admin/login", json={"email": "admin@test.local", "password": "testpass"}
            )
            client.post("/api/admin/accounts", json=TEST_ACCOUNT)
            client.post("/api/admin/logout")
            yield client

        app.auth._accounts = None


def _login(client: TestClient):
    return client.post(
        "/api/login", json={"email": TEST_ACCOUNT["email"], "password": TEST_ACCOUNT["password"]}
    )


class TestLogin:
    def test_login_valid_credentials(self, client):
        resp = _login(client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["display_name"] == "Test Account"

    def test_login_invalid_password(self, client):
        resp = client.post("/api/login", json={"email": TEST_ACCOUNT["email"], "password": "wrong"})
        assert resp.status_code == 401

    def test_login_invalid_email(self, client):
        resp = client.post("/api/login", json={"email": "nobody@example.com", "password": "x"})
        assert resp.status_code == 401

    def test_login_sets_session_cookie(self, client):
        resp = _login(client)
        assert resp.status_code == 200
        assert "linguagap_session" in resp.cookies


class TestLogout:
    def test_logout_clears_session(self, client):
        _login(client)
        resp = client.post("/api/logout")
        assert resp.status_code == 200
        me = client.get("/api/me")
        assert me.status_code == 401


class TestApiMe:
    def test_me_without_auth(self, client):
        resp = client.get("/api/me")
        assert resp.status_code == 401

    def test_me_with_auth(self, client):
        _login(client)
        resp = client.get("/api/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == TEST_ACCOUNT["email"]
        assert data["display_name"] == "Test Account"


class TestProtectedRoutes:
    def test_root_redirects_without_auth(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login"

    def test_root_serves_page_with_auth(self, client):
        _login(client)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_metrics_requires_auth(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_metrics_works_with_auth(self, client):
        _login(client)
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestPublicRoutes:
    def test_health_unprotected(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_login_page_unprotected(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_viewer_unprotected(self, client):
        resp = client.get("/viewer/sometoken")
        assert resp.status_code == 200

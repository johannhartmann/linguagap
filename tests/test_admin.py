"""Tests for admin authentication and account management."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

SEED_ACCOUNT = {
    "email": "seed@example.com",
    "password": "SeedPass#1",
    "display_name": "Seed Account",
    "logo_url": "/static/logos/synia.png",
}


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "logos").mkdir()
    return tmp_path


@pytest.fixture
def mock_models():
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
def client(mock_models, data_dir):  # noqa: ARG001
    with (
        patch("app.auth.DATA_DIR", data_dir),
        patch("app.auth.ACCOUNTS_FILE", data_dir / "accounts.json"),
        patch("app.auth.LOGOS_DIR", data_dir / "logos"),
        patch("app.main.LOGOS_DIR", data_dir / "logos"),
        patch("app.auth.ADMIN_EMAIL", "admin@test.local"),
        patch("app.auth.ADMIN_PASSWORD", "testpass"),
    ):
        import app.auth as auth_mod

        auth_mod._accounts = None

        from app.main import app

        with TestClient(app) as client:
            # Seed one account for tests that need it
            _admin_login(client)
            client.post("/api/admin/accounts", json=SEED_ACCOUNT)
            client.post("/api/admin/logout")
            yield client

        app.auth._accounts = None


def _admin_login(client, email="admin@test.local", password="testpass"):
    return client.post("/api/admin/login", json={"email": email, "password": password})


def _user_login(client):
    return client.post(
        "/api/login", json={"email": SEED_ACCOUNT["email"], "password": SEED_ACCOUNT["password"]}
    )


class TestAdminAuth:
    def test_admin_login_valid(self, client):
        resp = _admin_login(client)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_admin_login_invalid(self, client):
        resp = _admin_login(client, password="wrong")
        assert resp.status_code == 401

    def test_admin_page_redirects_without_auth(self, client):
        resp = client.get("/admin", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin/login"

    def test_admin_page_serves_with_auth(self, client):
        _admin_login(client)
        resp = client.get("/admin")
        assert resp.status_code == 200

    def test_admin_session_separate_from_user(self, client):
        _user_login(client)
        resp = client.get("/api/admin/accounts")
        assert resp.status_code == 403


class TestAccountCRUD:
    def test_list_accounts(self, client):
        _admin_login(client)
        resp = client.get("/api/admin/accounts")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1  # just the seeded account

    def test_create_account(self, client):
        _admin_login(client)
        resp = client.post(
            "/api/admin/accounts",
            json={
                "email": "new@example.com",
                "password": "pass123",
                "display_name": "New Account",
                "logo_url": "/static/logos/synia.png",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == "new@example.com"

        accounts = client.get("/api/admin/accounts").json()
        assert len(accounts) == 2

    def test_create_duplicate_email(self, client):
        _admin_login(client)
        resp = client.post(
            "/api/admin/accounts",
            json={
                "email": SEED_ACCOUNT["email"],
                "password": "x",
                "display_name": "x",
                "logo_url": "/static/logos/synia.png",
            },
        )
        assert resp.status_code == 409

    def test_update_account(self, client):
        _admin_login(client)
        resp = client.put(
            f"/api/admin/accounts/{SEED_ACCOUNT['email']}",
            json={
                "email": SEED_ACCOUNT["email"],
                "password": "NewPass!",
                "display_name": "Updated Name",
                "logo_url": "/static/logos/synia.png",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Updated Name"

    def test_update_nonexistent(self, client):
        _admin_login(client)
        resp = client.put(
            "/api/admin/accounts/nobody@example.com",
            json={
                "email": "nobody@example.com",
                "password": "x",
                "display_name": "x",
                "logo_url": "/static/logos/synia.png",
            },
        )
        assert resp.status_code == 404

    def test_delete_account(self, client):
        _admin_login(client)
        resp = client.delete(f"/api/admin/accounts/{SEED_ACCOUNT['email']}")
        assert resp.status_code == 200
        accounts = client.get("/api/admin/accounts").json()
        assert len(accounts) == 0

    def test_delete_nonexistent(self, client):
        _admin_login(client)
        resp = client.delete("/api/admin/accounts/nobody@example.com")
        assert resp.status_code == 404

    def test_persistence(self, client, data_dir):
        _admin_login(client)
        client.post(
            "/api/admin/accounts",
            json={
                "email": "persist@test.com",
                "password": "p",
                "display_name": "Persist",
                "logo_url": "/static/logos/synia.png",
            },
        )
        saved = json.loads((data_dir / "accounts.json").read_text())
        assert any(a["email"] == "persist@test.com" for a in saved)


class TestLogoUpload:
    def test_upload_logo(self, client):
        _admin_login(client)
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp = client.post(
            "/api/admin/upload-logo",
            files={"file": ("logo.png", png_header, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["logo_url"].startswith("/logos/")
        assert data["logo_url"].endswith(".png")

    def test_upload_invalid_type(self, client):
        _admin_login(client)
        resp = client.post(
            "/api/admin/upload-logo",
            files={"file": ("doc.pdf", b"fake pdf", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_upload_too_large(self, client):
        _admin_login(client)
        large = b"\x00" * (513 * 1024)
        resp = client.post(
            "/api/admin/upload-logo",
            files={"file": ("big.png", large, "image/png")},
        )
        assert resp.status_code == 400

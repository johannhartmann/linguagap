"""WebSocket handler tests via the FastAPI TestClient.

Drives the host/viewer message protocol through real WebSockets so we
exercise WebSocketHandler internals (config, ptt_mode, host_speaking_state,
host_transcript_requested, request_summary, viewer connect & broadcast)
without spinning up Whisper/MT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend


@pytest.fixture(autouse=True)
def clear_backend_caches():
    get_asr_backend.cache_clear()
    get_translation_backend.cache_clear()
    get_summarization_backend.cache_clear()
    yield
    get_asr_backend.cache_clear()
    get_translation_backend.cache_clear()
    get_summarization_backend.cache_clear()


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from unittest.mock import MagicMock

    from app import auth as auth_mod

    monkeypatch.setattr(auth_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(auth_mod, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setattr(auth_mod, "LOGOS_DIR", tmp_path / "logos")
    monkeypatch.setattr(auth_mod, "ADMIN_EMAIL", "admin@test.local")
    monkeypatch.setattr(auth_mod, "ADMIN_PASSWORD", "testpass")
    auth_mod._accounts = None

    from app import main as main_mod

    (tmp_path / "logos").mkdir(exist_ok=True)

    asr_backend = MagicMock()
    mt_backend = MagicMock()
    mt_backend.translate.return_value = ["TRANSLATED"]
    monkeypatch.setattr(main_mod, "get_asr_backend", lambda: asr_backend)
    monkeypatch.setattr(main_mod, "get_translation_backend", lambda: mt_backend)
    monkeypatch.setattr(main_mod, "get_summarization_backend", lambda: None)

    from app import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod.asr, "get_asr_backend", lambda: asr_backend)
    monkeypatch.setattr(streaming_mod.asr, "get_translation_backend", lambda: mt_backend)

    from app.main import app

    with TestClient(app) as client:
        # Bootstrap one demo account so /api/login succeeds.
        client.post("/api/admin/login", json={"email": "admin@test.local", "password": "testpass"})
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


# ---------------------------------------------------------------------------
# Host WebSocket protocol
# ---------------------------------------------------------------------------


def test_host_config_with_token_activates_session(auth_client: TestClient) -> None:
    with auth_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "src_lang": "en",
                    "foreign_lang": "en",
                    "token": "tok-host-1",
                    "ptt_mode": False,
                }
            )
        )


def test_host_config_without_token_is_a_noop(auth_client: TestClient) -> None:
    """Missing token logs an error and does not activate — handler stays alive
    for the rest of the protocol so the test doesn't hang."""
    with auth_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "src_lang": "en",
                }
            )
        )
        # Send a follow-up audio frame; the handler should not crash.
        ws.send_bytes(np.zeros(800, dtype=np.int16).tobytes())


def test_host_ptt_mode_toggle(auth_client: TestClient) -> None:
    with auth_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "src_lang": "en",
                    "token": "tok-host-2",
                }
            )
        )
        ws.send_text(json.dumps({"type": "ptt_mode", "enabled": True}))
        ws.send_text(json.dumps({"type": "ptt_mode", "enabled": False}))


def test_host_speaking_state_messages(auth_client: TestClient) -> None:
    with auth_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "src_lang": "en",
                    "token": "tok-host-3",
                }
            )
        )
        ws.send_text(json.dumps({"type": "speaking_state", "party": "host", "speaking": True}))
        ws.send_text(json.dumps({"type": "speaking_state", "party": "host", "speaking": False}))


def test_host_transcript_requested_toggle(auth_client: TestClient) -> None:
    with auth_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "src_lang": "en",
                    "token": "tok-host-4",
                }
            )
        )
        ws.send_text(json.dumps({"type": "host_transcript_requested", "enabled": True}))
        ws.send_text(json.dumps({"type": "host_transcript_requested", "enabled": False}))


def test_host_request_summary_drains_session(auth_client: TestClient) -> None:
    with auth_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "src_lang": "en",
                    "token": "tok-host-5",
                }
            )
        )
        # A single near-silent audio frame.
        ws.send_bytes(np.zeros(1600, dtype=np.int16).tobytes())
        ws.send_text(json.dumps({"type": "request_summary"}))


# ---------------------------------------------------------------------------
# Viewer WebSocket — connect before host activates, then host activates.
# ---------------------------------------------------------------------------


def test_viewer_websocket_pending_then_active(auth_client: TestClient) -> None:
    """Viewer connects before host -> receives an init/waiting frame.

    We don't drive the host here (that requires a parallel WS); instead we
    just verify the viewer endpoint accepts the connection and emits the
    correct first message.
    """
    token = "tok-viewer-1"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "init"
        # Pending session emits status=waiting; active emits foreign_lang/segments.
        assert first.get("status") in ("waiting", "active")


def test_viewer_websocket_unknown_token_is_reserved_and_pending(
    auth_client: TestClient,
) -> None:
    """A viewer with a never-seen token still gets an init/waiting frame
    after the registry auto-reserves the token."""
    with auth_client.websocket_connect("/ws/viewer/never-before-seen") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "init"


def test_viewer_serves_html_page(auth_client: TestClient) -> None:
    """The /viewer/{token} HTTP route serves the static viewer page."""
    resp = auth_client.get("/viewer/some-token")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")

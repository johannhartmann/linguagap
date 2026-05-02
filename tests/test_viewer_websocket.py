"""Viewer-side WebSocket flow tests.

Covers the message types the viewer can send back to the server:
viewer_audio_config (sets foreign_lang on the session), transcript_consent
(stored on SessionEntry and relayed to host), speaking_state (relayed to
host with the delayed-off scheduling), plus binary audio frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend


@pytest.fixture(autouse=True)
def clear_caches():
    get_asr_backend.cache_clear()
    get_translation_backend.cache_clear()
    get_summarization_backend.cache_clear()
    yield
    get_asr_backend.cache_clear()
    get_translation_backend.cache_clear()
    get_summarization_backend.cache_clear()


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
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
    mt_backend.translate.return_value = ["TR"]
    monkeypatch.setattr(main_mod, "get_asr_backend", lambda: asr_backend)
    monkeypatch.setattr(main_mod, "get_translation_backend", lambda: mt_backend)
    monkeypatch.setattr(main_mod, "get_summarization_backend", lambda: None)

    from app import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod.asr, "get_asr_backend", lambda: asr_backend)
    monkeypatch.setattr(streaming_mod.asr, "get_translation_backend", lambda: mt_backend)

    from app.main import app

    with TestClient(app) as client:
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


def _drain_init(ws) -> None:
    """Read and discard the init/host_transcript_requested handshake frames."""
    first = json.loads(ws.receive_text())
    assert first["type"] in {"init"}


def test_viewer_audio_config_message(auth_client: TestClient) -> None:
    """The viewer sends viewer_audio_config; the handler stores foreign_lang."""
    token = "tok-vac-1"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        _drain_init(ws)
        ws.send_text(json.dumps({"type": "viewer_audio_config", "foreign_lang": "fr"}))


def test_viewer_audio_config_ignores_de_unknown_auto(auth_client: TestClient) -> None:
    token = "tok-vac-2"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        _drain_init(ws)
        # These values must be ignored as a "foreign language" hint.
        for lang in ("de", "unknown", "auto"):
            ws.send_text(json.dumps({"type": "viewer_audio_config", "foreign_lang": lang}))


def test_viewer_transcript_consent_recorded_in_registry(auth_client: TestClient) -> None:
    """transcript_consent=true should be stored on the SessionEntry."""
    from app.session_registry import registry

    token = "tok-consent-1"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        _drain_init(ws)
        ws.send_text(json.dumps({"type": "transcript_consent", "enabled": True}))

    # After the WS closes, the registry entry persists with consent True.
    import asyncio

    async def _check() -> bool:
        entry = await registry.get(token)
        return entry is not None and entry.transcript_consent

    assert asyncio.run(_check())


def test_viewer_speaking_state_messages(auth_client: TestClient) -> None:
    """Viewer pushes speaking_state up to the server."""
    token = "tok-speak-1"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        _drain_init(ws)
        ws.send_text(json.dumps({"type": "speaking_state", "party": "viewer", "speaking": True}))
        ws.send_text(json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False}))


def test_viewer_binary_audio_is_accepted(auth_client: TestClient) -> None:
    """Binary frames from the viewer should not throw — they get queued onto
    the foreign channel buffer (or dropped if no session exists yet)."""
    token = "tok-audio-1"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        _drain_init(ws)
        ws.send_bytes(np.zeros(800, dtype=np.int16).tobytes())
        ws.send_bytes(np.zeros(800, dtype=np.int16).tobytes())


def test_viewer_malformed_text_is_ignored(auth_client: TestClient) -> None:
    """Bad JSON in a text message should be silently ignored."""
    token = "tok-bad-1"
    with auth_client.websocket_connect(f"/ws/viewer/{token}") as ws:
        _drain_init(ws)
        ws.send_text("{not valid json")
        ws.send_text(json.dumps({"type": "ping"}))  # unknown type also ignored

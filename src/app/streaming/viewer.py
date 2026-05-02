"""Read-only viewer WebSocket endpoint + delayed-off helper."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import WebSocket

from app.session_registry import registry
from app.streaming import trace as trace_mod
from app.streaming.broadcast import broadcast_to_viewers
from app.streaming.serialize import _serialize_segments
from app.streaming.session import TICK_SEC
from app.streaming_policy import STABILITY_SEC

logger = logging.getLogger(__name__)


async def _delayed_viewer_speaking_off(token: str, delay: float) -> None:
    """Send delayed 'viewer stopped speaking' to host, looking up fresh WS from registry.

    Also force-finalizes pending segments — in PTT mode, audio stops on release
    so get_current_time() stops advancing and live segments never finalize.
    """
    try:
        await asyncio.sleep(delay)
        entry = await registry.get(token)
        if not entry:
            return
        # Force-finalize pending segments so they get translated and marked final
        if entry.session is not None:
            newly_final = entry.session.segment_tracker.force_finalize_all()
            # Queue newly-finalized segments for translation — without this,
            # the viewer's PTT-captured foreign speech gets transcribed but
            # never translated to German.
            if newly_final and entry.session.translation_queue is not None:
                for seg in newly_final:
                    await entry.session.translation_queue.put(seg)
            # Drain the foreign channel buffer so the next viewer PTT press
            # starts from clean audio. Without this the 8s sliding window keeps
            # replaying the just-released utterance into the next transcript
            # (observed bleed: "Can you hear me clearly?" reappearing as a
            # prefix of the *next* viewer segment).
            entry.session.foreign_channel.reset()
            trace_mod._trace("channel_reset", tok=token[:8], party="viewer")
            if newly_final and entry.main_ws:
                all_segments = list(entry.session.segment_tracker.finalized_segments)
                segments_data = _serialize_segments(entry.session, all_segments)
                segments_msg = {
                    "type": "segments",
                    "t": entry.session.get_current_time(),
                    "src_lang": entry.session.foreign_lang or "unknown",
                    "foreign_lang": entry.session.foreign_lang,
                    "dual_channel": entry.session.is_dual_channel(),
                    "segments": segments_data,
                }
                msg_json = json.dumps(segments_msg)
                with contextlib.suppress(Exception):
                    await entry.main_ws.send_text(msg_json)
                await broadcast_to_viewers(entry, segments_msg)
        if entry.main_ws:
            trace_mod._trace("viewer_speaking_off_delayed", tok=token[:8])
            await entry.main_ws.send_text(
                json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False})
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Failed to relay viewer speaking_state off to host", exc_info=True)


async def _register_viewer(websocket: WebSocket, token: str) -> None:
    """Add the viewer, reserving the token if no session entry exists yet."""
    if not await registry.add_viewer(token, websocket):
        await registry.reserve(token)
        await registry.add_viewer(token, websocket)


async def _send_init(websocket: WebSocket, token: str) -> None:
    """Send the init payload (active state + segments OR waiting placeholder)."""
    entry = await registry.get(token)
    if entry and entry.session is not None:
        session = entry.session
        live_segments = [cs.segment for cs in session.segment_tracker.cumulative_segments]
        all_segments = list(session.segment_tracker.finalized_segments) + live_segments
        segments_data = _serialize_segments(session, all_segments)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "init",
                    "status": "active",
                    "foreign_lang": session.foreign_lang,
                    "segments": segments_data,
                    "ptt_mode": session.ptt_mode,
                }
            )
        )
    else:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "init",
                    "status": "waiting",
                    "message": "Waiting for recording to start...",
                }
            )
        )

    if entry and entry.host_transcript_requested:
        await websocket.send_text(
            json.dumps({"type": "host_transcript_requested", "enabled": True})
        )


async def _resolve_session(token: str, cached_session):
    """Return cached_session if set, otherwise pull a live one from the registry."""
    if cached_session is not None:
        return cached_session
    entry = await registry.get(token)
    if entry and entry.session:
        return entry.session
    return None


async def _handle_viewer_audio_config(token: str, data: dict, cached_session):
    """Apply a viewer-supplied foreign-language hint to the active session."""
    viewer_foreign_lang = data.get("foreign_lang")
    if not viewer_foreign_lang or viewer_foreign_lang in ("de", "unknown", "auto"):
        return cached_session
    cached_session = await _resolve_session(token, cached_session)
    if cached_session is None:
        return None
    if cached_session.foreign_lang != viewer_foreign_lang:
        logger.info(
            "Viewer updated foreign language: %s -> %s",
            cached_session.foreign_lang,
            viewer_foreign_lang,
        )
    cached_session.foreign_lang = viewer_foreign_lang
    return cached_session


async def _handle_transcript_consent(token: str, data: dict) -> None:
    """Record viewer consent on the entry and relay to host (best-effort)."""
    enabled = bool(data.get("enabled", False))
    entry = await registry.get(token)
    if entry and enabled:
        entry.transcript_consent = True
        if entry.session is not None:
            entry.session.transcript_consent = True
    if entry and entry.main_ws:
        msg = {"type": "transcript_consent", "enabled": entry.transcript_consent}
        try:
            await entry.main_ws.send_text(json.dumps(msg))
        except Exception as exc:  # noqa: BLE001 - relay best-effort
            logger.warning(
                "Failed to relay transcript_consent to host for token=%s: %s",
                token[:8],
                exc,
                exc_info=True,
            )


async def _handle_speaking_state(
    token: str, data: dict, viewer_speaking_off_task: asyncio.Task | None
) -> asyncio.Task | None:
    """Relay viewer speaking state to host with a delayed-off coalesce."""
    entry = await registry.get(token)
    if not entry or not entry.main_ws:
        return viewer_speaking_off_task
    speaking = bool(data.get("speaking", False))
    trace_mod._trace("viewer_speaking", tok=token[:8], speaking=speaking)
    if speaking:
        if viewer_speaking_off_task and not viewer_speaking_off_task.done():
            viewer_speaking_off_task.cancel()
        msg = {"type": "speaking_state", "party": "viewer", "speaking": True}
        try:
            await entry.main_ws.send_text(json.dumps(msg))
        except Exception:
            logger.warning("Failed to relay viewer speaking_state to host")
        return None
    return asyncio.create_task(
        _delayed_viewer_speaking_off(token, STABILITY_SEC + TICK_SEC)
    )


async def _dispatch_text_message(
    token: str, text: str, cached_session, viewer_speaking_off_task: asyncio.Task | None
):
    """Parse a JSON text frame and route to the right handler."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, KeyError):
        return cached_session, viewer_speaking_off_task

    msg_type = data.get("type")
    if msg_type == "viewer_audio_config":
        cached_session = await _handle_viewer_audio_config(token, data, cached_session)
    elif msg_type == "transcript_consent":
        await _handle_transcript_consent(token, data)
    elif msg_type == "speaking_state":
        viewer_speaking_off_task = await _handle_speaking_state(
            token, data, viewer_speaking_off_task
        )
    return cached_session, viewer_speaking_off_task


async def _send_speaking_off_on_disconnect(token: str) -> None:
    """Tell the host the viewer stopped speaking when the WS goes away."""
    entry = await registry.get(token)
    if entry and entry.main_ws:
        trace_mod._trace("viewer_speaking_off_disconnect", tok=token[:8])
        with contextlib.suppress(Exception):
            await entry.main_ws.send_text(
                json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False})
            )


async def handle_viewer_websocket(websocket: WebSocket, token: str) -> None:
    """Handle a read-only viewer WebSocket connection."""
    await websocket.accept()
    await _register_viewer(websocket, token)
    logger.info("Viewer connected to session %s...", token[:8])

    cached_session = None
    viewer_speaking_off_task: asyncio.Task | None = None

    try:
        await _send_init(websocket, token)

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
                continue

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message:
                cached_session = await _resolve_session(token, cached_session)
                if cached_session is not None:
                    cached_session.add_foreign_audio(message["bytes"])
            elif "text" in message:
                cached_session, viewer_speaking_off_task = await _dispatch_text_message(
                    token, message["text"], cached_session, viewer_speaking_off_task
                )

    except Exception as e:
        logger.error("Viewer websocket error: %s", e)
    finally:
        if viewer_speaking_off_task and not viewer_speaking_off_task.done():
            viewer_speaking_off_task.cancel()
        await _send_speaking_off_on_disconnect(token)
        await registry.remove_viewer(token, websocket)
        logger.info("Viewer disconnected from session %s...", token[:8])

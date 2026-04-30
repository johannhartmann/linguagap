"""
Real-time streaming transcription and translation pipeline.

This module implements the WebSocket streaming handler for real-time bilingual
conversation transcription. It orchestrates a diarization-first pipeline:

    Audio → Diarization → Per-Speaker Language Detection → ASR → Translation

Architecture:
    - StreamingSession: Manages audio buffering and session state
    - asr_loop: Runs every TICK_SEC, processes audio windows
    - mt_loop: Translates finalized segments asynchronously
    - handle_websocket: Main WebSocket handler, coordinates both loops

Key design decisions:
    - Diarization runs FIRST to identify speakers before ASR
    - Language is detected per-speaker using SpeechBrain (not window-level)
    - ASR runs per-speaker segment with correct language hint
    - Segments finalize based on time-stability (not text-stability)
    - Translation happens asynchronously to avoid blocking ASR updates

Threading model:
    - ASR/MT run in ThreadPoolExecutor to avoid blocking async event loop
    - Two async tasks (asr_loop, mt_loop) coordinate via asyncio.Queue
    - Viewers receive real-time updates via broadcast mechanism
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket

from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend
from app.languages import LANG_INFO
from app.session_registry import SessionEntry, registry
from app.streaming_policy import STABILITY_SEC, Segment, SegmentTracker

logger = logging.getLogger(__name__)

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "8.0"))
TICK_SEC = float(os.getenv("TICK_SEC", "0.5"))
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "30.0"))

# Crosstalk suppression: when both mics share a room, one picks up the other's
# speaker. Suppress the quieter channel only when (a) the louder side is clearly
# active and (b) it is much louder than the other.
_CROSSTALK_MIN_ACTIVE_RMS = 0.02  # below this a channel is treated as noise/silence
_CROSSTALK_DOMINANCE_RATIO = 4.0  # dominant side must be Nx louder to suppress the other

# Opt-in high-resolution tracing of the phrase lifecycle (ASR → MT → WS).
# Set LINGUAGAP_TRACE=1 to emit a TRACE line at each stage so lost-transcription
# bugs can be reconstructed from logs. Lines look like:
#   TRACE 12:34:56.789 ev=asr_emit tok=abc12345 seg=42 final=False src_lang=de text="Guten Tag"
TRACE_ENABLED = os.getenv("LINGUAGAP_TRACE", "1") == "1"


def _trace(event: str, **fields) -> None:
    """Emit a single high-resolution trace line if TRACE_ENABLED."""
    if not TRACE_ENABLED:
        return
    now = time.time()
    ts = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
    parts = [f"ev={event}"]
    for k, v in fields.items():
        if isinstance(v, str):
            flat = v.replace("\n", " ").replace("\r", " ")
            if len(flat) > 200:
                flat = flat[:197] + "..."
            parts.append(f'{k}="{flat}"')
        else:
            parts.append(f"{k}={v}")
    logger.info("TRACE %s %s", ts, " ".join(parts))


_executor = ThreadPoolExecutor(max_workers=2)

# Metrics
_metrics: dict[str, deque[float]] = {
    "asr_times": deque(maxlen=100),
    "mt_times": deque(maxlen=100),
    "diar_times": deque(maxlen=100),
    "tick_times": deque(maxlen=100),
}


def _role_from_lang(lang: str | None) -> str | None:
    """Map a source language to a semantic speaker role."""
    if lang == "de":
        return "german"
    if lang and lang != "unknown":
        return "foreign"
    return None


def _resolve_segment_role(segment: Segment, dual_channel: bool) -> str | None:
    """Resolve the role for a segment, preferring explicit role metadata."""
    if segment.speaker_role in {"german", "foreign"}:
        return segment.speaker_role

    if dual_channel:
        # In dual-channel mode, speaker IDs are fixed to roles by the pipeline.
        # If we don't have a known SPEAKER_xx, do NOT guess by language —
        # that flips the UI when Whisper misdetects.
        if segment.speaker_id == "SPEAKER_00":
            return "german"
        if segment.speaker_id == "SPEAKER_01":
            return "foreign"
        return None

    return _role_from_lang(segment.src_lang)


def _serialize_segments(session: "StreamingSession", segments: list[Segment]) -> list[dict]:
    """Convert Segment objects to dicts with resolved roles and translations."""
    dual_channel = session.is_dual_channel()
    result = []
    for seg in segments:
        seg_dict = asdict(seg)
        speaker_role = _resolve_segment_role(seg, dual_channel)
        seg_dict["speaker_role"] = speaker_role

        # Override src_lang if role is certain, to prevent Whisper's
        # misdetections from breaking the translation logic.
        if speaker_role == "german":
            seg_dict["src_lang"] = "de"
        elif (
            speaker_role == "foreign" and session.foreign_lang and session.foreign_lang in LANG_INFO
        ):
            seg_dict["src_lang"] = session.foreign_lang

        seg_dict["translations"] = session.translations.get(seg.id, {})
        result.append(seg_dict)
    return result


def _resolve_translation_pair(
    segment: Segment,
    role: str | None,
    foreign_lang: str | None,
) -> tuple[str, str] | None:
    """Determine (src_lang, tgt_lang) for a segment, or None to skip translation."""
    foreign = foreign_lang if foreign_lang in LANG_INFO else None

    if role == "german":
        return ("de", foreign) if foreign else None

    if role == "foreign":
        # Foreign channel always translates to German. Use the session's
        # foreign_lang, not segment.src_lang which may be misdetected.
        return (foreign, "de") if foreign else None

    src = segment.src_lang
    if src == "de" and foreign:
        return src, foreign
    return src, "de"


def _is_effective_silence(
    audio: np.ndarray, rms_threshold: float = 0.02, peak_threshold: float = 0.05
) -> bool:
    """Detect near-silent buffers that should not be sent to ASR."""
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    return rms < rms_threshold and peak < peak_threshold


def _avg_ms(samples: deque) -> float:
    return sum(samples) / len(samples) * 1000 if samples else 0


def get_metrics() -> dict:
    return {
        "avg_asr_time_ms": _avg_ms(_metrics["asr_times"]),
        "avg_mt_time_ms": _avg_ms(_metrics["mt_times"]),
        "avg_diar_time_ms": _avg_ms(_metrics["diar_times"]),
        "avg_tick_time_ms": _avg_ms(_metrics["tick_times"]),
        "sample_count": len(_metrics["tick_times"]),
    }


async def broadcast_to_viewers(entry: SessionEntry, message: dict) -> None:
    """Broadcast JSON message to all viewers, remove dead connections."""
    if not entry.viewers:
        return

    message_json = json.dumps(message)
    dead_viewers: list[WebSocket] = []

    for viewer_ws in list(entry.viewers):
        try:
            await viewer_ws.send_text(message_json)
        except Exception:
            dead_viewers.append(viewer_ws)

    # Clean up dead viewers
    for viewer_ws in dead_viewers:
        entry.viewers.discard(viewer_ws)


async def _maybe_broadcast(session_token: str | None, message: dict) -> None:
    """Broadcast to viewers if session_token is set and session exists."""
    if session_token:
        entry = await registry.get(session_token)
        if entry:
            await broadcast_to_viewers(entry, message)


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
            _trace("channel_reset", tok=token[:8], party="viewer")
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
            _trace("viewer_speaking_off_delayed", tok=token[:8])
            await entry.main_ws.send_text(
                json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False})
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Failed to relay viewer speaking_state off to host", exc_info=True)


@dataclass
class ChannelBuffer:
    """Audio buffer for a single channel (German or foreign speaker)."""

    audio: deque[bytes] = field(default_factory=deque)
    total_samples: int = 0
    trimmed_samples: int = 0
    start_offset_sec: float = 0.0
    started: bool = False

    def reset(self) -> None:
        """Drop buffered audio so the next utterance starts from a clean slate.

        Used after PTT release: without this, the 8s sliding ASR window keeps
        re-transcribing the just-finalized utterance every tick and the
        trailing words bleed into the *next* PTT press's transcript. Resetting
        ``started`` lets the next add_audio call re-anchor ``start_offset_sec``
        to the current wall-clock — combined with reset counters this keeps
        segment timestamps strictly ascending so SegmentTracker won't merge
        new segments with stale finalized ones.
        """
        self.audio.clear()
        self.total_samples = 0
        self.trimmed_samples = 0
        self.started = False


class StreamingSession:
    """
    Manages state for a single streaming transcription session.

    Maintains the audio buffer, segment tracking, and per-speaker language
    detection for a WebSocket connection. Each browser tab creates one session.

    Attributes:
        sample_rate: Audio sample rate (typically 16000 Hz)
        src_lang: User-selected source language or "auto" for detection
        audio_buffer: Rolling buffer of PCM16 audio bytes
        detected_lang: Currently detected language from ASR
        foreign_lang: The non-German language in the conversation
        segment_tracker: Tracks segment finalization state
        translations: Cached translations keyed by segment_id and language
    """

    def __init__(self, sample_rate: int = 16000, src_lang: str = "auto"):
        self.sample_rate = sample_rate
        self.src_lang = src_lang  # User-selected source language (or "auto")
        self.audio_buffer: deque[bytes] = deque()
        self.total_samples = 0
        self.start_time = time.time()  # Wall-clock time when session started
        self.detected_lang: str | None = None  # Currently detected language from ASR
        self.foreign_lang: str | None = (
            None  # The non-German language (auto-detected or user-selected)
        )
        self.segment_tracker = SegmentTracker()
        self.dropped_frames = 0
        self.trimmed_samples = 0  # Track total samples trimmed from buffer start
        self.translations: dict[int, dict[str, str]] = {}  # segment_id -> {lang -> translation}

        # Dual-channel buffers (German mic on main page, foreign mic on viewer)
        self.german_channel = ChannelBuffer()
        self.foreign_channel = ChannelBuffer()
        self.viewer_last_audio_time: float = 0.0
        self.dual_channel_locked: bool = False
        self.ptt_mode: bool = False
        # True once at least one viewer has explicitly consented to the
        # bilingual transcript download. Sticky: a later viewer opting out
        # does not revoke an earlier consent from another viewer.
        self.transcript_consent: bool = False
        # Set by WebSocketHandler after session creation so non-handler code
        # (e.g. viewer WebSocket loop) can queue segments for translation.
        self.translation_queue: asyncio.Queue[Segment] | None = None

    def add_audio(self, pcm16_bytes: bytes):
        self.audio_buffer.append(pcm16_bytes)
        self.total_samples += len(pcm16_bytes) // 2
        self._enforce_max_buffer()

    def _enforce_max_buffer(self) -> None:
        trimmed = self._trim_buffer(self.audio_buffer)
        if trimmed > 0:
            self.trimmed_samples += trimmed
            self.dropped_frames += 1

    def _trim_buffer(self, buffer: deque[bytes]) -> int:
        """Trim a buffer to MAX_BUFFER_SEC. Returns number of samples trimmed."""
        max_samples = int(MAX_BUFFER_SEC * self.sample_rate)
        all_bytes = b"".join(buffer)
        total_samples = len(all_bytes) // 2

        if total_samples > max_samples:
            excess_samples = total_samples - max_samples
            trimmed_bytes = all_bytes[excess_samples * 2 :]
            buffer.clear()
            buffer.append(trimmed_bytes)
            return excess_samples
        return 0

    def _add_channel_audio(self, pcm16_bytes: bytes, channel: ChannelBuffer) -> None:
        """Add audio to a channel buffer with first-chunk offset tracking."""
        if not channel.started:
            chunk_sec = (len(pcm16_bytes) // 2) / self.sample_rate
            elapsed = max(0.0, time.time() - self.start_time)
            channel.start_offset_sec = max(0.0, elapsed - chunk_sec)
            channel.started = True
        channel.audio.append(pcm16_bytes)
        channel.total_samples += len(pcm16_bytes) // 2
        trimmed = self._trim_buffer(channel.audio)
        channel.trimmed_samples += trimmed
        self.add_audio(pcm16_bytes)

    def add_german_audio(self, pcm16_bytes: bytes) -> None:
        """Add audio from the main page (German speaker) to the german channel buffer."""
        self._add_channel_audio(pcm16_bytes, self.german_channel)

    def add_foreign_audio(self, pcm16_bytes: bytes) -> None:
        """Add audio from the viewer (foreign speaker) to the foreign channel buffer."""
        self._add_channel_audio(pcm16_bytes, self.foreign_channel)
        self.viewer_last_audio_time = time.time()
        self.dual_channel_locked = True

    def _get_channel_window_audio(self, channel: ChannelBuffer) -> tuple[np.ndarray, float]:
        """Get windowed audio from a channel buffer as float32 array."""
        all_bytes = b"".join(channel.audio)
        if not all_bytes:
            return np.array([], dtype=np.float32), 0.0
        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_window_samples = int(WINDOW_SEC * self.sample_rate)
        if len(samples) > max_window_samples:
            samples = samples[-max_window_samples:]
        window_start = channel.start_offset_sec + (
            max(0, channel.total_samples - len(samples)) / self.sample_rate
        )
        return samples, window_start

    def get_german_window_audio(self) -> tuple[np.ndarray, float]:
        """Get the German channel audio buffer as a float32 array."""
        return self._get_channel_window_audio(self.german_channel)

    def get_foreign_window_audio(self) -> tuple[np.ndarray, float]:
        """Get the foreign channel audio buffer as a float32 array."""
        return self._get_channel_window_audio(self.foreign_channel)

    def resolve_foreign_lang(self) -> None:
        """Set foreign_lang from user selection if not yet auto-detected."""
        if self.foreign_lang is None and self.src_lang not in ("auto", "de"):
            self.foreign_lang = self.src_lang
            logger.info("Foreign language set from user selection: %s", self.foreign_lang)

    def is_dual_channel(self) -> bool:
        """Use dual-channel mode once foreign mic audio has been received at least once."""
        return self.dual_channel_locked

    def get_window_audio(self) -> tuple[np.ndarray, float]:
        """Get audio for ASR processing.

        Returns the full audio buffer to ensure complete coverage. Previous
        sliding window logic caused gaps when audio arrived faster than
        real-time (e.g., test scenarios with pre-recorded audio).

        Performance note: The diarization-first pipeline runs ASR only on
        individual speaker segments, not the full audio, so processing time
        scales with speech duration rather than total buffer duration.

        Buffer size is capped at MAX_BUFFER_SEC (default 30s) by
        _enforce_max_buffer(), which trims oldest audio when exceeded.
        """
        all_bytes = b"".join(self.audio_buffer)
        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        window_start = self.trimmed_samples / self.sample_rate
        return samples, window_start

    def get_current_time(self) -> float:
        if self.is_dual_channel():
            german_end = self.german_channel.start_offset_sec + (
                self.german_channel.total_samples / self.sample_rate
            )
            foreign_end = self.foreign_channel.start_offset_sec + (
                self.foreign_channel.total_samples / self.sample_rate
            )
            return max(german_end, foreign_end)
        return self.total_samples / self.sample_rate

    def get_buffered_seconds(self) -> float:
        if self.is_dual_channel():
            german_sec = len(b"".join(self.german_channel.audio)) / 2 / self.sample_rate
            foreign_sec = len(b"".join(self.foreign_channel.audio)) / 2 / self.sample_rate
            return max(german_sec, foreign_sec)
        all_bytes = b"".join(self.audio_buffer)
        return len(all_bytes) / 2 / self.sample_rate


def _build_prompt(backend, language: str | None, finalized_text: str = "") -> str | None:
    """Build ASR prompt from bilingual hint + recent finalized transcript."""
    prompt = backend.get_bilingual_prompt(language) if language else None
    if finalized_text:
        # Whisper prompt is limited to ~224 tokens; keep last ~200 chars
        context = finalized_text[-200:].strip()
        prompt = f"{prompt} {context}" if prompt else context
    return prompt


# NOTE: run_speaker_detection, extract_speaker_audio, _extract_segment_audio,
# _transcribe_speaker_segment, and run_asr (with diarization) were removed.
# Speakers and languages are now fixed via dual-channel architecture:
# - Desktop = German (SPEAKER_00)
# - Viewer = Foreign language (SPEAKER_01)
# See run_asr_german_channel() and run_asr_dual_channel().


def _transcribe_channel(
    backend,
    audio: np.ndarray,
    language: str | None,
    speaker_id: str,
    speaker_role: str,
    force_lang: str | None = None,
    finalized_text: str = "",
) -> list[dict]:
    """Transcribe a full channel buffer via the ASR backend."""
    if len(audio) < 1600:
        return []
    if _is_effective_silence(audio):
        return []

    prompt = _build_prompt(backend, language, finalized_text)
    asr_result = backend.transcribe(audio, language=language, initial_prompt=prompt)
    filtered = asr_result.segments

    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker_id": speaker_id,
            "lang": force_lang or seg.language,
            "speaker_role": speaker_role,
        }
        for seg in filtered
    ]


def run_asr_german_channel(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """
    Run ASR in deterministic desktop mode.

    When the foreign language is known and only desktop audio is active, treat
    desktop input as German-only to prevent role/language drift.
    """
    tick_start = time.time()
    now_sec = session.get_current_time()
    german_audio, german_offset = session.get_german_window_audio()

    if len(german_audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    backend = get_asr_backend()
    asr_start = time.time()

    # Only seed the German prompt with prior German history — leftover
    # foreign-role segments from a past dual-channel phase would bias Whisper
    # the wrong way.
    recent_final = session.segment_tracker.finalized_segments[-10:]
    finalized_text = " ".join(s.src for s in recent_final if s.speaker_role != "foreign")[-500:]
    german_results = _transcribe_channel(
        backend,
        german_audio,
        "de",
        "SPEAKER_00",
        "german",
        force_lang="de",
        finalized_text=finalized_text,
    )

    for seg in german_results:
        seg["start"] += german_offset
        seg["end"] += german_offset

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)
    logger.debug("German-channel ASR: %d segments in %.1fms", len(german_results), asr_time * 1000)

    session.resolve_foreign_lang()

    session.detected_lang = "de"
    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=german_results,
        window_start=0.0,
        now_sec=now_sec,
        src_lang="de",
    )

    tick_time = time.time() - tick_start
    _metrics["tick_times"].append(tick_time)
    return all_segments, newly_finalized


def run_asr_dual_channel(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """
    Run ASR with separate German and foreign audio channels.

    Used when the viewer is actively sending audio (dual-channel mode).
    Much simpler than run_asr() - no diarization or language detection needed
    since each channel is a known speaker with a known language.

    Returns:
        Tuple of (all_segments, newly_finalized)
    """
    tick_start = time.time()
    now_sec = session.get_current_time()

    german_audio, german_offset = session.get_german_window_audio()
    foreign_audio, foreign_offset = session.get_foreign_window_audio()

    if len(german_audio) < 1600 and len(foreign_audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    backend = get_asr_backend()

    # Why dominance-based, not floor-based: a phone mic across a desk picks up
    # the host at 0.05–0.1 RMS — loud enough to bleed through, not loud enough
    # to trip an absolute floor. Constants live at module top.
    german_rms = float(np.sqrt(np.mean(german_audio**2))) if len(german_audio) > 0 else 0.0
    foreign_rms = float(np.sqrt(np.mean(foreign_audio**2))) if len(foreign_audio) > 0 else 0.0

    if (
        german_rms > _CROSSTALK_MIN_ACTIVE_RMS
        and german_rms > foreign_rms * _CROSSTALK_DOMINANCE_RATIO
    ):
        logger.debug(
            "Suppressing foreign channel (crosstalk from german, rms %.3f vs %.3f)",
            german_rms,
            foreign_rms,
        )
        foreign_audio = np.array([], dtype=np.float32)
    elif (
        foreign_rms > _CROSSTALK_MIN_ACTIVE_RMS
        and foreign_rms > german_rms * _CROSSTALK_DOMINANCE_RATIO
    ):
        logger.debug(
            "Suppressing german channel (crosstalk from foreign, rms %.3f vs %.3f)",
            foreign_rms,
            german_rms,
        )
        german_audio = np.array([], dtype=np.float32)

    logger.debug(
        "Dual-channel pipeline: german=%d samples (rms=%.3f), foreign=%d samples (rms=%.3f)",
        len(german_audio),
        german_rms,
        len(foreign_audio),
        foreign_rms,
    )

    asr_start = time.time()

    # Channel-local finalized history. Whisper is strongly biased by its
    # initial_prompt: feeding the foreign channel German context early in a
    # session (when only German has finalized) makes it hallucinate a German
    # paraphrase of English audio despite language="en" being forced. Keep
    # each channel's prompt monolingual to prevent cross-language drift.
    recent_final = session.segment_tracker.finalized_segments[-10:]
    german_history = " ".join(s.src for s in recent_final if s.speaker_role == "german")[-500:]
    foreign_history = " ".join(s.src for s in recent_final if s.speaker_role == "foreign")[-500:]

    # Transcribe both channels via backend
    german_results = _transcribe_channel(
        backend,
        german_audio,
        "de",
        "SPEAKER_00",
        "german",
        force_lang="de",
        finalized_text=german_history,
    )
    foreign_results = _transcribe_channel(
        backend,
        foreign_audio,
        session.foreign_lang,
        "SPEAKER_01",
        "foreign",
        force_lang=session.foreign_lang,
        finalized_text=foreign_history,
    )

    # Offset segment timestamps to absolute time
    for seg in german_results:
        seg["start"] += german_offset
        seg["end"] += german_offset
    for seg in foreign_results:
        seg["start"] += foreign_offset
        seg["end"] += foreign_offset

    # force_lang on both channels ensures correct labels.
    # Only fall back to Whisper detection if foreign_lang was never configured.
    if session.foreign_lang is None:
        for seg in foreign_results:
            lang = seg.get("lang")
            if lang and lang not in ("unknown", "de") and lang in LANG_INFO:
                session.foreign_lang = lang
                logger.info("Dual-channel: foreign language detected as %s", lang)
                break

    # Filter cross-channel bleed: drop foreign segments that duplicate german ones
    if german_results and foreign_results:
        german_texts = {" ".join(seg["text"].lower().split()) for seg in german_results}
        foreign_results = [
            seg
            for seg in foreign_results
            if " ".join(seg["text"].lower().split()) not in german_texts
        ]

    hyp_segments = german_results + foreign_results
    hyp_segments.sort(key=lambda s: s["start"])

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    logger.debug("Dual-channel ASR: %d segments in %.1fms", len(hyp_segments), asr_time * 1000)

    session.detected_lang = session.foreign_lang or "unknown"

    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=hyp_segments,
        window_start=0.0,
        now_sec=now_sec,
        src_lang=session.detected_lang,
    )

    tick_time = time.time() - tick_start
    _metrics["tick_times"].append(tick_time)

    return all_segments, newly_finalized


def run_translation(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate a single text from src_lang to tgt_lang."""
    mt_start = time.time()
    mt_backend = get_translation_backend()
    result = mt_backend.translate([text], src_lang=src_lang, tgt_lang=tgt_lang)[0]
    mt_time = time.time() - mt_start
    _metrics["mt_times"].append(mt_time)
    return result


class WebSocketHandler:
    """Manages the lifecycle of a single WebSocket streaming session.

    Coordinates ASR and MT async loops, handles incoming messages (config,
    audio, request_summary), and broadcasts updates to viewers.

    Message protocol:
        Client → Server:
            - {"type": "config", "sample_rate": 16000, "src_lang": "auto", "token": "..."}
            - Binary PCM16 audio frames (16kHz mono)
            - {"type": "request_summary"} (stops recording, drains translations)

        Server → Client:
            - {"type": "config_ack", "status": "active"}
            - {"type": "segments", "segments": [...], "src_lang": "...", "foreign_lang": "..."}
            - {"type": "translation", "segment_id": N, "tgt_lang": "de", "text": "..."}
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.session: StreamingSession | None = None
        self.session_token: str | None = None
        self._asr_task: asyncio.Task | None = None
        self._mt_task: asyncio.Task | None = None
        self._running = True
        self._translation_queue: asyncio.Queue[Segment] = asyncio.Queue()
        self._msg_count = 0
        self._bytes_received = 0
        self._host_speaking_off_task: asyncio.Task | None = None
        # Lifecycle tracing state: remember the last text+final seen per
        # segment id so we can distinguish first-emit from update events,
        # and remember enqueue timestamps to report MT queue-wait durations.
        self._trace_seen: dict[int, tuple[str, bool]] = {}
        self._trace_mt_enq: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Accept the WebSocket and run the message loop until disconnect."""
        await self.websocket.accept()
        try:
            while True:
                message = await self.websocket.receive()
                self._msg_count += 1

                if message["type"] == "websocket.disconnect":
                    logger.info(
                        "WebSocket disconnect after %d msgs, %d bytes",
                        self._msg_count,
                        self._bytes_received,
                    )
                    break

                await self._handle_message(message)

        except Exception as e:
            logger.error("WebSocket error: %s", e)
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, message: Mapping[str, Any]) -> None:
        if "text" in message:
            data = json.loads(message["text"])
            msg_type = data.get("type")
            if msg_type == "config":
                await self._handle_config(data)
            elif msg_type == "request_summary":
                await self._handle_request_summary()
            elif msg_type == "ptt_mode":
                await self._handle_ptt_mode(data)
            elif msg_type == "speaking_state":
                await self._handle_host_speaking_state(data)
            elif msg_type == "host_transcript_requested":
                await self._handle_host_transcript_requested(data)
        elif "bytes" in message:
            self._handle_audio(message["bytes"])

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    async def _handle_config(self, data: dict) -> None:
        sample_rate = data.get("sample_rate", 16000)
        src_lang = data.get("src_lang", "auto")
        foreign_lang = data.get("foreign_lang")

        self.session_token = data.get("token")
        if not self.session_token:
            logger.error("No token provided in config message")
            return

        self.session = StreamingSession(sample_rate=sample_rate, src_lang=src_lang)
        self.session.translation_queue = self._translation_queue
        if foreign_lang and foreign_lang not in ("auto", "de", "unknown"):
            self.session.foreign_lang = foreign_lang
        # Honour a pre-activation PTT toggle so the session_active broadcast
        # below already carries the right mode and the viewer doesn't flicker
        # between non-PTT and PTT UI.
        self.session.ptt_mode = bool(data.get("ptt_mode", False))

        self._asr_task = asyncio.create_task(self._asr_loop())
        self._mt_task = asyncio.create_task(self._mt_loop())

        await registry.activate(self.session_token, self.session, self.websocket)
        logger.info(
            "Session activated: sample_rate=%d, src_lang=%s, foreign_lang=%s, session=%s...",
            sample_rate,
            src_lang,
            foreign_lang or "auto",
            self.session_token[:8],
        )

        _trace(
            "session_start",
            tok=self._trace_tok(),
            src_lang=src_lang,
            foreign_lang=foreign_lang or "auto",
            ptt=self.session.ptt_mode,
        )
        await self._send_json({"type": "config_ack", "status": "active"})

        entry = await registry.get(self.session_token)
        # A freshly-activated StreamingSession defaults transcript_consent=False.
        # Carry pre-activation viewer consent from the entry onto the new
        # session so the post-stop download button renders for the host.
        # Host notification is best-effort — a failure here must not abort
        # activation or block the viewer-side session_active broadcast.
        if entry and entry.transcript_consent:
            self.session.transcript_consent = True
            try:
                await self._send_json({"type": "transcript_consent", "enabled": True})
            except Exception as exc:  # noqa: BLE001 - best-effort relay
                logger.warning(
                    "transcript_consent host notify failed for token=%s: %s",
                    self.session_token[:8] if self.session_token else "?",
                    exc,
                )
        if entry and entry.viewers:
            await broadcast_to_viewers(
                entry,
                {
                    "type": "session_active",
                    "foreign_lang": self.session.foreign_lang,
                    "segments": [],
                    "ptt_mode": self.session.ptt_mode,
                },
            )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _handle_audio(self, audio_bytes: bytes) -> None:
        if self.session is not None:
            self._bytes_received += len(audio_bytes)
            self.session.add_german_audio(audio_bytes)
            if self._msg_count % 50 == 0:
                logger.debug(
                    "Audio: %d bytes, %.1fs buffered",
                    self._bytes_received,
                    self.session.get_buffered_seconds(),
                )

    # ------------------------------------------------------------------
    # Push-to-Talk
    # ------------------------------------------------------------------

    async def _handle_ptt_mode(self, data: dict) -> None:
        if self.session is None:
            return
        enabled = bool(data.get("enabled", False))
        self.session.ptt_mode = enabled
        logger.info("PTT mode %s", "enabled" if enabled else "disabled")
        await _maybe_broadcast(self.session_token, {"type": "ptt_mode", "enabled": enabled})

    async def _handle_host_transcript_requested(self, data: dict) -> None:
        if self.session_token is None:
            return
        enabled = bool(data.get("enabled", False))
        entry = await registry.get(self.session_token)
        if entry is not None:
            entry.host_transcript_requested = enabled
            # If the host turns the request off, the prior viewer consent
            # becomes irrelevant for this run — reset it so a later toggle-on
            # re-prompts the viewer explicitly.
            if not enabled:
                entry.transcript_consent = False
                if entry.session is not None:
                    entry.session.transcript_consent = False
        logger.info("Host transcript requested: %s", enabled)
        await _maybe_broadcast(
            self.session_token,
            {"type": "host_transcript_requested", "enabled": enabled},
        )

    async def _handle_host_speaking_state(self, data: dict) -> None:
        if self.session is None or self.session_token is None:
            return
        speaking = bool(data.get("speaking", False))
        if speaking:
            # Cancel any pending "speaking off" relay
            if self._host_speaking_off_task and not self._host_speaking_off_task.done():
                self._host_speaking_off_task.cancel()
                self._host_speaking_off_task = None
            # Broadcast immediately
            await _maybe_broadcast(
                self.session_token,
                {"type": "speaking_state", "party": "host", "speaking": True},
            )
        else:
            # Delay broadcast to let ASR finalize last segments
            delay = STABILITY_SEC + TICK_SEC
            self._host_speaking_off_task = asyncio.create_task(
                self._delayed_speaking_broadcast("host", delay)
            )

    async def _delayed_speaking_broadcast(self, party: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            # In PTT mode, audio stops when the key/button is released, so
            # get_current_time() (audio-based) no longer advances past the last
            # segment's end time. Without this, live segments never meet the
            # STABILITY_SEC finalization threshold and stay stuck as live.
            await self._finalize_pending_segments()
            await _maybe_broadcast(
                self.session_token,
                {"type": "speaking_state", "party": party, "speaking": False},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Failed to broadcast speaking_state off for %s", party, exc_info=True)

    async def _finalize_pending_segments(self) -> None:
        """Force-finalize live segments and queue them for translation.

        Used on PTT release: without new audio, get_current_time() freezes and
        live segments never finalize via the normal time-based threshold.
        Also drains the host (German) channel buffer so the next press starts
        from clean audio — otherwise the 8s sliding window keeps replaying the
        just-released utterance into subsequent transcripts.
        """
        if self.session is None:
            return
        newly_final = self.session.segment_tracker.force_finalize_all()
        await self._enqueue_for_translation(newly_final, reason="ptt_release")
        # Drop any audio still buffered for the host channel — the next PTT
        # press should start from a clean window. Done unconditionally because
        # even if no segment finalized this round, sub-second trailing audio
        # would otherwise bleed into the next utterance.
        self.session.german_channel.reset()
        _trace("channel_reset", tok=self._trace_tok(), party="host")
        if newly_final:
            # Broadcast updated segment state so clients see final=True
            all_segments = list(self.session.segment_tracker.finalized_segments)
            segments_data = _serialize_segments(self.session, all_segments)
            segments_msg = {
                "type": "segments",
                "t": self.session.get_current_time(),
                "src_lang": self.session.foreign_lang or "unknown",
                "foreign_lang": self.session.foreign_lang,
                "dual_channel": self.session.is_dual_channel(),
                "segments": segments_data,
            }
            await self._send_and_broadcast(segments_msg)

    # ------------------------------------------------------------------
    # Request summary (stop recording + summarize)
    # ------------------------------------------------------------------

    async def _handle_request_summary(self) -> None:
        _trace("stop_requested", tok=self._trace_tok(), qsize=self._translation_queue.qsize())
        await self._cancel_task("_asr_task")
        logger.debug("ASR loop stopped for stop request")

        if self.session is None:
            return

        # Finalize remaining segments
        now_sec = self.session.get_current_time()
        all_segs, time_finalized = self.session.segment_tracker.update_from_hypothesis(
            [], 0.0, now_sec, "unknown"
        )

        tok = self._trace_tok()
        if time_finalized:
            logger.debug("Queuing %d time-finalized segments", len(time_finalized))
            await self._enqueue_for_translation(time_finalized, reason="stop_time")

        live_segs = [s for s in all_segs if not s.final]
        if live_segs:
            logger.debug("Force-finalizing %d live segments", len(live_segs))
            newly_final = self.session.segment_tracker.force_finalize_all()
            await self._enqueue_for_translation(newly_final, reason="stop_force")

        # Wait for translations to complete
        _trace("stop_drain_begin", tok=tok, qsize=self._translation_queue.qsize())
        try:
            await asyncio.wait_for(self._translation_queue.join(), timeout=60)
            _trace("stop_drain_done", tok=tok)
        except TimeoutError:
            logger.warning(
                "Translation queue join timed out, %d items remaining",
                self._translation_queue.qsize(),
            )
            _trace(
                "stop_drain_timeout",
                tok=tok,
                remaining=self._translation_queue.qsize(),
            )

        # Send final segments with all translations. Broadcast to viewers FIRST
        # so their transcript download state is complete even if the host has
        # already disconnected mid-stop; then tolerate a host-send failure.
        finalized = self.session.segment_tracker.finalized_segments
        if finalized:
            segments_data = _serialize_segments(self.session, list(finalized))
            segments_msg = {
                "type": "segments",
                "t": self.session.get_current_time(),
                "src_lang": self.session.detected_lang or "unknown",
                "foreign_lang": self.session.foreign_lang or "en",
                "segments": segments_data,
            }
            await _maybe_broadcast(self.session_token, segments_msg)
            try:
                await self._send_json(segments_msg)
            except Exception as exc:  # noqa: BLE001 - host may be mid-disconnect
                logger.warning(
                    "Final segments host send failed for token=%s: %s",
                    self.session_token[:8] if self.session_token else "?",
                    exc,
                )

        await self._run_summarization(finalized)

        self._running = False
        logger.info("Recording stopped, closing connection")
        await self.websocket.close()

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    async def _run_summarization(self, finalized: list[Segment]) -> None:
        summ_backend = get_summarization_backend()
        if summ_backend is None or not finalized:
            return

        assert self.session is not None
        foreign_lang = self.session.foreign_lang or "en"
        summ_segments = [
            {
                "src": seg.src,
                "src_lang": seg.src_lang,
                "translations": self.session.translations.get(seg.id, {}),
            }
            for seg in finalized
        ]

        await self._send_json(
            {
                "type": "summary_progress",
                "step": "summarize",
                "message": "Generating summaries...",
            }
        )

        try:
            loop = asyncio.get_event_loop()
            foreign_summary, german_summary = await loop.run_in_executor(
                _executor,
                summ_backend.summarize_bilingual,
                summ_segments,
                foreign_lang,
            )
            summary_msg = {
                "type": "summary",
                "foreign_summary": foreign_summary,
                "german_summary": german_summary,
                "foreign_lang": foreign_lang,
            }
            await self._send_json(summary_msg)
            await _maybe_broadcast(self.session_token, summary_msg)
        except Exception as e:
            logger.error("Summarization error: %s", e)

    # ------------------------------------------------------------------
    # ASR loop
    # ------------------------------------------------------------------

    async def _asr_loop(self) -> None:
        """Process audio windows every TICK_SEC and send segment updates."""
        tick_count = 0
        last_segment_hash = None
        while self._running:
            await asyncio.sleep(TICK_SEC)
            if self.session is None or not self._running:
                continue

            tick_count += 1
            if tick_count <= 3 or tick_count % 20 == 0:
                logger.debug(
                    "ASR tick #%d: %.1fs buffered",
                    tick_count,
                    self.session.get_buffered_seconds(),
                )
            loop = asyncio.get_event_loop()
            try:
                if self.session.is_dual_channel():
                    asr_fn = run_asr_dual_channel
                else:
                    # Before the remote viewer joins, the web UI is the ONLY source of audio.
                    # Since the web UI is STRICTLY the German speaker's device, we must
                    # process this audio exclusively as German. Falling back to `run_asr`
                    # with diarization would cause the German speaker to be misidentified
                    # as 'foreign' if Whisper misdetects the language.
                    asr_fn = run_asr_german_channel

                all_segments, newly_finalized = await loop.run_in_executor(
                    _executor, asr_fn, self.session
                )

                if not self._running:
                    continue

                tok = self._trace_tok()
                for seg in all_segments:
                    prev = self._trace_seen.get(seg.id)
                    key = (seg.src, seg.final)
                    if prev is None:
                        _trace(
                            "asr_emit",
                            tok=tok,
                            seg=seg.id,
                            final=seg.final,
                            src_lang=seg.src_lang,
                            role=seg.speaker_role,
                            t=f"{seg.abs_start:.2f}-{seg.abs_end:.2f}",
                            text=seg.src,
                        )
                    elif prev != key:
                        _trace(
                            "asr_update",
                            tok=tok,
                            seg=seg.id,
                            final=seg.final,
                            src_lang=seg.src_lang,
                            text=seg.src,
                        )
                    self._trace_seen[seg.id] = key

                segments_data = _serialize_segments(self.session, all_segments)

                # Only send if segments changed (avoid redundant updates)
                current_hash = tuple(
                    (
                        s["id"],
                        s["src"],
                        s.get("src_lang"),
                        s.get("speaker_role"),
                        s["final"],
                        tuple(sorted(s["translations"].items())),
                    )
                    for s in segments_data
                )
                if current_hash != last_segment_hash:
                    last_segment_hash = current_hash
                    segments_msg = {
                        "type": "segments",
                        "t": self.session.get_current_time(),
                        "src_lang": self.session.detected_lang or "unknown",
                        "foreign_lang": self.session.foreign_lang,
                        "dual_channel": self.session.is_dual_channel(),
                        "segments": segments_data,
                    }
                    _trace(
                        "ws_segments",
                        tok=tok,
                        count=len(segments_data),
                        ids=",".join(str(s["id"]) for s in segments_data),
                    )
                    await self._send_and_broadcast(segments_msg)

                if newly_finalized:
                    logger.debug("Queuing %d segments for translation", len(newly_finalized))
                    await self._enqueue_for_translation(newly_finalized)

                if all_segments:
                    final_count = sum(1 for s in all_segments if s.final)
                    live_count = len(all_segments) - final_count
                    logger.debug("Segments: %d final, %d live", final_count, live_count)

            except Exception as e:
                logger.error("ASR tick error: %s", e)

    # ------------------------------------------------------------------
    # MT loop
    # ------------------------------------------------------------------

    async def _mt_loop(self) -> None:
        """Consume the translation queue and send updates when ready."""
        while self._running:
            try:
                segment = await asyncio.wait_for(
                    self._translation_queue.get(),
                    timeout=0.5,
                )
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(
                    "MT loop queue error (%s): %r",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                await asyncio.sleep(0.1)
                continue

            if not self._running or self.session is None:
                self._translation_queue.task_done()
                break

            tok = self._trace_tok()
            enq = self._trace_mt_enq.pop(segment.id, None)
            wait_ms = int((time.time() - enq) * 1000) if enq is not None else -1
            _trace("mt_get", tok=tok, seg=segment.id, wait_ms=wait_ms)

            tgt_lang: str | None = None
            try:
                role = _resolve_segment_role(segment, self.session.is_dual_channel())
                pair = _resolve_translation_pair(
                    segment,
                    role,
                    self.session.foreign_lang,
                )

                if pair is None:
                    _trace("mt_skip", tok=tok, seg=segment.id, reason="no_pair", role=role)
                    continue
                seg_src_lang, tgt_lang = pair

                if seg_src_lang not in LANG_INFO:
                    logger.warning(
                        "Skipping translation for segment %d: unsupported source language '%s'",
                        segment.id,
                        seg_src_lang,
                    )
                    _trace(
                        "mt_skip",
                        tok=tok,
                        seg=segment.id,
                        reason="bad_src_lang",
                        src_lang=seg_src_lang,
                    )
                    continue

                if self.session.translations.get(segment.id, {}).get(tgt_lang):
                    _trace(
                        "mt_skip",
                        tok=tok,
                        seg=segment.id,
                        reason="cached",
                        tgt_lang=tgt_lang,
                    )
                    continue

                logger.debug(
                    "Translating segment %d (%s→%s): %s",
                    segment.id,
                    seg_src_lang,
                    tgt_lang,
                    segment.src[:50],
                )
                _trace(
                    "mt_start",
                    tok=tok,
                    seg=segment.id,
                    src_lang=seg_src_lang,
                    tgt_lang=tgt_lang,
                    text=segment.src,
                )
                mt_t0 = time.time()
                loop = asyncio.get_event_loop()
                translation = await loop.run_in_executor(
                    _executor,
                    run_translation,
                    segment.src,
                    seg_src_lang,
                    tgt_lang,
                )
                mt_dur_ms = int((time.time() - mt_t0) * 1000)
                logger.debug("Translation done %d: %s", segment.id, translation[:50])
                _trace(
                    "mt_done",
                    tok=tok,
                    seg=segment.id,
                    tgt_lang=tgt_lang,
                    dur_ms=mt_dur_ms,
                    text=translation,
                )

                if segment.id not in self.session.translations:
                    self.session.translations[segment.id] = {}
                self.session.translations[segment.id][tgt_lang] = translation

                if self._running:
                    translation_msg = {
                        "type": "translation",
                        "segment_id": segment.id,
                        "tgt_lang": tgt_lang,
                        "text": translation,
                    }
                    _trace(
                        "ws_translation",
                        tok=tok,
                        seg=segment.id,
                        tgt_lang=tgt_lang,
                    )
                    await self._send_and_broadcast(translation_msg)

            except Exception as e:
                logger.error(
                    "Translation error for segment %d (%s): %r",
                    segment.id,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                _trace(
                    "mt_error",
                    tok=tok,
                    seg=segment.id,
                    tgt_lang=tgt_lang,
                    err=f"{type(e).__name__}: {e}",
                )
                # Surface the failure to clients so the segment isn't stuck on
                # a placeholder forever. The broadcast itself may fail (e.g. if
                # the websocket is what just died) so suppress to keep the loop
                # alive — the server log already has the original exception.
                if self._running:
                    error_msg = {
                        "type": "translation_error",
                        "segment_id": segment.id,
                        "tgt_lang": tgt_lang,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    with contextlib.suppress(Exception):
                        await self._send_and_broadcast(error_msg)
            finally:
                self._translation_queue.task_done()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trace_tok(self) -> str:
        """Short token identifier for trace lines (first 8 chars of session token)."""
        return self.session_token[:8] if self.session_token else "?"

    async def _enqueue_for_translation(
        self, segments: list[Segment], reason: str | None = None
    ) -> None:
        """Trace and queue finalized segments for the MT loop."""
        if not segments:
            return
        tok = self._trace_tok()
        for seg in segments:
            self._trace_mt_enq[seg.id] = time.time()
            fields = {
                "tok": tok,
                "seg": seg.id,
                "src_lang": seg.src_lang,
                "role": seg.speaker_role,
                "text": seg.src,
            }
            if reason is not None:
                fields["reason"] = reason
            _trace("asr_final", **fields)
            _trace("mt_put", tok=tok, seg=seg.id, qsize=self._translation_queue.qsize())
            await self._translation_queue.put(seg)

    async def _send_json(self, message: dict) -> None:
        await self.websocket.send_text(json.dumps(message))

    async def _send_and_broadcast(self, message: dict) -> None:
        await self._send_json(message)
        await _maybe_broadcast(self.session_token, message)

    async def _cancel_task(self, attr: str) -> None:
        task: asyncio.Task | None = getattr(self, attr)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            setattr(self, attr, None)

    async def _cleanup(self) -> None:
        self._running = False
        await _maybe_broadcast(self.session_token, {"type": "session_ended"})
        if self.session_token is not None:
            await registry.unregister(self.session_token)
        await self._cancel_task("_asr_task")
        await self._cancel_task("_mt_task")
        await self._cancel_task("_host_speaking_off_task")


async def handle_websocket(websocket: WebSocket) -> None:
    """Main WebSocket handler — thin wrapper around WebSocketHandler."""
    handler = WebSocketHandler(websocket)
    await handler.run()


async def handle_viewer_websocket(websocket: WebSocket, token: str) -> None:
    """Handle a read-only viewer WebSocket connection."""
    await websocket.accept()

    # Add viewer to session (creates pending entry if doesn't exist)
    if not await registry.add_viewer(token, websocket):
        # Token not reserved - reserve it now for the viewer
        await registry.reserve(token)
        await registry.add_viewer(token, websocket)

    logger.info("Viewer connected to session %s...", token[:8])

    try:
        # Check if session is already active
        entry = await registry.get(token)
        if entry and entry.session is not None:
            # Session is active, send current state
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
            # Inform late-joining viewer about an active host transcript request
            if entry.host_transcript_requested:
                await websocket.send_text(
                    json.dumps({"type": "host_transcript_requested", "enabled": True})
                )
        else:
            # Session is pending - tell viewer to wait
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

        # Cache session reference for audio routing
        cached_session = None
        viewer_speaking_off_task: asyncio.Task | None = None

        # Main loop: handle pong/keepalive, audio frames, and viewer config
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                if message["type"] == "websocket.disconnect":
                    break

                # Handle binary audio from viewer mic
                if "bytes" in message:
                    # Lazily resolve session reference
                    if cached_session is None:
                        entry = await registry.get(token)
                        if entry and entry.session:
                            cached_session = entry.session
                    if cached_session is not None:
                        cached_session.add_foreign_audio(message["bytes"])

                # Handle text messages (pong, viewer_audio_config)
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type")

                        if msg_type == "viewer_audio_config":
                            # Viewer sends foreign language hint
                            viewer_foreign_lang = data.get("foreign_lang")
                            if viewer_foreign_lang and viewer_foreign_lang not in (
                                "de",
                                "unknown",
                                "auto",
                            ):
                                if cached_session is None:
                                    entry = await registry.get(token)
                                    if entry and entry.session:
                                        cached_session = entry.session
                                if cached_session is not None:
                                    if cached_session.foreign_lang != viewer_foreign_lang:
                                        logger.info(
                                            "Viewer updated foreign language: %s -> %s",
                                            cached_session.foreign_lang,
                                            viewer_foreign_lang,
                                        )
                                        # Reset cached role/lang inferences to re-lock speakers with new hint.
                                    cached_session.foreign_lang = viewer_foreign_lang

                        elif msg_type == "transcript_consent":
                            # Viewer opted in to the bilingual transcript download.
                            # Consent is stored on the SessionEntry (not the
                            # session) so it survives the pre-activation window
                            # where the host hasn't started recording yet.
                            enabled = bool(data.get("enabled", False))
                            entry = await registry.get(token)
                            if entry and enabled:
                                entry.transcript_consent = True
                                if entry.session is not None:
                                    entry.session.transcript_consent = True
                            if entry and entry.main_ws:
                                msg = {
                                    "type": "transcript_consent",
                                    "enabled": entry.transcript_consent,
                                }
                                try:
                                    await entry.main_ws.send_text(json.dumps(msg))
                                except Exception as exc:  # noqa: BLE001 - relay best-effort
                                    logger.warning(
                                        "Failed to relay transcript_consent to host for "
                                        "token=%s: %s",
                                        token[:8],
                                        exc,
                                        exc_info=True,
                                    )

                        elif msg_type == "speaking_state":
                            # Relay viewer speaking state to host
                            entry = await registry.get(token)
                            if entry and entry.main_ws:
                                speaking = bool(data.get("speaking", False))
                                _trace(
                                    "viewer_speaking",
                                    tok=token[:8],
                                    speaking=speaking,
                                )
                                msg = {
                                    "type": "speaking_state",
                                    "party": "viewer",
                                    "speaking": speaking,
                                }
                                if speaking:
                                    if (
                                        viewer_speaking_off_task
                                        and not viewer_speaking_off_task.done()
                                    ):
                                        viewer_speaking_off_task.cancel()
                                        viewer_speaking_off_task = None
                                    try:
                                        await entry.main_ws.send_text(json.dumps(msg))
                                    except Exception:
                                        logger.warning(
                                            "Failed to relay viewer speaking_state to host"
                                        )
                                else:
                                    viewer_speaking_off_task = asyncio.create_task(
                                        _delayed_viewer_speaking_off(
                                            token, STABILITY_SEC + TICK_SEC
                                        )
                                    )

                        # pong is implicitly handled (no action needed)
                    except (json.JSONDecodeError, KeyError):
                        pass

            except TimeoutError:
                # Send ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break

    except Exception as e:
        logger.error("Viewer websocket error: %s", e)
    finally:
        # Cancel any pending delayed-off task — we send the speaking=false
        # below directly, which makes the delay redundant and prevents the
        # host indicator from latching ON forever if the viewer disconnected
        # mid-PTT-press without releasing.
        if viewer_speaking_off_task and not viewer_speaking_off_task.done():
            viewer_speaking_off_task.cancel()
        entry = await registry.get(token)
        if entry and entry.main_ws:
            _trace("viewer_speaking_off_disconnect", tok=token[:8])
            with contextlib.suppress(Exception):
                await entry.main_ws.send_text(
                    json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False})
                )
        await registry.remove_viewer(token, websocket)
        logger.info("Viewer disconnected from session %s...", token[:8])

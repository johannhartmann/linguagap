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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
from fastapi import WebSocket

from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend
from app.lang_id import SpeakerLanguageTracker, detect_language_from_audio
from app.languages import LANG_INFO
from app.session_registry import SessionEntry, registry
from app.speaker_tracker import SpeakerSegment, SpeakerTracker
from app.streaming_policy import Segment, SegmentTracker

logger = logging.getLogger(__name__)

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "8.0"))
TICK_SEC = float(os.getenv("TICK_SEC", "0.5"))
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "30.0"))

_executor = ThreadPoolExecutor(max_workers=2)

# Metrics
_metrics = {
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


def _resolve_segment_role(
    session: "StreamingSession", segment: Segment, dual_channel: bool
) -> str | None:
    """Resolve the role for a segment, preferring explicit role metadata."""
    if segment.speaker_role in {"german", "foreign"}:
        return segment.speaker_role

    if segment.speaker_id and segment.speaker_id in session.speaker_roles:
        return session.speaker_roles[segment.speaker_id]

    if dual_channel:
        if segment.speaker_id == "SPEAKER_00":
            return "german"
        if segment.speaker_id == "SPEAKER_01":
            return "foreign"

    return _role_from_lang(segment.src_lang)


def _serialize_segments(session: "StreamingSession", segments: list[Segment]) -> list[dict]:
    """Convert Segment objects to dicts with resolved roles and translations."""
    dual_channel = session.is_dual_channel()
    result = []
    for seg in segments:
        seg_dict = asdict(seg)
        speaker_role = _resolve_segment_role(session, seg, dual_channel)
        seg_dict["speaker_role"] = speaker_role
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
    foreign = (
        foreign_lang
        if (foreign_lang and foreign_lang != "unknown" and foreign_lang in LANG_INFO)
        else None
    )

    if role == "german":
        if not foreign:
            return None
        return "de", foreign

    if role == "foreign":
        if not foreign:
            return None
        if segment.src_lang == "de":
            return "de", foreign
        if segment.src_lang in LANG_INFO and segment.src_lang != "de":
            return segment.src_lang, "de"
        if foreign in LANG_INFO and segment.src_lang in {"unknown", None, ""}:
            return foreign, "de"
        return None

    # No role — fallback
    src = segment.src_lang
    tgt = foreign if (src == "de" and foreign) else "de"
    return src, tgt


def _is_effective_silence(
    audio: np.ndarray, rms_threshold: float = 0.003, peak_threshold: float = 0.02
) -> bool:
    """Detect near-silent buffers that should not be sent to ASR."""
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    return rms < rms_threshold and peak < peak_threshold


def get_metrics() -> dict:
    asr_times = list(_metrics["asr_times"])
    mt_times = list(_metrics["mt_times"])
    diar_times = list(_metrics["diar_times"])
    tick_times = list(_metrics["tick_times"])

    return {
        "avg_asr_time_ms": sum(asr_times) / len(asr_times) * 1000 if asr_times else 0,
        "avg_mt_time_ms": sum(mt_times) / len(mt_times) * 1000 if mt_times else 0,
        "avg_diar_time_ms": sum(diar_times) / len(diar_times) * 1000 if diar_times else 0,
        "avg_tick_time_ms": sum(tick_times) / len(tick_times) * 1000 if tick_times else 0,
        "sample_count": len(tick_times),
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


@dataclass
class ChannelBuffer:
    """Audio buffer for a single channel (German or foreign speaker)."""

    audio: deque[bytes] = field(default_factory=deque)
    total_samples: int = 0
    trimmed_samples: int = 0
    start_offset_sec: float = 0.0
    started: bool = False


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
        diarizer: Speaker diarization pipeline
        language_tracker: Per-speaker language detection cache
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
        # Speaker tracking (embedding-based) and language tracking
        self.speaker_tracker = SpeakerTracker(sample_rate=sample_rate)
        self.language_tracker = SpeakerLanguageTracker()  # Kept for compatibility
        self.last_diar_segments: list[SpeakerSegment] = []  # Cache for merging with ASR
        self.speaker_roles: dict[str, str] = {}  # speaker_id -> "german" | "foreign"

        # Dual-channel buffers (German mic on main page, foreign mic on viewer)
        self.german_channel = ChannelBuffer()
        self.foreign_channel = ChannelBuffer()
        self.viewer_last_audio_time: float = 0.0
        self.dual_channel_locked: bool = False

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


def run_speaker_detection(
    session: StreamingSession,
    audio: np.ndarray,
    window_start: float,
) -> list[SpeakerSegment]:
    """Run speaker detection using embedding-based tracking."""
    detect_start = time.time()
    try:
        speaker_segments = session.speaker_tracker.process_audio(audio, window_start)
        detect_time = time.time() - detect_start
        _metrics["diar_times"].append(detect_time)

        if speaker_segments:
            logger.debug(
                "Speaker detection: %d segments in %.1fms",
                len(speaker_segments),
                detect_time * 1000,
            )
            for seg in speaker_segments[:3]:
                logger.debug("  - [%.2f-%.2f] %s", seg.start, seg.end, seg.speaker_id)

        return speaker_segments
    except Exception as e:
        logger.error("Speaker detection error: %s", e)
        return []


def extract_speaker_audio(
    audio: np.ndarray,
    diar_segments: list[SpeakerSegment],
    speaker_id: str,
    window_start: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Extract and concatenate all audio for a specific speaker.

    Args:
        audio: Full audio window as float32 array
        diar_segments: Diarization segments with absolute times
        speaker_id: Target speaker ID to extract
        window_start: Absolute start time of the audio window
        sample_rate: Audio sample rate

    Returns:
        Concatenated audio from all speaker segments
    """
    chunks = []
    for seg in diar_segments:
        if seg.speaker_id == speaker_id:
            # Convert absolute times to window-relative sample indices
            start_rel = seg.start - window_start
            end_rel = seg.end - window_start
            start_sample = int(max(0, start_rel) * sample_rate)
            end_sample = int(min(len(audio) / sample_rate, end_rel) * sample_rate)

            if end_sample > start_sample and start_sample < len(audio):
                chunks.append(audio[start_sample : min(end_sample, len(audio))])

    if chunks:
        return np.concatenate(chunks)
    return np.array([], dtype=audio.dtype)


def _extract_segment_audio(
    audio: np.ndarray,
    diar_seg: SpeakerSegment,
    window_start: float,
    sample_rate: int = 16000,
    padding_sec: float = 0.3,
) -> tuple[np.ndarray | None, float]:
    """Extract audio for a diarization segment with padding.

    Returns:
        Tuple of (segment_audio, padded_start) or (None, 0.0) if too short.
    """
    start_rel = diar_seg.start - window_start
    end_rel = diar_seg.end - window_start

    padded_start = max(0, start_rel - padding_sec)
    padded_end = min(len(audio) / sample_rate, end_rel + padding_sec)

    start_sample = int(padded_start * sample_rate)
    end_sample = int(padded_end * sample_rate)

    segment_duration = diar_seg.end - diar_seg.start
    if segment_duration < 0.5:
        logger.debug("  SKIP short diar segment: %.2fs", segment_duration)
        return None, 0.0

    if end_sample - start_sample < int(0.5 * sample_rate):
        return None, 0.0

    return audio[start_sample:end_sample], padded_start


def _transcribe_speaker_segment(
    backend,
    audio: np.ndarray,
    language: str | None,
    diar_seg: SpeakerSegment,
    window_start: float,
    sample_rate: int = 16000,
    padding_sec: float = 0.3,
) -> list[dict]:
    """Transcribe a single speaker segment via the ASR backend.

    Extracts audio with padding, delegates to backend.transcribe() + post_process().
    """
    segment_audio, padded_start = _extract_segment_audio(
        audio, diar_seg, window_start, sample_rate, padding_sec
    )
    if segment_audio is None:
        return []

    prompt = backend.get_bilingual_prompt(language) if language else None
    asr_result = backend.transcribe(segment_audio, language=language, initial_prompt=prompt)
    filtered = backend.post_process(asr_result.segments)

    results = []
    for seg in filtered:
        results.append(
            {
                "start": padded_start + seg.start,
                "end": padded_start + seg.end,
                "text": seg.text,
                "speaker_id": diar_seg.speaker_id,
                "lang": seg.language,
            }
        )
    return results


def run_asr(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """
    Run ASR with diarization-first approach.

    This is the core transcription function called every TICK_SEC. It implements
    a pipeline that prevents language detection errors in bilingual conversations:

    Pipeline:
        1. Run diarization to identify speaker segments (SPEAKER_00, SPEAKER_01)
        2. Detect language per speaker using SpeechBrain on their audio
        3. Transcribe each speaker's segment with the correct language hint

    Why diarization-first?
        Window-level language detection fails when a German speaker and Arabic
        speaker alternate. The window might contain more Arabic audio, causing
        Whisper to transcribe German speech as Arabic. By detecting language
        per-speaker, each segment gets the correct language hint.

    Args:
        session: The streaming session with audio buffer and state

    Returns:
        Tuple of (all_segments, newly_finalized):
        - all_segments: Complete transcript (finalized + live segments)
        - newly_finalized: Segments that just became final (need translation)
    """
    tick_start = time.time()

    audio, window_start = session.get_window_audio()
    now_sec = session.get_current_time()

    if len(audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    backend = get_asr_backend()

    # Debug audio stats
    audio_rms = float(np.sqrt(np.mean(audio**2)))
    audio_max = float(np.max(np.abs(audio)))
    logger.debug("Pipeline: audio_len=%d, rms=%.4f, max=%.4f", len(audio), audio_rms, audio_max)

    # STEP 1: Run speaker detection to identify speaker segments (embedding-based)
    detect_start = time.time()
    diar_segments = run_speaker_detection(session, audio, window_start)
    session.last_diar_segments = diar_segments
    detect_time = time.time() - detect_start

    if not diar_segments:
        logger.debug("No speaker segments, falling back to full-window ASR")
        return _run_asr_fallback(session, audio, window_start, now_sec)

    logger.debug("Speaker detection: %d segments in %.1fms", len(diar_segments), detect_time * 1000)

    # STEP 2: Detect language per speaker from raw audio (BEFORE ASR)
    speaker_languages = _detect_speaker_languages(session, audio, diar_segments, window_start)

    # Build/maintain stable speaker roles so routing does not flip on noisy ticks.
    for speaker_id, (lang, confidence) in speaker_languages.items():
        if confidence < 0.5:
            continue
        if lang == "de":
            session.speaker_roles[speaker_id] = "german"
        elif (
            lang not in ("unknown", "de")
            and session.foreign_lang
            and session.foreign_lang in LANG_INFO
        ):
            session.speaker_roles[speaker_id] = "foreign"

    # If only one role is known and we have exactly two speakers, assign the opposite role.
    unique_speakers = sorted({seg.speaker_id for seg in diar_segments})
    if len(unique_speakers) == 2 and session.foreign_lang and session.foreign_lang in LANG_INFO:
        roles = {sid: session.speaker_roles.get(sid) for sid in unique_speakers}
        if list(roles.values()).count("german") == 1 and list(roles.values()).count(None) == 1:
            for sid, role in roles.items():
                if role is None:
                    session.speaker_roles[sid] = "foreign"
        elif list(roles.values()).count("foreign") == 1 and list(roles.values()).count(None) == 1:
            for sid, role in roles.items():
                if role is None:
                    session.speaker_roles[sid] = "german"

    # STEP 3: Transcribe per speaker with their detected language
    asr_start = time.time()
    hyp_segments: list[dict] = []

    for diar_seg in diar_segments:
        speaker_id = diar_seg.speaker_id
        lang, confidence = speaker_languages.get(speaker_id, ("unknown", 0.0))
        speaker_role = session.speaker_roles.get(speaker_id)

        # Skip segments that start right at window boundary (likely truncated from earlier)
        segment_rel_start = diar_seg.start - window_start
        if segment_rel_start < 0.5 and (diar_seg.end - diar_seg.start) < 1.0:
            logger.debug(
                "  SKIP boundary segment: %s [%.2f-%.2f] (truncated)",
                diar_seg.speaker_id,
                diar_seg.start,
                diar_seg.end,
            )
            continue

        if speaker_role == "german":
            use_lang = "de"
        elif speaker_role == "foreign" and session.foreign_lang:
            use_lang = session.foreign_lang
        else:
            use_lang = lang if confidence > 0.5 and lang != "unknown" else None

        # Transcribe via backend (includes post_process: delooping + hallucination filtering)
        seg_results = _transcribe_speaker_segment(backend, audio, use_lang, diar_seg, window_start)
        for seg in seg_results:
            seg["speaker_role"] = speaker_role
        hyp_segments.extend(seg_results)

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    logger.debug("Per-speaker ASR: %d segments in %.1fms", len(hyp_segments), asr_time * 1000)

    session.resolve_foreign_lang()

    # Keep language assignment aligned with stable speaker role.
    if session.foreign_lang:
        for seg in hyp_segments:
            role = seg.get("speaker_role")
            if role == "german":
                seg["lang"] = "de"
            elif role == "foreign" and seg.get("lang") in {None, "", "unknown"}:
                seg["lang"] = session.foreign_lang

    for seg in hyp_segments:
        seg["speaker_role"] = seg.get("speaker_role") or _role_from_lang(seg.get("lang"))

    for seg in hyp_segments[:3]:
        logger.debug(
            "  - [%.1f-%.1f] %s lang=%s: %s",
            seg["start"],
            seg["end"],
            seg["speaker_id"],
            seg["lang"],
            seg["text"][:40],
        )

    # Determine detected language for session from actual ASR results
    lang_counts: dict[str, int] = {}
    for lang, _ in speaker_languages.values():
        if lang != "unknown":
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session.detected_lang = (
        max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else "unknown"
    )

    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=hyp_segments,
        window_start=window_start,
        now_sec=now_sec,
        src_lang=session.detected_lang or "unknown",
    )

    tick_time = time.time() - tick_start
    _metrics["tick_times"].append(tick_time)

    return all_segments, newly_finalized


def _transcribe_channel(
    backend,
    audio: np.ndarray,
    language: str | None,
    speaker_id: str,
    speaker_role: str,
    force_lang: str | None = None,
) -> list[dict]:
    """Transcribe a full channel buffer via the ASR backend."""
    if len(audio) < 1600:
        return []
    if _is_effective_silence(audio):
        return []

    prompt = backend.get_bilingual_prompt(language) if language else None
    asr_result = backend.transcribe(audio, language=language, initial_prompt=prompt)
    filtered = backend.post_process(asr_result.segments)

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

    german_results = _transcribe_channel(
        backend,
        german_audio,
        "de",
        "SPEAKER_00",
        "german",
        force_lang="de",
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

    logger.debug(
        "Dual-channel pipeline: german=%d samples, foreign=%d samples",
        len(german_audio),
        len(foreign_audio),
    )

    asr_start = time.time()

    # Transcribe both channels via backend
    german_results = _transcribe_channel(
        backend,
        german_audio,
        "de",
        "SPEAKER_00",
        "german",
        force_lang="de",
    )
    foreign_results = _transcribe_channel(
        backend,
        foreign_audio,
        session.foreign_lang,
        "SPEAKER_01",
        "foreign",
    )

    # Offset segment timestamps to absolute time
    for seg in german_results:
        seg["start"] += german_offset
        seg["end"] += german_offset
    for seg in foreign_results:
        seg["start"] += foreign_offset
        seg["end"] += foreign_offset

    # Auto-detect foreign language from results if not yet known
    if session.foreign_lang is None:
        for seg in foreign_results:
            lang = seg.get("lang")
            if lang and lang not in ("unknown", "de") and lang in LANG_INFO:
                session.foreign_lang = lang
                logger.info("Dual-channel: foreign language detected as %s", lang)
                break

    # Only fill unknown labels; do not overwrite explicit "de"/other detections.
    if session.foreign_lang:
        for seg in foreign_results:
            if seg.get("lang") in {None, "", "unknown"}:
                seg["lang"] = session.foreign_lang
    else:
        for seg in foreign_results:
            if seg.get("lang") in {None, ""}:
                seg["lang"] = "unknown"

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


def _detect_speaker_languages(
    session: StreamingSession,
    audio: np.ndarray,
    diar_segments: list[SpeakerSegment],
    window_start: float,
) -> dict[str, tuple[str, float]]:
    """Detect language per speaker from raw audio using SpeechBrain.

    Returns:
        Dict mapping speaker_id -> (language_code, confidence)
    """
    speaker_languages: dict[str, tuple[str, float]] = {}
    unique_speakers = {seg.speaker_id for seg in diar_segments}

    # Language confusion groups - SpeechBrain often confuses these similar languages
    language_confusion_groups = [
        {"bg", "sl", "mk"},
        {"sr", "bs", "hr"},
        {"ku", "fa", "ps"},
        {"uk", "ru", "be"},
        {"cs", "sk"},
        {"no", "da", "sv"},
        {"id", "ms"},
    ]

    def correct_language_confusion(detected: str, user_hint: str | None, confidence: float) -> str:
        if not user_hint or detected == user_hint:
            return detected

        for group in language_confusion_groups:
            if detected in group and user_hint in group:
                logger.debug(
                    "    Correcting %s → %s (user hint, confusion group)", detected, user_hint
                )
                return user_hint

        if detected not in ("de", "unknown") and user_hint not in ("de", "unknown"):
            logger.debug("    Correcting %s → %s (user hint, foreign speaker)", detected, user_hint)
            return user_hint

        if detected == "de" and confidence < 0.9 and user_hint not in ("de", "unknown"):
            logger.debug(
                "    Correcting %s → %s (user hint, low confidence de)", detected, user_hint
            )
            return user_hint

        return detected

    for speaker_id in unique_speakers:
        speaker_audio = extract_speaker_audio(audio, diar_segments, speaker_id, window_start)

        if len(speaker_audio) < 8000:
            logger.debug("  %s: audio too short (%.2fs)", speaker_id, len(speaker_audio) / 16000)
            continue

        detected_lang, confidence = session.language_tracker.get_speaker_language(
            speaker_id, speaker_audio, 16000
        )

        lang = detected_lang
        corrected = correct_language_confusion(lang, session.foreign_lang, confidence)
        if corrected != lang:
            lang = corrected
            session.language_tracker.set_speaker_language(speaker_id, lang, confidence)

        speaker_languages[speaker_id] = (lang, confidence)
        logger.debug("  %s → %s (confidence=%.2f)", speaker_id, lang, confidence)

        if lang not in ("unknown", "de") and lang in LANG_INFO and session.foreign_lang is None:
            session.foreign_lang = lang
            logger.info("Foreign language detected: %s", lang)

    return speaker_languages


def _run_asr_fallback(
    session: StreamingSession,
    audio: np.ndarray,
    window_start: float,
    now_sec: float,
) -> tuple[list[Segment], list[Segment]]:
    """Fallback ASR when diarization fails - uses full window with multilingual."""
    backend = get_asr_backend()
    asr_start = time.time()

    asr_result = backend.transcribe(audio)
    filtered = backend.post_process(asr_result.segments)

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    logger.debug(
        "Fallback ASR: %d segs, lang=%s, %.1fms",
        len(filtered),
        asr_result.detected_language,
        asr_time * 1000,
    )

    hyp_segments = []
    for seg in filtered:
        # Use SpeechBrain for segment language detection
        seg_start_sample = int(seg.start * 16000)
        seg_end_sample = int(seg.end * 16000)
        segment_audio = audio[seg_start_sample : min(seg_end_sample, len(audio))]

        speechbrain_lang, _ = detect_language_from_audio(segment_audio, 16000)
        segment_lang = (
            speechbrain_lang if speechbrain_lang != "unknown" else asr_result.detected_language
        )

        if (
            segment_lang not in ("unknown", "de")
            and segment_lang in LANG_INFO
            and session.foreign_lang is None
        ):
            session.foreign_lang = segment_lang
            logger.info("Foreign language detected: %s", segment_lang)

        hyp_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker_id": None,
                "lang": segment_lang,
                "speaker_role": _role_from_lang(segment_lang),
            }
        )

    session.detected_lang = asr_result.detected_language

    session.resolve_foreign_lang()

    # Do not force all fallback segments to foreign language; keep detected labels.
    for seg in hyp_segments:
        seg["speaker_role"] = _role_from_lang(seg["lang"])

    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=hyp_segments,
        window_start=window_start,
        now_sec=now_sec,
        src_lang=session.detected_lang or "unknown",
    )

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

    async def _handle_message(self, message: dict) -> None:
        if "text" in message:
            data = json.loads(message["text"])
            msg_type = data.get("type")
            if msg_type == "config":
                await self._handle_config(data)
            elif msg_type == "request_summary":
                await self._handle_request_summary()
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
        if foreign_lang and foreign_lang not in ("auto", "de", "unknown"):
            self.session.foreign_lang = foreign_lang

        self._asr_task = asyncio.create_task(self._asr_loop())
        self._mt_task = asyncio.create_task(self._mt_loop())

        await registry.activate(self.session_token, self.session, self.websocket)
        logger.info(
            "Session activated: sample_rate=%d, src_lang=%s, foreign_lang=%s, token=%s...",
            sample_rate,
            src_lang,
            foreign_lang or "auto",
            self.session_token[:8],
        )

        await self._send_json({"type": "config_ack", "status": "active"})

        entry = await registry.get(self.session_token)
        if entry and entry.viewers:
            await broadcast_to_viewers(
                entry,
                {
                    "type": "session_active",
                    "foreign_lang": self.session.foreign_lang,
                    "segments": [],
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
    # Request summary (stop recording + summarize)
    # ------------------------------------------------------------------

    async def _handle_request_summary(self) -> None:
        await self._cancel_task("_asr_task")
        logger.debug("ASR loop stopped for stop request")

        if self.session is None:
            return

        # Finalize remaining segments
        now_sec = self.session.get_current_time()
        all_segs, time_finalized = self.session.segment_tracker.update_from_hypothesis(
            [], 0.0, now_sec, "unknown"
        )

        for seg in time_finalized:
            logger.debug("Queuing time-finalized segment %d: %s", seg.id, seg.src[:50])
            await self._translation_queue.put(seg)

        live_segs = [s for s in all_segs if not s.final]
        if live_segs:
            logger.debug("Force-finalizing %d live segments", len(live_segs))
            newly_final = self.session.segment_tracker.force_finalize_all()
            for seg in newly_final:
                logger.debug("Queuing force-finalized segment %d: %s", seg.id, seg.src[:50])
                await self._translation_queue.put(seg)

        # Wait for translations to complete
        try:
            await asyncio.wait_for(self._translation_queue.join(), timeout=60)
        except TimeoutError:
            logger.warning(
                "Translation queue join timed out, %d items remaining",
                self._translation_queue.qsize(),
            )

        # Send final segments with all translations
        finalized = self.session.segment_tracker.finalized_segments
        if finalized:
            segments_data = _serialize_segments(self.session, list(finalized))
            await self._send_json(
                {
                    "type": "segments",
                    "t": self.session.get_current_time(),
                    "src_lang": self.session.detected_lang or "unknown",
                    "foreign_lang": self.session.foreign_lang or "en",
                    "segments": segments_data,
                }
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
                elif (
                    self.session.src_lang not in ("auto", "de")
                    and self.session.foreign_lang
                    and self.session.foreign_lang in LANG_INFO
                ):
                    asr_fn = run_asr_german_channel
                else:
                    asr_fn = run_asr
                all_segments, newly_finalized = await loop.run_in_executor(
                    _executor, asr_fn, self.session
                )

                if not self._running:
                    continue

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
                    await self._send_and_broadcast(segments_msg)

                for seg in newly_finalized:
                    logger.debug("Queuing segment %d for translation: %s", seg.id, seg.src[:50])
                    await self._translation_queue.put(seg)

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
                logger.error("MT loop error: %s", e)
                continue

            if not self._running or self.session is None:
                self._translation_queue.task_done()
                break

            try:
                role = _resolve_segment_role(
                    self.session,
                    segment,
                    self.session.is_dual_channel(),
                )
                pair = _resolve_translation_pair(
                    segment,
                    role,
                    self.session.foreign_lang,
                )

                if pair is None:
                    continue
                seg_src_lang, tgt_lang = pair

                if seg_src_lang not in LANG_INFO:
                    logger.warning(
                        "Skipping translation for segment %d: unsupported source language '%s'",
                        segment.id,
                        seg_src_lang,
                    )
                    continue

                if self.session.translations.get(segment.id, {}).get(tgt_lang):
                    continue

                logger.debug(
                    "Translating segment %d (%s→%s): %s",
                    segment.id,
                    seg_src_lang,
                    tgt_lang,
                    segment.src[:50],
                )
                loop = asyncio.get_event_loop()
                translation = await loop.run_in_executor(
                    _executor,
                    run_translation,
                    segment.src,
                    seg_src_lang,
                    tgt_lang,
                )
                logger.debug("Translation done %d: %s", segment.id, translation[:50])

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
                    await self._send_and_broadcast(translation_msg)

            except Exception as e:
                logger.error("Translation error for segment %d: %s", segment.id, e)
            finally:
                self._translation_queue.task_done()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        await registry.unregister(self.session_token)
        await self._cancel_task("_asr_task")
        await self._cancel_task("_mt_task")


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
        if entry and entry.is_active:
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
                    }
                )
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

        # Cache session reference for audio routing
        cached_session = None

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
                                        cached_session.speaker_roles.clear()
                                        cached_session.language_tracker.clear_cache()
                                    cached_session.foreign_lang = viewer_foreign_lang
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
        await registry.remove_viewer(token, websocket)
        logger.info("Viewer disconnected from session %s...", token[:8])

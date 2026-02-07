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
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
from fastapi import WebSocket

from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend
from app.lang_id import SpeakerLanguageTracker, detect_language_from_audio
from app.mt import LANG_INFO
from app.session_registry import SessionEntry, registry
from app.speaker_tracker import SpeakerSegment, SpeakerTracker
from app.streaming_policy import Segment, SegmentTracker

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
        self.translations: dict[int, dict[str, str]] = {}  # segment_id -> {lang -> translation}
        # Speaker tracking (embedding-based) and language tracking
        self.speaker_tracker = SpeakerTracker(sample_rate=sample_rate)
        self.language_tracker = SpeakerLanguageTracker()  # Kept for compatibility
        self.last_diar_segments: list[SpeakerSegment] = []  # Cache for merging with ASR

        # Dual-channel buffers (German mic on main page, foreign mic on viewer)
        self.german_audio_buffer: deque[bytes] = deque()
        self.foreign_audio_buffer: deque[bytes] = deque()
        self.german_total_samples: int = 0
        self.foreign_total_samples: int = 0
        self.viewer_last_audio_time: float = 0.0

    def add_audio(self, pcm16_bytes: bytes):
        self.audio_buffer.append(pcm16_bytes)
        self.total_samples += len(pcm16_bytes) // 2
        self._enforce_max_buffer()

    def _enforce_max_buffer(self):
        max_samples = int(MAX_BUFFER_SEC * self.sample_rate)
        all_bytes = b"".join(self.audio_buffer)
        total_samples = len(all_bytes) // 2

        if total_samples > max_samples:
            excess_samples = total_samples - max_samples
            excess_bytes = excess_samples * 2

            trimmed_bytes = all_bytes[excess_bytes:]
            self.audio_buffer.clear()
            self.audio_buffer.append(trimmed_bytes)
            self.dropped_frames += 1

    def add_german_audio(self, pcm16_bytes: bytes):
        """Add audio from the main page (German speaker) to the german channel buffer."""
        self.german_audio_buffer.append(pcm16_bytes)
        self.german_total_samples += len(pcm16_bytes) // 2
        self._enforce_max_buffer_channel(self.german_audio_buffer)
        # Also add to combined buffer for get_current_time() / get_buffered_seconds()
        self.add_audio(pcm16_bytes)

    def add_foreign_audio(self, pcm16_bytes: bytes):
        """Add audio from the viewer (foreign speaker) to the foreign channel buffer."""
        self.foreign_audio_buffer.append(pcm16_bytes)
        self.foreign_total_samples += len(pcm16_bytes) // 2
        self.viewer_last_audio_time = time.time()
        self._enforce_max_buffer_channel(self.foreign_audio_buffer)
        # Also add to combined buffer for get_current_time() / get_buffered_seconds()
        self.add_audio(pcm16_bytes)

    def _enforce_max_buffer_channel(self, buffer: deque[bytes]):
        """Trim a per-channel buffer to MAX_BUFFER_SEC."""
        max_samples = int(MAX_BUFFER_SEC * self.sample_rate)
        all_bytes = b"".join(buffer)
        total_samples = len(all_bytes) // 2

        if total_samples > max_samples:
            excess_samples = total_samples - max_samples
            excess_bytes = excess_samples * 2
            trimmed_bytes = all_bytes[excess_bytes:]
            buffer.clear()
            buffer.append(trimmed_bytes)

    def get_german_window_audio(self) -> tuple[np.ndarray, float]:
        """Get the German channel audio buffer as a float32 array."""
        all_bytes = b"".join(self.german_audio_buffer)
        if not all_bytes:
            return np.array([], dtype=np.float32), 0.0
        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 0.0

    def get_foreign_window_audio(self) -> tuple[np.ndarray, float]:
        """Get the foreign channel audio buffer as a float32 array."""
        all_bytes = b"".join(self.foreign_audio_buffer)
        if not all_bytes:
            return np.array([], dtype=np.float32), 0.0
        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 0.0

    def is_dual_channel(self) -> bool:
        """Check if viewer is actively sending audio (within last 5 seconds)."""
        return (time.time() - self.viewer_last_audio_time) < 5.0

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
        return samples, 0.0

    def get_current_time(self) -> float:
        return self.total_samples / self.sample_rate

    def get_buffered_seconds(self) -> float:
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
            print(
                f"Speaker detection: {len(speaker_segments)} segments in {detect_time * 1000:.1f}ms"
            )
            for seg in speaker_segments[:3]:
                print(f"  - [{seg.start:.2f}-{seg.end:.2f}] {seg.speaker_id}")

        return speaker_segments
    except Exception as e:
        print(f"Speaker detection error: {e}")
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
        print(f"  SKIP short diar segment: {segment_duration:.2f}s")
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
    print(f"Pipeline: audio_len={len(audio)}, rms={audio_rms:.4f}, max={audio_max:.4f}")

    # STEP 1: Run speaker detection to identify speaker segments (embedding-based)
    detect_start = time.time()
    diar_segments = run_speaker_detection(session, audio, window_start)
    session.last_diar_segments = diar_segments
    detect_time = time.time() - detect_start

    if not diar_segments:
        print("No speaker segments, falling back to full-window ASR")
        return _run_asr_fallback(session, audio, window_start, now_sec)

    print(f"Speaker detection: {len(diar_segments)} segments in {detect_time * 1000:.1f}ms")

    # STEP 2: Detect language per speaker from raw audio (BEFORE ASR)
    speaker_languages = _detect_speaker_languages(session, audio, diar_segments, window_start)

    # STEP 3: Transcribe per speaker with their detected language
    asr_start = time.time()
    hyp_segments: list[dict] = []

    for diar_seg in diar_segments:
        speaker_id = diar_seg.speaker_id
        lang, confidence = speaker_languages.get(speaker_id, ("unknown", 0.0))

        # Skip segments that start right at window boundary (likely truncated from earlier)
        segment_rel_start = diar_seg.start - window_start
        if segment_rel_start < 0.5 and (diar_seg.end - diar_seg.start) < 1.0:
            print(
                f"  SKIP boundary segment: {diar_seg.speaker_id} "
                f"[{diar_seg.start:.2f}-{diar_seg.end:.2f}] (truncated)"
            )
            continue

        use_lang = lang if confidence > 0.5 and lang != "unknown" else None

        # Transcribe via backend (includes post_process: delooping + hallucination filtering)
        seg_results = _transcribe_speaker_segment(backend, audio, use_lang, diar_seg, window_start)
        hyp_segments.extend(seg_results)

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    print(f"Per-speaker ASR: {len(hyp_segments)} segments in {asr_time * 1000:.1f}ms")
    for seg in hyp_segments[:3]:
        print(
            f"  - [{seg['start']:.1f}-{seg['end']:.1f}] {seg['speaker_id']} "
            f"lang={seg['lang']}: {seg['text'][:40]}"
        )

    # Determine detected language for session (backwards compatibility)
    if session.src_lang == "auto":
        lang_counts: dict[str, int] = {}
        for lang, _ in speaker_languages.values():
            if lang != "unknown":
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        session.detected_lang = (
            max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else "unknown"
        )
    else:
        session.detected_lang = session.src_lang

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
) -> list[dict]:
    """Transcribe a full channel buffer via the ASR backend."""
    if len(audio) < 1600:
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
            "lang": seg.language,
        }
        for seg in filtered
    ]


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

    german_audio, _ = session.get_german_window_audio()
    foreign_audio, _ = session.get_foreign_window_audio()

    if len(german_audio) < 1600 and len(foreign_audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    backend = get_asr_backend()

    print(
        f"Dual-channel pipeline: german={len(german_audio)} samples, "
        f"foreign={len(foreign_audio)} samples"
    )

    asr_start = time.time()

    # Transcribe both channels via backend
    german_results = _transcribe_channel(backend, german_audio, "de", "SPEAKER_00")
    foreign_results = _transcribe_channel(
        backend, foreign_audio, session.foreign_lang, "SPEAKER_01"
    )

    # Auto-detect foreign language from results if not yet known
    if session.foreign_lang is None:
        for seg in foreign_results:
            lang = seg.get("lang")
            if lang and lang not in ("unknown", "de") and lang in LANG_INFO:
                session.foreign_lang = lang
                print(f"Dual-channel: foreign language detected as {lang}")
                break

    hyp_segments = german_results + foreign_results
    hyp_segments.sort(key=lambda s: s["start"])

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    print(f"Dual-channel ASR: {len(hyp_segments)} segments in {asr_time * 1000:.1f}ms")

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
                print(f"    Correcting {detected} → {user_hint} (user hint, confusion group)")
                return user_hint

        if detected not in ("de", "unknown") and user_hint not in ("de", "unknown"):
            print(f"    Correcting {detected} → {user_hint} (user hint, foreign speaker)")
            return user_hint

        if detected == "de" and confidence < 0.9 and user_hint not in ("de", "unknown"):
            print(f"    Correcting {detected} → {user_hint} (user hint, low confidence de)")
            return user_hint

        return detected

    for speaker_id in unique_speakers:
        speaker_audio = extract_speaker_audio(audio, diar_segments, speaker_id, window_start)

        if len(speaker_audio) < 8000:
            print(f"  {speaker_id}: audio too short ({len(speaker_audio) / 16000:.2f}s)")
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
        print(f"  {speaker_id} → {lang} (confidence={confidence:.2f})")

        if lang not in ("unknown", "de") and lang in LANG_INFO and session.foreign_lang is None:
            session.foreign_lang = lang
            print(f"Foreign language detected: {lang}")

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

    print(
        f"Fallback ASR: {len(filtered)} segs, "
        f"lang={asr_result.detected_language}, {asr_time * 1000:.1f}ms"
    )

    hyp_segments = []
    for seg in filtered:
        # Use SpeechBrain for segment language detection
        seg_start_sample = int(seg.start * 16000)
        seg_end_sample = int(seg.end * 16000)
        segment_audio = audio[seg_start_sample : min(seg_end_sample, len(audio))]

        speechbrain_lang, confidence = detect_language_from_audio(segment_audio, 16000)  # noqa: F841
        segment_lang = (
            speechbrain_lang if speechbrain_lang != "unknown" else asr_result.detected_language
        )

        if (
            segment_lang not in ("unknown", "de")
            and segment_lang in LANG_INFO
            and session.foreign_lang is None
        ):
            session.foreign_lang = segment_lang
            print(f"Foreign language detected: {segment_lang}")

        hyp_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker_id": None,
                "lang": segment_lang,
            }
        )

    if session.src_lang == "auto":
        session.detected_lang = asr_result.detected_language
    else:
        session.detected_lang = session.src_lang

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


async def handle_websocket(websocket: WebSocket):
    """
    Main WebSocket handler for real-time streaming transcription.

    Handles the complete lifecycle of a streaming session:
        1. Receive config message with session parameters
        2. Start ASR loop (runs every TICK_SEC, transcribes audio windows)
        3. Start MT loop (translates finalized segments asynchronously)
        4. Receive binary audio frames, buffer them
        5. Handle stop requests (drain translations, send final segments)
        6. Clean up and notify viewers on disconnect

    Message protocol:
        Client → Server:
            - {"type": "config", "sample_rate": 16000, "src_lang": "auto", "token": "..."}
            - Binary PCM16 audio frames (16kHz mono)
            - {"type": "request_summary"} (stops recording, drains translations)

        Server → Client:
            - {"type": "config_ack", "status": "active"}
            - {"type": "segments", "segments": [...], "src_lang": "...", "foreign_lang": "..."}
            - {"type": "translation", "segment_id": N, "tgt_lang": "de", "text": "..."}

    The handler spawns two async tasks that run concurrently:
        - asr_loop: Processes audio every TICK_SEC, sends segment updates
        - mt_loop: Consumes translation queue, sends translation updates
    """
    await websocket.accept()

    session: StreamingSession | None = None
    session_token: str | None = None
    asr_task: asyncio.Task | None = None
    mt_task: asyncio.Task | None = None
    running = True
    translation_queue: asyncio.Queue[Segment] = asyncio.Queue()

    async def asr_loop():
        """ASR loop - runs independently, sends segments immediately."""
        nonlocal running
        tick_count = 0
        last_segment_hash = None  # Track last sent state to avoid redundant sends
        while running:
            await asyncio.sleep(TICK_SEC)
            if session is not None and running:
                tick_count += 1
                if tick_count <= 3 or tick_count % 20 == 0:
                    print(f"ASR tick #{tick_count}: {session.get_buffered_seconds():.1f}s buffered")
                loop = asyncio.get_event_loop()
                try:
                    asr_fn = run_asr_dual_channel if session.is_dual_channel() else run_asr
                    all_segments, newly_finalized = await loop.run_in_executor(
                        _executor, asr_fn, session
                    )

                    if running:
                        # Auto-detect foreign language from first non-German speech
                        if session.foreign_lang is None and session.detected_lang:
                            if session.src_lang != "auto":
                                # User explicitly selected a foreign language
                                if session.src_lang != "de" and session.src_lang in LANG_INFO:
                                    session.foreign_lang = session.src_lang
                            elif (
                                session.detected_lang not in ("de", "unknown", None)
                                and session.detected_lang in LANG_INFO
                            ):
                                # Auto-detect: first non-German speech sets the foreign language
                                session.foreign_lang = session.detected_lang
                                print(f"Auto-detected foreign language: {session.foreign_lang}")

                        # Build segments with translations where available
                        segments_data = []
                        for seg in all_segments:
                            seg_dict = asdict(seg)
                            seg_dict["translations"] = session.translations.get(seg.id, {})
                            segments_data.append(seg_dict)

                        # Only send if segments changed (avoid redundant updates)
                        # Hash: (id, src, final, translations) for each segment
                        current_hash = tuple(
                            (
                                s["id"],
                                s["src"],
                                s["final"],
                                tuple(sorted(s["translations"].items())),
                            )
                            for s in segments_data
                        )
                        if current_hash != last_segment_hash:
                            last_segment_hash = current_hash

                            # Send ASR results
                            segments_msg = {
                                "type": "segments",
                                "t": session.get_current_time(),
                                "src_lang": session.detected_lang or "unknown",
                                "foreign_lang": session.foreign_lang,
                                "dual_channel": session.is_dual_channel(),
                                "segments": segments_data,
                            }
                            await websocket.send_text(json.dumps(segments_msg))

                            # Broadcast to viewers
                            if session_token:
                                entry = await registry.get(session_token)
                                if entry:
                                    await broadcast_to_viewers(entry, segments_msg)

                        # Queue newly finalized segments for translation
                        for seg in newly_finalized:
                            print(f"Queuing segment {seg.id} for translation: {seg.src[:50]}")
                            await translation_queue.put(seg)

                        if all_segments:
                            final_count = sum(1 for s in all_segments if s.final)
                            live_count = len(all_segments) - final_count
                            print(f"Segments: {final_count} final, {live_count} live")

                except Exception as e:
                    print(f"ASR tick error: {e}")

    async def mt_loop():
        """Translation loop - processes queue, sends updates when ready."""
        nonlocal running
        while running:
            try:
                # Wait for a segment to translate (with timeout to check running)
                try:
                    segment = await asyncio.wait_for(translation_queue.get(), timeout=0.5)
                except TimeoutError:
                    continue

                if not running or session is None:
                    break

                # Run translation in executor - bidirectional
                loop = asyncio.get_event_loop()
                seg_src_lang = segment.src_lang

                try:
                    # Determine translation direction: German → foreign, foreign → German
                    # Default to English if foreign language not yet detected or invalid
                    foreign = session.foreign_lang
                    if not foreign or foreign == "unknown" or foreign not in LANG_INFO:
                        foreign = "en"
                    tgt_lang = foreign if seg_src_lang == "de" else "de"

                    # Skip translation if source language is unsupported
                    if seg_src_lang not in LANG_INFO:
                        print(
                            f"Skipping translation for segment {segment.id}: "
                            f"unsupported source language '{seg_src_lang}'"
                        )
                        translation_queue.task_done()
                        continue

                    print(
                        f"Translating segment {segment.id} ({seg_src_lang}→{tgt_lang}): {segment.src[:50]}"
                    )
                    translation = await loop.run_in_executor(
                        _executor, run_translation, segment.src, seg_src_lang, tgt_lang
                    )
                    print(f"Translation done {segment.id}: {translation[:50]}")

                    # Store translation in dict
                    if segment.id not in session.translations:
                        session.translations[segment.id] = {}
                    session.translations[segment.id][tgt_lang] = translation

                    # Mark task as done for queue.join() to work
                    translation_queue.task_done()

                    if running:
                        # Send translation update
                        translation_msg = {
                            "type": "translation",
                            "segment_id": segment.id,
                            "tgt_lang": tgt_lang,
                            "text": translation,
                        }
                        await websocket.send_text(json.dumps(translation_msg))

                        # Broadcast to viewers
                        if session_token:
                            entry = await registry.get(session_token)
                            if entry:
                                await broadcast_to_viewers(entry, translation_msg)
                except Exception as e:
                    print(f"Translation error for segment {segment.id}: {e}")
                    # Still mark as done even on error
                    translation_queue.task_done()

            except Exception as e:
                print(f"MT loop error: {e}")

    try:
        msg_count = 0
        bytes_received = 0
        while True:
            message = await websocket.receive()
            msg_count += 1

            if message["type"] == "websocket.disconnect":
                print(f"WebSocket disconnect after {msg_count} msgs, {bytes_received} bytes")
                break

            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    sample_rate = data.get("sample_rate", 16000)
                    src_lang = data.get("src_lang", "auto")
                    foreign_lang = data.get("foreign_lang")  # Optional hint for non-German language
                    # Use client-provided token (generated on page load)
                    session_token = data.get("token")
                    if not session_token:
                        print("Error: No token provided in config message")
                        continue

                    session = StreamingSession(sample_rate=sample_rate, src_lang=src_lang)
                    # Set foreign language hint if provided (improves ASR for known language pairs)
                    if foreign_lang and foreign_lang not in ("auto", "de", "unknown"):
                        session.foreign_lang = foreign_lang
                    asr_task = asyncio.create_task(asr_loop())
                    mt_task = asyncio.create_task(mt_loop())

                    # Activate the session with the client-provided token
                    await registry.activate(session_token, session, websocket)
                    print(
                        f"Session activated: sample_rate={sample_rate}, src_lang={src_lang}, "
                        f"foreign_lang={foreign_lang or 'auto'}, token={session_token[:8]}..."
                    )

                    # Send config acknowledgment
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "config_ack",
                                "status": "active",
                            }
                        )
                    )

                    # Notify waiting viewers that session is now active
                    entry = await registry.get(session_token)
                    if entry and entry.viewers:
                        segments_data = []
                        await broadcast_to_viewers(
                            entry,
                            {
                                "type": "session_active",
                                "foreign_lang": session.foreign_lang,
                                "segments": segments_data,
                            },
                        )

                elif data.get("type") == "request_summary":
                    # Stop ASR loop
                    if asr_task:
                        asr_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await asr_task
                        asr_task = None
                        print("ASR loop stopped for stop request")

                    if session is not None:
                        # Finalize remaining segments
                        now_sec = session.get_current_time()
                        all_segs, time_finalized = session.segment_tracker.update_from_hypothesis(
                            [], 0.0, now_sec, "unknown"
                        )

                        for seg in time_finalized:
                            print(f"Queuing time-finalized segment {seg.id}: {seg.src[:50]}")
                            await translation_queue.put(seg)

                        live_segs = [s for s in all_segs if not s.final]
                        if live_segs:
                            print(f"Force-finalizing {len(live_segs)} live segments")
                            newly_final = session.segment_tracker.force_finalize_all(live_segs)
                            for seg in newly_final:
                                print(f"Queuing force-finalized segment {seg.id}: {seg.src[:50]}")
                                await translation_queue.put(seg)

                        # Wait for translations to complete
                        try:
                            await asyncio.wait_for(translation_queue.join(), timeout=60)
                        except TimeoutError:
                            print(
                                f"Warning: translation queue join timed out, "
                                f"{translation_queue.qsize()} items remaining"
                            )

                        # Send final segments with all translations
                        finalized = session.segment_tracker.finalized_segments
                        if finalized:
                            segments_data = []
                            for seg in finalized:
                                seg_dict = asdict(seg)
                                seg_dict["translations"] = session.translations.get(seg.id, {})
                                segments_data.append(seg_dict)

                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "segments",
                                        "t": session.get_current_time(),
                                        "src_lang": session.detected_lang or "unknown",
                                        "foreign_lang": session.foreign_lang or "en",
                                        "segments": segments_data,
                                    }
                                )
                            )

                        # Run summarization if backend is configured
                        summ_backend = get_summarization_backend()
                        if summ_backend is not None and finalized:
                            foreign_lang = session.foreign_lang or "en"
                            summ_segments = [
                                {
                                    "src": seg.src,
                                    "src_lang": seg.src_lang,
                                    "translations": session.translations.get(seg.id, {}),
                                }
                                for seg in finalized
                            ]
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "summary_progress",
                                        "step": "summarize",
                                        "message": "Generating summaries...",
                                    }
                                )
                            )
                            try:
                                loop = asyncio.get_event_loop()
                                foreign_summary, german_summary = await loop.run_in_executor(
                                    _executor,
                                    summ_backend.summarize_bilingual,
                                    summ_segments,
                                    foreign_lang,
                                )
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "summary",
                                            "foreign_summary": foreign_summary,
                                            "german_summary": german_summary,
                                            "foreign_lang": foreign_lang,
                                        }
                                    )
                                )
                                # Broadcast summary to viewers
                                if session_token:
                                    entry = await registry.get(session_token)
                                    if entry:
                                        await broadcast_to_viewers(
                                            entry,
                                            {
                                                "type": "summary",
                                                "foreign_summary": foreign_summary,
                                                "german_summary": german_summary,
                                                "foreign_lang": foreign_lang,
                                            },
                                        )
                            except Exception as e:
                                print(f"Summarization error: {e}")

                        running = False
                        print("Recording stopped, closing connection")
                        await websocket.close()

            elif "bytes" in message:
                if session is not None:
                    audio_bytes = message["bytes"]
                    bytes_received += len(audio_bytes)
                    session.add_german_audio(audio_bytes)
                    if msg_count % 50 == 0:
                        print(
                            f"Audio: {bytes_received} bytes, {session.get_buffered_seconds():.1f}s buffered"
                        )

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        running = False

        # Notify viewers that session has ended and unregister
        if session_token:
            entry = await registry.get(session_token)
            if entry:
                await broadcast_to_viewers(entry, {"type": "session_ended"})
            await registry.unregister(session_token)

        if asr_task:
            asr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asr_task
        if mt_task:
            mt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mt_task


async def handle_viewer_websocket(websocket: WebSocket, token: str) -> None:
    """Handle a read-only viewer WebSocket connection."""
    await websocket.accept()

    # Add viewer to session (creates pending entry if doesn't exist)
    if not await registry.add_viewer(token, websocket):
        # Token not reserved - reserve it now for the viewer
        await registry.reserve(token)
        await registry.add_viewer(token, websocket)

    print(f"Viewer connected to session {token[:8]}...")

    try:
        # Check if session is already active
        entry = await registry.get(token)
        if entry and entry.is_active:
            # Session is active, send current state
            session = entry.session
            segments_data = []
            live_segments = [cs.segment for cs in session.segment_tracker.cumulative_segments]
            all_segments = list(session.segment_tracker.finalized_segments) + live_segments
            for seg in all_segments:
                seg_dict = asdict(seg)
                seg_dict["translations"] = session.translations.get(seg.id, {})
                segments_data.append(seg_dict)

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
                            if viewer_foreign_lang:
                                if cached_session is None:
                                    entry = await registry.get(token)
                                    if entry and entry.session:
                                        cached_session = entry.session
                                if (
                                    cached_session is not None
                                    and cached_session.foreign_lang is None
                                ):
                                    cached_session.foreign_lang = viewer_foreign_lang
                                    print(f"Viewer set foreign language: {viewer_foreign_lang}")
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
        print(f"Viewer websocket error: {e}")
    finally:
        await registry.remove_viewer(token, websocket)
        print(f"Viewer disconnected from session {token[:8]}...")

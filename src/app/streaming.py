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

from app.asr import get_model
from app.diarization import SpeakerSegment, StreamingDiarizer
from app.lang_id import SpeakerLanguageTracker
from app.mt import (
    summarize_bilingual,
    translate_texts,
)
from app.session_registry import SessionEntry, registry
from app.streaming_policy import Segment, SegmentTracker

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "8.0"))
TICK_SEC = float(os.getenv("TICK_SEC", "0.5"))
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "30.0"))

# Bilingual example prompts for Whisper initial_prompt
# These are example text snippets (not instructions) that help Whisper recognize
# the expected languages. Whisper follows the style of the prompt, not instructions.
# See: https://cookbook.openai.com/examples/whisper_prompting_guide
BILINGUAL_PROMPTS = {
    # Must have languages
    "bg": "Guten Tag, wie kann ich Ihnen helfen? Здравейте, как мога да ви помогна?",
    "en": "Guten Tag, wie kann ich Ihnen helfen? Hello, how can I help you?",
    "es": "Guten Tag, wie kann ich Ihnen helfen? Hola, ¿cómo puedo ayudarle?",
    "fr": "Guten Tag, wie kann ich Ihnen helfen? Bonjour, comment puis-je vous aider?",
    "hr": "Guten Tag, wie kann ich Ihnen helfen? Dobar dan, kako vam mogu pomoći?",
    "hu": "Guten Tag, wie kann ich Ihnen helfen? Jó napot, miben segíthetek?",
    "it": "Guten Tag, wie kann ich Ihnen helfen? Buongiorno, come posso aiutarla?",
    "pl": "Guten Tag, wie kann ich Ihnen helfen? Dzień dobry, jak mogę pomóc?",
    "ro": "Guten Tag, wie kann ich Ihnen helfen? Bună ziua, cum vă pot ajuta?",
    "ru": "Guten Tag, wie kann ich Ihnen helfen? Здравствуйте, чем могу помочь?",
    "sq": "Guten Tag, wie kann ich Ihnen helfen? Mirëdita, si mund t'ju ndihmoj?",
    "tr": "Guten Tag, wie kann ich Ihnen helfen? Merhaba, size nasıl yardımcı olabilirim?",
    "uk": "Guten Tag, wie kann ich Ihnen helfen? Добрий день, чим можу допомогти?",
    # Nice to have languages
    "ar": "Guten Tag, wie kann ich Ihnen helfen? مرحباً، كيف يمكنني مساعدتك؟",
    "fa": "Guten Tag, wie kann ich Ihnen helfen? سلام، چطور می‌توانم کمکتان کنم؟",
    "ku": "Guten Tag, wie kann ich Ihnen helfen? Rojbaş, çawa dikarim alîkariya we bikim?",
    "sr": "Guten Tag, wie kann ich Ihnen helfen? Dobar dan, kako mogu da vam pomognem?",
}

# Bag of Hallucinations (BoH) - common Whisper hallucinations on silence/noise
# Based on research: https://arxiv.org/abs/2501.11378
HALLUCINATION_PHRASES = frozenset(
    phrase.lower()
    for phrase in [
        "Thank you.",
        "Thanks for watching.",
        "Thanks for watching!",
        "Thank you for watching.",
        "Thank you for watching!",
        "Please subscribe.",
        "Please subscribe!",
        "Subscribe to my channel.",
        "Like and subscribe.",
        "Subtitles by the Amara.org community",
        "Subtitles by",
        "ご視聴ありがとうございました",
        "Bye.",
        "Bye!",
        "Goodbye.",
        "See you next time.",
        "See you in the next video.",
        "...",
        "MBC 뉴스 , 뉴스를 전해 드립니다.",
        "Продолжение следует...",  # Russian "To be continued"
        "Продолжение следует",
        "To be continued...",
        "To be continued",
    ]
)

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
        # Speaker diarization and language tracking
        self.diarizer = StreamingDiarizer(sample_rate=sample_rate)
        self.language_tracker = SpeakerLanguageTracker()
        self.last_diar_segments: list[SpeakerSegment] = []  # Cache for merging with ASR

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

    def get_window_audio(self) -> tuple[np.ndarray, float]:
        """Get audio window for ASR processing.

        Uses full audio until it exceeds EXTENDED_WINDOW_SEC, then switches
        to sliding window. This ensures early speech has time to stabilize
        before the window starts sliding.
        """
        EXTENDED_WINDOW_SEC = 12.0  # Use full audio until this threshold
        window_samples = int(WINDOW_SEC * self.sample_rate)
        extended_samples = int(EXTENDED_WINDOW_SEC * self.sample_rate)

        all_bytes = b"".join(self.audio_buffer)
        total_samples = len(all_bytes) // 2

        window_start = 0.0

        # Use full audio until we exceed extended threshold
        # This gives early segments time to stabilize before window slides
        if total_samples <= extended_samples:
            # Use all audio
            pass
        elif total_samples > window_samples:
            # Switch to sliding window after extended period
            start_byte = (total_samples - window_samples) * 2
            all_bytes = all_bytes[start_byte:]
            window_start = (total_samples - window_samples) / self.sample_rate

        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, window_start

    def get_current_time(self) -> float:
        return self.total_samples / self.sample_rate

    def get_buffered_seconds(self) -> float:
        all_bytes = b"".join(self.audio_buffer)
        return len(all_bytes) / 2 / self.sample_rate


def run_diarization(
    session: StreamingSession,
    audio: np.ndarray,
    window_start: float,
) -> list[SpeakerSegment]:
    """Run speaker diarization on audio window."""
    diar_start = time.time()
    try:
        diar_segments = session.diarizer.process_audio(audio, window_start)
        diar_time = time.time() - diar_start
        _metrics["diar_times"].append(diar_time)

        if diar_segments:
            print(f"Diarization: {len(diar_segments)} speaker segments in {diar_time * 1000:.1f}ms")
            for seg in diar_segments[:3]:
                print(f"  - [{seg.start:.2f}-{seg.end:.2f}] {seg.speaker_id}")

        return diar_segments
    except Exception as e:
        print(f"Diarization error: {e}")
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


def transcribe_speaker_segment(
    model,
    audio: np.ndarray,
    language: str | None,
    diar_seg: SpeakerSegment,
    window_start: float,
    sample_rate: int = 16000,
    padding_sec: float = 0.3,
) -> list[dict]:
    """Transcribe a single speaker segment with explicit language hint.

    Args:
        model: Whisper model
        audio: Full audio window
        language: Language code to use (or None for multilingual)
        diar_seg: Diarization segment with start/end times
        window_start: Absolute start time of the audio window
        sample_rate: Audio sample rate
        padding_sec: Padding to add before/after segment boundaries

    Returns:
        List of segment dicts with text, timing, speaker, and language
    """
    # Extract audio for this speaker segment with padding for context
    start_rel = diar_seg.start - window_start
    end_rel = diar_seg.end - window_start

    # Add padding to capture context (helps with word boundaries)
    padded_start = max(0, start_rel - padding_sec)
    padded_end = min(len(audio) / sample_rate, end_rel + padding_sec)

    start_sample = int(padded_start * sample_rate)
    end_sample = int(padded_end * sample_rate)

    # Skip segments that are too short even with padding
    segment_duration = diar_seg.end - diar_seg.start
    if segment_duration < 0.5:  # Skip <0.5s diarization segments (likely truncated)
        print(f"  SKIP short diar segment: {segment_duration:.2f}s")
        return []

    if end_sample - start_sample < int(0.5 * sample_rate):  # Skip <0.5s audio
        return []

    segment_audio = audio[start_sample:end_sample]

    # Transcribe with explicit language (or multilingual if unknown)
    segments, info = model.transcribe(
        segment_audio,
        language=language if language and language != "unknown" else None,
        beam_size=1,
        patience=1.0,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "min_silence_duration_ms": 300,
            "min_speech_duration_ms": 200,
            "speech_pad_ms": 250,
        },
        compression_ratio_threshold=1.8,
        no_speech_threshold=0.3,
        log_prob_threshold=-0.8,
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=1.5,
        multilingual=(language is None or language == "unknown"),
        language_detection_threshold=0.5,
    )

    results = []
    for seg in segments:
        text = seg.text.strip()
        if len(text) < 2:
            continue

        # Convert times back to window-relative (accounting for padding)
        # seg.start/end are relative to padded_start, so add padded_start to get window-relative
        seg_start = padded_start + seg.start
        seg_end = padded_start + seg.end

        results.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "speaker_id": diar_seg.speaker_id,
                "lang": language if language and language != "unknown" else info.language,
            }
        )

    return results


def run_asr(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """Run ASR with diarization-first approach.

    Pipeline order:
    1. Run diarization to identify speaker segments
    2. Detect language per speaker using SpeechBrain
    3. Transcribe each speaker's audio with their detected language

    This prevents German speech from being transcribed as Arabic when
    the window contains more Arabic audio (wrong window-level language).
    """
    tick_start = time.time()

    audio, window_start = session.get_window_audio()
    now_sec = time.time() - session.start_time

    if len(audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    model = get_model()

    # Debug audio stats
    audio_rms = float(np.sqrt(np.mean(audio**2)))
    audio_max = float(np.max(np.abs(audio)))
    print(f"Pipeline: audio_len={len(audio)}, rms={audio_rms:.4f}, max={audio_max:.4f}")

    # STEP 1: Run diarization FIRST to identify speaker segments
    diar_start = time.time()
    diar_segments = run_diarization(session, audio, window_start)
    session.last_diar_segments = diar_segments
    diar_time = time.time() - diar_start

    if not diar_segments:
        # Fallback: no diarization, run full-window ASR with multilingual
        print("No diarization segments, falling back to full-window ASR")
        return _run_asr_fallback(model, session, audio, window_start, now_sec)

    print(f"Diarization: {len(diar_segments)} segments in {diar_time * 1000:.1f}ms")

    # STEP 2: Detect language per speaker from raw audio (BEFORE ASR)
    speaker_languages: dict[str, tuple[str, float]] = {}  # speaker_id -> (lang, confidence)
    unique_speakers = {seg.speaker_id for seg in diar_segments}

    for speaker_id in unique_speakers:
        # Extract all audio for this speaker
        speaker_audio = extract_speaker_audio(audio, diar_segments, speaker_id, window_start)

        if len(speaker_audio) < 8000:  # Need at least 0.5s for reliable detection
            print(f"  {speaker_id}: audio too short ({len(speaker_audio) / 16000:.2f}s)")
            continue

        # Use SpeechBrain language detection on raw audio
        lang, confidence = session.language_tracker.get_speaker_language(
            speaker_id, speaker_audio, 16000
        )

        speaker_languages[speaker_id] = (lang, confidence)
        print(f"  {speaker_id} → {lang} (confidence={confidence:.2f})")

        # Update foreign language tracking
        if lang not in ("unknown", "de") and session.foreign_lang is None:
            session.foreign_lang = lang
            print(f"Foreign language detected: {lang}")

    # STEP 3: Transcribe per speaker with their detected language
    asr_start = time.time()
    hyp_segments = []

    # Group consecutive segments by speaker to reduce ASR calls
    for diar_seg in diar_segments:
        speaker_id = diar_seg.speaker_id
        lang, confidence = speaker_languages.get(speaker_id, ("unknown", 0.0))

        # Skip segments that start right at window boundary (likely truncated from earlier)
        # These are the tail ends of segments that started before this window
        segment_rel_start = diar_seg.start - window_start
        if segment_rel_start < 0.5 and (diar_seg.end - diar_seg.start) < 1.0:
            print(
                f"  SKIP boundary segment: {diar_seg.speaker_id} "
                f"[{diar_seg.start:.2f}-{diar_seg.end:.2f}] (truncated)"
            )
            continue

        # Use detected language if confident, otherwise let Whisper decide
        use_lang = lang if confidence > 0.7 and lang != "unknown" else None

        # Transcribe this speaker segment
        seg_results = transcribe_speaker_segment(model, audio, use_lang, diar_seg, window_start)

        # Filter hallucinations
        for seg in seg_results:
            text = seg["text"]
            duration = seg["end"] - seg["start"]

            # Skip repeated words
            words = text.split()
            if len(words) > 1 and len(set(words)) == 1:
                continue

            # Skip long segments (hallucinations)
            if duration > 10.0:
                print(f"  SKIP hallucination (long): {duration:.1f}s: {text[:50]}")
                continue

            # Skip known hallucination phrases
            if text.lower() in HALLUCINATION_PHRASES:
                print(f"  SKIP hallucination (BoH): {text}")
                continue

            hyp_segments.append(seg)

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
        # Use most common non-unknown language from speakers
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


def _run_asr_fallback(
    model,
    session: StreamingSession,
    audio: np.ndarray,
    window_start: float,
    now_sec: float,
) -> tuple[list[Segment], list[Segment]]:
    """Fallback ASR when diarization fails - uses full window with multilingual."""
    asr_start = time.time()

    asr_segments, info = model.transcribe(
        audio,
        beam_size=1,
        patience=1.0,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "min_silence_duration_ms": 300,
            "min_speech_duration_ms": 200,
            "speech_pad_ms": 250,
        },
        compression_ratio_threshold=1.8,
        no_speech_threshold=0.3,
        log_prob_threshold=-0.8,
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=1.5,
        multilingual=True,
        language_detection_threshold=0.5,
    )
    asr_segments = list(asr_segments)
    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    print(f"Fallback ASR: {len(asr_segments)} segs, lang={info.language}, {asr_time * 1000:.1f}ms")

    hyp_segments = []
    for seg in asr_segments:
        text = seg.text.strip()
        duration = seg.end - seg.start

        if len(text) < 2:
            continue
        words = text.split()
        if len(words) > 1 and len(set(words)) == 1:
            continue
        if duration > 10.0:
            continue
        if text.lower() in HALLUCINATION_PHRASES:
            continue

        # Use SpeechBrain for segment language detection
        seg_start_sample = int(seg.start * 16000)
        seg_end_sample = int(seg.end * 16000)
        segment_audio = audio[seg_start_sample : min(seg_end_sample, len(audio))]

        from app.lang_id import detect_language_from_audio

        speechbrain_lang, confidence = detect_language_from_audio(segment_audio, 16000)
        segment_lang = speechbrain_lang if speechbrain_lang != "unknown" else info.language

        if segment_lang not in ("unknown", "de") and session.foreign_lang is None:
            session.foreign_lang = segment_lang
            print(f"Foreign language detected: {segment_lang}")

        hyp_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "speaker_id": None,
                "lang": segment_lang,
            }
        )

    if session.src_lang == "auto":
        session.detected_lang = info.language
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
    result = translate_texts([text], src_lang=src_lang, tgt_lang=tgt_lang)[0]
    mt_time = time.time() - mt_start
    _metrics["mt_times"].append(mt_time)
    return result


def run_summarization_pipeline(
    segments_data: list[dict],
    foreign_lang: str,
) -> dict:
    """
    Run summarization pipeline - generates both summaries in a single LLM call.
    """
    print(f"Starting summarization: {len(segments_data)} segments, foreign_lang={foreign_lang}")

    # Generate both summaries in one call
    print("Generating bilingual summary...")
    foreign_summary, german_summary = summarize_bilingual(
        segments=segments_data,
        foreign_lang=foreign_lang,
    )
    print(f"Foreign summary: {foreign_summary[:100]}...")
    print(f"German summary: {german_summary[:100]}...")

    return {
        "foreign_summary": foreign_summary,
        "german_summary": german_summary,
        "foreign_lang": foreign_lang,
        "validation": {"aligned": True, "issues": None},
        "regenerated": False,
    }


async def handle_websocket(websocket: WebSocket):
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
                    all_segments, newly_finalized = await loop.run_in_executor(
                        _executor, run_asr, session
                    )

                    if running:
                        # Auto-detect foreign language from first non-German speech
                        if session.foreign_lang is None and session.detected_lang:
                            if session.src_lang != "auto":
                                # User explicitly selected a foreign language
                                if session.src_lang != "de":
                                    session.foreign_lang = session.src_lang
                            elif session.detected_lang != "de":
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
                    # Default to English if foreign language not yet detected
                    tgt_lang = (session.foreign_lang or "en") if seg_src_lang == "de" else "de"

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
                    if foreign_lang:
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
                    # Handle summary request - stop ASR loop first
                    if asr_task:
                        asr_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await asr_task
                        asr_task = None
                        print("ASR loop stopped for summary generation")

                    if session is not None:
                        # Get all segments including live ones
                        # Use wall-clock time for stability tracking
                        now_sec = time.time() - session.start_time
                        all_segs, _ = session.segment_tracker.update_from_hypothesis(
                            [], 0.0, now_sec, "unknown"
                        )

                        # Identify live segments that need finalization
                        live_segs = [s for s in all_segs if not s.final]
                        if live_segs:
                            print(f"Force-finalizing {len(live_segs)} live segments")
                            newly_final = session.segment_tracker.force_finalize_all(live_segs)
                            # Queue for translation
                            for seg in newly_final:
                                print(f"Queuing force-finalized segment {seg.id}: {seg.src[:50]}")
                                await translation_queue.put(seg)

                        finalized = session.segment_tracker.finalized_segments
                        if finalized:
                            # Wait for translation queue to drain (up to 10s)
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "summary_progress",
                                        "step": "translate",
                                        "message": "Completing translations...",
                                    }
                                )
                            )

                            wait_start = time.time()
                            while not translation_queue.empty() and (time.time() - wait_start) < 10:
                                await asyncio.sleep(0.2)

                            if not translation_queue.empty():
                                print(
                                    f"Warning: {translation_queue.qsize()} translations still pending"
                                )

                            # Build segments data with translations
                            segments_data = []
                            for seg in finalized:
                                seg_dict = asdict(seg)
                                seg_dict["translations"] = session.translations.get(seg.id, {})
                                segments_data.append(seg_dict)

                            foreign_lang = session.foreign_lang or "en"

                            # Send progress update
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "summary_progress",
                                        "step": "summarize_foreign",
                                        "message": "Generating summaries...",
                                    }
                                )
                            )

                            # Run summarization in executor
                            loop = asyncio.get_event_loop()
                            try:
                                result = await loop.run_in_executor(
                                    _executor,
                                    run_summarization_pipeline,
                                    segments_data,
                                    foreign_lang,
                                )

                                # Send final summary
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "summary",
                                            "foreign_summary": result["foreign_summary"],
                                            "german_summary": result["german_summary"],
                                            "foreign_lang": result["foreign_lang"],
                                            "aligned": result["validation"]["aligned"],
                                            "issues": result["validation"].get("issues"),
                                            "regenerated": result["regenerated"],
                                        }
                                    )
                                )
                            except Exception as e:
                                print(f"Summarization error: {e}")
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "summary_error",
                                            "error": str(e),
                                        }
                                    )
                                )
                        else:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "summary_error",
                                        "error": "No conversation segments to summarize",
                                    }
                                )
                            )

            elif "bytes" in message:
                if session is not None:
                    audio_bytes = message["bytes"]
                    bytes_received += len(audio_bytes)
                    session.add_audio(audio_bytes)
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

        # Keep connection alive with ping/pong
        while True:
            try:
                # Wait for any message (used for pong/keepalive)
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                if message["type"] == "websocket.disconnect":
                    break
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

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
from app.mt import translate_texts
from app.streaming_policy import Segment, SegmentTracker

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "8.0"))
TICK_SEC = float(os.getenv("TICK_SEC", "0.5"))
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "30.0"))

_executor = ThreadPoolExecutor(max_workers=2)

# Metrics
_metrics = {
    "asr_times": deque(maxlen=100),
    "mt_times": deque(maxlen=100),
    "tick_times": deque(maxlen=100),
}


def get_metrics() -> dict:
    asr_times = list(_metrics["asr_times"])
    mt_times = list(_metrics["mt_times"])
    tick_times = list(_metrics["tick_times"])

    return {
        "avg_asr_time_ms": sum(asr_times) / len(asr_times) * 1000 if asr_times else 0,
        "avg_mt_time_ms": sum(mt_times) / len(mt_times) * 1000 if mt_times else 0,
        "avg_tick_time_ms": sum(tick_times) / len(tick_times) * 1000 if tick_times else 0,
        "sample_count": len(tick_times),
    }


class StreamingSession:
    def __init__(self, sample_rate: int = 16000, src_lang: str = "auto"):
        self.sample_rate = sample_rate
        self.src_lang = src_lang
        self.audio_buffer: deque[bytes] = deque()
        self.total_samples = 0
        self.detected_lang: str | None = None
        self.segment_tracker = SegmentTracker()
        self.dropped_frames = 0
        self.translations: dict[int, str] = {}  # segment_id -> translation

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
        window_samples = int(WINDOW_SEC * self.sample_rate)
        all_bytes = b"".join(self.audio_buffer)
        total_samples = len(all_bytes) // 2

        window_start = 0.0
        if total_samples > window_samples:
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


def run_asr(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """Run ASR and return (all_segments, newly_finalized)."""
    tick_start = time.time()

    audio, window_start = session.get_window_audio()
    now_sec = session.get_current_time()

    if len(audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    asr_start = time.time()
    model = get_model()
    asr_segments, info = model.transcribe(
        audio,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )
    asr_segments = list(asr_segments)  # Consume generator
    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    if session.src_lang == "auto":
        session.detected_lang = info.language
    else:
        session.detected_lang = session.src_lang

    # Debug: log raw ASR output
    if asr_segments:
        print(f"ASR raw: {len(asr_segments)} segments, lang={info.language}")
        for seg in asr_segments[:2]:  # Log first 2
            print(f"  - [{seg.start:.1f}-{seg.end:.1f}] {seg.text[:50]}")

    hyp_segments = []
    for seg in asr_segments:
        text = seg.text.strip()
        if len(text) < 2:
            continue
        words = text.split()
        if len(words) > 1 and len(set(words)) == 1:
            continue
        hyp_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": text,
            }
        )

    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=hyp_segments,
        window_start=window_start,
        now_sec=now_sec,
    )

    tick_time = time.time() - tick_start
    _metrics["tick_times"].append(tick_time)

    return all_segments, newly_finalized


def run_translation(text: str, src_lang: str) -> str:
    """Translate a single text."""
    mt_start = time.time()
    result = translate_texts([text], src_lang=src_lang, tgt_lang="de")[0]
    mt_time = time.time() - mt_start
    _metrics["mt_times"].append(mt_time)
    return result


async def handle_websocket(websocket: WebSocket):
    await websocket.accept()

    session: StreamingSession | None = None
    asr_task: asyncio.Task | None = None
    mt_task: asyncio.Task | None = None
    running = True
    translation_queue: asyncio.Queue[Segment] = asyncio.Queue()

    async def asr_loop():
        """ASR loop - runs independently, sends segments immediately."""
        nonlocal running
        tick_count = 0
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
                        # Build segments with translations where available
                        segments_data = []
                        for seg in all_segments:
                            seg_dict = asdict(seg)
                            seg_dict["de"] = session.translations.get(seg.id, "...")
                            segments_data.append(seg_dict)

                        # Send ASR results immediately
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "segments",
                                    "t": session.get_current_time(),
                                    "src_lang": session.detected_lang or "unknown",
                                    "segments": segments_data,
                                }
                            )
                        )

                        # Queue newly finalized segments for translation
                        for seg in newly_finalized:
                            await translation_queue.put(seg)

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

                # Run translation in executor
                loop = asyncio.get_event_loop()
                src_lang = session.detected_lang or "en"

                try:
                    translation = await loop.run_in_executor(
                        _executor, run_translation, segment.src, src_lang
                    )

                    # Store translation
                    session.translations[segment.id] = translation

                    if running:
                        # Send translation update
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "translation",
                                    "segment_id": segment.id,
                                    "de": translation,
                                }
                            )
                        )
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
                    session = StreamingSession(sample_rate=sample_rate, src_lang=src_lang)
                    asr_task = asyncio.create_task(asr_loop())
                    mt_task = asyncio.create_task(mt_loop())
                    print(f"Session started: sample_rate={sample_rate}, src_lang={src_lang}")

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
        if asr_task:
            asr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asr_task
        if mt_task:
            mt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mt_task

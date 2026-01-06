import asyncio
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
from fastapi import WebSocket

from app.asr import get_model
from app.mt import translate_texts
from app.streaming_policy import SegmentTracker

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "8.0"))
TICK_SEC = float(os.getenv("TICK_SEC", "0.5"))

_executor = ThreadPoolExecutor(max_workers=2)


class StreamingSession:
    def __init__(self, sample_rate: int = 16000, src_lang: str = "auto"):
        self.sample_rate = sample_rate
        self.src_lang = src_lang
        self.audio_buffer: deque[bytes] = deque()
        self.total_samples = 0
        self.detected_lang: str | None = None
        self.segment_tracker = SegmentTracker()

    def add_audio(self, pcm16_bytes: bytes):
        self.audio_buffer.append(pcm16_bytes)
        self.total_samples += len(pcm16_bytes) // 2

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


def transcribe_with_segments(session: StreamingSession) -> dict:
    audio, window_start = session.get_window_audio()
    now_sec = session.get_current_time()

    if len(audio) < 1600:
        return {
            "type": "segments",
            "t": now_sec,
            "src_lang": session.detected_lang or "unknown",
            "segments": [asdict(s) for s in session.segment_tracker.finalized_segments],
        }

    model = get_model()
    asr_segments, info = model.transcribe(audio)

    if session.src_lang == "auto":
        session.detected_lang = info.language
    else:
        session.detected_lang = session.src_lang

    hyp_segments = []
    for seg in asr_segments:
        hyp_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })

    segments = session.segment_tracker.update_from_hypothesis(
        hyp_segments=hyp_segments,
        window_start=window_start,
        now_sec=now_sec,
        translate_fn=translate_texts,
        src_lang=session.detected_lang or "en",
    )

    return {
        "type": "segments",
        "t": now_sec,
        "src_lang": session.detected_lang or "unknown",
        "segments": [asdict(s) for s in segments],
    }


async def handle_websocket(websocket: WebSocket):
    await websocket.accept()

    session: StreamingSession | None = None
    tick_task: asyncio.Task | None = None
    running = True

    async def tick_loop():
        nonlocal running
        while running:
            await asyncio.sleep(TICK_SEC)
            if session is not None and running:
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(_executor, transcribe_with_segments, session)
                    if running:
                        await websocket.send_text(json.dumps(result))
                except Exception:
                    pass

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    sample_rate = data.get("sample_rate", 16000)
                    src_lang = data.get("src_lang", "auto")
                    session = StreamingSession(sample_rate=sample_rate, src_lang=src_lang)
                    tick_task = asyncio.create_task(tick_loop())

            elif "bytes" in message:
                if session is not None:
                    session.add_audio(message["bytes"])

    except Exception:
        pass
    finally:
        running = False
        if tick_task:
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

"""
FastAPI application for LinguaGap real-time transcription and translation.

This is the main entry point for the application. It provides:
    - HTTP endpoints for health checks, metrics, and file upload transcription
    - WebSocket endpoints for real-time streaming transcription
    - Static file serving for the web UI

Endpoints:
    GET  /              - Web interface
    GET  /health        - Health check
    GET  /metrics       - Performance metrics (ASR, MT, diarization times)
    POST /transcribe_translate - File upload transcription
    WS   /ws            - Real-time streaming WebSocket
    GET  /viewer/{token} - Mobile viewer page
    WS   /ws/viewer/{token} - Read-only viewer WebSocket

Startup:
    All models are warmed up on startup via the lifespan handler to minimize
    cold-start latency on first request. This includes ASR, MT,
    diarization, and language ID models.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.asr import get_model, transcribe_wav_path
from app.lang_id import warmup_lang_id
from app.mt import get_llm, translate_texts
from app.scripts.asr_smoke import generate_silence_wav
from app.speaker_tracker import warmup_speaker_model
from app.streaming import get_metrics, handle_viewer_websocket, handle_websocket


def warmup_models():
    """
    Warm up all ML models on startup.

    This runs during application startup to ensure all models are loaded
    and ready before the first request. Without warmup, the first request
    would experience significant latency (minutes for large models).

    Models warmed up:
        - ASR (faster-whisper): ~2-3 GB VRAM
        - Translation (TranslateGemma 12B): ~8 GB VRAM
        - Speaker embeddings (SpeechBrain ECAPA): ~1 GB VRAM
        - Language ID (SpeechBrain): ~1 GB VRAM

    Total warmup time is typically 5-10 minutes depending on network speed
    for model downloads and GPU initialization.
    """
    # Load SpeechBrain models FIRST - they use CUDNN which must initialize
    # before llama-cpp-python's cuBLAS takes over CUDA context
    warmup_speaker_model()
    warmup_lang_id()

    print("Warming up ASR model...")
    asr_model = get_model()
    print("  Running test transcription...")
    silence = np.zeros(16000, dtype=np.float32)
    list(asr_model.transcribe(silence))
    print("ASR model ready")

    print("Warming up MT model...")
    get_llm()
    translate_texts(["Hello"], src_lang="en", tgt_lang="de")
    print("MT model ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup_models()
    yield


app = FastAPI(
    title="LinguaGap",
    description="Real-time speech transcription and translation",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return get_metrics()


@app.get("/asr_smoke")
async def asr_smoke():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    generate_silence_wav(wav_path, duration_sec=2.0)
    result = transcribe_wav_path(wav_path)
    return result


@app.get("/mt_smoke")
async def mt_smoke():
    texts = ["Hello world!"]
    result = translate_texts(texts, src_lang="en", tgt_lang="de")
    return {"input": texts, "output": result}


@app.post("/transcribe_translate")
async def transcribe_translate(
    file: UploadFile = File(...),
    src_lang: str = Form("auto"),
):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        audio_path = f.name

    asr_result = transcribe_wav_path(audio_path)

    detected_lang = asr_result["language"]
    if src_lang == "auto":
        src_lang = detected_lang

    segments = []
    for i, seg in enumerate(asr_result["segments"]):
        src_text = seg["text"].strip()
        if src_text:
            de_text = translate_texts([src_text], src_lang=src_lang, tgt_lang="de")[0]
        else:
            de_text = ""

        segments.append(
            {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "src": src_text,
                "de": de_text,
            }
        )

    os.unlink(audio_path)

    return {
        "src_lang_detected": detected_lang,
        "segments": segments,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)


@app.get("/viewer/{token}")
async def viewer_page(token: str):  # noqa: ARG001
    """Serve the mobile viewer page.

    The token is validated client-side via WebSocket connection.
    """
    viewer_html = STATIC_DIR / "viewer.html"
    if not viewer_html.exists():
        return {"error": "Viewer not available"}
    return FileResponse(viewer_html)


@app.websocket("/ws/viewer/{token}")
async def viewer_websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket endpoint for read-only viewers."""
    await handle_viewer_websocket(websocket, token)

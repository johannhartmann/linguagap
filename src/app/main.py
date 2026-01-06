import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.asr import get_model, transcribe_wav_path
from app.mt import get_tokenizer, get_translator, translate_texts
from app.scripts.asr_smoke import generate_silence_wav
from app.streaming import get_metrics, handle_websocket


def warmup_models():
    print("Warming up ASR model...")
    asr_model = get_model()
    silence = np.zeros(16000, dtype=np.float32)
    list(asr_model.transcribe(silence))
    print("ASR model ready")

    print("Warming up MT model...")
    get_translator()
    get_tokenizer()
    translate_texts(["Hello"], src_lang="en", tgt_lang="de")
    print("MT model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
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

        segments.append({
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "src": src_text,
            "de": de_text,
        })

    os.unlink(audio_path)

    return {
        "src_lang_detected": detected_lang,
        "segments": segments,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)

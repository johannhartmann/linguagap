import tempfile

from fastapi import FastAPI

from app.asr import transcribe_wav_path
from app.scripts.asr_smoke import generate_silence_wav

app = FastAPI(title="LinguaGap", description="Real-time speech transcription and translation")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/asr_smoke")
async def asr_smoke():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    generate_silence_wav(wav_path, duration_sec=2.0)
    result = transcribe_wav_path(wav_path)
    return result

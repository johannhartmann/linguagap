import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile

from app.asr import transcribe_wav_path
from app.mt import translate_texts
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

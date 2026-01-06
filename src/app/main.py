from fastapi import FastAPI

app = FastAPI(title="LinguaGap", description="Real-time speech transcription and translation")


@app.get("/health")
async def health():
    return {"status": "ok"}

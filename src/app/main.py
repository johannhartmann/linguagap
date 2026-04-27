"""
FastAPI application for LinguaGap real-time transcription and translation.

This is the main entry point for the application. It provides:
    - HTTP endpoints for health checks, metrics, and file upload transcription
    - WebSocket endpoints for real-time streaming transcription
    - Static file serving for the web UI
    - Demo authentication for the desktop interface
    - Admin UI for managing demo accounts and logos

Startup:
    All models are warmed up on startup via the lifespan handler to minimize
    cold-start latency on first request.
"""

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app.asr import transcribe_wav_path
from app.auth import (
    LOGOS_DIR,
    DemoAccount,
    get_accounts,
    get_current_user,
    save_accounts,
    verify_admin,
    verify_credentials,
)
from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend
from app.languages import speech_languages, translation_languages
from app.mt import translate_texts
from app.scripts.asr_smoke import generate_silence_wav
from app.streaming import get_metrics, handle_viewer_websocket, handle_websocket

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

SESSION_SECRET = os.getenv("SESSION_SECRET", "linguagap-dev-secret-change-me")


def warmup_models() -> None:
    """Load all ML models on startup so the first request doesn't pay the cold-start cost."""
    logger.info("Warming up ASR backend...")
    get_asr_backend().warmup()
    logger.info("ASR backend ready")

    logger.info("Warming up MT backend...")
    get_translation_backend().warmup()
    logger.info("MT backend ready")

    summ = get_summarization_backend()
    if summ is not None:
        logger.info("Warming up summarization backend...")
        summ.warmup()
        logger.info("Summarization backend ready")

    if os.getenv("TTS_BACKEND"):
        logger.info("Warming up TTS model...")
        from app.tts import get_tts_model

        get_tts_model()
        logger.info("TTS model ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    warmup_models()
    yield


app = FastAPI(
    title="LinguaGap",
    description="Real-time speech transcription and translation",
    lifespan=lifespan,
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="linguagap_session")

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------


async def require_auth(request: Request):
    if not request.session.get("email"):
        raise HTTPException(status_code=401, detail="Not authenticated")


async def require_admin(request: Request):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")


# ---------------------------------------------------------------------------
# User auth routes (public)
# ---------------------------------------------------------------------------


@app.get("/login")
async def login_page():
    return FileResponse(STATIC_DIR / "login.html")


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/login")
async def api_login(request: Request, body: LoginRequest):
    account, is_admin = verify_credentials(body.email, body.password)
    if account is None:
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    request.session["email"] = account.email
    request.session["display_name"] = account.display_name
    request.session["logo_url"] = account.logo_url
    if is_admin:
        request.session["is_admin"] = True
    return {"ok": True, "display_name": account.display_name, "logo_url": account.logo_url}


@app.post("/api/logout")
async def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
async def api_me(request: Request):
    user = get_current_user(request)
    if user is None:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    return user


# ---------------------------------------------------------------------------
# Admin auth routes
# ---------------------------------------------------------------------------


@app.get("/admin/login")
async def admin_login_page():
    return FileResponse(STATIC_DIR / "admin-login.html")


@app.post("/api/admin/login")
async def api_admin_login(request: Request, body: LoginRequest):
    if not verify_admin(body.email, body.password):
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    request.session["is_admin"] = True
    return {"ok": True}


@app.post("/api/admin/logout", dependencies=[Depends(require_admin)])
async def api_admin_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/admin")
async def admin_page(request: Request):
    if not request.session.get("is_admin"):
        return RedirectResponse("/admin/login", status_code=302)
    return FileResponse(STATIC_DIR / "admin.html")


# ---------------------------------------------------------------------------
# Admin API routes
# ---------------------------------------------------------------------------


@app.get("/api/admin/accounts", dependencies=[Depends(require_admin)])
async def list_accounts():
    return [asdict(a) for a in get_accounts()]


class AccountRequest(BaseModel):
    email: str
    password: str
    display_name: str
    logo_url: str = "/static/logos/synia.png"


@app.post("/api/admin/accounts", dependencies=[Depends(require_admin)])
async def create_account(body: AccountRequest):
    accounts = get_accounts()
    if any(a.email == body.email for a in accounts):
        raise HTTPException(status_code=409, detail="Account with this email already exists")
    account = DemoAccount(
        email=body.email,
        password=body.password,
        display_name=body.display_name,
        logo_url=body.logo_url,
    )
    accounts.append(account)
    save_accounts(accounts)
    return asdict(account)


@app.put("/api/admin/accounts/{email}", dependencies=[Depends(require_admin)])
async def update_account(email: str, body: AccountRequest):
    accounts = get_accounts()
    for i, a in enumerate(accounts):
        if a.email == email:
            accounts[i] = DemoAccount(
                email=body.email,
                password=body.password,
                display_name=body.display_name,
                logo_url=body.logo_url,
            )
            save_accounts(accounts)
            return asdict(accounts[i])
    raise HTTPException(status_code=404, detail="Account not found")


@app.delete("/api/admin/accounts/{email}", dependencies=[Depends(require_admin)])
async def delete_account(email: str):
    accounts = get_accounts()
    new_accounts = [a for a in accounts if a.email != email]
    if len(new_accounts) == len(accounts):
        raise HTTPException(status_code=404, detail="Account not found")
    save_accounts(new_accounts)
    return {"ok": True}


ALLOWED_LOGO_TYPES = {"image/png", "image/jpeg", "image/svg+xml", "image/webp"}
MAX_LOGO_SIZE = 512 * 1024  # 500KB


@app.post("/api/admin/upload-logo", dependencies=[Depends(require_admin)])
async def upload_logo(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_LOGO_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG, SVG, and WebP images allowed")
    content = await file.read()
    if len(content) > MAX_LOGO_SIZE:
        raise HTTPException(status_code=400, detail="Logo must be under 500KB")
    ext = Path(file.filename or "logo.png").suffix or ".png"
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    logo_path = LOGOS_DIR / filename
    logo_path.write_bytes(content)
    return {"logo_url": f"/logos/{filename}"}


@app.get("/logos/{filename}")
async def serve_logo(filename: str):
    path = LOGOS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root(request: Request):
    if not request.session.get("email"):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(require_auth)])
async def metrics():
    return get_metrics()


@app.get("/asr_smoke", dependencies=[Depends(require_auth)])
async def asr_smoke():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        generate_silence_wav(wav_path, duration_sec=2.0)
        return transcribe_wav_path(wav_path)
    finally:
        os.unlink(wav_path)


@app.get("/mt_smoke", dependencies=[Depends(require_auth)])
async def mt_smoke():
    texts = ["Hello world!"]
    result = translate_texts(texts, src_lang="en", tgt_lang="de")
    return {"input": texts, "output": result}


@app.get("/api/languages")
async def api_languages(scope: str = "speech"):
    """Single source of truth for UI language dropdowns.

    Public — viewer.html is served without a session cookie, and the list
    itself isn't sensitive. Frontend pages call this on load and populate
    both their dropdowns and any local code→label maps from the result.

    scope=speech (default): foreign speech languages for index/viewer.
    scope=translate: speech languages plus German (translate page).
    """
    if scope == "translate":
        return translation_languages()
    return speech_languages()


# Soft caps kept well below MT_N_CTX (4096 tokens). Latin/Cyrillic/Arabic
# scripts produce ~0.3–0.7 tokens per char with TranslateGemma's tokenizer;
# CJK scripts produce ~1.5–3 tokens per char, so allow far fewer characters
# for those source languages or a long input would blow the context window.
TRANSLATE_TEXT_MAX_CHARS = 4000
TRANSLATE_TEXT_MAX_CHARS_DENSE = 1500
_DENSE_TOKEN_LANGS: frozenset[str] = frozenset({"zh", "ja", "ko"})


def _max_translate_chars(src_lang: str) -> int:
    return (
        TRANSLATE_TEXT_MAX_CHARS_DENSE
        if src_lang in _DENSE_TOKEN_LANGS
        else TRANSLATE_TEXT_MAX_CHARS
    )


class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


@app.get("/translate", dependencies=[Depends(require_auth)])
async def translate_page():
    return FileResponse(STATIC_DIR / "translate.html")


@app.post("/api/translate", dependencies=[Depends(require_auth)])
async def api_translate(req: TranslateRequest):
    max_chars = _max_translate_chars(req.src_lang)
    if len(req.text) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Text zu lang ({len(req.text)} Zeichen, "
                f"Maximum {max_chars} für Quellsprache {req.src_lang!r})"
            ),
        )
    if not req.text.strip():
        return {"output": ""}
    if req.src_lang == req.tgt_lang:
        return {"output": req.text}
    try:
        output = await asyncio.to_thread(translate_texts, [req.text], req.src_lang, req.tgt_lang)
    except Exception:
        logger.exception("Text-to-text translation failed")
        raise HTTPException(status_code=500, detail="Übersetzung fehlgeschlagen") from None
    return {"output": output[0] if output else ""}


class TTSRequest(BaseModel):
    text: str
    lang: str


@app.post("/api/tts", dependencies=[Depends(require_auth)])
async def tts_endpoint(request: TTSRequest):
    from app.tts import TTS_SUPPORTED_LANGS, synthesize_wav

    if request.lang not in TTS_SUPPORTED_LANGS:
        raise HTTPException(status_code=404, detail="Language not supported for TTS")
    audio_bytes = await asyncio.to_thread(synthesize_wav, request.text, request.lang)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/transcribe_translate", dependencies=[Depends(require_auth)])
async def transcribe_translate(
    file: UploadFile = File(...),
    src_lang: str = Form("auto"),
):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        audio_path = f.name

    try:
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

        return {
            "src_lang_detected": detected_lang,
            "segments": segments,
        }
    finally:
        os.unlink(audio_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session = websocket.scope.get("session", {})
    if not session.get("email"):
        await websocket.close(code=4001, reason="Not authenticated")
        return
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

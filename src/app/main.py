"""FastAPI application entry point for LinguaGap.

Wires the lifespan/warmup, session middleware, static-files mount, route
modules, and WebSocket endpoints into a single ``app`` instance.

Each HTTP concern lives in its own router under :mod:`app.routes`:

- :mod:`app.routes.auth`      — user login / logout / current-user lookup
- :mod:`app.routes.admin`     — admin login, account CRUD, logo upload
- :mod:`app.routes.inference` — translation, TTS, ASR/MT smoke checks
- :mod:`app.routes.pages`     — root, viewer, /translate, /health, /metrics

The WebSocket handlers stay here because their lifecycle is tied to the
``app`` instance and they need direct access to the session middleware's
scope dict.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app import auth
from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend
from app.deps import STATIC_DIR
from app.routes import admin as admin_routes
from app.routes import auth as auth_routes
from app.routes import inference as inference_routes
from app.routes import pages as pages_routes
from app.streaming import handle_viewer_websocket, handle_websocket

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
    auth.LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    warmup_models()
    yield


app = FastAPI(
    title="LinguaGap",
    description="Real-time speech transcription and translation",
    lifespan=lifespan,
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="linguagap_session")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# HTTP routers
app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(inference_routes.router)
app.include_router(pages_routes.router)


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session = websocket.scope.get("session", {})
    if not session.get("email"):
        await websocket.close(code=4001, reason="Not authenticated")
        return
    await handle_websocket(websocket)


@app.websocket("/ws/viewer/{token}")
async def viewer_websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket endpoint for read-only viewers."""
    await handle_viewer_websocket(websocket, token)

"""
Machine translation - thin shim delegating to the translation backend.

Configuration:
    MT_BACKEND: Backend to use (default: "translategemma")
    MT_MODEL_REPO/MT_MODEL_FILE: Model config (passed to backend)
"""

from app.backends import get_translation_backend
from app.languages import LANG_INFO, LANG_NAMES

__all__ = ["LANG_INFO", "LANG_NAMES", "translate_texts"]


def translate_texts(texts: list[str], src_lang: str, tgt_lang: str = "de") -> list[str]:
    """Translate texts via the configured translation backend."""
    backend = get_translation_backend()
    return backend.translate(texts, src_lang=src_lang, tgt_lang=tgt_lang)

"""
Machine translation - thin shim delegating to the translation backend.

This module provides:
    - LANG_INFO / LANG_NAMES: Shared language registry used by all backends
    - translate_texts(): Backward-compatible function delegating to the backend
    - get_llm(): Backward-compatible function for warmup code

Configuration:
    MT_BACKEND: Backend to use (default: "translategemma")
    MT_MODEL_REPO/MT_MODEL_FILE: Model config (passed to backend)
"""

from app.backends import get_translation_backend

# Language name and code mapping for translation prompts
# Format: lang_code -> (full_name, iso_code)
# The full_name is used for summarization prompts (human-readable)
# The iso_code is used for TranslateGemma (BCP-47 style codes)
LANG_INFO = {
    "en": ("English", "en"),
    "de": ("German", "de"),
    "fr": ("French", "fr"),
    "es": ("Spanish", "es"),
    "it": ("Italian", "it"),
    "pl": ("Polish", "pl"),
    "ro": ("Romanian", "ro"),
    "hr": ("Croatian", "hr"),
    "bg": ("Bulgarian", "bg"),
    "sq": ("Albanian", "sq"),
    "tr": ("Turkish", "tr"),
    "ru": ("Russian", "ru"),
    "uk": ("Ukrainian", "uk"),
    "hu": ("Hungarian", "hu"),
    "ar": ("Arabic", "ar"),
    "fa": ("Persian", "fa"),
    "ku": ("Kurdish", "ku"),
    "sr": ("Serbian", "sr"),
    "zh": ("Chinese", "zh-Hans"),
    "ja": ("Japanese", "ja"),
    "ko": ("Korean", "ko"),
    "pt": ("Portuguese", "pt"),
    "nl": ("Dutch", "nl"),
}

# Keep LANG_NAMES for backward compatibility
LANG_NAMES = {k: v[0] for k, v in LANG_INFO.items()}


def get_llm():
    """Get the translation backend (for backward compatibility with warmup code)."""
    backend = get_translation_backend()
    backend.load_model()
    return backend


def translate_texts(texts: list[str], src_lang: str, tgt_lang: str = "de") -> list[str]:
    """Translate texts via the configured translation backend."""
    backend = get_translation_backend()
    return backend.translate(texts, src_lang=src_lang, tgt_lang=tgt_lang)

"""
Shared language registry for translation and summarization.

Format: lang_code -> (full_name, iso_code)
- full_name: Human-readable name for summarization prompts
- iso_code: BCP-47 style code for TranslateGemma
"""

LANG_INFO: dict[str, tuple[str, str]] = {
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

LANG_NAMES: dict[str, str] = {k: v[0] for k, v in LANG_INFO.items()}

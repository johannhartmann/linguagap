"""Voice configuration for Gemini TTS.

Defines voice assignments per language and speaker role.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceConfig:
    """Voice configuration for a speaker."""

    voice_name: str
    description: str


# Available Gemini TTS voices with characteristics
AVAILABLE_VOICES = {
    "Kore": VoiceConfig("Kore", "Firm, confident female voice"),
    "Puck": VoiceConfig("Puck", "Upbeat, energetic voice"),
    "Charon": VoiceConfig("Charon", "Deep, authoritative male voice"),
    "Fenrir": VoiceConfig("Fenrir", "Strong, commanding voice"),
    "Leda": VoiceConfig("Leda", "Clear, professional female voice"),
    "Enceladus": VoiceConfig("Enceladus", "Warm, friendly voice"),
    "Alnilam": VoiceConfig("Alnilam", "Calm, measured voice"),
    "Aoede": VoiceConfig("Aoede", "Melodic, pleasant voice"),
    "Zephyr": VoiceConfig("Zephyr", "Light, airy voice"),
}

# Language codes for Gemini TTS (BCP-47 format)
# Full list at: https://docs.cloud.google.com/text-to-speech/docs/gemini-tts
LANGUAGE_CODES = {
    # Core languages
    "de": "de-DE",  # German
    "en": "en-US",  # English
    # Target languages for bilingual testing
    "ar": "ar-EG",  # Arabic (Egypt)
    "bg": "bg-BG",  # Bulgarian
    "es": "es-ES",  # Spanish (Spain)
    "fa": "fa-IR",  # Farsi/Persian (Iran)
    "fr": "fr-FR",  # French
    "hr": "hr-HR",  # Croatian
    "hu": "hu-HU",  # Hungarian
    "it": "it-IT",  # Italian
    "ku": "ku-TR",  # Kurdish (Turkey) - NOTE: May not be supported
    "pl": "pl-PL",  # Polish
    "ro": "ro-RO",  # Romanian
    "ru": "ru-RU",  # Russian
    "sq": "sq-AL",  # Albanian (Albania)
    "sr": "sr-RS",  # Serbian (Serbia)
    "tr": "tr-TR",  # Turkish
    "uk": "uk-UA",  # Ukrainian
}

# Voice assignments per speaker role
# Speaker 1 = German speaker, Speaker 2 = Foreign language speaker
SPEAKER_VOICES = {
    "speaker_1": AVAILABLE_VOICES["Charon"],  # Deep male for German
    "speaker_2": AVAILABLE_VOICES["Kore"],  # Firm female for foreign language
}


def get_voice_for_speaker(speaker_id: str) -> str:
    """Get the voice name for a speaker ID.

    Args:
        speaker_id: Speaker identifier (e.g., "speaker_1", "speaker_2")

    Returns:
        Voice name for Gemini TTS
    """
    voice_config = SPEAKER_VOICES.get(speaker_id, AVAILABLE_VOICES["Puck"])
    return voice_config.voice_name


def get_language_code(lang: str) -> str:
    """Get the full language code for a language abbreviation.

    Args:
        lang: Language abbreviation (e.g., "de", "uk")

    Returns:
        Full language code (e.g., "de-DE", "uk-UA")
    """
    return LANGUAGE_CODES.get(lang, lang)

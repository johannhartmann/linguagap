"""Whisper ASR backend using faster-whisper (CTranslate2).

All Whisper-specific logic lives here: model loading, transcription parameters,
hallucination filtering, delooping, bilingual prompts, and language support.
"""

from __future__ import annotations

import os

import numpy as np

from app.backends.base import ASRBackend
from app.backends.types import ASRResult, ASRSegment

ASR_MODEL = os.getenv("ASR_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")


class WhisperASRBackend(ASRBackend):
    """faster-whisper ASR backend with hallucination filtering and delooping."""

    # Bilingual example prompts — help Whisper recognize expected language patterns
    # See: https://cookbook.openai.com/examples/whisper_prompting_guide
    _bilingual_prompts: dict[str, str] = {
        # Must have languages
        "bg": "Guten Tag, wie kann ich Ihnen helfen? Здравейте, как мога да ви помогна?",
        "en": "Guten Tag, wie kann ich Ihnen helfen? Hello, how can I help you?",
        "es": "Guten Tag, wie kann ich Ihnen helfen? Hola, ¿cómo puedo ayudarle?",
        "fr": "Guten Tag, wie kann ich Ihnen helfen? Bonjour, comment puis-je vous aider?",
        "hr": "Guten Tag, wie kann ich Ihnen helfen? Dobar dan, kako vam mogu pomoći?",
        "hu": "Guten Tag, wie kann ich Ihnen helfen? Jó napot, miben segíthetek?",
        "it": "Guten Tag, wie kann ich Ihnen helfen? Buongiorno, come posso aiutarla?",
        "pl": "Guten Tag, wie kann ich Ihnen helfen? Dzień dobry, jak mogę pomóc?",
        "ro": "Guten Tag, wie kann ich Ihnen helfen? Bună ziua, cum vă pot ajuta?",
        "ru": "Guten Tag, wie kann ich Ihnen helfen? Здравствуйте, чем могу помочь?",
        "sq": "Guten Tag, wie kann ich Ihnen helfen? Mirëdita, si mund t'ju ndihmoj?",
        "tr": "Guten Tag, wie kann ich Ihnen helfen? Merhaba, size nasıl yardımcı olabilirim?",
        "uk": "Guten Tag, wie kann ich Ihnen helfen? Добрий день, чим можу допомогти?",
        # Nice to have languages
        "ar": "Guten Tag, wie kann ich Ihnen helfen? مرحباً، كيف يمكنني مساعدتك؟",
        "fa": "Guten Tag, wie kann ich Ihnen helfen? سلام، چطور می‌توانم کمکتان کنم؟",
        "ku": "Guten Tag, wie kann ich Ihnen helfen? Rojbaş, çawa dikarim alîkariya we bikim?",
        "sr": "Guten Tag, wie kann ich Ihnen helfen? Dobar dan, kako mogu da vam pomognem?",
    }

    # Bag of Hallucinations (BoH) - common Whisper hallucinations on silence/noise
    # Based on research: https://arxiv.org/abs/2501.11378
    _hallucination_phrases: frozenset[str] = frozenset(
        phrase.lower()
        for phrase in [
            # Top English hallucinations (from research - >0.5% frequency)
            "Thank you",
            "Thank you.",
            "Thanks for watching",
            "Thanks for watching.",
            "Thanks for watching!",
            "Thank you for watching",
            "Thank you for watching.",
            "Thank you for watching!",
            "So",
            "So.",
            "The",
            "You",
            "Oh",
            "Oh.",
            "Okay",
            "Okay.",
            "I'm sorry",
            "I'm sorry.",
            "Oh my god",
            "Oh my god.",
            "Bye",
            "Bye.",
            "Bye!",
            "Uh",
            "Uh.",
            "Meow",
            "I'm not sure what I'm doing here",
            "I'm not sure what I'm doing here.",
            # Subscription/channel hallucinations
            "Please subscribe",
            "Please subscribe.",
            "Please subscribe!",
            "Subscribe to my channel",
            "Subscribe to my channel.",
            "Like and subscribe",
            "Like and subscribe.",
            "Hello everyone welcome to my channel",
            "See you next time",
            "See you next time.",
            "See you in the next video",
            "See you in the next video.",
            # Subtitle attribution hallucinations
            "Subtitles by the Amara.org community",
            "Subtitles by the Amara org community",
            "Subtitles by steamteamextra",
            "Subtitles by",
            # Non-English hallucinations
            "ご視聴ありがとうございました",  # Japanese "Thank you for watching"
            "MBC 뉴스 , 뉴스를 전해 드립니다.",  # Korean news intro
            "Продолжение следует...",  # Russian "To be continued"
            "Продолжение следует",
            "字幕由Amara.org社区提供",  # Chinese subtitle attribution
            "感谢收看",  # Chinese "Thanks for watching"
            "شكرا للمشاهدة",  # Arabic "Thanks for watching"
            "متابعتكم",  # Arabic "Your following"
            "مرحبا",  # Arabic "Hello" (when alone)
            # Common continuation/filler hallucinations
            "To be continued...",
            "To be continued",
            "Goodbye",
            "Goodbye.",
            "...",
            "…",
            # Single word/sound hallucinations
            "Hmm",
            "Hmm.",
            "Huh",
            "Huh.",
            "Yeah",
            "Yeah.",
            "Yes",
            "Yes.",
            "No",
            "No.",
            "Um",
            "Um.",
            "Ah",
            "Ah.",
        ]
    )

    # Whisper supported language codes
    _supported_languages: frozenset[str] = frozenset(
        [
            "af",
            "am",
            "ar",
            "as",
            "az",
            "ba",
            "be",
            "bg",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jw",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "la",
            "lb",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sa",
            "sd",
            "si",
            "sk",
            "sl",
            "sn",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yo",
            "zh",
            "yue",
        ]
    )

    # Map unsupported languages to closest supported alternatives
    _language_fallbacks: dict[str, str | None] = {
        "ku": None,  # Kurdish - no close match, use multilingual
    }

    def __init__(self) -> None:
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        print(f"  Loading WhisperModel: {ASR_MODEL}")
        print(f"  Device: {ASR_DEVICE}, Compute type: {ASR_COMPUTE_TYPE}")
        self._model = WhisperModel(
            ASR_MODEL,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        )
        print("  WhisperModel loaded")

    def warmup(self) -> None:
        self.load_model()
        print("  Running test transcription...")
        silence = np.zeros(16000, dtype=np.float32)
        list(self._model.transcribe(silence))
        print("  ASR warmup complete")

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        self.load_model()

        # Handle unsupported languages
        whisper_lang = language
        if language and language != "unknown" and language not in self._supported_languages:
            whisper_lang = self._language_fallbacks.get(language)
            if whisper_lang is None:
                print(f"  Language {language} not supported by Whisper, using multilingual")

        use_multilingual = whisper_lang is None or whisper_lang == "unknown"

        segments, info = self._model.transcribe(
            audio,
            language=whisper_lang if not use_multilingual else None,
            beam_size=1,
            patience=1.0,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.35,
                "min_silence_duration_ms": 300,
                "min_speech_duration_ms": 200,
                "speech_pad_ms": 250,
            },
            compression_ratio_threshold=1.8,
            no_speech_threshold=0.3,
            log_prob_threshold=-0.8,
            condition_on_previous_text=False,
            word_timestamps=True,
            hallucination_silence_threshold=1.5,
            multilingual=use_multilingual,
            language_detection_threshold=0.5,
            initial_prompt=initial_prompt,
        )

        result_segments = []
        for seg in segments:
            text = seg.text.strip()
            if len(text) < 2:
                continue
            result_segments.append(
                ASRSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    language=language if language and language != "unknown" else info.language,
                )
            )

        return ASRResult(
            segments=result_segments,
            detected_language=info.language,
            language_probability=info.language_probability,
        )

    def post_process(self, segments: list[ASRSegment]) -> list[ASRSegment]:
        """Filter hallucinations and deloop repeated text."""
        result = []
        for seg in segments:
            text = seg.text
            duration = seg.end - seg.start

            # Apply delooping
            delooped = self._deloop_text(text)
            if delooped != text:
                print(f"  DELOOP: '{text[:40]}...' -> '{delooped[:40]}...'")
                text = delooped

            # Check hallucinations
            is_hal, reason = self._is_hallucination(text, duration)
            if is_hal:
                print(f"  SKIP hallucination ({reason}): {text[:50]}")
                continue

            if text != seg.text:
                seg = ASRSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    language=seg.language,
                    confidence=seg.confidence,
                )
            result.append(seg)
        return result

    def supports_language(self, lang_code: str) -> bool:
        return lang_code in self._supported_languages

    def get_language_fallback(self, lang_code: str) -> str | None:
        if lang_code in self._supported_languages:
            return lang_code
        return self._language_fallbacks.get(lang_code)

    def get_bilingual_prompt(self, foreign_lang: str) -> str | None:
        return self._bilingual_prompts.get(foreign_lang)

    @staticmethod
    def _deloop_text(
        text: str,
        min_ngram: int = 2,
        max_ngram: int = 6,
        min_repeats: int = 3,
    ) -> str:
        """Remove repeated n-gram patterns from text."""
        if not text or not text.strip():
            return text

        words = text.split()
        if len(words) < min_ngram * min_repeats:
            return text

        result_words = words.copy()
        changed = True

        while changed:
            changed = False
            for n in range(max_ngram, min_ngram - 1, -1):
                i = 0
                new_words: list[str] = []
                while i < len(result_words):
                    if i + n * min_repeats <= len(result_words):
                        ngram = tuple(result_words[i : i + n])
                        repeat_count = 1

                        j = i + n
                        while j + n <= len(result_words):
                            next_ngram = tuple(result_words[j : j + n])
                            if next_ngram == ngram:
                                repeat_count += 1
                                j += n
                            else:
                                break

                        if repeat_count >= min_repeats:
                            new_words.extend(result_words[i : i + n])
                            i = j
                            changed = True
                            continue

                    new_words.append(result_words[i])
                    i += 1

                result_words = new_words

        return " ".join(result_words)

    @staticmethod
    def _is_hallucination(text: str, duration: float) -> tuple[bool, str]:
        """Check if text is likely a hallucination."""
        if not text or not text.strip():
            return True, "empty"

        text_clean = text.strip()
        text_lower = text_clean.lower()

        if text_lower in WhisperASRBackend._hallucination_phrases:
            return True, "boh_exact"

        text_no_punct = text_lower.rstrip(".,!?…")
        if text_no_punct in WhisperASRBackend._hallucination_phrases:
            return True, "boh_stripped"

        words = text_clean.split()
        if len(words) > 1 and len({w.lower() for w in words}) == 1:
            return True, "single_word_repeat"

        if duration > 10.0:
            return True, "too_long"

        if duration > 3.0 and len(text_clean) < 10:
            return True, "short_text_long_duration"

        word_count = len(words)
        if duration > 2.0 and word_count / duration < 0.5:
            return True, "low_word_rate"

        return False, ""

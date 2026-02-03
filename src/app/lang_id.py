"""
Language identification module using SpeechBrain ECAPA-TDNN.

Provides per-speaker language detection that runs once per new speaker,
caching the result for subsequent segments from the same speaker.
"""

import os

import numpy as np
import torch
import torchaudio

# Monkey-patch for torchaudio 2.x which removed list_audio_backends
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

LANG_ID_MODEL = os.getenv("LANG_ID_MODEL", "speechbrain/lang-id-voxlingua107-ecapa")
LANG_ID_DEVICE = os.getenv("LANG_ID_DEVICE", "cuda")
LANG_ID_ENABLED = os.getenv("LANG_ID_ENABLED", "true").lower() == "true"

# Lazy-loaded model
_lang_model = None


def get_lang_model():
    """Get the SpeechBrain language identification model (lazy-loaded singleton)."""
    global _lang_model
    if _lang_model is None:
        from speechbrain.inference.classifiers import EncoderClassifier

        _lang_model = EncoderClassifier.from_hparams(
            source=LANG_ID_MODEL,
            savedir=os.path.join(os.getenv("HF_HOME", "/data/hf"), "speechbrain"),
            run_opts={"device": LANG_ID_DEVICE if torch.cuda.is_available() else "cpu"},
        )
    return _lang_model


def detect_language_from_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> tuple[str, float]:
    """
    Detect language from audio using SpeechBrain.

    Args:
        audio: Audio samples as float32 array (normalized to [-1, 1])
        sample_rate: Sample rate of the audio (should be 16000)

    Returns:
        Tuple of (language_code, confidence)
        Language code is ISO 639-1 (e.g., "en", "de", "fr")
    """
    if not LANG_ID_ENABLED:
        return "unknown", 0.0

    if len(audio) < sample_rate * 0.5:  # Need at least 0.5s
        return "unknown", 0.0

    try:
        model = get_lang_model()

        # Convert to torch tensor
        signal = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

        # Run classification
        prediction = model.classify_batch(signal)

        # prediction = (scores, softmax_probs, index, label)
        # label format is "en: English" or similar
        label = prediction[3][0]
        lang_code = label.split(":")[0].strip()

        # Get confidence (max probability)
        confidence = float(prediction[1].exp().max())

        # Fix known ISO code issues in the model
        if lang_code == "iw":
            lang_code = "he"  # Hebrew
        elif lang_code == "jw":
            lang_code = "jv"  # Javanese

        return lang_code, confidence

    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown", 0.0


class SpeakerLanguageTracker:
    """
    Tracks language assignments per speaker.

    Performs language detection once per new speaker and caches the result.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.speaker_languages: dict[str, str] = {}  # speaker_id -> language
        self.speaker_confidences: dict[str, float] = {}  # speaker_id -> confidence
        self.confidence_threshold = confidence_threshold

    def get_speaker_language(
        self,
        speaker_id: str,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """
        Get the language for a speaker, detecting it if not already known.

        Args:
            speaker_id: The speaker identifier
            audio: Audio samples for detection (only used if language not cached)
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (language_code, confidence)
            Language code is "en", "de", etc., or "unknown"
        """
        # Return cached language if available
        if speaker_id in self.speaker_languages:
            return self.speaker_languages[speaker_id], self.speaker_confidences.get(speaker_id, 1.0)

        # Detect language from audio if provided
        if audio is not None and len(audio) > 0:
            lang, confidence = detect_language_from_audio(audio, sample_rate)

            if confidence >= self.confidence_threshold:
                self.speaker_languages[speaker_id] = lang
                self.speaker_confidences[speaker_id] = confidence
                print(f"Detected language for {speaker_id}: {lang} ({confidence:.2f})")
                return lang, confidence

        return "unknown", 0.0

    def set_speaker_language(self, speaker_id: str, language: str, confidence: float = 1.0):
        """Manually set a speaker's language."""
        self.speaker_languages[speaker_id] = language
        self.speaker_confidences[speaker_id] = confidence

    def get_all_speakers(self) -> dict[str, str]:
        """Get all speaker -> language mappings."""
        return self.speaker_languages.copy()

    def clear_cache(self) -> None:
        """Clear all cached language detections.

        Useful for session boundaries or when starting a new dialogue context.
        """
        self.speaker_languages.clear()
        self.speaker_confidences.clear()
        print("  Language tracker cache cleared")


def warmup_lang_id():
    """Warm up language ID model to reduce first-inference latency."""
    if not LANG_ID_ENABLED:
        return

    print("Warming up language ID model...")
    try:
        _ = get_lang_model()
        print("Language ID model loaded")
    except Exception as e:
        print(f"Language ID warmup failed: {e}")

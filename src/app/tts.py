"""TTS module using KugelAudio with 4-bit quantization."""

import logging
import os
import struct
import time
from collections import deque

import numpy as np
import torch

logger = logging.getLogger(__name__)

TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "kugelaudio/kugelaudio-0-open")
TTS_CFG_SCALE = float(os.getenv("TTS_CFG_SCALE", "3.0"))
TTS_SAMPLE_RATE = 24000  # KugelAudio output sample rate

TTS_SUPPORTED_LANGS = {
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pl",
    "ro",
    "hr",
    "bg",
    "tr",
    "ru",
    "uk",
    "hu",
    "sr",
    "pt",
    "nl",
}

_tts_model = None
_tts_processor = None

# Metrics
_tts_metrics = {
    "tts_times": deque(maxlen=100),
}


def get_tts_metrics() -> dict:
    """Get TTS performance metrics."""
    tts_times = list(_tts_metrics["tts_times"])
    return {
        "avg_tts_time_ms": sum(tts_times) / len(tts_times) * 1000 if tts_times else 0,
        "tts_sample_count": len(tts_times),
    }


def get_tts_model():
    """Lazy-load TTS model with 4-bit quantization."""
    global _tts_model, _tts_processor

    if _tts_model is None:
        logger.info("Loading TTS model with 4-bit quantization...")

        from kugelaudio_open import (
            KugelAudioForConditionalGenerationInference,
            KugelAudioProcessor,
        )
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        _tts_model = KugelAudioForConditionalGenerationInference.from_pretrained(
            TTS_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
        )
        _tts_model.eval()
        _tts_processor = KugelAudioProcessor.from_pretrained(TTS_MODEL_ID)
        logger.info("TTS model loaded")

    return _tts_model, _tts_processor


def _make_wav(pcm16_bytes: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono audio in a WAV container."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm16_bytes)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm16_bytes


@torch.compiler.disable()
def synthesize_speech(text: str, lang: str = "en") -> bytes:
    """
    Generate speech audio from text.

    Args:
        text: Text to synthesize
        lang: Target language code passed to KugelAudio

    Returns:
        PCM16 24kHz mono audio bytes
    """
    tts_start = time.time()

    model, processor = get_tts_model()

    # Prepare inputs with language hint
    inputs = processor(text=text, language=lang, return_tensors="pt")
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # Generate audio
    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=TTS_CFG_SCALE)

    # Convert float32 audio to PCM16 bytes
    # KugelAudio outputs audio in speech_outputs attribute
    audio = outputs.speech_outputs[0].cpu().numpy()

    # Normalize and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_pcm16 = (audio * 32767).astype(np.int16).tobytes()

    tts_time = time.time() - tts_start
    _tts_metrics["tts_times"].append(tts_time)

    logger.debug(
        "TTS synthesized %d chars in %.1fms, %d bytes", len(text), tts_time * 1000, len(audio_pcm16)
    )

    return audio_pcm16


def synthesize_wav(text: str, lang: str = "en") -> bytes:
    """Generate speech as a WAV file (PCM16 24kHz mono)."""
    pcm16 = synthesize_speech(text, lang)
    return _make_wav(pcm16)

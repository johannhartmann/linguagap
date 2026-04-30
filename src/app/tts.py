"""TTS module using Piper (rhasspy/piper-voices).

Replaces the prior KugelAudio backend. Piper is a small VITS-based ONNX
runtime that synthesizes ~50–300x realtime per voice on CPU and even
faster with onnxruntime-gpu, which keeps end-to-end TTS latency well
below the translation delivery latency.
"""

import logging
import os
import struct
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

PIPER_DATA_DIR = Path(os.getenv("PIPER_DATA_DIR", "/data/piper"))
PIPER_HF_REPO = "rhasspy/piper-voices"
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "0") not in ("0", "false", "False", "")

# Map our BCP-47 short codes to a Piper voice. Format: voice_id, hf_subpath.
# voice_id is the file basename (without .onnx) and hf_subpath is the
# directory inside rhasspy/piper-voices that contains <voice_id>.onnx
# and <voice_id>.onnx.json.
PIPER_VOICES: dict[str, tuple[str, str]] = {
    "en": ("en_US-lessac-medium", "en/en_US/lessac/medium"),
    "de": ("de_DE-thorsten-medium", "de/de_DE/thorsten/medium"),
    "fr": ("fr_FR-siwis-medium", "fr/fr_FR/siwis/medium"),
    "es": ("es_ES-davefx-medium", "es/es_ES/davefx/medium"),
    "it": ("it_IT-paola-medium", "it/it_IT/paola/medium"),
    "pl": ("pl_PL-darkman-medium", "pl/pl_PL/darkman/medium"),
    "ro": ("ro_RO-mihai-medium", "ro/ro_RO/mihai/medium"),
    "bg": ("bg_BG-dimitar-medium", "bg/bg_BG/dimitar/medium"),
    "tr": ("tr_TR-dfki-medium", "tr/tr_TR/dfki/medium"),
    "ru": ("ru_RU-ruslan-medium", "ru/ru_RU/ruslan/medium"),
    "uk": ("uk_UA-ukrainian_tts-medium", "uk/uk_UA/ukrainian_tts/medium"),
    "hu": ("hu_HU-anna-medium", "hu/hu_HU/anna/medium"),
    "sr": ("sr_RS-serbski_institut-medium", "sr/sr_RS/serbski_institut/medium"),
    "pt": ("pt_BR-faber-medium", "pt/pt_BR/faber/medium"),
    "nl": ("nl_NL-mls-medium", "nl/nl_NL/mls/medium"),
    # hr: no Croatian voice in rhasspy/piper-voices.
}

TTS_SUPPORTED_LANGS: set[str] = set(PIPER_VOICES.keys())

_voices: dict[str, object] = {}
_metrics: dict[str, deque[float]] = {"tts_times": deque(maxlen=100)}


def get_tts_metrics() -> dict:
    times = list(_metrics["tts_times"])
    return {
        "avg_tts_time_ms": sum(times) / len(times) * 1000 if times else 0,
        "tts_sample_count": len(times),
    }


def _download_voice(lang: str) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    voice_id, hf_subpath = PIPER_VOICES[lang]
    PIPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    onnx = Path(
        hf_hub_download(
            repo_id=PIPER_HF_REPO,
            filename=f"{hf_subpath}/{voice_id}.onnx",
            cache_dir=str(PIPER_DATA_DIR),
        )
    )
    cfg = Path(
        hf_hub_download(
            repo_id=PIPER_HF_REPO,
            filename=f"{hf_subpath}/{voice_id}.onnx.json",
            cache_dir=str(PIPER_DATA_DIR),
        )
    )
    return onnx, cfg


def _load_voice(lang: str):
    from piper.voice import PiperVoice

    onnx_path, cfg_path = _download_voice(lang)
    logger.info("Loading Piper voice %s (cuda=%s)", PIPER_VOICES[lang][0], PIPER_USE_CUDA)
    return PiperVoice.load(str(onnx_path), config_path=str(cfg_path), use_cuda=PIPER_USE_CUDA)


def get_voice(lang: str):
    if lang not in PIPER_VOICES:
        raise ValueError(f"Language {lang!r} not supported by Piper TTS")
    cached = _voices.get(lang)
    if cached is not None:
        return cached
    voice = _load_voice(lang)
    _voices[lang] = voice
    return voice


def get_tts_model():
    """Eagerly load every configured voice. Called at warmup so the first
    synthesis request after startup does not pay the load cost."""
    for lang in PIPER_VOICES:
        get_voice(lang)
    return None, None


def _make_wav(pcm16_bytes: bytes, sample_rate: int) -> bytes:
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
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm16_bytes


def _synthesize_pcm16(text: str, lang: str) -> tuple[bytes, int]:
    """Run Piper synthesis and return concatenated PCM16 bytes + sample rate.

    Piper's streaming synthesize() emits one or more AudioChunk objects, each
    carrying its own (consistent) sample_rate. We collect raw int16 bytes and
    take the rate from the first chunk.
    """
    if lang not in PIPER_VOICES:
        raise ValueError(f"Language {lang!r} not supported by Piper TTS")
    t0 = time.time()
    voice = get_voice(lang)
    sample_rate: int | None = None
    chunks: list[bytes] = []
    for chunk in voice.synthesize(text):
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        chunks.append(chunk.audio_int16_bytes)
    pcm16 = b"".join(chunks)
    if sample_rate is None:
        sample_rate = 22050  # Empty input fallback; should not happen.
    dt = time.time() - t0
    _metrics["tts_times"].append(dt)
    logger.debug(
        "Piper %s: %d chars -> %d bytes in %.0fms (sr=%d)",
        lang,
        len(text),
        len(pcm16),
        dt * 1000,
        sample_rate,
    )
    return pcm16, sample_rate


def synthesize_speech(text: str, lang: str = "en") -> bytes:
    """Synthesize text to raw PCM16 mono bytes at the voice's sample rate."""
    pcm16, _sr = _synthesize_pcm16(text, lang)
    return pcm16


def synthesize_wav(text: str, lang: str = "en") -> bytes:
    """Synthesize text and wrap the PCM16 output in a WAV container."""
    pcm16, sample_rate = _synthesize_pcm16(text, lang)
    return _make_wav(pcm16, sample_rate)

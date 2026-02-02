"""Audio file caching for TTS synthesis.

Uses SHA256 hash of text + voice + language to avoid repeated API calls.
"""

import hashlib
import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the audio cache directory.

    Returns:
        Path to cache directory
    """
    cache_dir = os.getenv("E2E_AUDIO_CACHE_DIR", "./tests/fixtures/e2e_audio")
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_cache_key(
    dialogue_yaml: str,
    voices: dict[str, str],
) -> str:
    """Compute cache key for a dialogue.

    Args:
        dialogue_yaml: YAML string of the dialogue scenario
        voices: Dict mapping speaker IDs to voice names

    Returns:
        SHA256 hash string
    """
    # Create a stable string representation
    voices_str = ",".join(f"{k}={v}" for k, v in sorted(voices.items()))
    content = f"{dialogue_yaml}|{voices_str}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_cached_audio(cache_key: str) -> Path | None:
    """Get cached audio file if it exists.

    Args:
        cache_key: Cache key from compute_cache_key

    Returns:
        Path to cached WAV file or None if not cached
    """
    cache_dir = get_cache_dir()
    cache_path = cache_dir / f"{cache_key}.wav"
    if cache_path.exists():
        return cache_path
    return None


def save_to_cache(cache_key: str, audio_data: bytes) -> Path:
    """Save audio data to cache.

    Args:
        cache_key: Cache key from compute_cache_key
        audio_data: WAV audio bytes

    Returns:
        Path to saved file
    """
    cache_dir = get_cache_dir()
    cache_path = cache_dir / f"{cache_key}.wav"
    cache_path.write_bytes(audio_data)
    return cache_path

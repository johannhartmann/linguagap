"""Gemini TTS client for generating dialogue audio.

Uses Google Cloud Text-to-Speech API with Gemini TTS model.
Per-turn synthesis with explicit language codes for bilingual dialogues.

Requires: GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to
a service account JSON file with Cloud Text-to-Speech permissions.

Reference: https://docs.cloud.google.com/text-to-speech/docs/gemini-tts
"""

import io
import os
import struct
import wave
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech

from tests.e2e.dialogues.templates import DialogueScenario, DialogueTurn
from tests.e2e.tts.cache import compute_cache_key, get_cached_audio, save_to_cache
from tests.e2e.tts.voices import get_language_code, get_voice_for_speaker

# Gemini TTS model name
TTS_MODEL = "gemini-2.5-pro-tts"

# Sample rates
CLOUD_TTS_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 16000

# Default pause between turns (ms)
DEFAULT_TURN_GAP_MS = 350

# Style prompt for consistent voice across all turns
DEFAULT_STYLE_PROMPT = "Sprich natürlich, klar und freundlich. " "Nutze kurze, natürliche Pausen."


class GeminiTTSClient:
    """Client for synthesizing dialogue audio using Google Cloud TTS with Gemini model.

    Requires GOOGLE_APPLICATION_CREDENTIALS to be set.
    """

    def __init__(self, region: str | None = None):
        """Initialize the TTS client.

        Args:
            region: Cloud TTS region (e.g., "eu", "us", "global"). Default from env.

        Raises:
            ValueError: If GOOGLE_APPLICATION_CREDENTIALS is not set.
        """
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable must be set. "
                "See: https://cloud.google.com/docs/authentication/application-default-credentials"
            )

        self.region = region or os.getenv("GOOGLE_CLOUD_REGION", "eu")
        self.client = self._make_client(self.region)

    def _make_client(self, region: str) -> texttospeech.TextToSpeechClient:
        """Create a Cloud TTS client for the specified region."""
        api_endpoint = (
            f"{region}-texttospeech.googleapis.com"
            if region and region != "global"
            else "texttospeech.googleapis.com"
        )
        return texttospeech.TextToSpeechClient(
            client_options=ClientOptions(api_endpoint=api_endpoint)
        )

    def synthesize_dialogue(
        self,
        scenario: DialogueScenario,
        use_cache: bool = True,
        style_prompt: str | None = None,
    ) -> Path:
        """Synthesize audio for an entire dialogue using per-turn TTS.

        Args:
            scenario: DialogueScenario containing all turns
            use_cache: Whether to use cached audio if available
            style_prompt: Optional style prompt for voice consistency

        Returns:
            Path to the generated WAV file (16kHz mono)
        """
        voices = {speaker_id: get_voice_for_speaker(speaker_id) for speaker_id in scenario.speakers}

        pause_ms = DEFAULT_TURN_GAP_MS
        synthesis_method = f"cloud_tts_{pause_ms}ms_v3"
        cache_key = compute_cache_key(scenario.to_yaml(), voices, synthesis_method)

        if use_cache:
            cached = get_cached_audio(cache_key)
            if cached:
                print(f"Using cached audio: {cached}")
                return cached

        print(f"  Synthesizing dialogue with {len(scenario.turns)} turns (pause={pause_ms}ms)...")

        prompt = style_prompt or DEFAULT_STYLE_PROMPT

        wav_parts = []
        for i, turn in enumerate(scenario.turns):
            lang_code = get_language_code(turn.language)
            print(f"    Turn {i + 1}/{len(scenario.turns)}: {lang_code} - {turn.text[:30]}...")
            turn_audio = self.synthesize_turn(turn, style_prompt=prompt)
            wav_parts.append(turn_audio)

        pause_sec = pause_ms / 1000.0
        audio = self._concatenate_with_silence(wav_parts, pause_sec)

        cache_path = save_to_cache(cache_key, audio)
        print(f"Saved audio to cache: {cache_path}")
        return cache_path

    def synthesize_turn(
        self,
        turn: DialogueTurn,
        style_prompt: str | None = None,
    ) -> bytes:
        """Synthesize audio for a single dialogue turn.

        Args:
            turn: DialogueTurn to synthesize
            style_prompt: Optional style/direction prompt

        Returns:
            WAV audio bytes (16kHz mono)
        """
        voice_name = get_voice_for_speaker(turn.speaker_id)
        language_code = get_language_code(turn.language)

        synthesis_input = texttospeech.SynthesisInput(
            text=turn.text,
            prompt=style_prompt or DEFAULT_STYLE_PROMPT,
        )

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            model_name=TTS_MODEL,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=CLOUD_TTS_SAMPLE_RATE,
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        return self._audio_to_wav(response.audio_content)

    def _audio_to_wav(self, audio_content: bytes) -> bytes:
        """Convert Cloud TTS audio response to 16kHz mono WAV."""
        if (
            len(audio_content) > 12
            and audio_content[:4] == b"RIFF"
            and audio_content[8:12] == b"WAVE"
        ):
            with wave.open(io.BytesIO(audio_content), "rb") as wf:
                src_rate = wf.getframerate()
                pcm_data = wf.readframes(wf.getnframes())
        else:
            src_rate = CLOUD_TTS_SAMPLE_RATE
            pcm_data = audio_content

        if src_rate != TARGET_SAMPLE_RATE:
            pcm_data = self._resample_pcm(pcm_data, src_rate, TARGET_SAMPLE_RATE)

        return self._pcm_to_wav(pcm_data, TARGET_SAMPLE_RATE)

    def _concatenate_with_silence(
        self,
        wav_parts: list[bytes],
        silence_sec: float,
    ) -> bytes:
        """Concatenate WAV audio parts with silence gaps between turns."""
        silence_samples = int(silence_sec * TARGET_SAMPLE_RATE)
        silence_pcm = b"\x00\x00" * silence_samples

        all_pcm = []
        for i, wav_bytes in enumerate(wav_parts):
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                pcm = wav.readframes(wav.getnframes())
                wav_rate = wav.getframerate()

            if wav_rate != TARGET_SAMPLE_RATE:
                pcm = self._resample_pcm(pcm, wav_rate, TARGET_SAMPLE_RATE)

            all_pcm.append(pcm)

            if i < len(wav_parts) - 1:
                all_pcm.append(silence_pcm)

        return self._pcm_to_wav(b"".join(all_pcm), TARGET_SAMPLE_RATE)

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
        return buffer.getvalue()

    def _resample_pcm(self, pcm_data: bytes, src_rate: int, tgt_rate: int) -> bytes:
        """Simple linear resampling of PCM data."""
        if src_rate == tgt_rate:
            return pcm_data

        num_samples = len(pcm_data) // 2
        samples = list(struct.unpack(f"<{num_samples}h", pcm_data))

        ratio = tgt_rate / src_rate
        new_len = int(num_samples * ratio)

        resampled = []
        for i in range(new_len):
            src_idx = i / ratio
            idx_low = int(src_idx)
            idx_high = min(idx_low + 1, num_samples - 1)
            frac = src_idx - idx_low

            sample = int(samples[idx_low] * (1 - frac) + samples[idx_high] * frac)
            resampled.append(max(-32768, min(32767, sample)))

        return struct.pack(f"<{len(resampled)}h", *resampled)

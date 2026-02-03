"""Gemini TTS client for generating dialogue audio.

Uses gemini-2.5-flash-preview-tts with multi-speaker support.
"""

import io
import os
import struct
import wave
from pathlib import Path

from tests.e2e.dialogues.templates import DialogueScenario, DialogueTurn
from tests.e2e.tts.cache import compute_cache_key, get_cached_audio, save_to_cache
from tests.e2e.tts.voices import get_voice_for_speaker

# Gemini TTS model
TTS_MODEL = "gemini-2.5-flash-preview-tts"


class GeminiTTSClient:
    """Client for synthesizing dialogue audio using Gemini TTS."""

    def __init__(self, api_key: str | None = None):
        """Initialize the TTS client.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Import here to avoid import errors when running without the package
        from google import genai

        self.client = genai.Client(api_key=self.api_key)
        self._genai = genai

    def synthesize_dialogue(
        self,
        scenario: DialogueScenario,
        use_cache: bool = True,
        inter_turn_silence_sec: float = 0.7,
    ) -> Path:
        """Synthesize audio for an entire dialogue with natural breaks.

        Uses per-turn synthesis with silence gaps between turns to ensure
        proper VAD/diarization separation.

        Args:
            scenario: DialogueScenario containing all turns
            use_cache: Whether to use cached audio if available
            inter_turn_silence_sec: Silence duration between turns (default 0.7s)
                - > 0.3s: VAD detects speech boundary
                - > 0.5s: Pre-ASR won't merge same-speaker segments
                - 0.5-0.8s: Natural conversational pause range

        Returns:
            Path to the generated WAV file (16kHz mono)
        """
        # Build voices dict
        voices = {speaker_id: get_voice_for_speaker(speaker_id) for speaker_id in scenario.speakers}

        # Include synthesis method in cache key for differentiation
        synthesis_method = f"per_turn_{inter_turn_silence_sec}s"
        cache_key = compute_cache_key(scenario.to_yaml(), voices, synthesis_method)

        # Check cache
        if use_cache:
            cached = get_cached_audio(cache_key)
            if cached:
                print(f"Using cached audio: {cached}")
                return cached

        # Synthesize each turn separately
        audio_parts = []
        for i, turn in enumerate(scenario.turns):
            print(f"  Synthesizing turn {i + 1}/{len(scenario.turns)}: {turn.text[:40]}...")
            turn_audio = self.synthesize_turn(turn)
            audio_parts.append(turn_audio)

        # Concatenate with natural silence gaps
        combined = self._concatenate_with_silence(audio_parts, inter_turn_silence_sec)

        # Save to cache
        cache_path = save_to_cache(cache_key, combined)
        print(f"Saved audio to cache: {cache_path}")
        return cache_path

    def _concatenate_with_silence(
        self,
        wav_parts: list[bytes],
        silence_sec: float,
        sample_rate: int = 16000,
    ) -> bytes:
        """Concatenate WAV audio parts with silence gaps between turns.

        Args:
            wav_parts: List of WAV audio bytes for each turn
            silence_sec: Silence duration between turns in seconds
            sample_rate: Target sample rate (default 16kHz)

        Returns:
            Combined WAV audio bytes
        """
        # Generate silence (16-bit PCM, mono)
        silence_samples = int(silence_sec * sample_rate)
        silence_pcm = b"\x00\x00" * silence_samples

        # Extract PCM from each WAV and concatenate
        all_pcm = []
        for i, wav_bytes in enumerate(wav_parts):
            buffer = io.BytesIO(wav_bytes)
            with wave.open(buffer, "rb") as wav:
                pcm = wav.readframes(wav.getnframes())
                wav_rate = wav.getframerate()

            # Resample if needed
            if wav_rate != sample_rate:
                pcm = self._resample_pcm(pcm, wav_rate, sample_rate)

            all_pcm.append(pcm)

            # Add silence after each turn (except the last)
            if i < len(wav_parts) - 1:
                all_pcm.append(silence_pcm)

        # Create output WAV
        combined_pcm = b"".join(all_pcm)
        out_buffer = io.BytesIO()
        with wave.open(out_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(combined_pcm)

        return out_buffer.getvalue()

    def synthesize_turn(
        self,
        turn: DialogueTurn,
    ) -> bytes:
        """Synthesize audio for a single dialogue turn.

        Args:
            turn: DialogueTurn to synthesize

        Returns:
            WAV audio bytes (16kHz mono)
        """
        from google.genai import types

        voice_name = get_voice_for_speaker(turn.speaker_id)

        response = self.client.models.generate_content(
            model=TTS_MODEL,
            contents=turn.text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
            ),
        )

        # Extract audio data from response
        return self._extract_audio(response)

    def _generate_audio(
        self,
        prompt: str,
        voices: dict[str, str],
        speakers: dict[str, str],
    ) -> bytes:
        """Generate audio using multi-speaker TTS.

        Args:
            prompt: Multi-speaker formatted prompt
            voices: Dict mapping speaker IDs to voice names
            speakers: Dict mapping speaker IDs to display names

        Returns:
            WAV audio bytes (16kHz mono)
        """
        from google.genai import types

        # Build multi-speaker voice config
        speaker_voice_configs = []
        for speaker_id, voice_name in voices.items():
            display_name = speakers.get(speaker_id, speaker_id)
            speaker_voice_configs.append(
                types.SpeakerVoiceConfig(
                    speaker=display_name,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    ),
                )
            )

        response = self.client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=speaker_voice_configs
                    )
                ),
            ),
        )

        return self._extract_audio(response)

    def _extract_audio(self, response) -> bytes:
        """Extract audio data from Gemini response.

        Args:
            response: GenerateContentResponse from Gemini

        Returns:
            WAV audio bytes (16kHz mono PCM16)
        """
        # Get inline data from response
        audio_part = response.candidates[0].content.parts[0]

        if hasattr(audio_part, "inline_data") and audio_part.inline_data:
            audio_bytes = audio_part.inline_data.data
            mime_type = audio_part.inline_data.mime_type

            # Gemini returns raw PCM, convert to WAV
            if "audio/pcm" in mime_type or "audio/L16" in mime_type:
                return self._pcm_to_wav(audio_bytes, sample_rate=24000)
            elif "audio/wav" in mime_type:
                # Already WAV, resample to 16kHz if needed
                return self._resample_wav(audio_bytes, target_rate=16000)
            else:
                # Try to handle as raw PCM
                return self._pcm_to_wav(audio_bytes, sample_rate=24000)

        raise ValueError("No audio data in response")

    def _pcm_to_wav(
        self,
        pcm_data: bytes,
        sample_rate: int = 24000,
        target_rate: int = 16000,
    ) -> bytes:
        """Convert raw PCM to WAV format.

        Args:
            pcm_data: Raw PCM bytes (16-bit signed)
            sample_rate: Input sample rate
            target_rate: Output sample rate

        Returns:
            WAV file bytes
        """
        # Resample if needed
        if sample_rate != target_rate:
            pcm_data = self._resample_pcm(pcm_data, sample_rate, target_rate)
            sample_rate = target_rate

        # Create WAV file
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)

        return buffer.getvalue()

    def _resample_pcm(
        self,
        pcm_data: bytes,
        src_rate: int,
        tgt_rate: int,
    ) -> bytes:
        """Simple linear resampling of PCM data.

        Args:
            pcm_data: Input PCM bytes (16-bit signed)
            src_rate: Source sample rate
            tgt_rate: Target sample rate

        Returns:
            Resampled PCM bytes
        """
        # Parse samples
        num_samples = len(pcm_data) // 2
        samples = list(struct.unpack(f"<{num_samples}h", pcm_data))

        # Calculate new length
        ratio = tgt_rate / src_rate
        new_len = int(num_samples * ratio)

        # Linear interpolation
        resampled = []
        for i in range(new_len):
            src_idx = i / ratio
            idx_low = int(src_idx)
            idx_high = min(idx_low + 1, num_samples - 1)
            frac = src_idx - idx_low

            sample = int(samples[idx_low] * (1 - frac) + samples[idx_high] * frac)
            resampled.append(max(-32768, min(32767, sample)))

        return struct.pack(f"<{len(resampled)}h", *resampled)

    def _resample_wav(self, wav_data: bytes, target_rate: int) -> bytes:
        """Resample WAV file to target sample rate.

        Args:
            wav_data: Input WAV bytes
            target_rate: Target sample rate

        Returns:
            Resampled WAV bytes
        """
        # Parse input WAV
        buffer = io.BytesIO(wav_data)
        with wave.open(buffer, "rb") as wav:
            src_rate = wav.getframerate()
            if src_rate == target_rate:
                return wav_data

            pcm_data = wav.readframes(wav.getnframes())

        # Resample
        resampled = self._resample_pcm(pcm_data, src_rate, target_rate)

        # Write new WAV
        out_buffer = io.BytesIO()
        with wave.open(out_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(target_rate)
            wav.writeframes(resampled)

        return out_buffer.getvalue()

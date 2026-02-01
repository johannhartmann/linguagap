"""
Speaker diarization module using pyannote community pipeline.

Uses pyannote/speaker-diarization-community-1 for high-quality
speaker diarization in real-time streaming scenarios.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
DIARIZATION_DEVICE = os.getenv("DIARIZATION_DEVICE", "cuda")
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "true").lower() == "true"
# For bilingual conversations, we expect 2 speakers
DIARIZATION_NUM_SPEAKERS = int(os.getenv("DIARIZATION_NUM_SPEAKERS", "2"))

# Lazy-loaded pipeline
_pipeline = None


@dataclass
class SpeakerSegment:
    """A segment with speaker information."""

    start: float
    end: float
    speaker_id: str


def get_pipeline():
    """Get the pyannote diarization pipeline (lazy-loaded singleton)."""
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline

        print(f"Loading diarization pipeline: {DIARIZATION_MODEL}")
        _pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=os.getenv("HF_TOKEN"),
        )
        if DIARIZATION_DEVICE == "cuda" and torch.cuda.is_available():
            _pipeline = _pipeline.to(torch.device(DIARIZATION_DEVICE))
            print(f"Diarization pipeline moved to {DIARIZATION_DEVICE}")
        print("Diarization pipeline loaded")
    return _pipeline


class StreamingDiarizer:
    """
    Speaker diarization using pyannote community pipeline.

    Processes audio chunks and returns speaker segments.
    For bilingual conversations, uses num_speakers=2 by default.
    """

    def __init__(self, sample_rate: int = 16000, num_speakers: int | None = None):
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers or DIARIZATION_NUM_SPEAKERS
        # Track speaker mapping across windows for consistency
        self.speaker_mapping: dict[str, str] = {}
        self.next_speaker_idx = 0

    def _normalize_speaker_id(self, raw_speaker: str) -> str:
        """Map pipeline speaker IDs to consistent SPEAKER_XX format."""
        if raw_speaker not in self.speaker_mapping:
            self.speaker_mapping[raw_speaker] = f"SPEAKER_{self.next_speaker_idx:02d}"
            self.next_speaker_idx += 1
        return self.speaker_mapping[raw_speaker]

    def process_audio(
        self,
        audio: np.ndarray,
        window_start: float = 0.0,
    ) -> list[SpeakerSegment]:
        """
        Process an audio chunk and return speaker segments.

        Args:
            audio: Audio samples as float32 array (normalized to [-1, 1])
            window_start: Absolute start time of this window in seconds

        Returns:
            List of SpeakerSegment with speaker assignments
        """
        if not DIARIZATION_ENABLED:
            return []

        if len(audio) < self.sample_rate * 0.5:  # Need at least 0.5s of audio
            return []

        try:
            pipeline = get_pipeline()

            # Convert to torch tensor for pipeline input
            # Pipeline expects (channels, samples) format
            waveform = torch.from_numpy(audio).unsqueeze(0).float()  # (1, samples)

            # Run diarization pipeline
            # Specify num_speakers for bilingual conversations
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": self.sample_rate},
                num_speakers=self.num_speakers,
            )

            # Convert pipeline output to SpeakerSegment list
            # Output has speaker_diarization attribute with (turn, speaker) tuples
            speaker_segments = []
            for turn, speaker in diarization.speaker_diarization:
                # Normalize speaker ID for consistency across windows
                normalized_speaker = self._normalize_speaker_id(speaker)
                speaker_segments.append(
                    SpeakerSegment(
                        start=window_start + turn.start,
                        end=window_start + turn.end,
                        speaker_id=normalized_speaker,
                    )
                )

            return speaker_segments

        except Exception as e:
            print(f"Diarization error: {e}")
            return []

    def get_speaker_at_time(
        self,
        segments: list[SpeakerSegment],
        time: float,
    ) -> str | None:
        """Find which speaker is active at a given time."""
        for seg in segments:
            if seg.start <= time <= seg.end:
                return seg.speaker_id
        return None

    def get_dominant_speaker(
        self,
        segments: list[SpeakerSegment],
        start: float,
        end: float,
    ) -> str | None:
        """Find the dominant speaker in a time range based on overlap duration."""
        speaker_durations: dict[str, float] = {}

        for seg in segments:
            # Calculate overlap
            overlap_start = max(seg.start, start)
            overlap_end = min(seg.end, end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                speaker_durations[seg.speaker_id] = (
                    speaker_durations.get(seg.speaker_id, 0) + overlap
                )

        if not speaker_durations:
            return None

        return max(speaker_durations, key=lambda k: speaker_durations[k])


def warmup_diarization():
    """Warm up diarization pipeline to reduce first-inference latency."""
    if not DIARIZATION_ENABLED:
        return

    print("Warming up diarization pipeline...")
    try:
        pipeline = get_pipeline()
        # Run a quick inference to fully initialize
        dummy_audio = torch.zeros(1, 16000)  # 1 second of silence
        _ = pipeline({"waveform": dummy_audio, "sample_rate": 16000}, num_speakers=2)
        print("Diarization pipeline warmed up")
    except Exception as e:
        print(f"Diarization warmup failed: {e}")

"""
Speaker diarization module using pyannote for real-time streaming.

Provides speaker segmentation that runs in parallel with ASR,
identifying which speaker is talking at each moment.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

DIARIZATION_DEVICE = os.getenv("DIARIZATION_DEVICE", "cuda")
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "true").lower() == "true"

# Lazy-loaded models
_segmentation_model = None
_embedding_model = None


@dataclass
class SpeakerSegment:
    """A segment with speaker information."""

    start: float
    end: float
    speaker_id: str


def get_segmentation_model():
    """Get the pyannote segmentation model (lazy-loaded singleton)."""
    global _segmentation_model
    if _segmentation_model is None:
        from pyannote.audio import Model

        _segmentation_model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            token=os.getenv("HF_TOKEN"),  # pyannote 4.x uses 'token' instead of 'use_auth_token'
        )
        if DIARIZATION_DEVICE == "cuda" and torch.cuda.is_available():
            _segmentation_model = _segmentation_model.to(DIARIZATION_DEVICE)
    return _segmentation_model


def get_embedding_model():
    """Get the pyannote embedding model (lazy-loaded singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from pyannote.audio import Model

        _embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            token=os.getenv("HF_TOKEN"),  # pyannote 4.x uses 'token' instead of 'use_auth_token'
        )
        if DIARIZATION_DEVICE == "cuda" and torch.cuda.is_available():
            _embedding_model = _embedding_model.to(DIARIZATION_DEVICE)
    return _embedding_model


class StreamingDiarizer:
    """
    Streaming speaker diarization using incremental clustering.

    Processes audio chunks and maintains speaker state across the session.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.speaker_embeddings: dict[str, np.ndarray] = {}  # speaker_id -> embedding
        self.next_speaker_id = 0
        self.embedding_threshold = 0.5  # Cosine similarity threshold for same speaker

    def _get_next_speaker_id(self) -> str:
        """Generate the next speaker ID."""
        speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
        self.next_speaker_id += 1
        return speaker_id

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _find_or_create_speaker(self, embedding: np.ndarray) -> str:
        """Find matching speaker or create new one based on embedding similarity."""
        best_match = None
        best_score = -1.0

        for speaker_id, stored_emb in self.speaker_embeddings.items():
            score = self._cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_match = speaker_id

        if best_match is not None and best_score > self.embedding_threshold:
            # Update embedding with exponential moving average
            alpha = 0.3
            self.speaker_embeddings[best_match] = (
                alpha * embedding + (1 - alpha) * self.speaker_embeddings[best_match]
            )
            return best_match

        # Create new speaker
        new_speaker = self._get_next_speaker_id()
        self.speaker_embeddings[new_speaker] = embedding
        return new_speaker

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
            segmentation_model = get_segmentation_model()
            embedding_model = get_embedding_model()

            # Convert to torch tensor
            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
            if DIARIZATION_DEVICE == "cuda" and torch.cuda.is_available():
                waveform = waveform.to(DIARIZATION_DEVICE)

            # Run segmentation to get speaker activity
            with torch.no_grad():
                segmentation = segmentation_model(waveform)  # (1, frames, speakers)

            # Get frame-level speaker probabilities
            seg_probs = segmentation.squeeze(0).cpu().numpy()  # (frames, speakers)
            num_frames = seg_probs.shape[0]
            frame_duration = len(audio) / self.sample_rate / num_frames

            # Find active speaker regions
            speaker_segments = []
            current_speaker = None
            segment_start = None

            for frame_idx in range(num_frames):
                frame_probs = seg_probs[frame_idx]
                # Get dominant speaker if any speaker is active
                if frame_probs.max() > 0.5:
                    # Extract embedding for this frame's audio region
                    frame_start_sample = int(frame_idx * frame_duration * self.sample_rate)
                    frame_end_sample = min(
                        int((frame_idx + 1) * frame_duration * self.sample_rate),
                        len(audio),
                    )

                    if frame_end_sample - frame_start_sample > self.sample_rate * 0.1:
                        frame_audio = audio[frame_start_sample:frame_end_sample]
                        frame_waveform = torch.from_numpy(frame_audio).unsqueeze(0)
                        if DIARIZATION_DEVICE == "cuda" and torch.cuda.is_available():
                            frame_waveform = frame_waveform.to(DIARIZATION_DEVICE)

                        with torch.no_grad():
                            embedding = embedding_model(frame_waveform)
                            embedding = embedding.squeeze().cpu().numpy()

                        speaker_id = self._find_or_create_speaker(embedding)

                        if speaker_id != current_speaker:
                            # Save previous segment
                            if current_speaker is not None and segment_start is not None:
                                speaker_segments.append(
                                    SpeakerSegment(
                                        start=window_start + segment_start,
                                        end=window_start + frame_idx * frame_duration,
                                        speaker_id=current_speaker,
                                    )
                                )
                            current_speaker = speaker_id
                            segment_start = frame_idx * frame_duration
                else:
                    # No active speaker
                    if current_speaker is not None and segment_start is not None:
                        speaker_segments.append(
                            SpeakerSegment(
                                start=window_start + segment_start,
                                end=window_start + frame_idx * frame_duration,
                                speaker_id=current_speaker,
                            )
                        )
                    current_speaker = None
                    segment_start = None

            # Close final segment
            if current_speaker is not None and segment_start is not None:
                speaker_segments.append(
                    SpeakerSegment(
                        start=window_start + segment_start,
                        end=window_start + len(audio) / self.sample_rate,
                        speaker_id=current_speaker,
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
    """Warm up diarization models to reduce first-inference latency."""
    if not DIARIZATION_ENABLED:
        return

    print("Warming up diarization models...")
    try:
        _ = get_segmentation_model()
        _ = get_embedding_model()
        print("Diarization models loaded")
    except Exception as e:
        print(f"Diarization warmup failed: {e}")

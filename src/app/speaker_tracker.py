"""
Speaker tracking using SpeechBrain embeddings.

Replaces pyannote diarization with embedding-based speaker identification.
This approach maintains consistent speaker IDs across processing ticks by
matching voice embeddings rather than relying on arbitrary diarization labels.

Key concepts:
    - Speaker Embeddings: 192-dim vectors representing voice characteristics
    - Cosine Similarity: Used to match new audio to known speakers
    - Energy-based VAD: Simple voice activity detection using RMS threshold
    - Consistent IDs: Speaker IDs are assigned based on embedding clusters,
      not arbitrary labels that can flip between ticks

Pipeline:
    1. Energy VAD detects speech regions in audio
    2. For each region, extract speaker embedding using ECAPA-TDNN
    3. Match embedding to known speakers (cosine similarity > threshold)
    4. If no match, register as new speaker
    5. Return speaker segments with consistent IDs
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch

SPEAKER_MODEL = os.getenv("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "cuda")
SPEAKER_ENABLED = os.getenv("SPEAKER_ENABLED", "true").lower() == "true"

# Similarity threshold for matching speakers
# ECAPA embeddings for same speaker typically have similarity > 0.75
# Using 0.5 to be more lenient and avoid over-segmentation
SPEAKER_SIMILARITY_THRESHOLD = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.5"))

# Maximum number of speakers expected (for bilingual conversations)
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "2"))

# VAD parameters
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.01"))  # RMS threshold
VAD_MIN_SPEECH_SEC = float(os.getenv("VAD_MIN_SPEECH_SEC", "0.3"))  # Min speech duration
VAD_MIN_SILENCE_SEC = float(os.getenv("VAD_MIN_SILENCE_SEC", "0.3"))  # Min silence to split

# Lazy-loaded model
_speaker_model = None


def get_speaker_model():
    """Get the SpeechBrain speaker embedding model (lazy-loaded singleton)."""
    global _speaker_model
    if _speaker_model is None:
        from speechbrain.inference import EncoderClassifier

        print(f"Loading speaker embedding model: {SPEAKER_MODEL}")
        _speaker_model = EncoderClassifier.from_hparams(
            source=SPEAKER_MODEL,
            savedir=os.path.join(os.getenv("HF_HOME", "/data/hf"), "speechbrain_spk"),
            run_opts={"device": SPEAKER_DEVICE if torch.cuda.is_available() else "cpu"},
        )
        print("Speaker embedding model loaded")
    return _speaker_model


@dataclass
class SpeakerSegment:
    """A segment with speaker information (compatible with old diarization API)."""

    start: float
    end: float
    speaker_id: str


@dataclass
class KnownSpeaker:
    """A speaker identified by their voice embedding."""

    speaker_id: str
    embedding: np.ndarray
    language: str | None = None
    language_confidence: float = 0.0


@dataclass
class SpeakerTracker:
    """
    Tracks speakers using voice embeddings for consistent identification.

    Unlike pyannote diarization which assigns arbitrary speaker IDs that can
    flip between processing ticks, this tracker uses voice embeddings to
    maintain consistent speaker identity across the entire session.
    """

    sample_rate: int = 16000
    known_speakers: list[KnownSpeaker] = field(default_factory=list)
    next_speaker_idx: int = 0

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray | None:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio samples as float32 array [-1, 1]

        Returns:
            192-dimensional embedding vector, or None if extraction fails
        """
        if len(audio) < self.sample_rate * 0.3:  # Need at least 0.3s
            return None

        try:
            model = get_speaker_model()
            signal = torch.from_numpy(audio).unsqueeze(0).float()
            embedding = model.encode_batch(signal)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _find_matching_speaker(
        self, embedding: np.ndarray, force_match: bool = False
    ) -> KnownSpeaker | None:
        """Find a known speaker matching the given embedding.

        Args:
            embedding: Speaker embedding to match
            force_match: If True, return best match even if below threshold
                        (used when MAX_SPEAKERS reached)

        Returns:
            Matching KnownSpeaker or None if no match above threshold
        """
        best_match = None
        best_similarity = -1.0 if force_match else SPEAKER_SIMILARITY_THRESHOLD

        for speaker in self.known_speakers:
            similarity = self._cosine_similarity(embedding, speaker.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker

        if best_match:
            print(f"  Matched speaker {best_match.speaker_id} (similarity={best_similarity:.3f})")

        return best_match

    def _register_speaker(self, embedding: np.ndarray) -> KnownSpeaker:
        """Register a new speaker with the given embedding."""
        speaker_id = f"SPEAKER_{self.next_speaker_idx:02d}"
        self.next_speaker_idx += 1

        speaker = KnownSpeaker(
            speaker_id=speaker_id,
            embedding=embedding,
        )
        self.known_speakers.append(speaker)
        print(f"  Registered new speaker: {speaker_id}")
        return speaker

    def _energy_vad(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Simple energy-based voice activity detection.

        Args:
            audio: Audio samples as float32 array

        Returns:
            List of (start_sample, end_sample) tuples for speech regions
        """
        # Calculate frame-level energy (10ms frames)
        frame_size = int(self.sample_rate * 0.01)
        hop_size = frame_size // 2
        num_frames = (len(audio) - frame_size) // hop_size + 1

        if num_frames <= 0:
            return []

        energies = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start : start + frame_size]
            energies[i] = np.sqrt(np.mean(frame**2))

        # Apply threshold
        is_speech = energies > VAD_ENERGY_THRESHOLD

        # Smooth: apply minimum duration constraints
        min_speech_frames = int(VAD_MIN_SPEECH_SEC / 0.005)  # 5ms hop
        min_silence_frames = int(VAD_MIN_SILENCE_SEC / 0.005)

        # Find speech regions
        regions = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    speech_start = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # End of speech region
                        speech_end = i - silence_count
                        if speech_end - speech_start >= min_speech_frames:
                            start_sample = speech_start * hop_size
                            end_sample = min(speech_end * hop_size + frame_size, len(audio))
                            regions.append((start_sample, end_sample))
                        in_speech = False
                        silence_count = 0

        # Handle trailing speech
        if in_speech:
            speech_end = len(is_speech) - 1
            if speech_end - speech_start >= min_speech_frames:
                start_sample = speech_start * hop_size
                end_sample = len(audio)
                regions.append((start_sample, end_sample))

        return regions

    def _merge_adjacent_same_speaker(
        self, segments: list[SpeakerSegment], max_gap: float = 0.5
    ) -> list[SpeakerSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments:
            return []

        # Sort by start time
        sorted_segs = sorted(segments, key=lambda s: s.start)
        merged = [sorted_segs[0]]

        for seg in sorted_segs[1:]:
            last = merged[-1]
            if seg.speaker_id == last.speaker_id and seg.start - last.end <= max_gap:
                # Merge: extend last segment
                merged[-1] = SpeakerSegment(
                    start=last.start,
                    end=seg.end,
                    speaker_id=last.speaker_id,
                )
            else:
                merged.append(seg)

        return merged

    def process_audio(
        self,
        audio: np.ndarray,
        window_start: float = 0.0,
    ) -> list[SpeakerSegment]:
        """
        Process audio and return speaker segments with consistent IDs.

        Unlike pyannote which assigns arbitrary IDs that can flip, this method
        uses voice embeddings to ensure speakers are identified consistently.

        Args:
            audio: Audio samples as float32 array (normalized to [-1, 1])
            window_start: Absolute start time of this window in seconds

        Returns:
            List of SpeakerSegment with consistent speaker assignments
        """
        if not SPEAKER_ENABLED:
            return []

        if len(audio) < self.sample_rate * 0.5:
            return []

        # Step 1: Find speech regions using energy VAD
        speech_regions = self._energy_vad(audio)

        if not speech_regions:
            print("  No speech detected by VAD")
            return []

        print(f"  VAD found {len(speech_regions)} speech regions")

        # Step 2: For each region, extract embedding and match/register speaker
        segments = []

        for start_sample, end_sample in speech_regions:
            region_audio = audio[start_sample:end_sample]
            duration = len(region_audio) / self.sample_rate

            # Skip very short regions
            if duration < 0.3:
                continue

            # Extract embedding
            embedding = self._extract_embedding(region_audio)
            if embedding is None:
                continue

            # Match to known speaker or register new
            # If we've reached MAX_SPEAKERS, force match to closest speaker
            at_max_speakers = len(self.known_speakers) >= MAX_SPEAKERS
            speaker = self._find_matching_speaker(embedding, force_match=at_max_speakers)

            if speaker is None:
                speaker = self._register_speaker(embedding)
            else:
                # Update embedding with exponential moving average for robustness
                alpha = 0.3
                speaker.embedding = alpha * embedding + (1 - alpha) * speaker.embedding

            # Create segment
            start_sec = window_start + start_sample / self.sample_rate
            end_sec = window_start + end_sample / self.sample_rate

            segments.append(
                SpeakerSegment(
                    start=start_sec,
                    end=end_sec,
                    speaker_id=speaker.speaker_id,
                )
            )

        # Merge adjacent segments from same speaker
        merged = self._merge_adjacent_same_speaker(segments)
        print(f"  Speaker tracking: {len(segments)} regions → {len(merged)} segments")

        return merged

    def get_speaker_language(self, speaker_id: str) -> tuple[str | None, float]:
        """Get the cached language for a speaker."""
        for speaker in self.known_speakers:
            if speaker.speaker_id == speaker_id:
                return speaker.language, speaker.language_confidence
        return None, 0.0

    def set_speaker_language(self, speaker_id: str, language: str, confidence: float = 1.0) -> None:
        """Set/update the language for a speaker."""
        for speaker in self.known_speakers:
            if speaker.speaker_id == speaker_id:
                speaker.language = language
                speaker.language_confidence = confidence
                print(f"  Set {speaker_id} language: {language} ({confidence:.2f})")
                return


def warmup_speaker_model():
    """Warm up speaker embedding model."""
    if not SPEAKER_ENABLED:
        return

    print("Warming up speaker embedding model...")
    try:
        model = get_speaker_model()
        # Quick inference to fully initialize
        dummy = torch.zeros(1, 16000)
        _ = model.encode_batch(dummy)
        print("Speaker embedding model warmed up")
    except Exception as e:
        print(f"Speaker model warmup failed: {e}")

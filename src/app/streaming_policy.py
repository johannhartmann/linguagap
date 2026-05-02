"""
Segment tracking and finalization for streaming transcription.

This module maintains a cumulative transcript that persists across sliding
ASR windows. It solves a key problem with streaming ASR: segments from
earlier in the conversation would be lost when the window slides forward.

Key concepts:
    - Cumulative Transcript: All segments ever detected are kept, even when
      no longer in the current ASR window
    - Segment Merging: New hypotheses merge with existing segments based on
      50%+ time overlap
    - Time-Based Finalization: Segments finalize when their end time is
      STABILITY_SEC in the past (not when text stabilizes)

Finalization state machine:
    1. NEW: Segment detected in ASR window, added to cumulative_segments
    2. UPDATING: Segment re-detected, text/timing may change
    3. STABLE: Segment end time is STABILITY_SEC ago
    4. FINALIZED: Moved to finalized_segments, queued for translation

Why time-based finalization?
    Text-based stability detection is unreliable - Whisper may refine text
    even for old audio. Time-based approach ensures segments finalize
    predictably when they're safely in the past.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

STABILITY_SEC = float(os.getenv("STABILITY_SEC", "1.5"))


@dataclass
class Segment:
    """
    A transcribed speech segment with timing and speaker information.

    Attributes:
        id: Unique identifier (monotonically increasing)
        abs_start: Absolute start time in seconds from session start
        abs_end: Absolute end time in seconds from session start
        src: Transcribed text in the source language
        src_lang: ISO language code (e.g., "de", "bg")
        final: True if segment is finalized and won't change
        speaker_id: Speaker identifier from diarization (e.g., "SPEAKER_00")
    """

    id: int
    abs_start: float
    abs_end: float
    src: str
    src_lang: str
    final: bool
    speaker_id: str | None = None
    speaker_role: str | None = None  # "german" | "foreign" | None


@dataclass
class CumulativeSegment:
    """A segment in the cumulative transcript with update tracking."""

    segment: Segment
    last_updated: float  # Timestamp when this segment was last seen/updated
    stable_since: float | None = None  # Timestamp when text became stable (for finalization)


@dataclass
class SegmentTracker:
    """
    Maintains a cumulative transcript that persists across sliding windows.

    Key behavior:
    - All detected segments are kept in cumulative_segments (never lost)
    - Segments merge based on time overlap when re-detected
    - Segments finalize when stable for STABILITY_SEC
    - Finalized segments are moved to finalized_segments list
    """

    next_id: int = 0
    finalized_segments: list[Segment] = field(default_factory=list)
    cumulative_segments: list[CumulativeSegment] = field(default_factory=list)

    def _calc_overlap_ratio(self, start1: float, end1: float, start2: float, end2: float) -> float:
        """Overlap as fraction of the FIRST range. 0.0 = disjoint, 1.0 = first ⊆ second."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = max(0, overlap_end - overlap_start)
        duration1 = end1 - start1
        if duration1 <= 0:
            return 0.0
        return overlap_duration / duration1

    def _overlaps_majority(
        self, a_start: float, a_end: float, b_start: float, b_end: float, threshold: float = 0.5
    ) -> bool:
        """True if either range covers >threshold of the other (bidirectional overlap)."""
        forward = self._calc_overlap_ratio(a_start, a_end, b_start, b_end)
        reverse = self._calc_overlap_ratio(b_start, b_end, a_start, a_end)
        return forward > threshold or reverse > threshold

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _is_substring_match(a: str, b: str, min_ratio: float) -> bool:
        """True if the shorter of (a, b) is a substring of the longer with len ratio ≥ threshold."""
        if not a or not b:
            return False
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return shorter in longer and len(shorter) / len(longer) >= min_ratio

    def _strip_finalized_prefix(self, text: str) -> str:
        """Strip text from hypothesis that already belongs to a finalized segment.

        Whisper's sliding window can re-transcribe already-finalized audio,
        producing hypotheses like "Do you work in Germany? How long would you
        like to work in Germany?" when only the second sentence is new.
        """
        if not self.finalized_segments or not text:
            return text

        text_norm = self._normalize_text(text)

        # Check the most recent finalized segments (reverse order, limit scope)
        for seg in reversed(self.finalized_segments[-10:]):
            seg_norm = self._normalize_text(seg.src)
            if not seg_norm or len(seg_norm) < 6:
                continue

            # Check if the hypothesis starts with finalized text (normalized)
            if text_norm.startswith(seg_norm):
                # Strip the finalized prefix from the original text
                # Find the split point by matching word count
                prefix_word_count = len(seg_norm.split())
                words = text.split()
                if prefix_word_count < len(words):
                    stripped = " ".join(words[prefix_word_count:]).strip()
                    if stripped:
                        logger.debug(
                            "  STRIP PREFIX: '%s' -> '%s'",
                            text[:40],
                            stripped[:40],
                        )
                        return stripped

        return text

    def _is_compatible_segment(
        self,
        existing: Segment,
        speaker_id: str | None,
        src_lang: str,
        speaker_role: str | None = None,
    ) -> bool:
        """Check whether two segments can represent the same utterance."""
        # Same speaker_id is always compatible — Whisper's language detection
        # can flip between ticks, but the speaker identity is stable.
        if existing.speaker_id and speaker_id:
            if existing.speaker_id == speaker_id:
                return True
            same_role = bool(
                existing.speaker_role and speaker_role and existing.speaker_role == speaker_role
            )
            if not same_role:
                return False

        # Reject when one side says German and the other says non-German
        # (and both languages are known) — they cannot be the same utterance.
        existing_lang = existing.src_lang
        if existing_lang == "unknown" or src_lang == "unknown":
            return True
        return (existing_lang == "de") == (src_lang == "de")

    def _text_match(
        self, src_text: str, cs: CumulativeSegment, abs_start: float
    ) -> bool:
        """Return True if the candidate text matches the cumulative segment by text."""
        if not src_text or not cs.segment.src:
            return False
        text_a = self._normalize_text(src_text)
        text_b = self._normalize_text(cs.segment.src)
        if len(text_a) < 6 or len(text_b) < 6:
            return False
        if text_a == text_b and abs(abs_start - cs.segment.abs_start) <= 3.0:
            return True
        return (
            self._is_substring_match(text_a, text_b, min_ratio=0.8)
            and abs(abs_start - cs.segment.abs_start) <= 2.0
        )

    def _find_matching_cumulative(
        self,
        abs_start: float,
        abs_end: float,
        src_text: str,
        speaker_id: str | None,
        src_lang: str,
        speaker_role: str | None,
    ) -> CumulativeSegment | None:
        """
        Find a cumulative segment that overlaps significantly with the given range.

        Uses bidirectional overlap checking - a match is found if either:
            - >50% of the new range overlaps with existing segment, OR
            - >50% of existing segment overlaps with new range
        """
        for cs in self.cumulative_segments:
            if cs.segment.final:
                continue
            if not self._is_compatible_segment(cs.segment, speaker_id, src_lang, speaker_role):
                continue
            if self._text_match(src_text, cs, abs_start):
                return cs
            if self._overlaps_majority(
                abs_start, abs_end, cs.segment.abs_start, cs.segment.abs_end
            ):
                return cs
        return None

    def _text_duplicate(self, normalized_text: str, abs_start: float, seg: Segment) -> bool:
        """Detect drifted re-detection by text equality / substring match."""
        if not normalized_text or not seg.src:
            return False
        seg_text_norm = self._normalize_text(seg.src)

        if normalized_text == seg_text_norm:
            max_drift = 2.0 if len(normalized_text) < 6 else 5.0
            if abs(abs_start - seg.abs_start) < max_drift:
                return True

        return (
            len(normalized_text) > 4
            and len(seg_text_norm) > 4
            and self._is_substring_match(normalized_text, seg_text_norm, min_ratio=0.7)
            and abs(abs_start - seg.abs_start) < 6.0
        )

    def _is_duplicate_segment(
        self,
        abs_start: float,
        abs_end: float,
        text: str = "",
        segments_to_check: list[Segment] | None = None,
    ) -> bool:
        """Check if a segment duplicates any segment in the provided list.

        Uses two strategies:
            1. Time-based: bidirectional >50% overlap
            2. Text-based: identical or near-identical text content
        """
        normalized_text = self._normalize_text(text) if text else ""
        candidates = segments_to_check if segments_to_check is not None else self.finalized_segments

        for seg in candidates:
            if self._overlaps_majority(abs_start, abs_end, seg.abs_start, seg.abs_end):
                return True
            if self._text_duplicate(normalized_text, abs_start, seg):
                return True
        return False

    def _find_mergeable_segment(
        self,
        abs_start: float,
        speaker_id: str | None,
        src_lang: str,
    ) -> CumulativeSegment | None:
        """
        Find a segment that should be merged with a new segment.

        Merging happens when:
        - Same speaker_id (both non-None)
        - Same language
        - Small gap or slight overlap between existing segment end and new start

        This handles the case where a speaker pauses mid-sentence, causing
        diarization to create separate segments that should be one turn.

        Note: MAX_MERGE_GAP_SEC was reduced from 2.0s to 0.8s to prevent
        cross-turn merging in bilingual dialogues where speaker turns
        may have short pauses between them.
        """
        MAX_MERGE_GAP_SEC = 0.8  # Reduced from 2.0 to prevent cross-turn merging
        MAX_MERGE_OVERLAP_SEC = 0.5  # Allow small overlap

        if speaker_id is None:
            return None

        for cs in self.cumulative_segments:
            if cs.segment.final:
                continue
            if cs.segment.speaker_id != speaker_id:
                continue
            if cs.segment.src_lang != src_lang:
                continue

            # Check if new segment starts shortly after (or slightly overlaps) this one
            # gap > 0: new segment starts after existing ends (gap)
            # gap < 0: new segment starts before existing ends (overlap)
            # gap = 0: segments are adjacent
            gap = abs_start - cs.segment.abs_end
            if -MAX_MERGE_OVERLAP_SEC <= gap < MAX_MERGE_GAP_SEC:
                return cs

        return None

    def _merge_into_existing(
        self,
        match: CumulativeSegment,
        abs_end: float,
        src_text: str,
        now_sec: float,
    ) -> None:
        old_text = match.segment.src
        match.segment.src = old_text + " " + src_text
        match.segment.abs_end = abs_end
        match.last_updated = now_sec
        match.stable_since = None
        logger.debug(
            "  MERGED: [%.1f-%.1f] '%s...' + '%s...'",
            match.segment.abs_start,
            abs_end,
            old_text[:20],
            src_text[:20],
        )

    def _update_existing(
        self,
        match: CumulativeSegment,
        abs_start: float,
        abs_end: float,
        src_text: str,
        seg_lang: str,
        speaker_id: str | None,
        speaker_role: str | None,
        now_sec: float,
    ) -> None:
        old_len = len(match.segment.src)
        new_len = len(src_text)
        match.segment.abs_start = min(match.segment.abs_start, abs_start)
        match.segment.abs_end = max(match.segment.abs_end, abs_end)
        match.last_updated = now_sec
        if new_len < old_len * 0.7:
            # New text is significantly shorter — keep existing text, just stretch the times.
            return
        text_changed = match.segment.src != src_text
        match.segment.src = src_text
        match.segment.src_lang = seg_lang
        if speaker_id:
            match.segment.speaker_id = speaker_id
        if speaker_role:
            match.segment.speaker_role = speaker_role
        if text_changed:
            match.stable_since = None
        elif match.stable_since is None:
            match.stable_since = now_sec

    def _create_new(
        self,
        abs_start: float,
        abs_end: float,
        src_text: str,
        seg_lang: str,
        speaker_id: str | None,
        speaker_role: str | None,
        now_sec: float,
    ) -> bool:
        """Append a new live segment if it isn't a duplicate of an existing one."""
        live_segments = [cs.segment for cs in self.cumulative_segments]
        if self._is_duplicate_segment(
            abs_start, abs_end, src_text, segments_to_check=live_segments
        ):
            return False

        logger.debug(
            "  NEW SEG: [%.1f-%.1f] '%s' (total: %d)",
            abs_start,
            abs_end,
            src_text[:30],
            len(self.cumulative_segments) + 1,
        )
        new_segment = Segment(
            id=self.next_id,
            abs_start=abs_start,
            abs_end=abs_end,
            src=src_text,
            src_lang=seg_lang,
            final=False,
            speaker_id=speaker_id,
            speaker_role=speaker_role,
        )
        self.cumulative_segments.append(
            CumulativeSegment(
                segment=new_segment,
                last_updated=now_sec,
                stable_since=None,
            )
        )
        self.next_id += 1
        return True

    def _ingest_hypothesis_segment(
        self, seg: dict, window_start: float, now_sec: float, default_lang: str
    ) -> None:
        """Fold one ASR hypothesis segment into the cumulative segment list."""
        abs_start = window_start + seg["start"]
        abs_end = window_start + seg["end"]
        src_text = seg["text"].strip()

        if not src_text or len(src_text) < 2:
            return

        src_text = self._strip_finalized_prefix(src_text)
        if not src_text or len(src_text) < 2:
            return

        if self._is_duplicate_segment(abs_start, abs_end, src_text):
            return

        seg_lang = seg.get("lang", default_lang)
        speaker_id = seg.get("speaker_id")
        speaker_role = seg.get("speaker_role")

        match = self._find_matching_cumulative(
            abs_start=abs_start,
            abs_end=abs_end,
            src_text=src_text,
            speaker_id=speaker_id,
            src_lang=seg_lang,
            speaker_role=speaker_role,
        )
        is_merge = False
        if not match:
            match = self._find_mergeable_segment(abs_start, speaker_id, seg_lang)
            if match:
                is_merge = True

        if match is None:
            self._create_new(
                abs_start, abs_end, src_text, seg_lang, speaker_id, speaker_role, now_sec
            )
        elif is_merge:
            self._merge_into_existing(match, abs_end, src_text, now_sec)
        else:
            self._update_existing(
                match,
                abs_start,
                abs_end,
                src_text,
                seg_lang,
                speaker_id,
                speaker_role,
                now_sec,
            )

    def _finalize_stale(self, now_sec: float) -> list[Segment]:
        """Move segments whose end time is older than STABILITY_SEC into finalized."""
        stability_threshold = now_sec - STABILITY_SEC
        newly_finalized: list[Segment] = []
        still_live: list[CumulativeSegment] = []

        for cs in self.cumulative_segments:
            if cs.segment.final:
                continue
            if cs.segment.abs_end <= stability_threshold:
                cs.segment.final = True
                self.finalized_segments.append(cs.segment)
                newly_finalized.append(cs.segment)
                logger.debug(
                    "  FINALIZED (time): [%.1f-%.1f] %s",
                    cs.segment.abs_start,
                    cs.segment.abs_end,
                    cs.segment.src[:50],
                )
            else:
                still_live.append(cs)

        self.cumulative_segments = still_live
        return newly_finalized

    def _assemble_chronological(self) -> list[Segment]:
        self.finalized_segments.sort(key=lambda s: s.id)
        live_segments = [cs.segment for cs in self.cumulative_segments]
        all_segments = self.finalized_segments + live_segments
        all_segments.sort(key=lambda s: s.id)
        return all_segments

    def update_from_hypothesis(
        self,
        hyp_segments: list[dict],
        window_start: float,
        now_sec: float,
        src_lang: str = "unknown",
    ) -> tuple[list[Segment], list[Segment]]:
        """
        Process hypothesis segments and return (all_segments, newly_finalized).

        This maintains a cumulative transcript - segments are never lost.
        """
        for seg in hyp_segments:
            self._ingest_hypothesis_segment(seg, window_start, now_sec, src_lang)

        newly_finalized = self._finalize_stale(now_sec)
        all_segments = self._assemble_chronological()
        return all_segments, newly_finalized

    def force_finalize_all(self) -> list[Segment]:
        """
        Force-finalize any remaining live segments.
        Call this when recording stops to ensure all segments get translated.
        """
        newly_finalized = []

        for cs in self.cumulative_segments:
            if cs.segment.final:
                continue
            cs.segment.final = True
            self.finalized_segments.append(cs.segment)
            newly_finalized.append(cs.segment)
            logger.debug(
                "  FORCE-FINALIZED: [%.1f-%.1f] %s",
                cs.segment.abs_start,
                cs.segment.abs_end,
                cs.segment.src[:50],
            )

        self.cumulative_segments = []
        self.finalized_segments.sort(key=lambda s: s.id)

        return newly_finalized

    # Legacy property for compatibility
    @property
    def finalized_end_time(self) -> float:
        if not self.finalized_segments:
            return 0.0
        return max(s.abs_end for s in self.finalized_segments)

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

STABILITY_SEC = float(os.getenv("STABILITY_SEC", "1.25"))


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
        """
        Calculate overlap ratio between two time ranges.

        The ratio is: overlap_duration / duration_of_first_range

        This is used for segment matching - if a new hypothesis overlaps >50%
        with an existing cumulative segment, they're considered the same segment.

        Args:
            start1, end1: First time range
            start2, end2: Second time range

        Returns:
            Overlap ratio from 0.0 (no overlap) to 1.0 (complete overlap)
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = max(0, overlap_end - overlap_start)
        duration1 = end1 - start1
        if duration1 <= 0:
            return 0.0
        return overlap_duration / duration1

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    def _is_compatible_segment(
        self,
        existing: Segment,
        speaker_id: str | None,
        src_lang: str,
        speaker_role: str | None = None,
    ) -> bool:
        """Check whether two segments can represent the same utterance."""
        if (
            existing.speaker_id
            and speaker_id
            and existing.speaker_id != speaker_id
            and not (
                existing.speaker_role and speaker_role and existing.speaker_role == speaker_role
            )
        ):
            return False

        existing_lang = existing.src_lang
        return not (
            existing_lang != "unknown"
            and src_lang != "unknown"
            and (existing_lang == "de") != (src_lang == "de")
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

        This handles both cases:
            - New hypothesis slightly shorter than existing (refinement)
            - New hypothesis slightly longer than existing (extension)

        Args:
            abs_start: Absolute start time of new hypothesis
            abs_end: Absolute end time of new hypothesis

        Returns:
            Matching CumulativeSegment or None if no match found
        """
        for cs in self.cumulative_segments:
            if cs.segment.final:
                continue
            if not self._is_compatible_segment(cs.segment, speaker_id, src_lang, speaker_role):
                continue

            # Check both directions of overlap
            overlap1 = self._calc_overlap_ratio(
                abs_start, abs_end, cs.segment.abs_start, cs.segment.abs_end
            )
            overlap2 = self._calc_overlap_ratio(
                cs.segment.abs_start, cs.segment.abs_end, abs_start, abs_end
            )
            # Match if either direction has >50% overlap
            if overlap1 > 0.5 or overlap2 > 0.5:
                return cs

            # Also match near-identical text in a short time window.
            # This catches repeated ASR outputs of the same phrase with shifted timestamps.
            if src_text and cs.segment.src:
                text_a = self._normalize_text(src_text)
                text_b = self._normalize_text(cs.segment.src)
                if len(text_a) >= 8 and len(text_b) >= 8:
                    if text_a == text_b:
                        if abs(abs_start - cs.segment.abs_start) <= 2.0:
                            return cs
                    else:
                        shorter, longer = (
                            (text_a, text_b) if len(text_a) <= len(text_b) else (text_b, text_a)
                        )
                        if (
                            shorter in longer
                            and len(shorter) / len(longer) >= 0.85
                            and abs(abs_start - cs.segment.abs_start) <= 1.5
                        ):
                            return cs
        return None

    def _is_duplicate_segment(self, abs_start: float, abs_end: float, text: str = "") -> bool:
        """Check if a segment duplicates any existing segment (finalized or live).

        Uses two strategies:
            1. Time-based: bidirectional >50% overlap
            2. Text-based: identical or near-identical text content

        The text check is critical because the sliding ASR window causes the
        same speech to be re-detected at shifted absolute positions. E.g.
        "Hallo Johann" might appear at [10.2-14.5] then [8.8-10.6] then
        [7.6-9.4] as the window slides — these don't overlap in time but
        are clearly the same speech. Hallucinations also repeat text rapidly.
        """
        normalized_text = self._normalize_text(text) if text else ""
        
        all_existing_segments = self.finalized_segments + [cs.segment for cs in self.cumulative_segments]
        
        for seg in all_existing_segments:
            # Check 1: Time-based overlap (bidirectional)
            overlap1 = self._calc_overlap_ratio(abs_start, abs_end, seg.abs_start, seg.abs_end)
            overlap2 = self._calc_overlap_ratio(seg.abs_start, seg.abs_end, abs_start, abs_end)
            if overlap1 > 0.5 or overlap2 > 0.5:
                return True

            # Check 2: Text-based deduplication for sliding window drift and hallucinations
            if normalized_text and seg.src:
                seg_text_norm = self._normalize_text(seg.src)
                
                # Identical text in a short time proximity (within 5 seconds)
                # Catching Whisper hallucinations and re-emitted segments.
                if normalized_text == seg_text_norm:
                    if abs(abs_start - seg.abs_start) < 5.0:
                        return True
                
                # Substring check for partial redetections
                if len(normalized_text) > 8 and len(seg_text_norm) > 8:
                    shorter, longer = (
                        (normalized_text, seg_text_norm) 
                        if len(normalized_text) <= len(seg_text_norm) 
                        else (seg_text_norm, normalized_text)
                    )
                    if shorter in longer and len(shorter) / len(longer) > 0.6:
                        if abs(abs_start - seg.abs_start) < 4.0:
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
        newly_finalized = []

        # Process each hypothesis segment
        for seg in hyp_segments:
            abs_start = window_start + seg["start"]
            abs_end = window_start + seg["end"]
            src_text = seg["text"].strip()

            if not src_text or len(src_text) < 2:
                continue

            # Skip if overlaps with already finalized or live segment (time or text match)
            if self._is_duplicate_segment(abs_start, abs_end, src_text):
                continue

            seg_lang = seg.get("lang", src_lang)
            speaker_id = seg.get("speaker_id")
            speaker_role = seg.get("speaker_role")

            # Try to find matching cumulative segment (by time overlap)
            match = self._find_matching_cumulative(
                abs_start=abs_start,
                abs_end=abs_end,
                src_text=src_text,
                speaker_id=speaker_id,
                src_lang=seg_lang,
                speaker_role=speaker_role,
            )
            is_merge = False

            # If no overlap match, check for mergeable segment (same speaker, small gap)
            if not match:
                match = self._find_mergeable_segment(abs_start, speaker_id, seg_lang)
                if match:
                    is_merge = True

            if match:
                if is_merge:
                    # Merge: append text and extend end time (same speaker continuation)
                    old_text = match.segment.src
                    match.segment.src = old_text + " " + src_text
                    match.segment.abs_end = abs_end
                    match.last_updated = now_sec
                    match.stable_since = None  # Reset stability since text changed
                    logger.debug(
                        "  MERGED: [%.1f-%.1f] '%s...' + '%s...'",
                        match.segment.abs_start,
                        abs_end,
                        old_text[:20],
                        src_text[:20],
                    )
                else:
                    # Update existing cumulative segment (overlap case)
                    # Don't replace if new text is much shorter (would lose merged content)
                    old_len = len(match.segment.src)
                    new_len = len(src_text)
                    if new_len < old_len * 0.7:
                        # New text is significantly shorter - keep existing, just update times
                        match.segment.abs_start = min(match.segment.abs_start, abs_start)
                        match.segment.abs_end = max(match.segment.abs_end, abs_end)
                        match.last_updated = now_sec
                    else:
                        # Normal update - replace text
                        text_changed = match.segment.src != src_text
                        match.segment.abs_start = min(match.segment.abs_start, abs_start)
                        match.segment.abs_end = max(match.segment.abs_end, abs_end)
                        match.segment.src = src_text
                        match.segment.src_lang = seg_lang
                        if speaker_id:
                            match.segment.speaker_id = speaker_id
                        if speaker_role:
                            match.segment.speaker_role = speaker_role

                        match.last_updated = now_sec
                        if text_changed:
                            match.stable_since = None  # Reset stability tracking
                        elif match.stable_since is None:
                            # Text is same as before - mark when it became stable
                            match.stable_since = now_sec
            else:
                # Create new cumulative segment
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

        # Check for segments that should be finalized
        # Use time-based finalization: segment is final if its end time is STABILITY_SEC ago
        stability_threshold = now_sec - STABILITY_SEC
        still_live = []

        for cs in self.cumulative_segments:
            if cs.segment.final:
                continue

            # Finalize if segment ended STABILITY_SEC ago (time-based, not text-based)
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

        # Sort finalized segments by start time
        self.finalized_segments.sort(key=lambda s: s.abs_start)

        # Build result: finalized + live segments (sorted by time)
        live_segments = [cs.segment for cs in self.cumulative_segments]
        all_segments = self.finalized_segments + sorted(live_segments, key=lambda s: s.abs_start)
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
        self.finalized_segments.sort(key=lambda s: s.abs_start)

        return newly_finalized

    # Legacy property for compatibility
    @property
    def finalized_end_time(self) -> float:
        if not self.finalized_segments:
            return 0.0
        return max(s.abs_end for s in self.finalized_segments)

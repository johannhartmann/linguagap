import os
from dataclasses import dataclass, field

STABILITY_SEC = float(os.getenv("STABILITY_SEC", "1.25"))
# How many ticks a live segment can be missing before being dropped
LIVE_SEGMENT_GRACE_TICKS = int(os.getenv("LIVE_SEGMENT_GRACE_TICKS", "3"))


@dataclass
class Segment:
    id: int
    abs_start: float
    abs_end: float
    src: str
    src_lang: str
    final: bool
    speaker_id: str | None = None  # Speaker identifier from diarization


@dataclass
class LiveSegmentState:
    """Track state of a live (non-finalized) segment."""

    segment: Segment
    missing_ticks: int = 0  # How many ticks since last seen


@dataclass
class SegmentTracker:
    next_id: int = 0
    finalized_segments: list[Segment] = field(default_factory=list)
    finalized_end_time: float = 0.0
    # Persistent live segments: keyed by a rough time bucket for matching
    live_segment_states: list[LiveSegmentState] = field(default_factory=list)

    def _segments_overlap(self, seg1_start: float, seg1_end: float, seg2: Segment) -> bool:
        """Check if two segments overlap significantly."""
        overlap_start = max(seg1_start, seg2.abs_start)
        overlap_end = min(seg1_end, seg2.abs_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        seg1_duration = seg1_end - seg1_start
        # Consider overlapping if > 50% of the new segment overlaps
        return overlap_duration > seg1_duration * 0.5 if seg1_duration > 0 else False

    def update_from_hypothesis(
        self,
        hyp_segments: list[dict],
        window_start: float,
        now_sec: float,
        src_lang: str = "unknown",
    ) -> tuple[list[Segment], list[Segment]]:
        """
        Process hypothesis segments and return (all_segments, newly_finalized).

        Returns:
            all_segments: Complete list of finalized + live segments
            newly_finalized: Segments that just became final (need translation)
        """
        stability_threshold = now_sec - STABILITY_SEC

        newly_finalized = []
        seen_live_indices: set[int] = set()

        for seg in hyp_segments:
            abs_start = window_start + seg["start"]
            abs_end = window_start + seg["end"]
            src_text = seg["text"].strip()

            # Skip only if segment is ENTIRELY within already-finalized time range
            # (Previously checked abs_start which was too aggressive with sliding window)
            if abs_end <= self.finalized_end_time:
                continue

            if not src_text:
                continue

            is_final = abs_end <= stability_threshold

            # Get speaker_id and per-segment language from hypothesis if available
            speaker_id = seg.get("speaker_id")
            seg_lang = seg.get("lang", src_lang)  # Use per-segment lang or fallback

            if is_final:
                # Finalize this segment
                segment = Segment(
                    id=self.next_id,
                    abs_start=abs_start,
                    abs_end=abs_end,
                    src=src_text,
                    src_lang=seg_lang,
                    final=True,
                    speaker_id=speaker_id,
                )
                self.finalized_segments.append(segment)
                newly_finalized.append(segment)
                self.finalized_end_time = abs_end
                self.next_id += 1

                # Remove any live segments that overlap with the finalized segment
                self.live_segment_states = [
                    ls
                    for ls in self.live_segment_states
                    if not self._segments_overlap(ls.segment.abs_start, ls.segment.abs_end, segment)
                ]
            else:
                # Try to match with existing live segment
                matched = False
                for i, ls in enumerate(self.live_segment_states):
                    if self._segments_overlap(abs_start, abs_end, ls.segment):
                        # Update existing live segment
                        ls.segment.abs_start = abs_start
                        ls.segment.abs_end = abs_end
                        ls.segment.src = src_text
                        ls.segment.src_lang = seg_lang
                        if speaker_id is not None:
                            ls.segment.speaker_id = speaker_id
                        ls.missing_ticks = 0
                        seen_live_indices.add(i)
                        matched = True
                        break

                if not matched:
                    # Create new live segment
                    new_segment = Segment(
                        id=self.next_id,
                        abs_start=abs_start,
                        abs_end=abs_end,
                        src=src_text,
                        src_lang=seg_lang,
                        final=False,
                        speaker_id=speaker_id,
                    )
                    self.live_segment_states.append(LiveSegmentState(segment=new_segment))
                    seen_live_indices.add(len(self.live_segment_states) - 1)
                    self.next_id += 1

        # Increment missing count for unseen live segments and remove expired ones
        updated_live_states = []
        for i, ls in enumerate(self.live_segment_states):
            if i not in seen_live_indices:
                ls.missing_ticks += 1
            if ls.missing_ticks <= LIVE_SEGMENT_GRACE_TICKS:
                updated_live_states.append(ls)
        self.live_segment_states = updated_live_states

        # Build result: finalized + current live segments
        live_segments = [ls.segment for ls in self.live_segment_states]
        return self.finalized_segments + live_segments, newly_finalized

    def force_finalize_all(self, live_segments: list[Segment]) -> list[Segment]:
        """
        Force-finalize any remaining live segments.
        Call this when recording stops to ensure all segments get translated.

        Args:
            live_segments: List of live (non-final) segments to finalize

        Returns:
            List of newly finalized segments
        """
        newly_finalized = []
        for seg in live_segments:
            if seg.final:
                continue
            finalized_seg = Segment(
                id=self.next_id,
                abs_start=seg.abs_start,
                abs_end=seg.abs_end,
                src=seg.src,
                src_lang=seg.src_lang,
                final=True,
                speaker_id=seg.speaker_id,
            )
            self.finalized_segments.append(finalized_seg)
            newly_finalized.append(finalized_seg)
            self.finalized_end_time = max(self.finalized_end_time, seg.abs_end)
            self.next_id += 1
        return newly_finalized

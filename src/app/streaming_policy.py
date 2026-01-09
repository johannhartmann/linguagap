import os
from dataclasses import dataclass, field

STABILITY_SEC = float(os.getenv("STABILITY_SEC", "1.25"))


@dataclass
class Segment:
    id: int
    abs_start: float
    abs_end: float
    src: str
    src_lang: str
    final: bool


@dataclass
class SegmentTracker:
    next_id: int = 0
    finalized_segments: list[Segment] = field(default_factory=list)
    finalized_end_time: float = 0.0

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

        live_segments = []
        newly_finalized = []

        for seg in hyp_segments:
            abs_start = window_start + seg["start"]
            abs_end = window_start + seg["end"]
            src_text = seg["text"].strip()

            if abs_start < self.finalized_end_time:
                continue

            is_final = abs_end <= stability_threshold

            if is_final:
                if src_text:
                    segment = Segment(
                        id=self.next_id,
                        abs_start=abs_start,
                        abs_end=abs_end,
                        src=src_text,
                        src_lang=src_lang,
                        final=True,
                    )
                    self.finalized_segments.append(segment)
                    newly_finalized.append(segment)
                    self.finalized_end_time = abs_end
                    self.next_id += 1
            else:
                if src_text:
                    segment = Segment(
                        id=self.next_id + len(live_segments),
                        abs_start=abs_start,
                        abs_end=abs_end,
                        src=src_text,
                        src_lang=src_lang,
                        final=False,
                    )
                    live_segments.append(segment)

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
            )
            self.finalized_segments.append(finalized_seg)
            newly_finalized.append(finalized_seg)
            self.finalized_end_time = max(self.finalized_end_time, seg.abs_end)
            self.next_id += 1
        return newly_finalized

import os
from dataclasses import dataclass, field

STABILITY_SEC = float(os.getenv("STABILITY_SEC", "1.25"))


@dataclass
class Segment:
    id: int
    abs_start: float
    abs_end: float
    src: str
    de: str
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
        translate_fn,
        src_lang: str,
    ) -> list[Segment]:
        stability_threshold = now_sec - STABILITY_SEC

        live_segments = []

        for seg in hyp_segments:
            abs_start = window_start + seg["start"]
            abs_end = window_start + seg["end"]
            src_text = seg["text"].strip()

            if abs_start < self.finalized_end_time:
                continue

            is_final = abs_end <= stability_threshold

            if is_final:
                if src_text:
                    de_text = translate_fn([src_text], src_lang=src_lang, tgt_lang="de")[0]
                else:
                    de_text = ""

                segment = Segment(
                    id=self.next_id,
                    abs_start=abs_start,
                    abs_end=abs_end,
                    src=src_text,
                    de=de_text,
                    final=True,
                )
                self.finalized_segments.append(segment)
                self.finalized_end_time = abs_end
                self.next_id += 1
            else:
                if src_text:
                    de_text = translate_fn([src_text], src_lang=src_lang, tgt_lang="de")[0]
                else:
                    de_text = ""

                segment = Segment(
                    id=self.next_id + len(live_segments),
                    abs_start=abs_start,
                    abs_end=abs_end,
                    src=src_text,
                    de=de_text,
                    final=False,
                )
                live_segments.append(segment)

        return self.finalized_segments + live_segments

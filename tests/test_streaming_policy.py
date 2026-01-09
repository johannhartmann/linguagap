"""Tests for streaming policy module."""

from app.streaming_policy import Segment, SegmentTracker


class TestSegment:
    """Tests for Segment dataclass."""

    def test_segment_creation(self):
        """Test creating a segment with all fields."""
        segment = Segment(
            id=1,
            abs_start=0.0,
            abs_end=1.5,
            src="Hello world",
            src_lang="en",
            final=True,
        )
        assert segment.id == 1
        assert segment.abs_start == 0.0
        assert segment.abs_end == 1.5
        assert segment.src == "Hello world"
        assert segment.src_lang == "en"
        assert segment.final is True

    def test_segment_live(self):
        """Test creating a live (non-final) segment."""
        segment = Segment(
            id=0,
            abs_start=0.0,
            abs_end=0.5,
            src="Test",
            src_lang="de",
            final=False,
        )
        assert segment.final is False
        assert segment.src_lang == "de"


class TestSegmentTracker:
    """Tests for SegmentTracker."""

    def test_initial_state(self):
        """Test initial tracker state."""
        tracker = SegmentTracker()
        assert tracker.next_id == 0
        assert tracker.finalized_segments == []
        assert tracker.finalized_end_time == 0.0

    def test_empty_hypothesis(self):
        """Test with empty hypothesis segments."""
        tracker = SegmentTracker()
        all_segments, newly_finalized = tracker.update_from_hypothesis(
            hyp_segments=[],
            window_start=0.0,
            now_sec=5.0,
        )
        assert all_segments == []
        assert newly_finalized == []

    def test_live_segment_not_finalized(self):
        """Test that recent segments remain live."""
        tracker = SegmentTracker()
        # Segment ends at 4.5, now is 5.0, stability threshold is ~3.75 (5.0 - 1.25)
        # Since 4.5 > 3.75, segment should be live
        hyp = [{"start": 0.0, "end": 4.5, "text": "Hello"}]
        all_segments, newly_finalized = tracker.update_from_hypothesis(
            hyp_segments=hyp,
            window_start=0.0,
            now_sec=5.0,
        )
        assert len(all_segments) == 1
        assert all_segments[0].final is False
        assert newly_finalized == []

    def test_segment_finalization(self):
        """Test that old segments get finalized."""
        tracker = SegmentTracker()
        # Segment ends at 1.0, now is 5.0, stability threshold is ~3.75 (5.0 - 1.25)
        # Since 1.0 <= 3.75, segment should be finalized
        hyp = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        all_segments, newly_finalized = tracker.update_from_hypothesis(
            hyp_segments=hyp,
            window_start=0.0,
            now_sec=5.0,
        )
        assert len(all_segments) == 1
        assert all_segments[0].final is True
        assert len(newly_finalized) == 1
        assert newly_finalized[0].src == "Hello"

    def test_skip_overlapping_segments(self):
        """Test that segments overlapping with finalized are skipped."""
        tracker = SegmentTracker()

        # First call: finalize a segment
        hyp1 = [{"start": 0.0, "end": 1.0, "text": "First"}]
        tracker.update_from_hypothesis(hyp1, window_start=0.0, now_sec=5.0)

        # Second call: new segment that overlaps with finalized end time
        hyp2 = [{"start": 0.5, "end": 2.0, "text": "Overlapping"}]
        all_segments, newly_finalized = tracker.update_from_hypothesis(
            hyp2, window_start=0.0, now_sec=6.0
        )

        # Should only have the first finalized segment
        assert len(all_segments) == 1
        assert all_segments[0].src == "First"
        assert newly_finalized == []

    def test_empty_text_skipped(self):
        """Test that segments with empty text are skipped."""
        tracker = SegmentTracker()
        hyp = [{"start": 0.0, "end": 1.0, "text": "   "}]
        all_segments, newly_finalized = tracker.update_from_hypothesis(
            hyp_segments=hyp,
            window_start=0.0,
            now_sec=5.0,
        )
        assert all_segments == []
        assert newly_finalized == []

    def test_window_offset(self):
        """Test that window_start offsets are applied correctly."""
        tracker = SegmentTracker()
        # Window starts at 10.0, segment is at 0-1 relative = 10-11 absolute
        hyp = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        all_segments, _ = tracker.update_from_hypothesis(
            hyp_segments=hyp,
            window_start=10.0,
            now_sec=15.0,
        )
        assert len(all_segments) == 1
        assert all_segments[0].abs_start == 10.0
        assert all_segments[0].abs_end == 11.0

    def test_id_assignment(self):
        """Test that segment IDs are assigned correctly."""
        tracker = SegmentTracker()

        # Finalize first segment
        hyp1 = [{"start": 0.0, "end": 1.0, "text": "First"}]
        tracker.update_from_hypothesis(hyp1, window_start=0.0, now_sec=5.0)
        assert tracker.next_id == 1

        # Finalize second segment
        hyp2 = [{"start": 2.0, "end": 3.0, "text": "Second"}]
        all_segments, _ = tracker.update_from_hypothesis(hyp2, window_start=0.0, now_sec=10.0)

        assert tracker.next_id == 2
        assert all_segments[0].id == 0
        assert all_segments[1].id == 1

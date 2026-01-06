"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_hypothesis_segments():
    """Sample ASR hypothesis segments for testing."""
    return [
        {"start": 0.0, "end": 1.5, "text": "Hello world"},
        {"start": 2.0, "end": 3.5, "text": "This is a test"},
        {"start": 4.0, "end": 5.0, "text": "Final segment"},
    ]


@pytest.fixture
def empty_hypothesis_segments():
    """Empty hypothesis segments."""
    return []

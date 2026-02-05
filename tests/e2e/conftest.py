"""Pytest configuration and fixtures for E2E tests.

Session-scoped fixtures for TTS client, streaming client, and judge.
Environment variable validation.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Register custom markers and load .env file."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    load_dotenv(Path(__file__).parent / ".env", override=True)


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """Get Gemini API key from environment.

    Raises:
        pytest.skip: If GEMINI_API_KEY is not set
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="session")
def ws_url() -> str:
    """Get WebSocket URL from environment.

    Returns:
        WebSocket URL, defaults to localhost if not set
    """
    return os.getenv("LINGUAGAP_WS_URL", "ws://localhost:8000/ws")


@pytest.fixture(scope="session")
def audio_cache_dir() -> Path:
    """Get or create audio cache directory.

    Returns:
        Path to cache directory
    """
    cache_dir = os.getenv("E2E_AUDIO_CACHE_DIR", "./tests/fixtures/e2e_audio")
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def tts_client():
    """Create a session-scoped TTS client.

    Requires GOOGLE_APPLICATION_CREDENTIALS to be set.

    Returns:
        GeminiTTSClient instance
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    from tests.e2e.tts.client import GeminiTTSClient

    return GeminiTTSClient()


@pytest.fixture(scope="session")
def streaming_client(ws_url):
    """Create a session-scoped streaming client.

    Args:
        ws_url: WebSocket URL fixture

    Returns:
        StreamingClient instance
    """
    from tests.e2e.streaming.client import StreamingClient

    return StreamingClient(ws_url=ws_url)


@pytest.fixture(scope="session")
def judge(gemini_api_key):
    """Create a session-scoped Gemini judge.

    Args:
        gemini_api_key: API key fixture

    Returns:
        GeminiJudge instance
    """
    from tests.e2e.evaluation.judge import GeminiJudge

    return GeminiJudge(api_key=gemini_api_key)


@pytest.fixture(scope="session")
def dialogue_generator(gemini_api_key):
    """Create a session-scoped dialogue generator.

    Args:
        gemini_api_key: API key fixture

    Returns:
        DialogueGenerator instance
    """
    from tests.e2e.dialogues.generator import DialogueGenerator

    return DialogueGenerator(api_key=gemini_api_key)


@pytest.fixture
def sample_scenario():
    """Create a realistic test scenario for E2E testing.

    Returns:
        DialogueScenario with 6 turns for realistic diarization/ASR testing
    """
    from tests.e2e.dialogues.templates import DialogueScenario, DialogueTurn

    return DialogueScenario(
        name="test_scenario",
        description="Customer service dialogue about order inquiry",
        german_lang="de",
        foreign_lang="en",
        speakers={
            "speaker_1": "Anna",
            "speaker_2": "John",
        },
        turns=[
            DialogueTurn(
                speaker_id="speaker_1",
                language="de",
                text="Guten Tag, hier ist der Kundenservice. Wie kann ich Ihnen heute helfen?",
            ),
            DialogueTurn(
                speaker_id="speaker_2",
                language="en",
                text="Hello, I placed an order last week but I haven't received any updates.",
                expected_translation="Hallo, ich habe letzte Woche eine Bestellung aufgegeben, aber ich habe keine Updates erhalten.",
            ),
            DialogueTurn(
                speaker_id="speaker_1",
                language="de",
                text="Das tut mir leid zu hören. Können Sie mir bitte Ihre Bestellnummer geben?",
            ),
            DialogueTurn(
                speaker_id="speaker_2",
                language="en",
                text="Yes, the order number is twelve thirty-four fifty-six. I ordered a laptop.",
                expected_translation="Ja, die Bestellnummer ist zwölf vierunddreißig sechsundfünfzig. Ich habe einen Laptop bestellt.",
            ),
            DialogueTurn(
                speaker_id="speaker_1",
                language="de",
                text="Ich sehe die Bestellung. Das Paket wurde gestern versendet und sollte morgen ankommen.",
            ),
            DialogueTurn(
                speaker_id="speaker_2",
                language="en",
                text="That's great news! Thank you so much for your help.",
                expected_translation="Das sind tolle Neuigkeiten! Vielen Dank für Ihre Hilfe.",
            ),
        ],
        expected_summary_topics=["order inquiry", "shipping status", "laptop order"],
    )

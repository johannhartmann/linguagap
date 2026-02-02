#!/usr/bin/env python3
"""Generate TTS audio fixtures for E2E tests.

This script pre-generates audio files for all test scenarios so they can be
committed to git and reused without calling the Gemini TTS API repeatedly.

Usage:
    uv run python tests/e2e/scripts/generate_audio_fixtures.py
"""

import os
import re
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.e2e.dialogues.templates import DialogueScenario
from tests.e2e.tts.client import GeminiTTSClient

# Directory containing pre-defined scenario YAML files
SCENARIOS_DIR = Path(__file__).parent.parent.parent / "fixtures" / "scenarios"
AUDIO_DIR = Path(__file__).parent.parent.parent / "fixtures" / "e2e_audio"

# Rate limiting: Gemini TTS free tier allows 3 requests/minute
REQUESTS_PER_MINUTE = 3
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE + 1  # ~21 seconds


def extract_retry_delay(error_msg: str) -> int:
    """Extract retry delay from error message."""
    match = re.search(r"retry in (\d+)", str(error_msg))
    if match:
        return int(match.group(1)) + 5  # Add 5s buffer
    return 40  # Default 40 seconds


def main():
    """Generate all audio fixtures from pre-defined scenarios."""
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it in tests/e2e/.env or export it")
        sys.exit(1)

    tts_client = GeminiTTSClient(api_key=api_key)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Find all scenario YAML files
    scenario_files = sorted(SCENARIOS_DIR.glob("*.yaml"))
    if not scenario_files:
        print(f"Error: No scenario files found in {SCENARIOS_DIR}")
        sys.exit(1)

    print(f"Found {len(scenario_files)} scenario files")
    print(f"Audio output directory: {AUDIO_DIR}")
    print(
        f"Rate limit: {REQUESTS_PER_MINUTE} req/min → {DELAY_BETWEEN_REQUESTS:.0f}s between requests"
    )
    print()

    generated = []
    cached = []
    failed = []

    requests_made = 0

    for i, yaml_path in enumerate(scenario_files, 1):
        scenario_name = yaml_path.stem
        print(f"[{i}/{len(scenario_files)}] {scenario_name}...", end=" ", flush=True)

        try:
            # Load scenario from YAML
            scenario = DialogueScenario.from_yaml_file(str(yaml_path))

            # Check if already cached first
            from tests.e2e.tts.cache import compute_cache_key, get_cached_audio
            from tests.e2e.tts.voices import get_voice_for_speaker

            voices = {sid: get_voice_for_speaker(sid) for sid in scenario.speakers}
            cache_key = compute_cache_key(scenario.to_yaml(), voices)
            existing = get_cached_audio(cache_key)

            if existing:
                cached.append((scenario_name, existing))
                print(f"CACHED: {existing.name}")
                continue

            # Not cached - need to make API call
            # Retry loop with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    audio_path = tts_client.synthesize_dialogue(scenario, use_cache=True)
                    generated.append((scenario_name, audio_path))
                    print(f"OK: {audio_path.name}")
                    requests_made += 1
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        retry_delay = extract_retry_delay(error_str)
                        if attempt < max_retries - 1:
                            print(f"RATE LIMITED, waiting {retry_delay}s...", end=" ", flush=True)
                            time.sleep(retry_delay)
                        else:
                            raise
                    elif "500" in error_str or "INTERNAL" in error_str:
                        if attempt < max_retries - 1:
                            print("SERVER ERROR, retrying in 10s...", end=" ", flush=True)
                            time.sleep(10)
                        else:
                            raise
                    else:
                        raise

            # Rate limit delay after successful request
            if i < len(scenario_files) and requests_made > 0:
                remaining = len(scenario_files) - i
                # Check if next file is cached
                next_yaml = scenario_files[i] if i < len(scenario_files) else None
                if next_yaml:
                    next_scenario = DialogueScenario.from_yaml_file(str(next_yaml))
                    next_voices = {
                        sid: get_voice_for_speaker(sid) for sid in next_scenario.speakers
                    }
                    next_key = compute_cache_key(next_scenario.to_yaml(), next_voices)
                    if not get_cached_audio(next_key):
                        # Next file needs API call, apply rate limit
                        print(
                            f"  (waiting {DELAY_BETWEEN_REQUESTS:.0f}s for rate limit, {remaining} remaining)"
                        )
                        time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            failed.append((scenario_name, str(e)))
            print(f"FAIL: {e}")

    # Summary
    print()
    print("=" * 60)
    print(f"Generated: {len(generated)} new audio files")
    print(f"Cached:    {len(cached)} existing files")
    print(f"Failed:    {len(failed)}")

    if failed:
        print("\nFailed scenarios:")
        for name, error in failed:
            print(f"  - {name}: {error[:100]}...")
        print("\nRe-run the script to retry failed scenarios.")
        sys.exit(1)

    print(f"\nAudio files in: {AUDIO_DIR}")
    print("\nTo commit fixtures to git:")
    print("  git add tests/fixtures/scenarios/*.yaml")
    print("  git add tests/fixtures/e2e_audio/*.wav")


if __name__ == "__main__":
    main()

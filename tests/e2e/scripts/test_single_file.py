#!/usr/bin/env python3
"""Test a single audio file through the ASR/MT/summarization pipeline.

Usage:
    uv run python tests/e2e/scripts/test_single_file.py [scenario_name]

Example:
    uv run python tests/e2e/scripts/test_single_file.py sample_scenario
"""

import asyncio
import json
import sys
import uuid
import wave
from pathlib import Path

import websockets

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.e2e.dialogues.templates import DialogueScenario
from tests.e2e.tts.cache import compute_cache_key, get_cached_audio
from tests.e2e.tts.voices import get_voice_for_speaker

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "fixtures" / "scenarios"
WS_URL = "ws://localhost:8000/ws"


async def stream_audio_file(audio_path: Path, ws_url: str = WS_URL):
    """Stream an audio file through the WebSocket and collect results."""

    # Read WAV file
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        audio_data = wav.readframes(wav.getnframes())

    print(f"Audio: {audio_path.name}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {len(audio_data) / (sample_rate * sample_width * n_channels):.1f}s")
    print()

    results = {
        "segments": [],
        "translations": {},
        "summary": None,
        "errors": [],
    }

    try:
        async with websockets.connect(ws_url) as ws:
            # Send config
            config = {
                "type": "config",
                "sample_rate": sample_rate,
                "token": str(uuid.uuid4()),
            }
            await ws.send(json.dumps(config))
            print("Sent config...")

            # Wait for config_ack
            ack = await asyncio.wait_for(ws.recv(), timeout=10)
            ack_data = json.loads(ack)
            if ack_data.get("type") == "config_ack":
                print(f"Config acknowledged: {ack_data.get('status')}")
            else:
                print(f"Unexpected response: {ack_data}")

            # Stream audio in chunks
            chunk_size = sample_rate * sample_width  # 1 second chunks
            print(
                f"Streaming {len(audio_data)} bytes in {len(audio_data) // chunk_size + 1} chunks..."
            )

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            print("Audio sent, waiting for results...")

            # Wait for transcription results with longer timeout
            end_time = asyncio.get_event_loop().time() + 30  # 30 second timeout

            while asyncio.get_event_loop().time() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(msg)
                    msg_type = data.get("type", "")

                    if msg_type in ("transcription", "segments"):
                        segments = data.get("segments", [])
                        for seg in segments:
                            src = seg.get("src", "")
                            lang = seg.get("src_lang", "??")
                            final = seg.get("final", False)
                            marker = "✓" if final else "…"
                            print(f"  {marker} [{lang}] {src[:70]}")
                            if final and seg not in results["segments"]:
                                results["segments"].append(seg)

                    elif msg_type == "translation":
                        seg_id = data.get("segment_id")
                        tgt_lang = data.get("tgt_lang")
                        text = data.get("text", "")
                        if seg_id not in results["translations"]:
                            results["translations"][seg_id] = {}
                        results["translations"][seg_id][tgt_lang] = text
                        print(f"    → [{tgt_lang}] {text[:70]}")

                    elif msg_type == "summary":
                        results["summary"] = data.get("summary")
                        print(f"\n📝 Summary: {results['summary']}")

                    elif msg_type == "error":
                        results["errors"].append(data.get("message"))
                        print(f"❌ ERROR: {data.get('message')}")

                    elif msg_type == "end":
                        print("End signal received")
                        break
                    else:
                        print(f"  [debug] {msg_type}: {str(data)[:100]}")

                except TimeoutError:
                    # No message in 5 seconds, check if we have results
                    if results["segments"]:
                        print("No more messages, have results")
                        break
                    continue

    except Exception as e:
        results["errors"].append(str(e))
        print(f"Connection error: {e}")

    return results


def main():
    # Get scenario name from args or use default
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "sample_scenario"

    # Load scenario
    yaml_path = SCENARIOS_DIR / f"{scenario_name}.yaml"
    if not yaml_path.exists():
        print(f"Error: Scenario not found: {yaml_path}")
        print("Available scenarios:")
        for f in sorted(SCENARIOS_DIR.glob("*.yaml")):
            print(f"  - {f.stem}")
        sys.exit(1)

    scenario = DialogueScenario.from_yaml_file(str(yaml_path))
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Languages: {scenario.german_lang} + {scenario.foreign_lang}")
    print()

    # Find cached audio
    voices = {sid: get_voice_for_speaker(sid) for sid in scenario.speakers}
    cache_key = compute_cache_key(scenario.to_yaml(), voices)
    audio_path = get_cached_audio(cache_key)

    if not audio_path:
        print(f"Error: No cached audio for {scenario_name}")
        print("Run: uv run python tests/e2e/scripts/generate_audio_fixtures.py")
        sys.exit(1)

    # Print expected content
    print("Expected dialogue:")
    for turn in scenario.turns:
        print(f"  [{turn.language}] {turn.text}")
    print()

    # Run test
    print("=" * 60)
    print("Streaming to backend...")
    print("=" * 60)

    results = asyncio.run(stream_audio_file(audio_path))

    # Summary
    print()
    print("=" * 60)
    print("Results:")
    print(f"  Final segments: {len(results['segments'])}")
    print(f"  Translations: {len(results['translations'])}")
    print(f"  Summary: {'Yes' if results['summary'] else 'No'}")
    print(f"  Errors: {len(results['errors'])}")

    if results["errors"]:
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()

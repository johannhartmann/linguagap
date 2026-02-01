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


async def stream_audio_file(
    audio_path: Path,
    ws_url: str = WS_URL,
    wait_for_summary: bool = True,
    foreign_lang: str | None = None,
):
    """Stream an audio file through the WebSocket and collect results.

    Args:
        audio_path: Path to WAV file
        ws_url: WebSocket URL
        wait_for_summary: Whether to wait for summary generation
        foreign_lang: Optional language code hint for non-German language (e.g., "ar", "tr")
    """

    # Read WAV file
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        audio_data = wav.readframes(wav.getnframes())

    audio_duration = len(audio_data) / (sample_rate * sample_width * n_channels)
    print(f"Audio: {audio_path.name}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {audio_duration:.1f}s")
    if foreign_lang:
        print(f"  Foreign language hint: {foreign_lang}")
    print()

    results = {
        "segments": [],
        "translations": {},
        "summary": None,
        "errors": [],
    }

    try:
        # Long timeouts for summarization which can take 5+ minutes on CPU
        async with websockets.connect(
            ws_url,
            ping_interval=60,
            ping_timeout=360,
            close_timeout=30,
        ) as ws:
            # Send config with optional foreign language hint
            config = {
                "type": "config",
                "sample_rate": sample_rate,
                "token": str(uuid.uuid4()),
            }
            if foreign_lang:
                config["foreign_lang"] = foreign_lang
            await ws.send(json.dumps(config))
            print(f"Sent config (foreign_lang={foreign_lang or 'auto'})...")

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

            print("Audio sent, collecting transcriptions...")

            # First phase: collect transcriptions and translations
            # Wait for segments to stabilize and translations to complete
            # Timeout: audio duration + 10s for ASR + 8s per segment for translation
            estimated_segments = max(2, int(audio_duration / 3))
            transcription_wait = audio_duration + 10 + estimated_segments * 10
            phase1_end = asyncio.get_event_loop().time() + transcription_wait
            no_new_data_count = 0

            while asyncio.get_event_loop().time() < phase1_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
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
                            if final:
                                # Track by segment ID to avoid duplicates
                                seg_id = seg.get("id")
                                if not any(s.get("id") == seg_id for s in results["segments"]):
                                    results["segments"].append(seg)

                    elif msg_type == "translation":
                        seg_id = data.get("segment_id")
                        tgt_lang = data.get("tgt_lang")
                        text = data.get("text", "")
                        if seg_id not in results["translations"]:
                            results["translations"][seg_id] = {}
                        results["translations"][seg_id][tgt_lang] = text
                        print(f"    → [{tgt_lang}] {text[:70]}")

                    elif msg_type == "error":
                        results["errors"].append(data.get("message"))
                        print(f"❌ ERROR: {data.get('message')}")

                except TimeoutError:
                    no_new_data_count += 1
                    # Exit if we have segments with translations and no new data
                    if results["segments"] and results["translations"]:
                        if len(results["translations"]) >= len(results["segments"]):
                            print(f"All {len(results['segments'])} segments translated")
                            break
                        if no_new_data_count >= 3:
                            pending = len(results["segments"]) - len(results["translations"])
                            print(f"Waiting for {pending} more translations...")
                            no_new_data_count = 0
                    continue

            # Second phase: request summary (forces finalization + translations)
            if wait_for_summary:
                print("\nRequesting summary (finalizing segments)...")
                await ws.send(json.dumps({"type": "request_summary"}))

                # Wait for summary with longer timeout
                # Summarization runs on CPU (~60s per LLM call x 5 calls = ~5 min)
                # Plus translations (~6s each on GPU)
                estimated_segments = max(len(results["segments"]), int(audio_duration / 3))
                summary_timeout = 360 + estimated_segments * 10  # 6 min base + segment time
                summary_end = asyncio.get_event_loop().time() + summary_timeout

                while asyncio.get_event_loop().time() < summary_end:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=120)
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
                                if final:
                                    seg_id = seg.get("id")
                                    if not any(s.get("id") == seg_id for s in results["segments"]):
                                        results["segments"].append(seg)

                        elif msg_type == "translation":
                            seg_id = data.get("segment_id")
                            tgt_lang = data.get("tgt_lang")
                            text = data.get("text", "")
                            if seg_id not in results["translations"]:
                                results["translations"][seg_id] = {}
                            results["translations"][seg_id][tgt_lang] = text
                            print(f"    → [{tgt_lang}] {text[:70]}")

                        elif msg_type == "summary_progress":
                            step = data.get("step", "")
                            message = data.get("message", "")
                            print(f"  ⏳ {step}: {message}")

                        elif msg_type == "summary":
                            results["summary"] = {
                                "foreign": data.get("foreign_summary"),
                                "german": data.get("german_summary"),
                                "foreign_lang": data.get("foreign_lang"),
                                "aligned": data.get("aligned"),
                            }
                            print(f"\n📝 German summary: {data.get('german_summary', '')[:100]}")
                            print(f"📝 Foreign summary: {data.get('foreign_summary', '')[:100]}")
                            break

                        elif msg_type == "summary_error":
                            error = data.get("error", "Unknown error")
                            results["errors"].append(f"Summary error: {error}")
                            print(f"❌ Summary error: {error}")
                            break

                        elif msg_type == "error":
                            results["errors"].append(data.get("message"))
                            print(f"❌ ERROR: {data.get('message')}")

                    except TimeoutError:
                        # Check if we're still expecting translations
                        pending = len(results["segments"]) - len(results["translations"])
                        if pending > 0:
                            print(f"  (waiting for {pending} translations...)")
                        else:
                            print("  (waiting for summary...)")
                        continue  # Keep waiting for summary until overall timeout

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

    results = asyncio.run(stream_audio_file(audio_path, foreign_lang=scenario.foreign_lang))

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

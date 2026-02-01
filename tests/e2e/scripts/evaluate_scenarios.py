#!/usr/bin/env python3
"""Evaluate E2E scenarios with transcription, translation, and summary quality scores.

Usage:
    # Evaluate all customer_service scenarios
    uv run python tests/e2e/scripts/evaluate_scenarios.py

    # Evaluate specific scenarios
    uv run python tests/e2e/scripts/evaluate_scenarios.py en_customer_service uk_customer_service

    # Evaluate with pattern matching
    uv run python tests/e2e/scripts/evaluate_scenarios.py --pattern "*_customer_service"

    # Skip TTS generation (use existing cache only)
    uv run python tests/e2e/scripts/evaluate_scenarios.py --no-generate

    # Output JSON report
    uv run python tests/e2e/scripts/evaluate_scenarios.py --output report.json

Environment:
    GEMINI_API_KEY: Required for TTS generation and LLM-as-Judge evaluation
    LINGUAGAP_WS_URL: WebSocket URL (default: ws://localhost:8000/ws)
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load .env before importing clients that need API keys (override=True to use .env over existing env vars)
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from tests.e2e.dialogues.templates import DialogueScenario  # noqa: E402
from tests.e2e.evaluation.judge import GeminiJudge  # noqa: E402
from tests.e2e.tts.cache import compute_cache_key, get_cached_audio  # noqa: E402
from tests.e2e.tts.client import GeminiTTSClient  # noqa: E402
from tests.e2e.tts.voices import get_voice_for_speaker  # noqa: E402

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "fixtures" / "scenarios"
WS_URL = "ws://localhost:8000/ws"


@dataclass
class TurnEvaluation:
    """Evaluation of a single dialogue turn."""

    speaker_id: str
    language: str
    expected_text: str
    actual_text: str | None
    transcription_score: int = 0
    transcription_reasoning: str = ""
    expected_translation: str | None = None
    actual_translation: str | None = None
    translation_score: int | None = None
    translation_reasoning: str | None = None


@dataclass
class ScenarioEvaluation:
    """Complete evaluation of a scenario."""

    name: str
    description: str
    foreign_lang: str
    turns: list[TurnEvaluation] = field(default_factory=list)
    summary_score: int = 0
    summary_reasoning: str = ""
    avg_transcription_score: float = 0.0
    avg_translation_score: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall score (average of all metrics)."""
        scores = [self.avg_transcription_score, self.summary_score]
        if self.avg_translation_score > 0:
            scores.append(self.avg_translation_score)
        return sum(scores) / len(scores) if scores else 0.0


async def stream_scenario(
    audio_path: Path,
    foreign_lang: str,
    ws_url: str = WS_URL,
) -> dict:
    """Stream audio through WebSocket and collect results."""
    import uuid
    import wave

    import websockets

    # Read WAV file
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        n_channels = wav.getnchannels()
        audio_data = wav.readframes(wav.getnframes())

    results = {
        "segments": [],
        "translations": {},
        "summary": None,
        "errors": [],
    }

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=60,
            ping_timeout=360,
            close_timeout=30,
        ) as ws:
            # Send config
            config = {
                "type": "config",
                "sample_rate": sample_rate,
                "token": str(uuid.uuid4()),
                "foreign_lang": foreign_lang,
            }
            await ws.send(json.dumps(config))

            # Wait for ack
            ack = await asyncio.wait_for(ws.recv(), timeout=10)
            ack_data = json.loads(ack)
            if ack_data.get("type") != "config_ack":
                results["errors"].append(f"Unexpected config response: {ack_data}")
                return results

            # Stream audio in chunks
            chunk_size = sample_rate * sample_width  # 1 second chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            # Wait for transcriptions
            audio_duration = len(audio_data) / (sample_rate * sample_width * n_channels)
            phase1_timeout = audio_duration + 30
            phase1_end = asyncio.get_event_loop().time() + phase1_timeout

            while asyncio.get_event_loop().time() < phase1_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    data = json.loads(msg)
                    msg_type = data.get("type", "")

                    if msg_type in ("transcription", "segments"):
                        for seg in data.get("segments", []):
                            if seg.get("final"):
                                seg_id = seg.get("id")
                                # Update existing segment or add new one
                                existing_idx = next(
                                    (
                                        i
                                        for i, s in enumerate(results["segments"])
                                        if s.get("id") == seg_id
                                    ),
                                    None,
                                )
                                if existing_idx is not None:
                                    # Update with newer version (may have more translations)
                                    results["segments"][existing_idx] = seg
                                else:
                                    results["segments"].append(seg)

                    elif msg_type == "translation":
                        seg_id = data.get("segment_id")
                        tgt_lang = data.get("tgt_lang")
                        text = data.get("text", "")
                        if seg_id not in results["translations"]:
                            results["translations"][seg_id] = {}
                        results["translations"][seg_id][tgt_lang] = text

                    elif msg_type == "error":
                        results["errors"].append(data.get("message"))

                except TimeoutError:
                    if results["segments"] and len(results["translations"]) >= len(
                        results["segments"]
                    ):
                        break
                    continue

            # Request summary
            await ws.send(json.dumps({"type": "request_summary"}))
            summary_end = asyncio.get_event_loop().time() + 420  # 7 min timeout

            while asyncio.get_event_loop().time() < summary_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(msg)
                    msg_type = data.get("type", "")

                    if msg_type in ("transcription", "segments"):
                        for seg in data.get("segments", []):
                            if seg.get("final"):
                                seg_id = seg.get("id")
                                # Update existing segment or add new one
                                existing_idx = next(
                                    (
                                        i
                                        for i, s in enumerate(results["segments"])
                                        if s.get("id") == seg_id
                                    ),
                                    None,
                                )
                                if existing_idx is not None:
                                    # Update with newer version (may have more translations)
                                    results["segments"][existing_idx] = seg
                                else:
                                    results["segments"].append(seg)

                    elif msg_type == "translation":
                        seg_id = data.get("segment_id")
                        tgt_lang = data.get("tgt_lang")
                        text = data.get("text", "")
                        if seg_id not in results["translations"]:
                            results["translations"][seg_id] = {}
                        results["translations"][seg_id][tgt_lang] = text

                    elif msg_type == "summary":
                        results["summary"] = {
                            "german": data.get("german_summary"),
                            "foreign": data.get("foreign_summary"),
                            "foreign_lang": data.get("foreign_lang"),
                        }
                        break

                    elif msg_type == "summary_error":
                        results["errors"].append(f"Summary: {data.get('error')}")
                        break

                except TimeoutError:
                    continue

    except Exception as e:
        results["errors"].append(str(e))

    return results


def match_segments_to_turns(
    scenario: DialogueScenario,
    segments: list[dict],
    translations: dict,
) -> list[tuple[dict | None, dict | None]]:
    """Match streaming segments to expected dialogue turns.

    Uses text similarity to align segments with expected turns.
    Returns list of (turn, matched_segment_data) tuples.
    """
    from difflib import SequenceMatcher

    matched = []

    for turn in scenario.turns:
        best_match = None
        best_score = 0.0

        for seg in segments:
            src = seg.get("src", "")
            # Simple similarity matching
            score = SequenceMatcher(None, turn.text.lower(), src.lower()).ratio()
            if score > best_score:
                best_score = score
                seg_id = seg.get("id")
                # Check embedded translations first, then fallback to separate translations dict
                embedded_trans = seg.get("translations", {})
                separate_trans = translations.get(seg_id, {})
                trans = embedded_trans if embedded_trans else separate_trans
                best_match = {
                    "segment": seg,
                    "translation": trans.get("de") if turn.language != "de" else None,
                    "similarity": best_score,
                }

        if best_match and best_match["similarity"] > 0.3:
            matched.append((turn, best_match))
        else:
            matched.append((turn, None))

    return matched


def evaluate_scenario(
    scenario: DialogueScenario,
    results: dict,
    judge: GeminiJudge,
) -> ScenarioEvaluation:
    """Evaluate a scenario's results using LLM-as-Judge."""
    eval_result = ScenarioEvaluation(
        name=scenario.name,
        description=scenario.description,
        foreign_lang=scenario.foreign_lang,
        errors=results.get("errors", []),
    )

    segments = results.get("segments", [])
    translations = results.get("translations", {})
    summary = results.get("summary")

    if not segments:
        eval_result.errors.append("No segments received")
        return eval_result

    # Match segments to turns
    matched = match_segments_to_turns(scenario, segments, translations)

    transcription_scores = []
    translation_scores = []

    for turn, match_data in matched:
        turn_eval = TurnEvaluation(
            speaker_id=turn.speaker_id,
            language=turn.language,
            expected_text=turn.text,
            actual_text=match_data["segment"]["src"] if match_data else None,
            expected_translation=turn.expected_translation,
            actual_translation=match_data["translation"] if match_data else None,
        )

        if match_data:
            # Evaluate transcription
            trans_result = judge.evaluate_transcription(
                expected_text=turn.text,
                actual_text=match_data["segment"]["src"],
                language=turn.language,
            )
            turn_eval.transcription_score = trans_result.score
            turn_eval.transcription_reasoning = trans_result.reasoning
            transcription_scores.append(trans_result.score)

            # Evaluate translation if expected
            if turn.expected_translation and match_data["translation"]:
                translation_result = judge.evaluate_translation(
                    source_text=turn.text,
                    expected_translation=turn.expected_translation,
                    actual_translation=match_data["translation"],
                    src_lang=turn.language,
                    tgt_lang="de",
                )
                turn_eval.translation_score = translation_result.score
                turn_eval.translation_reasoning = translation_result.reasoning
                translation_scores.append(translation_result.score)

        eval_result.turns.append(turn_eval)

    # Calculate averages
    if transcription_scores:
        eval_result.avg_transcription_score = sum(transcription_scores) / len(transcription_scores)

    if translation_scores:
        eval_result.avg_translation_score = sum(translation_scores) / len(translation_scores)

    # Evaluate summary
    if summary and summary.get("german"):
        summary_result = judge.evaluate_summary(
            conversation_segments=[
                {"src": s.get("src"), "src_lang": s.get("src_lang")} for s in segments
            ],
            expected_topics=scenario.expected_summary_topics,
            actual_summary={
                "german_summary": summary.get("german"),
                "foreign_summary": summary.get("foreign"),
            },
            foreign_lang=scenario.foreign_lang,
        )
        eval_result.summary_score = summary_result.score
        eval_result.summary_reasoning = summary_result.reasoning
    else:
        eval_result.errors.append("No summary received")

    return eval_result


def print_evaluation_report(evaluations: list[ScenarioEvaluation]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 80)
    print("E2E EVALUATION REPORT")
    print("=" * 80)

    # Summary table header
    print(
        f"\n{'Scenario':<30} {'Lang':<6} {'Trans':<8} {'Transl':<8} {'Summary':<8} {'Overall':<8}"
    )
    print("-" * 80)

    for ev in evaluations:
        trans_str = f"{ev.avg_transcription_score:.1f}" if ev.avg_transcription_score > 0 else "N/A"
        transl_str = f"{ev.avg_translation_score:.1f}" if ev.avg_translation_score > 0 else "N/A"
        summary_str = f"{ev.summary_score}" if ev.summary_score > 0 else "N/A"
        overall_str = f"{ev.overall_score:.1f}" if ev.overall_score > 0 else "N/A"

        # Color coding based on score
        status = ""
        if ev.overall_score >= 4:
            status = "+"
        elif ev.overall_score >= 3:
            status = "~"
        elif ev.overall_score > 0:
            status = "-"
        else:
            status = "!"

        print(
            f"{status} {ev.name:<28} {ev.foreign_lang:<6} {trans_str:<8} {transl_str:<8} {summary_str:<8} {overall_str:<8}"
        )

    print("-" * 80)

    # Statistics
    valid_evals = [e for e in evaluations if e.overall_score > 0]
    if valid_evals:
        avg_overall = sum(e.overall_score for e in valid_evals) / len(valid_evals)
        avg_trans = sum(e.avg_transcription_score for e in valid_evals) / len(valid_evals)
        trans_with_transl = [e for e in valid_evals if e.avg_translation_score > 0]
        avg_transl = (
            sum(e.avg_translation_score for e in trans_with_transl) / len(trans_with_transl)
            if trans_with_transl
            else 0
        )
        avg_summary = sum(e.summary_score for e in valid_evals) / len(valid_evals)

        print(
            f"\n{'AVERAGE':<30} {'':<6} {avg_trans:.1f}     {avg_transl:.1f}     {avg_summary:.1f}     {avg_overall:.1f}"
        )

    # Legend
    print("\nLegend: + (>=4 Good), ~ (>=3 Average), - (<3 Poor), ! (Error)")
    print("Scores: 1-5 scale (1=Very Poor, 3=Average, 5=Excellent)")

    # Error summary
    errors = [(e.name, err) for e in evaluations for err in e.errors]
    if errors:
        print("\nErrors:")
        for name, err in errors:
            print(f"  {name}: {err}")

    # Detailed results for low scores
    low_scores = [e for e in evaluations if 0 < e.overall_score < 3]
    if low_scores:
        print("\n" + "=" * 80)
        print("LOW SCORE DETAILS")
        print("=" * 80)

        for ev in low_scores:
            print(f"\n{ev.name} ({ev.foreign_lang}):")
            print(f"  Description: {ev.description}")

            for turn in ev.turns:
                if turn.transcription_score > 0 and turn.transcription_score < 3:
                    print(f"\n  Turn [{turn.language}]:")
                    print(f"    Expected: {turn.expected_text[:60]}...")
                    print(
                        f"    Actual:   {turn.actual_text[:60] if turn.actual_text else 'N/A'}..."
                    )
                    print(f"    Transcription Score: {turn.transcription_score}")
                    print(f"    Reason: {turn.transcription_reasoning[:100]}...")

                if turn.translation_score and turn.translation_score < 3:
                    print(f"    Translation Score: {turn.translation_score}")
                    print(
                        f"    Expected: {turn.expected_translation[:60] if turn.expected_translation else 'N/A'}..."
                    )
                    print(
                        f"    Actual:   {turn.actual_translation[:60] if turn.actual_translation else 'N/A'}..."
                    )

            if ev.summary_score < 3:
                print(f"\n  Summary Score: {ev.summary_score}")
                print(f"  Reason: {ev.summary_reasoning[:150]}...")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate E2E scenarios")
    parser.add_argument("scenarios", nargs="*", help="Scenario names to evaluate")
    parser.add_argument("--pattern", "-p", help="Glob pattern to match scenario names")
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip TTS generation, use cache only"
    )
    parser.add_argument("--output", "-o", help="Output JSON report to file")
    parser.add_argument("--ws-url", default=WS_URL, help="WebSocket URL")
    args = parser.parse_args()

    # Find scenarios
    all_scenarios = sorted(SCENARIOS_DIR.glob("*.yaml"))

    if args.scenarios:
        scenario_paths = [SCENARIOS_DIR / f"{name}.yaml" for name in args.scenarios]
        scenario_paths = [p for p in scenario_paths if p.exists()]
    elif args.pattern:
        scenario_paths = [p for p in all_scenarios if fnmatch(p.stem, args.pattern)]
    else:
        # Default: all *_customer_service scenarios
        scenario_paths = [p for p in all_scenarios if p.stem.endswith("_customer_service")]

    if not scenario_paths:
        print("No scenarios found. Available:")
        for p in all_scenarios:
            print(f"  - {p.stem}")
        sys.exit(1)

    print(f"Evaluating {len(scenario_paths)} scenarios:")
    for p in scenario_paths:
        print(f"  - {p.stem}")
    print()

    # Initialize clients
    try:
        judge = GeminiJudge()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    tts_client = None
    if not args.no_generate:
        try:
            tts_client = GeminiTTSClient()
        except ValueError:
            print("Warning: GEMINI_API_KEY not set, TTS generation disabled")

    evaluations = []

    for scenario_path in scenario_paths:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario_path.stem}")
        print("=" * 60)

        scenario = DialogueScenario.from_yaml_file(str(scenario_path))
        print(f"Description: {scenario.description}")
        print(f"Languages: {scenario.german_lang} + {scenario.foreign_lang}")

        # Get or generate audio
        voices = {sid: get_voice_for_speaker(sid) for sid in scenario.speakers}
        cache_key = compute_cache_key(scenario.to_yaml(), voices)
        audio_path = get_cached_audio(cache_key)

        if not audio_path:
            if tts_client:
                print("Generating TTS audio...")
                try:
                    audio_path = tts_client.synthesize_dialogue(scenario)
                except Exception as e:
                    print(f"TTS generation failed: {e}")
                    evaluations.append(
                        ScenarioEvaluation(
                            name=scenario.name,
                            description=scenario.description,
                            foreign_lang=scenario.foreign_lang,
                            errors=[f"TTS failed: {e}"],
                        )
                    )
                    continue
            else:
                print("No cached audio and TTS disabled, skipping")
                evaluations.append(
                    ScenarioEvaluation(
                        name=scenario.name,
                        description=scenario.description,
                        foreign_lang=scenario.foreign_lang,
                        errors=["No audio available"],
                    )
                )
                continue

        print(f"Audio: {audio_path}")

        # Stream and collect results
        print("Streaming to backend...")
        results = await stream_scenario(audio_path, scenario.foreign_lang, args.ws_url)
        print(f"  Segments: {len(results['segments'])}")
        print(f"  Translations: {len(results['translations'])}")
        print(f"  Summary: {'Yes' if results['summary'] else 'No'}")

        if results["errors"]:
            print(f"  Errors: {results['errors']}")

        # Evaluate with LLM-as-Judge
        print("Evaluating with LLM-as-Judge...")
        evaluation = evaluate_scenario(scenario, results, judge)
        evaluations.append(evaluation)

        print(f"  Transcription: {evaluation.avg_transcription_score:.1f}")
        print(f"  Translation: {evaluation.avg_translation_score:.1f}")
        print(f"  Summary: {evaluation.summary_score}")
        print(f"  Overall: {evaluation.overall_score:.1f}")

    # Print report
    print_evaluation_report(evaluations)

    # Save JSON report if requested
    if args.output:
        report_data = {
            "evaluations": [
                {
                    "name": e.name,
                    "description": e.description,
                    "foreign_lang": e.foreign_lang,
                    "avg_transcription_score": e.avg_transcription_score,
                    "avg_translation_score": e.avg_translation_score,
                    "summary_score": e.summary_score,
                    "overall_score": e.overall_score,
                    "errors": e.errors,
                    "turns": [
                        {
                            "speaker_id": t.speaker_id,
                            "language": t.language,
                            "expected_text": t.expected_text,
                            "actual_text": t.actual_text,
                            "transcription_score": t.transcription_score,
                            "expected_translation": t.expected_translation,
                            "actual_translation": t.actual_translation,
                            "translation_score": t.translation_score,
                        }
                        for t in e.turns
                    ],
                }
                for e in evaluations
            ],
        }
        with open(args.output, "w") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

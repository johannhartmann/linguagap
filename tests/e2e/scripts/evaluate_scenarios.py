#!/usr/bin/env python3
"""Evaluate all dialogue scenarios through the ASR/MT pipeline.

Runs all scenarios, evaluates with LLM-as-Judge, and generates detailed reports.

Usage:
    uv run python tests/e2e/scripts/evaluate_scenarios.py [--url URL] [--output FILE]

Example:
    uv run python tests/e2e/scripts/evaluate_scenarios.py --output report.json
"""

import argparse
import asyncio
import json
import sys
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import websockets
from dotenv import load_dotenv

# Load .env from tests/e2e directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import os  # noqa: E402

from tests.e2e.dialogues.templates import DialogueScenario  # noqa: E402
from tests.e2e.evaluation.judge import GeminiJudge  # noqa: E402
from tests.e2e.tts.cache import compute_cache_key, get_cached_audio  # noqa: E402
from tests.e2e.tts.voices import get_voice_for_speaker  # noqa: E402

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "fixtures" / "scenarios"
DEFAULT_WS_URL = os.getenv("LINGUAGAP_WS_URL", "ws://localhost:8000/ws")


@dataclass
class TurnEvaluation:
    """Detailed evaluation of a single turn."""

    speaker_id: str
    language: str
    expected_text: str
    actual_text: str | None
    transcription_score: int
    transcription_reasoning: str
    expected_translation: str | None
    actual_translation: str | None
    translation_score: int | None
    translation_reasoning: str | None


@dataclass
class ScenarioEvaluation:
    """Complete evaluation of a scenario."""

    name: str
    description: str
    foreign_lang: str
    avg_transcription_score: float
    avg_translation_score: float
    summary_score: int
    summary_reasoning: str
    overall_score: float
    errors: list[str]
    turns: list[TurnEvaluation]
    actual_summary: dict | None = None
    expected_summary_topics: list[str] = field(default_factory=list)


async def stream_audio_file(audio_path: Path, ws_url: str) -> dict:
    """Stream an audio file through the WebSocket and collect results."""

    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        audio_data = wav.readframes(wav.getnframes())

    results = {
        "segments": [],
        "translations": {},
        "summary": None,
        "errors": [],
    }

    try:
        async with websockets.connect(ws_url) as ws:
            config = {
                "type": "config",
                "sample_rate": sample_rate,
                "token": str(uuid.uuid4()),
            }
            await ws.send(json.dumps(config))

            ack = await asyncio.wait_for(ws.recv(), timeout=10)
            ack_data = json.loads(ack)
            if ack_data.get("type") != "config_ack":
                results["errors"].append(f"Unexpected config response: {ack_data}")
                return results

            # Stream audio in chunks
            chunk_size = sample_rate * sample_width
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            # Wait for ASR to process final audio
            await asyncio.sleep(2.0)

            # Request summary generation
            await ws.send(json.dumps({"type": "request_summary"}))

            # Collect results - wait for segments first, then summary
            # Summary generation takes 2-8 min (2-4 LLM calls with validation/regen)
            end_time = asyncio.get_event_loop().time() + 600  # 10 minutes

            while asyncio.get_event_loop().time() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    print(f"    [WS] {msg_type}", end="", flush=True)

                    if msg_type in ("transcription", "segments"):
                        for seg in data.get("segments", []):
                            if seg.get("final") and seg not in results["segments"]:
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
                            "foreign": data.get("foreign_summary"),
                            "german": data.get("german_summary"),
                        }
                        break  # Got summary, we're done

                    elif msg_type == "summary_progress":
                        # Summary is being generated, keep waiting
                        continue

                    elif msg_type == "summary_error":
                        results["errors"].append(f"Summary error: {data.get('error')}")
                        break

                    elif msg_type == "error":
                        results["errors"].append(data.get("message"))

                    elif msg_type == "end":
                        break

                except TimeoutError:
                    # Keep waiting if we have segments but no summary yet
                    if results["segments"] and not results["summary"]:
                        continue
                    # No segments at all after timeout - give up
                    if not results["segments"]:
                        break
                    continue

    except Exception as e:
        results["errors"].append(str(e))

    return results


def match_segments_to_turns(
    scenario: DialogueScenario,
    results: dict,
) -> list[tuple[dict, dict | None]]:
    """Match expected turns to actual segments.

    Returns list of (expected_turn, actual_segment) tuples.
    """
    segments = results.get("segments", [])
    translations = results.get("translations", {})

    matched = []
    seg_idx = 0

    for turn in scenario.turns:
        actual_seg = None
        if seg_idx < len(segments):
            actual_seg = segments[seg_idx].copy()
            # Attach translation if available
            seg_id = actual_seg.get("id")
            if seg_id and seg_id in translations:
                actual_seg["translations"] = translations[seg_id]
            seg_idx += 1

        matched.append((turn, actual_seg))

    return matched


async def evaluate_scenario(
    scenario: DialogueScenario,
    judge: GeminiJudge,
    ws_url: str,
) -> ScenarioEvaluation:
    """Evaluate a single scenario."""

    print(f"  Evaluating: {scenario.name}")

    # Get cached audio
    voices = {sid: get_voice_for_speaker(sid) for sid in scenario.speakers}
    cache_key = compute_cache_key(scenario.to_yaml(), voices)
    audio_path = get_cached_audio(cache_key)

    if not audio_path:
        return ScenarioEvaluation(
            name=scenario.name,
            description=scenario.description,
            foreign_lang=scenario.foreign_lang,
            avg_transcription_score=0,
            avg_translation_score=0,
            summary_score=0,
            summary_reasoning="No audio fixture available",
            overall_score=0,
            errors=["No cached audio - run generate_audio_fixtures.py first"],
            turns=[],
            expected_summary_topics=scenario.expected_summary_topics,
        )

    # Stream and get results
    results = await stream_audio_file(audio_path, ws_url)

    # Match segments to expected turns
    matched = match_segments_to_turns(scenario, results)

    # Evaluate each turn
    turn_evals = []
    transcription_scores = []
    translation_scores = []

    for turn, actual_seg in matched:
        actual_text = actual_seg.get("src") if actual_seg else None
        actual_translation = None

        if actual_seg and "translations" in actual_seg:
            # Get German translation for non-German turns
            actual_translation = actual_seg["translations"].get("de")

        # Evaluate transcription
        if actual_text:
            trans_result = judge.evaluate_transcription(
                expected_text=turn.text,
                actual_text=actual_text,
                language=turn.language,
            )
            trans_score = trans_result.score
            trans_reasoning = trans_result.reasoning
        else:
            trans_score = 0
            trans_reasoning = "No transcription output for this segment"

        transcription_scores.append(trans_score)

        # Evaluate translation (only for non-German with expected translation)
        transl_score = None
        transl_reasoning = None

        if turn.expected_translation and turn.language != "de":
            if actual_translation:
                transl_result = judge.evaluate_translation(
                    source_text=turn.text,
                    expected_translation=turn.expected_translation,
                    actual_translation=actual_translation,
                    src_lang=turn.language,
                    tgt_lang="de",
                )
                transl_score = transl_result.score
                transl_reasoning = transl_result.reasoning
            else:
                transl_score = 0
                transl_reasoning = "No translation output for this segment"

            translation_scores.append(transl_score)

        turn_evals.append(
            TurnEvaluation(
                speaker_id=turn.speaker_id,
                language=turn.language,
                expected_text=turn.text,
                actual_text=actual_text,
                transcription_score=trans_score,
                transcription_reasoning=trans_reasoning,
                expected_translation=turn.expected_translation,
                actual_translation=actual_translation,
                translation_score=transl_score,
                translation_reasoning=transl_reasoning,
            )
        )

    # Evaluate summary
    summary_score = 0
    summary_reasoning = "No summary generated"

    if results.get("summary"):
        summary_result = judge.evaluate_summary(
            conversation_segments=results.get("segments", []),
            expected_topics=scenario.expected_summary_topics,
            actual_summary=results["summary"],
            foreign_lang=scenario.foreign_lang,
        )
        summary_score = summary_result.score
        summary_reasoning = summary_result.reasoning

    # Calculate averages
    avg_trans = sum(transcription_scores) / len(transcription_scores) if transcription_scores else 0
    avg_transl = sum(translation_scores) / len(translation_scores) if translation_scores else 0

    # Overall score: weighted average
    all_scores = transcription_scores + translation_scores + [summary_score]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0

    return ScenarioEvaluation(
        name=scenario.name,
        description=scenario.description,
        foreign_lang=scenario.foreign_lang,
        avg_transcription_score=avg_trans,
        avg_translation_score=avg_transl,
        summary_score=summary_score,
        summary_reasoning=summary_reasoning,
        overall_score=overall,
        errors=results.get("errors", []),
        turns=turn_evals,
        actual_summary=results.get("summary"),
        expected_summary_topics=scenario.expected_summary_topics,
    )


def turn_to_dict(turn: TurnEvaluation) -> dict:
    """Convert TurnEvaluation to dict for JSON serialization."""
    return {
        "speaker_id": turn.speaker_id,
        "language": turn.language,
        "expected_text": turn.expected_text,
        "actual_text": turn.actual_text,
        "transcription_score": turn.transcription_score,
        "transcription_reasoning": turn.transcription_reasoning,
        "expected_translation": turn.expected_translation,
        "actual_translation": turn.actual_translation,
        "translation_score": turn.translation_score,
        "translation_reasoning": turn.translation_reasoning,
    }


def scenario_to_dict(scenario: ScenarioEvaluation) -> dict:
    """Convert ScenarioEvaluation to dict for JSON serialization."""
    return {
        "name": scenario.name,
        "description": scenario.description,
        "foreign_lang": scenario.foreign_lang,
        "avg_transcription_score": scenario.avg_transcription_score,
        "avg_translation_score": scenario.avg_translation_score,
        "summary_score": scenario.summary_score,
        "summary_reasoning": scenario.summary_reasoning,
        "overall_score": scenario.overall_score,
        "errors": scenario.errors,
        "turns": [turn_to_dict(t) for t in scenario.turns],
        "actual_summary": scenario.actual_summary,
        "expected_summary_topics": scenario.expected_summary_topics,
    }


def generate_markdown_report(evaluations: list[ScenarioEvaluation], ws_url: str) -> str:
    """Generate a detailed markdown report from evaluations."""
    lines = []

    # Header
    lines.append("# LinguaGap E2E Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**WebSocket URL:** {ws_url}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Language | Scenario | Transcription | Translation | Summary | Overall |")
    lines.append("|----------|----------|:-------------:|:-----------:|:-------:|:-------:|")

    for e in sorted(evaluations, key=lambda x: x.overall_score, reverse=True):
        lines.append(
            f"| {e.foreign_lang.upper()} | {e.description[:40]} | "
            f"{e.avg_transcription_score:.2f} | {e.avg_translation_score:.2f} | "
            f"{e.summary_score} | **{e.overall_score:.2f}** |"
        )

    lines.append("")

    # Calculate averages
    if evaluations:
        avg_trans = sum(e.avg_transcription_score for e in evaluations) / len(evaluations)
        avg_transl = sum(e.avg_translation_score for e in evaluations) / len(evaluations)
        avg_summ = sum(e.summary_score for e in evaluations) / len(evaluations)
        avg_overall = sum(e.overall_score for e in evaluations) / len(evaluations)

        lines.append(
            f"**Averages:** Transcription: {avg_trans:.2f} | Translation: {avg_transl:.2f} | Summary: {avg_summ:.2f} | Overall: {avg_overall:.2f}"
        )
        lines.append("")

    # Detailed results per scenario
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Results by Scenario")
    lines.append("")

    for e in evaluations:
        lines.append(f"### {e.foreign_lang.upper()} - {e.description}")
        lines.append(f"**Overall Score: {e.overall_score:.2f}**")
        lines.append("")

        if e.errors:
            lines.append("**Errors:**")
            for err in e.errors:
                lines.append(f"- {err}")
            lines.append("")

        # Turns table with all details
        lines.append("#### Turns")
        lines.append("")

        for i, turn in enumerate(e.turns, 1):
            lines.append(f"**Turn {i}** ({turn.speaker_id}, {turn.language.upper()})")
            lines.append("")
            lines.append("| | Expected | Actual |")
            lines.append("|---|----------|--------|")

            # Transcription row
            expected_text = turn.expected_text.replace("|", "\\|").replace("\n", " ")
            actual_text = (turn.actual_text or "—").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| **Transcription** | {expected_text} | {actual_text} |")

            # Translation row (only for non-German)
            if turn.expected_translation:
                expected_transl = turn.expected_translation.replace("|", "\\|").replace("\n", " ")
                actual_transl = (
                    (turn.actual_translation or "—").replace("|", "\\|").replace("\n", " ")
                )
                lines.append(f"| **Translation** | {expected_transl} | {actual_transl} |")

            lines.append("")

            # Scores
            score_line = f"Transcription Score: **{turn.transcription_score}**/5"
            if turn.translation_score is not None:
                score_line += f" | Translation Score: **{turn.translation_score}**/5"
            lines.append(score_line)
            lines.append("")

        # Summary section
        lines.append("#### Summary")
        lines.append("")

        if e.expected_summary_topics:
            lines.append("**Expected Topics:**")
            for topic in e.expected_summary_topics:
                lines.append(f"- {topic}")
            lines.append("")

        if e.actual_summary:
            lines.append("**Generated Summary:**")
            if isinstance(e.actual_summary, dict):
                for lang, text in e.actual_summary.items():
                    lines.append(f"- **{lang.upper()}:** {text}")
            else:
                lines.append(f"{e.actual_summary}")
            lines.append("")
        else:
            lines.append("*No summary generated*")
            lines.append("")

        lines.append(f"**Summary Score:** {e.summary_score}/5")
        lines.append(f"**Reasoning:** {e.summary_reasoning}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate all dialogue scenarios")
    parser.add_argument(
        "--url",
        default=DEFAULT_WS_URL,
        help=f"WebSocket URL (default: {DEFAULT_WS_URL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="report.json",
        help="Output file path (default: report.json)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        help="Run only specific scenario (by name)",
    )
    args = parser.parse_args()

    print("LinguaGap E2E Evaluation")
    print("=" * 60)
    print(f"WebSocket URL: {args.url}")
    print(f"Output: {args.output}")
    print()

    # Initialize judge
    judge = GeminiJudge()

    # Find all scenarios
    scenario_files = sorted(SCENARIOS_DIR.glob("*.yaml"))
    if not scenario_files:
        print(f"No scenarios found in {SCENARIOS_DIR}")
        sys.exit(1)

    print(f"Found {len(scenario_files)} scenarios")

    # Filter if specific scenario requested
    if args.scenario:
        scenario_files = [f for f in scenario_files if args.scenario in f.stem]
        if not scenario_files:
            print(f"No scenario matching '{args.scenario}' found")
            sys.exit(1)

    # Evaluate each scenario
    evaluations = []

    for scenario_file in scenario_files:
        try:
            scenario = DialogueScenario.from_yaml_file(str(scenario_file))
            evaluation = await evaluate_scenario(scenario, judge, args.url)
            evaluations.append(evaluation)
            print(f"    Score: {evaluation.overall_score:.2f}")
        except Exception as e:
            print(f"  Error evaluating {scenario_file.stem}: {e}")
            continue

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "ws_url": args.url,
        "evaluations": [scenario_to_dict(e) for e in evaluations],
        "summary": {
            "total_scenarios": len(evaluations),
            "avg_transcription": (
                sum(e.avg_transcription_score for e in evaluations) / len(evaluations)
                if evaluations
                else 0
            ),
            "avg_translation": (
                sum(e.avg_translation_score for e in evaluations) / len(evaluations)
                if evaluations
                else 0
            ),
            "avg_summary": (
                sum(e.summary_score for e in evaluations) / len(evaluations) if evaluations else 0
            ),
            "avg_overall": (
                sum(e.overall_score for e in evaluations) / len(evaluations) if evaluations else 0
            ),
        },
    }

    # Save JSON report
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save markdown report
    md_output_path = output_path.with_suffix(".md")
    markdown_report = generate_markdown_report(evaluations, args.url)
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print()
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Markdown report: {md_output_path}")
    print(f"Average transcription: {report['summary']['avg_transcription']:.2f}")
    print(f"Average translation: {report['summary']['avg_translation']:.2f}")
    print(f"Average summary: {report['summary']['avg_summary']:.2f}")
    print(f"Average overall: {report['summary']['avg_overall']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

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


def compute_cer(expected: str, actual: str) -> float:
    """Compute Character Error Rate using Levenshtein distance.

    Returns a value between 0.0 (perfect) and 1.0+ (completely wrong).
    """
    # Normalize: lowercase, strip punctuation for fair comparison
    import re

    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)  # Remove punctuation
        s = re.sub(r"\s+", " ", s)  # Normalize whitespace
        return s

    expected = normalize(expected)
    actual = normalize(actual)

    if not expected:
        return 0.0 if not actual else 1.0

    # Levenshtein distance
    m, n = len(expected), len(actual)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / m


def cer_to_score(cer: float) -> int:
    """Convert CER to 1-5 score.

    CER 0.00-0.05 = 5 (Excellent)
    CER 0.05-0.15 = 4 (Good)
    CER 0.15-0.30 = 3 (Average)
    CER 0.30-0.50 = 2 (Poor)
    CER 0.50+     = 1 (Very Poor)
    """
    if cer <= 0.05:
        return 5
    elif cer <= 0.15:
        return 4
    elif cer <= 0.30:
        return 3
    elif cer <= 0.50:
        return 2
    else:
        return 1


def compute_bleu(expected: str, actual: str) -> float:
    """Compute simplified BLEU score (unigram + bigram).

    Returns a value between 0.0 and 1.0.
    """
    import re
    from collections import Counter

    def tokenize(s: str) -> list[str]:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return s.split()

    ref_tokens = tokenize(expected)
    hyp_tokens = tokenize(actual)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Unigram precision
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    unigram_matches = sum(min(hyp_counts[w], ref_counts[w]) for w in hyp_counts)
    unigram_precision = unigram_matches / len(hyp_tokens) if hyp_tokens else 0

    # Bigram precision
    ref_bigrams = Counter(zip(ref_tokens[:-1], ref_tokens[1:], strict=False))
    hyp_bigrams = Counter(zip(hyp_tokens[:-1], hyp_tokens[1:], strict=False))
    bigram_matches = sum(min(hyp_bigrams[b], ref_bigrams[b]) for b in hyp_bigrams)
    bigram_precision = bigram_matches / len(hyp_bigrams) if hyp_bigrams else 0

    # Combined score (geometric mean)
    if unigram_precision == 0 or bigram_precision == 0:
        return unigram_precision * 0.5  # Fallback to unigram only

    import math

    bleu = math.sqrt(unigram_precision * bigram_precision)

    # Brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        bleu *= bp

    return bleu


def bleu_to_score(bleu: float) -> int:
    """Convert BLEU to 1-5 score.

    BLEU 0.80-1.00 = 5 (Excellent)
    BLEU 0.60-0.80 = 4 (Good)
    BLEU 0.40-0.60 = 3 (Average)
    BLEU 0.20-0.40 = 2 (Poor)
    BLEU 0.00-0.20 = 1 (Very Poor)
    """
    if bleu >= 0.80:
        return 5
    elif bleu >= 0.60:
        return 4
    elif bleu >= 0.40:
        return 3
    elif bleu >= 0.20:
        return 2
    else:
        return 1


@dataclass
class TurnEvaluation:
    """Evaluation of a single dialogue turn."""

    speaker_id: str
    language: str
    expected_text: str
    actual_text: str | None
    transcription_score: int = 0
    transcription_reasoning: str = ""
    cer_value: float | None = None  # Raw CER value for debugging
    expected_translation: str | None = None
    actual_translation: str | None = None
    translation_score: int | None = None
    translation_reasoning: str | None = None
    bleu_value: float | None = None  # Raw BLEU value for debugging


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
    summary_data: dict | None = None  # Stored for batch evaluation

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
            print(
                f"  [stream] Phase 1: waiting for transcriptions ({phase1_timeout:.0f}s timeout)",
                flush=True,
            )

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
            print("  [stream] Phase 2: requesting summary", flush=True)
            await ws.send(json.dumps({"type": "request_summary"}))
            print("  [stream] request_summary sent, waiting for response...", flush=True)
            summary_end = asyncio.get_event_loop().time() + 420  # 7 min timeout

            while asyncio.get_event_loop().time() < summary_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    print(f"  [stream] Phase 2 msg: {msg_type}", flush=True)

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
                        print("  [stream] Received summary!", flush=True)
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
                except websockets.exceptions.ConnectionClosed:
                    print("  [stream] WebSocket closed by server", flush=True)
                    break

            print("  [stream] Exiting summary loop", flush=True)

    except Exception as e:
        print(f"  [stream] Exception: {e}", flush=True)
        results["errors"].append(str(e))

    print("  [stream] Returning results", flush=True)
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


async def evaluate_scenario(
    scenario: DialogueScenario,
    results: dict,
    judge: GeminiJudge,  # noqa: ARG001 - kept for API compatibility, batch eval uses it
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
            # Evaluate transcription using CER (no LLM needed)
            actual_text = match_data["segment"]["src"]
            cer = compute_cer(turn.text, actual_text)
            score = cer_to_score(cer)
            turn_eval.transcription_score = score
            turn_eval.cer_value = cer
            turn_eval.transcription_reasoning = f"CER={cer:.2%}"
            transcription_scores.append(score)

            # Evaluate translation using BLEU (no LLM needed)
            if turn.expected_translation and match_data["translation"]:
                bleu = compute_bleu(turn.expected_translation, match_data["translation"])
                score = bleu_to_score(bleu)
                turn_eval.translation_score = score
                turn_eval.bleu_value = bleu
                turn_eval.translation_reasoning = f"BLEU={bleu:.2%}"
                translation_scores.append(score)

        eval_result.turns.append(turn_eval)

    # Calculate averages
    if transcription_scores:
        eval_result.avg_transcription_score = sum(transcription_scores) / len(transcription_scores)

    if translation_scores:
        eval_result.avg_translation_score = sum(translation_scores) / len(translation_scores)

    # Summary evaluation is done in batch later to save LLM calls
    # Just store the summary data for now
    eval_result.summary_data = {
        "segments": segments,
        "expected_topics": scenario.expected_summary_topics,
        "summary": summary,
        "foreign_lang": scenario.foreign_lang,
    }

    if not summary or not summary.get("german"):
        eval_result.errors.append("No summary received")

    return eval_result


async def batch_evaluate_summaries(
    evaluations: list[ScenarioEvaluation],
    judge: GeminiJudge,
) -> None:
    """Evaluate all summaries in a single batched LLM call to save quota."""
    # Collect all summaries that need evaluation
    summaries_to_eval = []
    for eval_result in evaluations:
        data = getattr(eval_result, "summary_data", None)
        if data and data.get("summary") and data["summary"].get("german"):
            summaries_to_eval.append((eval_result, data))

    if not summaries_to_eval:
        return

    print(f"Batch evaluating {len(summaries_to_eval)} summaries...")

    # Build a single prompt with all summaries
    batch_prompt = "Evaluate the following conversation summaries. For each, rate 1-5.\n\n"

    for i, (eval_result, data) in enumerate(summaries_to_eval, 1):
        segments_text = "\n".join(
            f"  - [{s.get('src_lang')}]: {s.get('src', '')[:100]}" for s in data["segments"][:6]
        )
        batch_prompt += f"""
--- Summary {i}: {eval_result.name} ({data["foreign_lang"]}) ---
Conversation:
{segments_text}

Expected topics: {", ".join(data["expected_topics"]) if data["expected_topics"] else "N/A"}

German summary: {data["summary"].get("german", "N/A")[:200]}
Foreign summary: {data["summary"].get("foreign", "N/A")[:200]}

"""

    batch_prompt += """
For each summary, respond with ONLY a JSON array like:
[{"scenario": 1, "score": 4, "reason": "brief reason"}, ...]

Scoring: 5=Excellent (all topics covered), 4=Good, 3=Average, 2=Poor, 1=Very Poor
"""

    try:
        response = await judge.client.aio.models.generate_content(
            model="gemini-2.0-flash",  # Use fast model for batch eval
            contents=batch_prompt,
        )
        response_text = response.text.strip() if response.text else ""

        # Parse JSON response
        import re

        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match:
            results = json.loads(json_match.group())
            for result in results:
                idx = result.get("scenario", 0) - 1
                if 0 <= idx < len(summaries_to_eval):
                    eval_result, _ = summaries_to_eval[idx]
                    eval_result.summary_score = result.get("score", 3)
                    eval_result.summary_reasoning = result.get("reason", "Batch evaluated")
        else:
            print(f"  Warning: Could not parse batch response: {response_text[:200]}")
            # Default scores
            for eval_result, _ in summaries_to_eval:
                eval_result.summary_score = 3
                eval_result.summary_reasoning = "Batch eval parse error"

    except Exception as e:
        print(f"  Batch summary evaluation failed: {e}")
        for eval_result, _ in summaries_to_eval:
            eval_result.summary_score = 3
            eval_result.summary_reasoning = f"Batch eval error: {e}"


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


def generate_markdown_report(
    evaluations: list[ScenarioEvaluation],
    results_map: dict[str, dict],
) -> str:
    """Generate a detailed markdown test protocol for debugging.

    Args:
        evaluations: List of scenario evaluations
        results_map: Map of scenario name to raw results (segments, translations, summary)

    Returns:
        Markdown string with detailed report including CER/BLEU metrics
    """
    from datetime import datetime

    lines = []
    lines.append("# E2E Test Protocol - Debug Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Scenarios:** {len(evaluations)}")
    lines.append("")

    # Calculate overall stats
    valid_evals = [e for e in evaluations if e.overall_score > 0]
    if valid_evals:
        avg_trans = sum(e.avg_transcription_score for e in valid_evals) / len(valid_evals)
        trans_with_transl = [e for e in valid_evals if e.avg_translation_score > 0]
        avg_transl = (
            sum(e.avg_translation_score for e in trans_with_transl) / len(trans_with_transl)
            if trans_with_transl
            else 0
        )
        avg_summ = sum(e.summary_score for e in valid_evals) / len(valid_evals)
        lines.append("## Overall Statistics")
        lines.append("")
        lines.append(f"- **Average Transcription Score:** {avg_trans:.2f}/5")
        lines.append(f"- **Average Translation Score:** {avg_transl:.2f}/5")
        lines.append(f"- **Average Summary Score:** {avg_summ:.2f}/5")
        lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Scenario | Language | Transcription | Translation | Summary | Overall |")
    lines.append("|----------|----------|:-------------:|:-----------:|:-------:|:-------:|")

    for ev in evaluations:
        trans = f"{ev.avg_transcription_score:.1f}" if ev.avg_transcription_score > 0 else "—"
        transl = f"{ev.avg_translation_score:.1f}" if ev.avg_translation_score > 0 else "—"
        summ = f"{ev.summary_score}" if ev.summary_score > 0 else "—"
        overall = f"**{ev.overall_score:.1f}**" if ev.overall_score > 0 else "—"

        # Status emoji
        if ev.overall_score >= 4:
            status = "✅"
        elif ev.overall_score >= 3:
            status = "⚠️"
        elif ev.overall_score > 0:
            status = "❌"
        else:
            status = "💥"

        lines.append(
            f"| {status} {ev.name} | {ev.foreign_lang.upper()} | {trans} | {transl} | {summ} | {overall} |"
        )

    lines.append("")
    lines.append("Legend: ✅ Good (≥4) | ⚠️ Average (≥3) | ❌ Poor (<3) | 💥 Error")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed results per scenario
    for ev in evaluations:
        results = results_map.get(ev.name, {})

        lines.append(f"## {ev.name}")
        lines.append("")
        lines.append(f"**Description:** {ev.description}")
        lines.append(f"**Languages:** German (DE) ↔ {ev.foreign_lang.upper()}")
        lines.append(f"**Overall Score:** {ev.overall_score:.1f}/5")
        lines.append("")

        if ev.errors:
            lines.append("### ⚠️ Errors")
            lines.append("")
            for err in ev.errors:
                lines.append(f"- `{err}`")
            lines.append("")

        # Turns detail
        lines.append("### Dialogue Turns")
        lines.append("")

        for i, turn in enumerate(ev.turns, 1):
            # Status indicator for turn
            turn_status = (
                "✅"
                if turn.transcription_score >= 4
                else ("⚠️" if turn.transcription_score >= 3 else "❌")
            )

            lines.append(
                f"#### {turn_status} Turn {i} — {turn.speaker_id} ({turn.language.upper()})"
            )
            lines.append("")

            # Transcription comparison
            lines.append("**Transcription:**")
            lines.append("")
            lines.append("| | Text |")
            lines.append("|---|------|")
            lines.append(f"| **Expected** | {turn.expected_text} |")
            lines.append(f"| **Actual** | {turn.actual_text or '*(no transcription)*'} |")
            lines.append("")

            # Transcription metrics
            if turn.transcription_score > 0:
                cer_display = f"{turn.cer_value:.2%}" if turn.cer_value is not None else "N/A"
                lines.append(
                    f"📊 **Transcription Score:** {turn.transcription_score}/5 (CER: {cer_display})"
                )
                lines.append("")

            # Translation comparison (if applicable)
            if turn.expected_translation:
                lines.append("**Translation:**")
                lines.append("")
                lines.append("| | Text |")
                lines.append("|---|------|")
                lines.append(f"| **Expected** | {turn.expected_translation} |")
                lines.append(f"| **Actual** | {turn.actual_translation or '*(no translation)*'} |")
                lines.append("")

                # Translation metrics
                if turn.translation_score:
                    bleu_display = (
                        f"{turn.bleu_value:.2%}" if turn.bleu_value is not None else "N/A"
                    )
                    lines.append(
                        f"📊 **Translation Score:** {turn.translation_score}/5 (BLEU: {bleu_display})"
                    )
                    lines.append("")

            lines.append("")

        # Summary section
        lines.append("### Summary Evaluation")
        lines.append("")

        summary = results.get("summary", {})
        if summary:
            lines.append("**Generated Summaries:**")
            lines.append("")
            lines.append(f"**{ev.foreign_lang.upper()} (Foreign):**")
            lines.append("")
            lines.append(f"> {summary.get('foreign', '*(no summary)*')}")
            lines.append("")
            lines.append("**DE (German):**")
            lines.append("")
            lines.append(f"> {summary.get('german', '*(no summary)*')}")
            lines.append("")
        else:
            lines.append("*No summary generated*")
            lines.append("")

        # Summary score and reasoning
        if ev.summary_score > 0:
            summ_status = (
                "✅" if ev.summary_score >= 4 else ("⚠️" if ev.summary_score >= 3 else "❌")
            )
            lines.append(f"📊 {summ_status} **Summary Score:** {ev.summary_score}/5")
            lines.append("")

        if ev.summary_reasoning:
            lines.append(f"**Reasoning:** {ev.summary_reasoning}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate E2E scenarios")
    parser.add_argument("scenarios", nargs="*", help="Scenario names to evaluate")
    parser.add_argument("--pattern", "-p", help="Glob pattern to match scenario names")
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip TTS generation, use cache only"
    )
    parser.add_argument("--output", "-o", help="Output JSON report to file")
    parser.add_argument("--markdown", "-m", help="Output detailed markdown report to file")
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
    results_map = {}  # Store raw results for markdown report

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
        print("  [main] Stream complete", flush=True)
        print(f"  Segments: {len(results['segments'])}", flush=True)
        print(f"  Translations: {len(results['translations'])}", flush=True)
        print(f"  Summary: {'Yes' if results['summary'] else 'No'}", flush=True)

        if results["errors"]:
            print(f"  Errors: {results['errors']}", flush=True)

        # Store results for markdown report
        results_map[scenario.name] = results

        # Evaluate transcription/translation with CER/BLEU (no LLM needed)
        print("Evaluating with CER/BLEU...", flush=True)
        evaluation = await evaluate_scenario(scenario, results, judge)
        evaluations.append(evaluation)

        print(f"  Transcription: {evaluation.avg_transcription_score:.1f}", flush=True)
        print(f"  Translation: {evaluation.avg_translation_score:.1f}", flush=True)

    # Batch evaluate all summaries with a single LLM call
    await batch_evaluate_summaries(evaluations, judge)

    # Print final scores
    for evaluation in evaluations:
        print(f"  {evaluation.name}: Summary={evaluation.summary_score}")

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
                    "summary_reasoning": e.summary_reasoning,
                    "overall_score": e.overall_score,
                    "errors": e.errors,
                    "summary": results_map.get(e.name, {}).get("summary"),
                    "turns": [
                        {
                            "speaker_id": t.speaker_id,
                            "language": t.language,
                            "expected_text": t.expected_text,
                            "actual_text": t.actual_text,
                            "transcription_score": t.transcription_score,
                            "cer": t.cer_value,
                            "expected_translation": t.expected_translation,
                            "actual_translation": t.actual_translation,
                            "translation_score": t.translation_score,
                            "bleu": t.bleu_value,
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

    # Save markdown report if requested
    if args.markdown:
        md_report = generate_markdown_report(evaluations, results_map)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"Markdown report saved to: {args.markdown}")


if __name__ == "__main__":
    asyncio.run(main())

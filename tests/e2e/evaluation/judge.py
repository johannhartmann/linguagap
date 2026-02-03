"""Gemini-based LLM-as-Judge for evaluating system outputs.

Uses Gemini 2.0 Flash for fast, capable evaluation with chain-of-thought reasoning.
"""

import json
import os
from dataclasses import dataclass

from tests.e2e.evaluation.rubrics import get_rubric

# Gemini model for evaluation
JUDGE_MODEL = "gemini-3-pro-preview"


@dataclass
class EvaluationResult:
    """Result of an LLM-as-Judge evaluation.

    Attributes:
        score: Score from 1-5
        reasoning: Chain-of-thought reasoning for the score
        details: Additional evaluation details
    """

    score: int
    reasoning: str
    details: dict | None = None

    def passed(self, min_score: int = 3) -> bool:
        """Check if evaluation passed minimum threshold.

        Args:
            min_score: Minimum passing score (default 3 = Average)

        Returns:
            True if score >= min_score
        """
        return self.score >= min_score


EVALUATION_PROMPT_TEMPLATE = """You are an expert evaluator for a speech-to-text translation system.

{rubric}

## Task
Evaluate the following {eval_type} and provide:
1. Your reasoning (think step by step)
2. A score from 1-5 based on the rubric

{context}

## Output Format
Respond with JSON only:
{{
    "reasoning": "Your step-by-step analysis here...",
    "score": <1-5>
}}
"""


class GeminiJudge:
    """LLM-as-Judge using Gemini for evaluation."""

    def __init__(self, api_key: str | None = None):
        """Initialize the judge.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        from google import genai
        from google.genai import types

        # Set explicit timeout to avoid hanging (known SDK issue)
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=120_000),  # 120s timeout
        )

    async def evaluate_transcription(
        self,
        expected_text: str,
        actual_text: str,
        language: str,
    ) -> EvaluationResult:
        """Evaluate transcription accuracy.

        Args:
            expected_text: Original text that was spoken
            actual_text: Transcribed text from the system
            language: Language code

        Returns:
            EvaluationResult with score and reasoning
        """
        context = f"""
## Expected Text (Original)
Language: {language}
```
{expected_text}
```

## Actual Transcription
```
{actual_text}
```
"""
        return await self._evaluate("transcription", context)

    async def evaluate_translation(
        self,
        source_text: str,
        expected_translation: str,
        actual_translation: str,
        src_lang: str,
        tgt_lang: str,
    ) -> EvaluationResult:
        """Evaluate translation quality.

        Args:
            source_text: Original source text
            expected_translation: Reference translation
            actual_translation: System's translation
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            EvaluationResult with score and reasoning
        """
        context = f"""
## Source Text
Language: {src_lang}
```
{source_text}
```

## Expected Translation
Language: {tgt_lang}
```
{expected_translation}
```

## Actual Translation
```
{actual_translation}
```
"""
        return await self._evaluate("translation", context)

    async def evaluate_summary(
        self,
        conversation_segments: list[dict],
        expected_topics: list[str],
        actual_summary: dict,
        foreign_lang: str,
    ) -> EvaluationResult:
        """Evaluate conversation summary quality.

        Args:
            conversation_segments: List of segment dicts with src, src_lang
            expected_topics: Topics that should be covered
            actual_summary: Summary dict from system
            foreign_lang: The non-German language in the conversation

        Returns:
            EvaluationResult with score and reasoning
        """
        # Format conversation for context
        conv_text = "\n".join(
            f"[{s.get('src_lang', '?')}] {s.get('src', '')}" for s in conversation_segments
        )

        context = f"""
## Conversation (German and {foreign_lang})
```
{conv_text}
```

## Expected Topics to Cover
{json.dumps(expected_topics, ensure_ascii=False)}

## Actual Summary
German:
```
{actual_summary.get("german_summary", "N/A")}
```

Foreign ({foreign_lang}):
```
{actual_summary.get("foreign_summary", "N/A")}
```
"""
        return await self._evaluate("summary", context)

    async def evaluate_language_detection(
        self,
        expected_segments: list[dict],
        actual_segments: list[dict],
    ) -> EvaluationResult:
        """Evaluate language detection accuracy.

        Args:
            expected_segments: Expected segments with language labels
            actual_segments: Actual segments from system

        Returns:
            EvaluationResult with score and reasoning
        """
        # Format for comparison
        expected_langs = [
            f"Segment {i}: {s.get('language', '?')}" for i, s in enumerate(expected_segments)
        ]
        actual_langs = [
            f"Segment {i}: {s.get('src_lang', '?')}" for i, s in enumerate(actual_segments)
        ]

        context = f"""
## Expected Language Labels
{chr(10).join(expected_langs)}

## Actual Language Labels (from system)
{chr(10).join(actual_langs)}
"""
        return await self._evaluate("language_detection", context)

    async def evaluate_speaker_diarization(
        self,
        expected_speakers: list[str],
        actual_speakers: list[str],
        num_expected_speakers: int,
    ) -> EvaluationResult:
        """Evaluate speaker diarization quality.

        Args:
            expected_speakers: Expected speaker IDs per segment
            actual_speakers: Actual speaker IDs from system
            num_expected_speakers: Number of distinct speakers expected

        Returns:
            EvaluationResult with score and reasoning
        """
        context = f"""
## Expected Configuration
Number of speakers: {num_expected_speakers}
Speaker sequence: {expected_speakers}

## Actual Speaker Detection
Speakers detected: {len(set(actual_speakers))}
Speaker sequence: {actual_speakers}
"""
        return await self._evaluate("speaker_diarization", context)

    async def _evaluate(self, eval_type: str, context: str) -> EvaluationResult:
        """Run evaluation with the judge.

        Args:
            eval_type: Type of evaluation
            context: Context-specific content

        Returns:
            EvaluationResult
        """
        rubric = get_rubric(eval_type)
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            rubric=rubric,
            eval_type=eval_type,
            context=context,
        )

        response = await self.client.aio.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
        )

        # Parse JSON response
        if not response.text:
            return EvaluationResult(
                score=1,
                reasoning="Failed to get evaluation response",
                details={"error": "Empty response"},
            )

        response_text = response.text.strip()
        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            data = json.loads(response_text)
            return EvaluationResult(
                score=data.get("score", 1),
                reasoning=data.get("reasoning", "No reasoning provided"),
                details=data,
            )
        except json.JSONDecodeError as e:
            return EvaluationResult(
                score=1,
                reasoning=f"Failed to parse evaluation response: {e}",
                details={"raw_response": response_text},
            )

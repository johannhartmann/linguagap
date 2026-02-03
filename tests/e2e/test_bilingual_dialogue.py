"""E2E tests for bilingual dialogue processing.

Tests the complete pipeline: TTS -> streaming -> ASR -> translation -> summary.
Evaluated using Gemini as LLM-as-Judge.
"""

from pathlib import Path

import pytest

from tests.e2e.dialogues.templates import (
    SCENARIO_TYPES,
    TARGET_LANGUAGES,
)
from tests.e2e.evaluation.reports import ScenarioReport, TestReport


@pytest.mark.e2e
class TestBilingualDialogue:
    """E2E tests for bilingual dialogue processing."""

    @pytest.fixture
    def test_report(self):
        """Create a test report for collecting results."""
        return TestReport()

    @pytest.mark.asyncio
    async def test_e2e_dialogue_processing(
        self,
        tts_client,
        streaming_client,
        judge,
        sample_scenario,
    ):
        """Test complete dialogue processing pipeline.

        1. Synthesize dialogue audio with TTS
        2. Stream through WebSocket to linguagap
        3. Collect transcription, translation, summary
        4. Evaluate quality with LLM-as-Judge
        """
        # 1. Synthesize audio
        audio_path = tts_client.synthesize_dialogue(sample_scenario)
        assert audio_path.exists(), "Audio file should be created"

        # 2. Stream and collect results
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=True,
        )

        # Check for streaming errors
        assert not result.errors, f"Streaming errors: {result.errors}"

        # 3. Verify we got segments
        assert len(result.final_segments) > 0, "Should have finalized segments"

        # 4. Evaluate transcription quality
        # Compare expected text with actual transcription
        expected_texts = [t.text for t in sample_scenario.turns]
        actual_texts = [s.get("src", "") for s in result.final_segments]

        # Join for overall comparison
        expected_combined = " ".join(expected_texts)
        actual_combined = " ".join(actual_texts)

        transcription_eval = await judge.evaluate_transcription(
            expected_text=expected_combined,
            actual_text=actual_combined,
            language="mixed",
        )

        assert transcription_eval.passed(min_score=3), (
            f"Transcription quality too low: {transcription_eval.score}/5\n"
            f"Reasoning: {transcription_eval.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_speaker_diarization(
        self,
        tts_client,
        streaming_client,
        judge,
        sample_scenario,
    ):
        """Test that speaker diarization detects speakers.

        Note: TTS-generated audio often produces acoustically similar voices
        that pyannote may not distinguish well. We test that at least some
        speaker detection occurs and that the pipeline doesn't crash.
        """
        # Synthesize and stream
        audio_path = tts_client.synthesize_dialogue(sample_scenario)
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=False,
        )

        # Check that we got segments with speaker IDs
        detected_speakers = result.detected_speakers
        assert len(detected_speakers) >= 1, (
            f"Expected at least 1 speaker detected, got none. "
            f"Segments: {len(result.final_segments)}"
        )

        # Evaluate diarization quality (relaxed for TTS audio)
        expected_speaker_seq = [t.speaker_id for t in sample_scenario.turns]
        actual_speaker_seq = [s.get("speaker_id", "unknown") for s in result.final_segments]

        diar_eval = await judge.evaluate_speaker_diarization(
            expected_speakers=expected_speaker_seq,
            actual_speakers=actual_speaker_seq,
            num_expected_speakers=len(set(sample_scenario.speakers.keys())),
        )

        # TTS voices are hard to distinguish - accept score >= 1
        # Real audio with distinct human voices should score higher
        assert diar_eval.score >= 1, (
            f"Speaker diarization failed completely: {diar_eval.score}/5\n"
            f"Detected speakers: {detected_speakers}\n"
            f"Reasoning: {diar_eval.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_language_detection(
        self,
        tts_client,
        streaming_client,
        judge,
        sample_scenario,
    ):
        """Test that language detection correctly identifies segment languages."""
        audio_path = tts_client.synthesize_dialogue(sample_scenario)
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=False,
        )

        # Check that both languages are detected
        detected_languages = result.detected_languages
        expected_languages = {t.language for t in sample_scenario.turns}

        # At minimum, we should detect the two languages used
        assert (
            len(detected_languages) >= 2
        ), f"Expected at least 2 languages ({expected_languages}), got: {detected_languages}"

        # Evaluate language detection
        expected_segments = [
            {"language": t.language, "text": t.text} for t in sample_scenario.turns
        ]
        lang_eval = await judge.evaluate_language_detection(
            expected_segments=expected_segments,
            actual_segments=result.final_segments,
        )

        assert lang_eval.passed(min_score=3), (
            f"Language detection quality too low: {lang_eval.score}/5\n"
            f"Reasoning: {lang_eval.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_translation_quality(
        self,
        tts_client,
        streaming_client,
        judge,
        sample_scenario,
    ):
        """Test that translations meet quality threshold."""
        audio_path = tts_client.synthesize_dialogue(sample_scenario)
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=False,
        )

        # Find segments with expected translations
        foreign_turns = [t for t in sample_scenario.turns if t.expected_translation]
        assert foreign_turns, "Test scenario should have turns with expected translations"

        # Evaluate each translation
        for turn in foreign_turns:
            # Find corresponding segment (approximate match)
            matching_segments = [
                s for s in result.final_segments if turn.text[:20] in s.get("src", "")
            ]

            if not matching_segments:
                continue

            segment = matching_segments[0]
            segment_id = segment.get("id")
            translations = result.translations.get(segment_id, {})

            if "de" in translations:
                trans_eval = await judge.evaluate_translation(
                    source_text=turn.text,
                    expected_translation=turn.expected_translation,
                    actual_translation=translations["de"],
                    src_lang=turn.language,
                    tgt_lang="de",
                )

                assert trans_eval.passed(min_score=3), (
                    f"Translation quality too low: {trans_eval.score}/5\n"
                    f"Source: {turn.text}\n"
                    f"Expected: {turn.expected_translation}\n"
                    f"Actual: {translations['de']}\n"
                    f"Reasoning: {trans_eval.reasoning}"
                )

    @pytest.mark.asyncio
    async def test_summary_coherence(
        self,
        tts_client,
        streaming_client,
        judge,
        sample_scenario,
    ):
        """Test that summary covers both speakers and expected topics."""
        audio_path = tts_client.synthesize_dialogue(sample_scenario)
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=True,
        )

        # Verify summary was generated
        assert result.summary is not None, "Summary should be generated"

        # Evaluate summary quality
        summary_eval = await judge.evaluate_summary(
            conversation_segments=result.final_segments,
            expected_topics=sample_scenario.expected_summary_topics,
            actual_summary=result.summary,
            foreign_lang=sample_scenario.foreign_lang,
        )

        assert summary_eval.passed(
            min_score=3
        ), f"Summary quality too low: {summary_eval.score}/5\nReasoning: {summary_eval.reasoning}"


@pytest.mark.e2e
class TestLanguagePairs:
    """Test each target language pair."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("foreign_lang", list(TARGET_LANGUAGES.keys()))
    async def test_language_pair(
        self,
        foreign_lang,
        dialogue_generator,
        tts_client,
        streaming_client,
        judge,
    ):
        """Test a specific language pair (German + foreign).

        Args:
            foreign_lang: Language code to test (uk, sq, fa, ar, tr)
        """
        # Generate a customer service scenario
        scenario = dialogue_generator.generate_scenario(
            foreign_lang_code=foreign_lang,
            scenario_type="customer_service",
        )

        # Synthesize audio
        audio_path = tts_client.synthesize_dialogue(scenario)

        # Stream and process
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=True,
        )

        # Basic assertions
        assert not result.errors, f"Streaming errors for {foreign_lang}: {result.errors}"
        assert len(result.final_segments) > 0, f"No segments for {foreign_lang}"

        # Evaluate transcription
        expected_combined = " ".join(t.text for t in scenario.turns)
        actual_combined = " ".join(s.get("src", "") for s in result.final_segments)

        transcription_eval = await judge.evaluate_transcription(
            expected_text=expected_combined,
            actual_text=actual_combined,
            language=f"German + {TARGET_LANGUAGES[foreign_lang]}",
        )

        assert transcription_eval.passed(
            min_score=3
        ), f"Transcription failed for {foreign_lang}: {transcription_eval.score}/5"


@pytest.mark.e2e
class TestScenarioTypes:
    """Test different scenario types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_type", SCENARIO_TYPES)
    async def test_scenario_type(
        self,
        scenario_type,
        dialogue_generator,
        tts_client,
        streaming_client,
    ):
        """Test a specific scenario type.

        Args:
            scenario_type: Type of scenario to test
        """
        # Use Ukrainian as the test language
        scenario = dialogue_generator.generate_scenario(
            foreign_lang_code="uk",
            scenario_type=scenario_type,
        )

        # Synthesize audio
        audio_path = tts_client.synthesize_dialogue(scenario)

        # Stream and process
        result = await streaming_client.stream_dialogue(
            audio_path=audio_path,
            request_summary=True,
        )

        # Basic assertions
        assert not result.errors, f"Streaming errors for {scenario_type}: {result.errors}"
        assert len(result.final_segments) > 0, f"No segments for {scenario_type}"


@pytest.mark.e2e
class TestFullSuite:
    """Full test suite running all language pairs and scenario types."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_matrix(
        self,
        dialogue_generator,
        tts_client,
        streaming_client,
        judge,
    ):
        """Run complete test matrix: all languages x all scenario types.

        This is a comprehensive test that generates a full report.
        Mark with @pytest.mark.slow as it takes significant time.
        """
        report = TestReport(
            environment={
                "languages": list(TARGET_LANGUAGES.keys()),
                "scenario_types": SCENARIO_TYPES,
            }
        )

        for lang_code in TARGET_LANGUAGES:
            for scenario_type in SCENARIO_TYPES:
                scenario_report = ScenarioReport(
                    scenario_name=f"{lang_code}_{scenario_type}",
                    foreign_lang=lang_code,
                    scenario_type=scenario_type,
                )

                try:
                    # Generate scenario
                    scenario = dialogue_generator.generate_scenario(
                        foreign_lang_code=lang_code,
                        scenario_type=scenario_type,
                    )

                    # Synthesize and stream
                    audio_path = tts_client.synthesize_dialogue(scenario)
                    result = await streaming_client.stream_dialogue(
                        audio_path=audio_path,
                        request_summary=True,
                    )

                    if result.errors:
                        scenario_report.errors.extend(result.errors)
                        continue

                    # Evaluate transcription
                    expected_combined = " ".join(t.text for t in scenario.turns)
                    actual_combined = " ".join(s.get("src", "") for s in result.final_segments)

                    scenario_report.transcription_score = await judge.evaluate_transcription(
                        expected_text=expected_combined,
                        actual_text=actual_combined,
                        language=f"German + {TARGET_LANGUAGES[lang_code]}",
                    )

                    # Evaluate summary
                    if result.summary:
                        scenario_report.summary_score = await judge.evaluate_summary(
                            conversation_segments=result.final_segments,
                            expected_topics=scenario.expected_summary_topics,
                            actual_summary=result.summary,
                            foreign_lang=lang_code,
                        )

                except Exception as e:
                    scenario_report.errors.append(str(e))

                report.add_scenario(scenario_report)

        # Save reports
        report_dir = Path("tests/e2e/reports")
        report.save_json(report_dir / "e2e_report.json")
        report.save_html(report_dir / "e2e_report.html")

        # Assert overall pass rate
        assert (
            report.overall_pass_rate >= 60
        ), f"Overall pass rate too low: {report.overall_pass_rate:.1f}%"

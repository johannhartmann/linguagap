"""Scoring rubrics for LLM-as-Judge evaluation.

Uses 1-5 categorical scale with explicit criteria per level.
Based on best practices from LLM evaluation research.
"""

# Transcription Quality Rubric
TRANSCRIPTION_RUBRIC = """
Rate the transcription quality on a scale of 1-5:

5 - Excellent: Near-perfect transcription with no meaningful errors.
    Minor differences in punctuation or capitalization are acceptable.
    All words are correctly transcribed with proper word boundaries.

4 - Good: Very accurate transcription with only minor errors.
    1-2 minor word substitutions or omissions that don't change meaning.
    Speaker intent and content are fully preserved.

3 - Average: Reasonably accurate with some noticeable errors.
    Several word-level errors but overall meaning is preserved.
    A human reader could understand the intended content.

2 - Poor: Significant errors that impact comprehension.
    Multiple incorrect words, missing phrases, or garbled sections.
    Meaning is partially lost but some content is recognizable.

1 - Very Poor: Severely inaccurate or largely unintelligible.
    Most content is wrong, missing, or nonsensical.
    Little to no correspondence with the original speech.
"""

# Translation Quality Rubric
TRANSLATION_RUBRIC = """
Rate the translation quality on a scale of 1-5:

5 - Excellent: Accurate, fluent, and natural-sounding translation.
    Meaning is fully preserved with appropriate register and tone.
    Reads like it was originally written in the target language.

4 - Good: Accurate translation with minor fluency issues.
    Core meaning is preserved, phrasing may be slightly awkward.
    No significant meaning errors or omissions.

3 - Average: Understandable translation with some issues.
    Main meaning is conveyed but with noticeable errors or awkward phrasing.
    Some nuance or context may be lost.

2 - Poor: Problematic translation with significant errors.
    Meaning is partially distorted or key information is missing.
    Reader may misunderstand the intended message.

1 - Very Poor: Severely flawed or incomprehensible translation.
    Major meaning errors, nonsensical phrases, or completely wrong.
    Does not convey the original message.
"""

# Summary Quality Rubric
SUMMARY_RUBRIC = """
Rate the conversation summary quality on a scale of 1-5:

5 - Excellent: Comprehensive summary covering both speakers' perspectives.
    All key topics and outcomes are captured accurately.
    Balanced representation of the bilingual conversation.

4 - Good: Accurate summary with good coverage.
    Most important points from both speakers are included.
    Minor details may be omitted but nothing crucial.

3 - Average: Adequate summary with some gaps.
    Main topics are covered but some speaker perspectives may be underrepresented.
    Captures the general nature of the conversation.

2 - Poor: Incomplete or imbalanced summary.
    Missing important topics or heavily favoring one speaker.
    Reader gets an incomplete picture of the conversation.

1 - Very Poor: Inaccurate or severely lacking summary.
    Major topics missing, factual errors, or only covers one language.
    Does not reflect what was discussed.
"""

# Language Detection Rubric
LANGUAGE_DETECTION_RUBRIC = """
Rate the language detection accuracy on a scale of 1-5:

5 - Excellent: All segments correctly identified by language.
    Per-segment language labels match the actual spoken language.
    No misattributions.

4 - Good: Nearly all segments correctly identified.
    1-2 minor misattributions on short or ambiguous segments.
    Overall language distribution is accurate.

3 - Average: Mostly correct language identification.
    Several errors but majority of segments are correct.
    Both languages are detected and distinguished.

2 - Poor: Significant language detection errors.
    Many segments mislabeled or one language dominates incorrectly.
    Pattern of consistent misidentification.

1 - Very Poor: Language detection failed.
    Most segments have wrong language labels.
    System could not distinguish between the two languages.
"""

# Speaker Diarization Rubric
SPEAKER_DIARIZATION_RUBRIC = """
Rate the speaker diarization quality on a scale of 1-5:

5 - Excellent: All speaker changes correctly identified.
    Each segment attributed to the correct speaker.
    Clean speaker boundaries without fragmentation.

4 - Good: Very accurate speaker identification.
    1-2 minor errors at speaker boundaries.
    Overall speaker attribution is reliable.

3 - Average: Reasonable speaker diarization.
    Some errors in speaker attribution but pattern is recognizable.
    Both speakers are detected and mostly distinguished.

2 - Poor: Problematic speaker identification.
    Frequent errors in speaker attribution.
    Speakers may be merged or incorrectly split.

1 - Very Poor: Speaker diarization failed.
    Cannot reliably determine who said what.
    Single speaker detected or random attribution.
"""

# Mapping of evaluation types to rubrics
RUBRICS = {
    "transcription": TRANSCRIPTION_RUBRIC,
    "translation": TRANSLATION_RUBRIC,
    "summary": SUMMARY_RUBRIC,
    "language_detection": LANGUAGE_DETECTION_RUBRIC,
    "speaker_diarization": SPEAKER_DIARIZATION_RUBRIC,
}


def get_rubric(eval_type: str) -> str:
    """Get the rubric for an evaluation type.

    Args:
        eval_type: One of "transcription", "translation", "summary",
                   "language_detection", "speaker_diarization"

    Returns:
        Rubric text
    """
    if eval_type not in RUBRICS:
        raise ValueError(f"Unknown evaluation type: {eval_type}")
    return RUBRICS[eval_type]

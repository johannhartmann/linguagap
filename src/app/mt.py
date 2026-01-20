import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MT_MODEL_REPO = os.getenv("MT_MODEL_REPO", "bullerwins/translategemma-27b-it-GGUF")
MT_MODEL_FILE = os.getenv("MT_MODEL_FILE", "translategemma-27b-it-Q4_K_M.gguf")
MT_N_GPU_LAYERS = int(os.getenv("MT_N_GPU_LAYERS", "-1"))  # -1 = all layers on GPU
MT_N_CTX = int(os.getenv("MT_N_CTX", "4096"))

# Language name and code mapping for TranslateGemma prompts
# Format: lang_code -> (full_name, iso_code)
LANG_INFO = {
    "en": ("English", "en"),
    "de": ("German", "de"),
    "fr": ("French", "fr"),
    "es": ("Spanish", "es"),
    "it": ("Italian", "it"),
    "pl": ("Polish", "pl"),
    "ro": ("Romanian", "ro"),
    "hr": ("Croatian", "hr"),
    "bg": ("Bulgarian", "bg"),
    "sq": ("Albanian", "sq"),
    "tr": ("Turkish", "tr"),
    "ru": ("Russian", "ru"),
    "uk": ("Ukrainian", "uk"),
    "hu": ("Hungarian", "hu"),
    "ar": ("Arabic", "ar"),
    "fa": ("Persian", "fa"),
    "sr": ("Serbian", "sr"),
    "zh": ("Chinese", "zh-Hans"),
    "ja": ("Japanese", "ja"),
    "ko": ("Korean", "ko"),
    "pt": ("Portuguese", "pt"),
    "nl": ("Dutch", "nl"),
}

# Keep LANG_NAMES for backward compatibility
LANG_NAMES = {k: v[0] for k, v in LANG_INFO.items()}

_llm: Llama | None = None


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        print(f"Downloading MT model: {MT_MODEL_REPO}/{MT_MODEL_FILE}")
        model_path = hf_hub_download(  # nosec B615
            repo_id=MT_MODEL_REPO,
            filename=MT_MODEL_FILE,
        )
        print(f"Loading MT model from: {model_path}")
        _llm = Llama(
            model_path=model_path,
            n_gpu_layers=MT_N_GPU_LAYERS,
            n_ctx=MT_N_CTX,
            verbose=False,
            use_mmap=False,  # Sequential read - faster on network storage
        )
        print("MT model loaded")
    return _llm


def translate_texts(texts: list[str], src_lang: str, tgt_lang: str = "de") -> list[str]:
    llm = get_llm()
    _, src_code = LANG_INFO.get(src_lang, (src_lang, src_lang))
    _, tgt_code = LANG_INFO.get(tgt_lang, (tgt_lang, tgt_lang))

    results = []
    for text in texts:
        if not text.strip():
            results.append("")
            continue

        # TranslateGemma requires structured content with type, lang codes, and text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_code,
                        "target_lang_code": tgt_code,
                        "text": text,
                    }
                ],
            },
        ]

        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
            top_p=0.9,
        )

        response = output["choices"][0]["message"]["content"].strip()
        results.append(response)

    return results


def summarize_conversation(segments: list[dict], _foreign_lang: str, target_lang: str) -> str:
    """
    Summarize conversation segments in the target language.

    Args:
        segments: List of segment dicts with 'src', 'src_lang', and 'translations'
        _foreign_lang: The non-German language code (unused, kept for API compatibility)
        target_lang: Language to generate summary in ('de' or foreign_lang)

    Returns:
        Summary text in target_lang
    """
    llm = get_llm()
    target_name = LANG_NAMES.get(target_lang, target_lang)

    # Build conversation text with speaker labels
    conversation_lines = []
    for seg in segments:
        speaker = "German speaker" if seg["src_lang"] == "de" else "Foreign speaker"
        # Get text in target language
        if seg["src_lang"] == target_lang:
            text = seg["src"]
        else:
            translations = seg.get("translations", {})
            text = translations.get(target_lang, seg["src"])
        conversation_lines.append(f"{speaker}: {text}")

    conversation_text = "\n".join(conversation_lines)

    # TranslateGemma-style prompt for summarization
    prompt = (
        f"You are a professional summarizer. Create a concise summary of the following conversation "
        f"in {target_name}. Capture the key points and main topics discussed. "
        f"Output only the summary in 2-4 sentences."
        f"\n\n\n{conversation_text}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()
    return response


def validate_summary_alignment(
    german_summary: str,
    original_german_segments: list[str],
) -> dict:
    """
    Validate that the German summary aligns with original German segments.

    Args:
        german_summary: The generated German summary
        original_german_segments: List of original German text from conversation

    Returns:
        Dict with 'aligned' (bool), 'issues' (str or None), 'feedback' (str)
    """
    llm = get_llm()

    original_text = "\n".join(original_german_segments)

    prompt = (
        "You are a quality checker for translations and summaries. "
        "Compare the German summary against the original German conversation segments below. "
        "Check if key information is preserved and there are no mistranslations or missing points. "
        "Respond in this exact format:\n"
        "ALIGNED: yes or no\n"
        "ISSUES: description of issues (or 'none')\n"
        "FEEDBACK: specific suggestions for improvement (or 'none')"
        f"\n\n\nOriginal German segments:\n{original_text}\n\n"
        f"German summary to validate:\n{german_summary}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.3,
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()

    # Parse response
    aligned = "ALIGNED: yes" in response.lower() or "aligned: yes" in response.lower()
    issues = None
    feedback = None

    lines = response.split("\n")
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("issues:"):
            issues_text = line.split(":", 1)[1].strip()
            if issues_text.lower() != "none":
                issues = issues_text
        elif line_lower.startswith("feedback:"):
            feedback_text = line.split(":", 1)[1].strip()
            if feedback_text.lower() != "none":
                feedback = feedback_text

    return {
        "aligned": aligned,
        "issues": issues,
        "feedback": feedback,
    }


def regenerate_summary_with_feedback(
    segments: list[dict],
    _foreign_lang: str,
    target_lang: str,
    previous_issues: str,
    previous_feedback: str,
) -> str:
    """
    Regenerate summary with knowledge of previous mistakes.

    Args:
        segments: List of segment dicts
        _foreign_lang: The non-German language code (unused, kept for API compatibility)
        target_lang: Language to generate summary in
        previous_issues: Issues from validation
        previous_feedback: Feedback from validation

    Returns:
        Improved summary text
    """
    llm = get_llm()
    target_name = LANG_NAMES.get(target_lang, target_lang)

    # Build conversation text
    conversation_lines = []
    for seg in segments:
        speaker = "German speaker" if seg["src_lang"] == "de" else "Foreign speaker"
        if seg["src_lang"] == target_lang:
            text = seg["src"]
        else:
            translations = seg.get("translations", {})
            text = translations.get(target_lang, seg["src"])
        conversation_lines.append(f"{speaker}: {text}")

    conversation_text = "\n".join(conversation_lines)

    # Include feedback in prompt
    feedback_section = ""
    if previous_issues:
        feedback_section += f" Previous attempt had these issues: {previous_issues}."
    if previous_feedback:
        feedback_section += f" Please address: {previous_feedback}."

    prompt = (
        f"You are a professional summarizer. Create a concise summary of the following conversation "
        f"in {target_name}. Capture the key points and main topics discussed. "
        f"Output only the summary in 2-4 sentences.{feedback_section}"
        f"\n\n\n{conversation_text}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.4,  # Slightly lower for more focused regeneration
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()
    return response

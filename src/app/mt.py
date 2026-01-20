import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MT_MODEL_REPO = os.getenv("MT_MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
MT_MODEL_FILE = os.getenv("MT_MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
MT_N_GPU_LAYERS = int(os.getenv("MT_N_GPU_LAYERS", "-1"))  # -1 = all layers on GPU
MT_N_CTX = int(os.getenv("MT_N_CTX", "2048"))

# Language name mapping for prompts
LANG_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pl": "Polish",
    "ro": "Romanian",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "sq": "Albanian",
    "tr": "Turkish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "hu": "Hungarian",
    "ar": "Arabic",
    "fa": "Persian",
    "sr": "Serbian",
}

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
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)
    src_name = LANG_NAMES.get(src_lang, src_lang)

    results = []
    for text in texts:
        if not text.strip():
            results.append("")
            continue

        # Use chat completion with /no_think to disable thinking mode
        # /no_think must be at the END of the user message
        messages = [
            {
                "role": "system",
                "content": f"You are a translator. Translate {src_name} to {tgt_name}. Output only the translation.",
            },
            {
                "role": "user",
                "content": f"{text} /no_think",
            },
        ]

        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
            top_p=0.9,
        )

        response = output["choices"][0]["message"]["content"].strip()

        # Clean up any residual thinking tags
        if "<think>" in response:
            response = response.split("</think>")[-1].strip() if "</think>" in response else ""

        results.append(response)

    return results


def summarize_conversation(segments: list[dict], _foreign_lang: str, target_lang: str) -> str:
    """
    Summarize conversation segments in the target language.

    Args:
        segments: List of segment dicts with 'src', 'src_lang', and 'translations'
        foreign_lang: The non-German language code (e.g., 'en')
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

    messages = [
        {
            "role": "system",
            "content": f"You are a summarizer. Create a concise summary of the conversation in {target_name}. "
            f"Capture the key points and main topics discussed. Output only the summary, 2-4 sentences.",
        },
        {
            "role": "user",
            "content": f"Summarize this conversation:\n\n{conversation_text} /no_think",
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()

    # Clean up any residual thinking tags
    if "<think>" in response:
        response = response.split("</think>")[-1].strip() if "</think>" in response else ""

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

    messages = [
        {
            "role": "system",
            "content": "You are a quality checker for translations and summaries. "
            "Compare a German summary against original German conversation segments. "
            "Check if key information is preserved and there are no mistranslations or missing points. "
            "Respond in this exact format:\n"
            "ALIGNED: yes or no\n"
            "ISSUES: description of issues (or 'none')\n"
            "FEEDBACK: specific suggestions for improvement (or 'none')",
        },
        {
            "role": "user",
            "content": f"Original German segments from conversation:\n{original_text}\n\n"
            f"German summary to validate:\n{german_summary} /no_think",
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.3,
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()

    # Clean up any residual thinking tags
    if "<think>" in response:
        response = response.split("</think>")[-1].strip() if "</think>" in response else ""

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
        foreign_lang: The non-German language code
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
        feedback_section += f"\n\nPrevious attempt had these issues: {previous_issues}"
    if previous_feedback:
        feedback_section += f"\nPlease address: {previous_feedback}"

    messages = [
        {
            "role": "system",
            "content": f"You are a summarizer. Create a concise summary of the conversation in {target_name}. "
            f"Capture the key points and main topics discussed. Output only the summary, 2-4 sentences."
            f"{feedback_section}",
        },
        {
            "role": "user",
            "content": f"Summarize this conversation:\n\n{conversation_text} /no_think",
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.4,  # Slightly lower for more focused regeneration
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()

    if "<think>" in response:
        response = response.split("</think>")[-1].strip() if "</think>" in response else ""

    return response

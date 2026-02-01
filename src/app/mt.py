import os
import re

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MT_MODEL_REPO = os.getenv("MT_MODEL_REPO", "bullerwins/translategemma-12b-it-GGUF")
MT_MODEL_FILE = os.getenv("MT_MODEL_FILE", "translategemma-12b-it-Q4_K_M.gguf")
MT_N_GPU_LAYERS = int(os.getenv("MT_N_GPU_LAYERS", "-1"))  # -1 = all layers on GPU
MT_N_CTX = int(os.getenv("MT_N_CTX", "4096"))

# Summarization model configuration (Qwen3-4B - general purpose)
SUMM_MODEL_REPO = os.getenv("SUMM_MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
SUMM_MODEL_FILE = os.getenv("SUMM_MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
SUMM_N_GPU_LAYERS = int(os.getenv("SUMM_N_GPU_LAYERS", "-1"))
SUMM_N_CTX = int(os.getenv("SUMM_N_CTX", "4096"))

# Regex to strip Qwen3 <think>...</think> blocks from responses
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Also match incomplete/truncated think blocks (no closing tag)
_INCOMPLETE_THINK_PATTERN = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think_block(text: str) -> str:
    """Strip Qwen3 thinking blocks from model output, including truncated ones."""
    original = text
    text = _THINK_PATTERN.sub("", text)
    text = _INCOMPLETE_THINK_PATTERN.sub("", text)
    result = text.strip()
    # If stripping left us with nothing, return original without think tags
    if not result:
        # Try to extract any content before <think>
        before_think = original.split("<think>")[0].strip()
        if before_think:
            return before_think
        # Last resort: return a placeholder
        return "(Summary generation incomplete)"
    return result


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
_summ_llm: Llama | None = None


def get_llm() -> Llama:
    """Get the translation LLM (TranslateGemma - specialized for translation)."""
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


def get_summ_llm() -> Llama:
    """Get the summarization LLM (Qwen3-4B - general purpose model)."""
    global _summ_llm
    if _summ_llm is None:
        print(f"Downloading summarization model: {SUMM_MODEL_REPO}/{SUMM_MODEL_FILE}")
        model_path = hf_hub_download(  # nosec B615
            repo_id=SUMM_MODEL_REPO,
            filename=SUMM_MODEL_FILE,
        )
        print(f"Loading summarization model from: {model_path}")
        _summ_llm = Llama(
            model_path=model_path,
            n_gpu_layers=SUMM_N_GPU_LAYERS,
            n_ctx=SUMM_N_CTX,
            n_batch=512,
            verbose=False,
            use_mmap=False,
        )
        print("Summarization model loaded")
    return _summ_llm


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


def summarize_bilingual(segments: list[dict], foreign_lang: str) -> tuple[str, str]:
    """
    Generate both foreign and German summaries in a single LLM call.

    Args:
        segments: List of segment dicts with 'src', 'src_lang', and 'translations'
        foreign_lang: The non-German language code

    Returns:
        Tuple of (foreign_summary, german_summary)
    """
    llm = get_summ_llm()
    foreign_name = LANG_NAMES.get(foreign_lang, foreign_lang)

    # Build conversation with original text (each line in its original language)
    conversation_lines = []
    for seg in segments:
        speaker = "German speaker" if seg["src_lang"] == "de" else "Foreign speaker"
        lang_label = "German" if seg["src_lang"] == "de" else foreign_name
        conversation_lines.append(f"{speaker} ({lang_label}): {seg['src']}")

    conversation_text = "\n".join(conversation_lines)

    prompt = f"""Summarize this bilingual dialogue. Generate TWO summaries:

1. A summary in {foreign_name} (2-3 sentences)
2. The same summary translated to German (2-3 sentences)

Both summaries must cover what BOTH speakers said.

Conversation:
{conversation_text}

Respond in this exact format:
{foreign_name.upper()}: [summary in {foreign_name}]
GERMAN: [same summary in German]"""

    messages = [{"role": "user", "content": prompt}]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.5,
        top_p=0.9,
    )

    response = _strip_think_block(output["choices"][0]["message"]["content"].strip())

    # Parse the response
    foreign_summary = ""
    german_summary = ""

    lines = response.split("\n")
    current_section = None

    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith(foreign_name.upper() + ":"):
            current_section = "foreign"
            foreign_summary = line.split(":", 1)[1].strip() if ":" in line else ""
        elif line_upper.startswith("GERMAN:") or line_upper.startswith("DEUTSCH:"):
            current_section = "german"
            german_summary = line.split(":", 1)[1].strip() if ":" in line else ""
        elif current_section == "foreign" and line.strip():
            foreign_summary += " " + line.strip()
        elif current_section == "german" and line.strip():
            german_summary += " " + line.strip()

    # Fallback if parsing failed
    if not foreign_summary or not german_summary:
        # Just use the whole response as both
        foreign_summary = foreign_summary or response
        german_summary = german_summary or response

    return foreign_summary.strip(), german_summary.strip()


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
    llm = get_summ_llm()

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
        max_tokens=4096,
        temperature=0.3,
        top_p=0.9,
    )

    response = _strip_think_block(output["choices"][0]["message"]["content"])

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
    llm = get_summ_llm()
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
        f"Summarize this dialogue between two speakers in {target_name}. "
        f"Include what BOTH the German speaker and the Foreign speaker said. "
        f"Write 2-4 sentences covering the main topics from both sides.{feedback_section}"
        f"\n\nConversation:\n{conversation_text}\n\nSummary:"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=4096,
        temperature=0.4,  # Slightly lower for more focused regeneration
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()
    return _strip_think_block(response)

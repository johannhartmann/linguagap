"""Qwen3-4B summarization backend using llama-cpp-python.

Generates dual-language summaries (foreign + German) of bilingual conversations.
Uses structured prompts with LANGUAGE: [summary] format.

Qwen3 uses <think>...</think> blocks for chain-of-thought reasoning which are
stripped from the final output.
"""

from __future__ import annotations

import os
import re

from app.backends.base import SummarizationBackend

SUMM_MODEL_REPO = os.getenv("SUMM_MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
SUMM_MODEL_FILE = os.getenv("SUMM_MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
SUMM_N_GPU_LAYERS = int(os.getenv("SUMM_N_GPU_LAYERS", "-1"))
SUMM_N_CTX = int(os.getenv("SUMM_N_CTX", "4096"))

# Regex to strip Qwen3 <think>...</think> blocks from responses
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_INCOMPLETE_THINK_PATTERN = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think_block(text: str) -> str:
    """Strip Qwen3 thinking blocks from model output, including truncated ones."""
    original = text
    text = _THINK_PATTERN.sub("", text)
    text = _INCOMPLETE_THINK_PATTERN.sub("", text)
    result = text.strip()
    if not result:
        before_think = original.split("<think>")[0].strip()
        if before_think:
            return before_think
        return "(Summary generation incomplete)"
    return result


class Qwen3SummarizationBackend(SummarizationBackend):
    """Qwen3-4B summarization backend via llama-cpp-python."""

    def __init__(self) -> None:
        self._llm = None

    def load_model(self) -> None:
        if self._llm is not None:
            return
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        print(f"Downloading summarization model: {SUMM_MODEL_REPO}/{SUMM_MODEL_FILE}")
        model_path = hf_hub_download(  # nosec B615
            repo_id=SUMM_MODEL_REPO,
            filename=SUMM_MODEL_FILE,
        )
        print(f"Loading summarization model from: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=SUMM_N_GPU_LAYERS,
            n_ctx=SUMM_N_CTX,
            n_batch=512,
            verbose=False,
            use_mmap=False,
        )
        print("Summarization model loaded")

    def warmup(self) -> None:
        self.load_model()
        # Quick warmup: generate a short completion
        self._llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
        )
        print("  Summarization warmup complete")

    def summarize_bilingual(
        self,
        segments: list[dict],
        foreign_lang: str,
    ) -> tuple[str, str]:
        """Generate both foreign and German summaries in a single LLM call."""
        from app.mt import LANG_NAMES

        if foreign_lang not in LANG_NAMES:
            print(f"Warning: Unsupported language '{foreign_lang}' for summary, using 'English'")
            foreign_lang = "en"

        self.load_model()
        foreign_name = LANG_NAMES.get(foreign_lang, foreign_lang)

        # Build conversation with original text
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

        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.5,
            top_p=0.9,
        )

        response = _strip_think_block(output["choices"][0]["message"]["content"].strip())

        # Parse the response
        foreign_summary = ""
        german_summary = ""
        current_section = None

        for line in response.split("\n"):
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
            foreign_summary = foreign_summary or response
            german_summary = german_summary or response

        return foreign_summary.strip(), german_summary.strip()

"""
Machine translation using GGUF models via llama-cpp-python.

This module provides translation via TranslateGemma 12B, a specialized
translation model loaded as a lazy singleton from HuggingFace Hub.

TranslateGemma prompt format:
    Uses structured content with source/target language codes:
    {"type": "text", "source_lang_code": "en", "target_lang_code": "de", "text": "..."}
"""

import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MT_MODEL_REPO = os.getenv("MT_MODEL_REPO", "bullerwins/translategemma-12b-it-GGUF")
MT_MODEL_FILE = os.getenv("MT_MODEL_FILE", "translategemma-12b-it-Q4_K_M.gguf")
MT_N_GPU_LAYERS = int(os.getenv("MT_N_GPU_LAYERS", "-1"))  # -1 = all layers on GPU
MT_N_CTX = int(os.getenv("MT_N_CTX", "4096"))

# Language name and code mapping for TranslateGemma prompts
# Format: lang_code -> (full_name, iso_code)
# The iso_code is used for TranslateGemma (BCP-47 style codes)
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
    "ku": ("Kurdish", "ku"),  # Added for bilingual support
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


def translate_texts(texts: list[str], src_lang: str, tgt_lang: str = "de") -> list[str]:
    """
    Translate a list of texts using TranslateGemma.

    TranslateGemma requires a specific message format with structured content
    containing source/target language codes. This is different from typical
    instruction-following models.

    Args:
        texts: List of texts to translate
        src_lang: Source language code (e.g., "en", "bg")
        tgt_lang: Target language code (default "de" for German)

    Returns:
        List of translated texts in the same order as input
    """
    # Check for unsupported languages - fallback to English for unknown codes
    if src_lang not in LANG_INFO:
        print(f"Warning: Unsupported source language '{src_lang}', falling back to 'en'")
        src_lang = "en"
    if tgt_lang not in LANG_INFO:
        print(f"Warning: Unsupported target language '{tgt_lang}', falling back to 'en'")
        tgt_lang = "en"

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

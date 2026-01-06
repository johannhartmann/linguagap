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
        )
        print("MT model loaded")
    return _llm


def translate_texts(texts: list[str], src_lang: str, tgt_lang: str = "de") -> list[str]:
    llm = get_llm()

    src_name = LANG_NAMES.get(src_lang, src_lang)
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

    results = []
    for text in texts:
        if not text.strip():
            results.append("")
            continue

        # Use chat completion with system prompt for better results
        messages = [
            {
                "role": "system",
                "content": f"You are a translator. Translate the user's {src_name} text to {tgt_name}. Output only the translation, nothing else. Do not explain, do not add notes.",
            },
            {"role": "user", "content": text},
        ]

        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
        )

        response = output["choices"][0]["message"]["content"].strip()

        # Remove thinking tags if present
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        results.append(response)

    return results

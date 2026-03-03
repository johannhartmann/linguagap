"""TranslateGemma translation backend using llama-cpp-python.

TranslateGemma is a specialized translation model that uses structured content
messages with source/target language codes (BCP-47 style).
"""

from __future__ import annotations

import logging
import os

from app.backends.base import TranslationBackend

logger = logging.getLogger(__name__)

MT_MODEL_REPO = os.getenv("MT_MODEL_REPO", "bullerwins/translategemma-12b-it-GGUF")
MT_MODEL_FILE = os.getenv("MT_MODEL_FILE", "translategemma-12b-it-Q4_K_M.gguf")
MT_N_GPU_LAYERS = int(os.getenv("MT_N_GPU_LAYERS", "-1"))
MT_N_CTX = int(os.getenv("MT_N_CTX", "4096"))


class TranslateGemmaBackend(TranslationBackend):
    """TranslateGemma 12B translation backend via llama-cpp-python."""

    def __init__(self) -> None:
        self._llm = None

    def load_model(self) -> None:
        if self._llm is not None:
            return
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        logger.info("Downloading MT model: %s/%s", MT_MODEL_REPO, MT_MODEL_FILE)
        model_path = hf_hub_download(  # nosec B615
            repo_id=MT_MODEL_REPO,
            filename=MT_MODEL_FILE,
        )
        logger.info("Loading MT model from: %s", model_path)
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=MT_N_GPU_LAYERS,
            n_ctx=MT_N_CTX,
            verbose=False,
            use_mmap=False,  # Sequential read - faster on network storage
        )
        logger.info("MT model loaded")

    def warmup(self) -> None:
        self.load_model()
        self.translate(["Hello"], src_lang="en", tgt_lang="de")
        logger.info("  MT warmup complete")

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        from app.languages import LANG_INFO

        # Check for unsupported languages - fallback to English for unknown codes
        if src_lang not in LANG_INFO:
            logger.warning("Unsupported source language '%s', falling back to 'en'", src_lang)
            src_lang = "en"
        if tgt_lang not in LANG_INFO:
            logger.warning("Unsupported target language '%s', falling back to 'en'", tgt_lang)
            tgt_lang = "en"

        self.load_model()
        _, src_code = LANG_INFO[src_lang]
        _, tgt_code = LANG_INFO[tgt_lang]

        results = []
        for text in texts:
            if not text.strip():
                results.append("")
                continue

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

            output = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.3,
                top_p=0.9,
            )

            response = output["choices"][0]["message"]["content"].strip()
            results.append(response)

        return results

    def supports_language_pair(self, src: str, tgt: str) -> bool:
        from app.languages import LANG_INFO

        return src in LANG_INFO and tgt in LANG_INFO

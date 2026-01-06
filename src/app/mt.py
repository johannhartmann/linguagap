import os

import ctranslate2
from huggingface_hub import snapshot_download
from transformers import M2M100Tokenizer

MT_MODEL = os.getenv("MT_MODEL", "michaelfeil/ct2fast-m2m100_418M")
MT_DEVICE = os.getenv("MT_DEVICE", "cuda")
MT_COMPUTE_TYPE = os.getenv("MT_COMPUTE_TYPE", "int8_float16")

_translator: ctranslate2.Translator | None = None
_tokenizer: M2M100Tokenizer | None = None


def get_translator() -> ctranslate2.Translator:
    global _translator
    if _translator is None:
        model_path = snapshot_download(MT_MODEL)
        _translator = ctranslate2.Translator(
            model_path,
            device=MT_DEVICE,
            compute_type=MT_COMPUTE_TYPE,
        )
    return _translator


def get_tokenizer() -> M2M100Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return _tokenizer


def translate_texts(texts: list[str], src_lang: str, tgt_lang: str = "de") -> list[str]:
    translator = get_translator()
    tokenizer = get_tokenizer()

    tokenizer.src_lang = src_lang
    target_prefix = [[tokenizer.lang_code_to_token[tgt_lang]]]

    results = []
    for text in texts:
        source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        output = translator.translate_batch(
            [source_tokens],
            target_prefix=target_prefix,
        )
        translated_tokens = output[0].hypotheses[0][1:]
        translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(translated_tokens))
        results.append(translated_text)

    return results

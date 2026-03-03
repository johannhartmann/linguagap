import logging

from app.mt import translate_texts

logger = logging.getLogger(__name__)


def main():
    texts = ["Hello world!"]
    result = translate_texts(texts, src_lang="en", tgt_lang="de")

    logger.info("Input: %s", texts)
    logger.info("Output: %s", result)

    assert len(result) > 0
    assert result[0]
    logger.info("MT smoke test passed!")


if __name__ == "__main__":
    main()

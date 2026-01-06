from app.mt import translate_texts


def main():
    texts = ["Hello world!"]
    result = translate_texts(texts, src_lang="en", tgt_lang="de")

    print(f"Input: {texts}")
    print(f"Output: {result}")

    assert len(result) > 0
    assert result[0]
    print("\nMT smoke test passed!")


if __name__ == "__main__":
    main()

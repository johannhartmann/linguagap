import tempfile

import httpx

from app.scripts.asr_smoke import generate_silence_wav


def main():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    generate_silence_wav(wav_path, duration_sec=2.0)

    with open(wav_path, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = httpx.post(
            "http://localhost:8000/transcribe_translate",
            files=files,
            timeout=60.0,
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    result = response.json()
    print(f"Response: {result}")

    assert "src_lang_detected" in result
    assert "segments" in result
    assert isinstance(result["segments"], list)

    print("\nPipeline smoke test passed!")


if __name__ == "__main__":
    main()

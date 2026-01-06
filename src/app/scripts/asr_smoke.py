import json
import struct
import tempfile
import wave

import numpy as np

from app.asr import transcribe_wav_path


def generate_silence_wav(path: str, duration_sec: float = 2.0, sample_rate: int = 16000):
    num_samples = int(duration_sec * sample_rate)
    samples = np.zeros(num_samples, dtype=np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def main():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    generate_silence_wav(wav_path, duration_sec=2.0)
    result = transcribe_wav_path(wav_path)

    print(json.dumps(result, indent=2))

    assert "language" in result
    assert "segments" in result
    print("\nASR smoke test passed!")


if __name__ == "__main__":
    main()

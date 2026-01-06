import asyncio
import json

import numpy as np
import websockets


async def main():
    uri = "ws://localhost:8000/ws"
    sample_rate = 16000
    frame_duration_ms = 100
    total_duration_sec = 10

    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    total_frames = int(total_duration_sec * 1000 / frame_duration_ms)

    print(f"Connecting to {uri}")
    print(f"Streaming {total_duration_sec}s of silence at {sample_rate}Hz")
    print(f"Frame size: {samples_per_frame} samples ({frame_duration_ms}ms)")
    print()

    messages_received = []

    async with websockets.connect(uri) as ws:
        config = {
            "type": "config",
            "sample_rate": sample_rate,
            "src_lang": "auto",
        }
        await ws.send(json.dumps(config))
        print(f"Sent config: {config}")
        print()

        async def receive_messages():
            try:
                async for message in ws:
                    data = json.loads(message)
                    messages_received.append(data)
                    asr_text = data.get("asr_text", "")
                    de_text = data.get("de_text", "")
                    if len(asr_text) > 50:
                        asr_text = asr_text[:50] + "..."
                    if len(de_text) > 50:
                        de_text = de_text[:50] + "..."
                    print(f"Received: t={data['t']:.2f}s, lang={data['src_lang']}, "
                          f"asr='{asr_text}', de='{de_text}'")
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"Receive error: {e}")

        receiver = asyncio.create_task(receive_messages())

        for i in range(total_frames):
            silence = np.zeros(samples_per_frame, dtype=np.int16)
            await ws.send(silence.tobytes())
            await asyncio.sleep(frame_duration_ms / 1000)

            if (i + 1) % 10 == 0:
                print(f"Sent {(i + 1) * frame_duration_ms / 1000:.1f}s of audio")

        print("\nFinished streaming, waiting for final responses...")
        await asyncio.sleep(10)

        receiver.cancel()
        try:
            await receiver
        except asyncio.CancelledError:
            pass

    print(f"\nTotal messages received: {len(messages_received)}")
    print("Stream client test completed!")


if __name__ == "__main__":
    asyncio.run(main())

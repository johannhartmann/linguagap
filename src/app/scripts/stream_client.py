import asyncio
import contextlib
import json

import numpy as np
import websockets


async def main():
    uri = "ws://localhost:8000/ws"
    sample_rate = 16000
    frame_duration_ms = 100
    total_duration_sec = 60

    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    total_frames = int(total_duration_sec * 1000 / frame_duration_ms)

    print(f"Connecting to {uri}")
    print(f"Streaming {total_duration_sec}s of silence at {sample_rate}Hz")
    print(f"Frame size: {samples_per_frame} samples ({frame_duration_ms}ms)")
    print()

    messages_received = []
    seen_segment_ids = set()

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

                    if data.get("type") == "segments":
                        segments = data.get("segments", [])
                        print(
                            f"t={data['t']:.2f}s, lang={data['src_lang']}, {len(segments)} segment(s):"
                        )
                        for seg in segments:
                            seg_id = seg["id"]
                            final = seg["final"]
                            src = seg["src"][:30] + "..." if len(seg["src"]) > 30 else seg["src"]
                            de = seg["de"][:30] + "..." if len(seg["de"]) > 30 else seg["de"]
                            status = "FINAL" if final else "LIVE"
                            print(f"  [{seg_id}] {status}: '{src}' -> '{de}'")
                            seen_segment_ids.add(seg_id)
                    else:
                        print(f"Received: {data}")

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
                print(f"--- Sent {(i + 1) * frame_duration_ms / 1000:.1f}s of audio ---")

        print("\nFinished streaming, waiting for final responses...")
        await asyncio.sleep(10)

        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver

    print(f"\nTotal messages received: {len(messages_received)}")
    print(f"Unique segment IDs seen: {sorted(seen_segment_ids)}")
    print("Stream client test completed!")


if __name__ == "__main__":
    asyncio.run(main())

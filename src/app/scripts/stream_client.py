import asyncio
import contextlib
import json
import logging

import numpy as np
import websockets

logger = logging.getLogger(__name__)


async def main():
    uri = "ws://localhost:8000/ws"
    sample_rate = 16000
    frame_duration_ms = 100
    total_duration_sec = 60

    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    total_frames = int(total_duration_sec * 1000 / frame_duration_ms)

    logger.info("Connecting to %s", uri)
    logger.info("Streaming %ds of silence at %dHz", total_duration_sec, sample_rate)
    logger.info("Frame size: %d samples (%dms)", samples_per_frame, frame_duration_ms)

    messages_received = []
    seen_segment_ids = set()

    async with websockets.connect(uri) as ws:
        config = {
            "type": "config",
            "sample_rate": sample_rate,
            "src_lang": "auto",
        }
        await ws.send(json.dumps(config))
        logger.info("Sent config: %s", config)

        async def receive_messages():
            try:
                async for message in ws:
                    data = json.loads(message)
                    messages_received.append(data)

                    if data.get("type") == "segments":
                        segments = data.get("segments", [])
                        logger.debug(
                            "t=%.2fs, lang=%s, %d segment(s):",
                            data["t"],
                            data["src_lang"],
                            len(segments),
                        )
                        for seg in segments:
                            seg_id = seg["id"]
                            final = seg["final"]
                            src = seg["src"][:30] + "..." if len(seg["src"]) > 30 else seg["src"]
                            de = seg["de"][:30] + "..." if len(seg["de"]) > 30 else seg["de"]
                            status = "FINAL" if final else "LIVE"
                            logger.debug("  [%s] %s: '%s' -> '%s'", seg_id, status, src, de)
                            seen_segment_ids.add(seg_id)
                    else:
                        logger.debug("Received: %s", data)

            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.error("Receive error: %s", e)

        receiver = asyncio.create_task(receive_messages())

        for i in range(total_frames):
            silence = np.zeros(samples_per_frame, dtype=np.int16)
            await ws.send(silence.tobytes())
            await asyncio.sleep(frame_duration_ms / 1000)

            if (i + 1) % 10 == 0:
                logger.debug("--- Sent %.1fs of audio ---", (i + 1) * frame_duration_ms / 1000)

        logger.info("Finished streaming, waiting for final responses...")
        await asyncio.sleep(10)

        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver

    logger.info("Total messages received: %d", len(messages_received))
    logger.info("Unique segment IDs seen: %s", sorted(seen_segment_ids))
    logger.info("Stream client test completed!")


if __name__ == "__main__":
    asyncio.run(main())

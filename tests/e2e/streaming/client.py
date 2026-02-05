"""WebSocket streaming client for E2E tests.

Streams audio to linguagap WebSocket API and collects results.
"""

import asyncio
import contextlib
import json
import os
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StreamingResult:
    """Results collected from a streaming session.

    Attributes:
        segments: List of transcription segments
        translations: Dict mapping segment_id to translations
        summary: Summary result if requested
        errors: List of error messages
        duration_sec: Total audio duration streamed
    """

    segments: list[dict] = field(default_factory=list)
    translations: dict[int, dict[str, str]] = field(default_factory=dict)
    summary: dict | None = None
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def final_segments(self) -> list[dict]:
        """Get only finalized segments."""
        return [s for s in self.segments if s.get("final")]

    @property
    def detected_languages(self) -> set[str]:
        """Get unique languages detected in segments."""
        return {s.get("src_lang", "unknown") for s in self.segments}

    @property
    def detected_speakers(self) -> set[str]:
        """Get unique speaker IDs detected in segments."""
        return {s["speaker_id"] for s in self.segments if s.get("speaker_id")}


class StreamingClient:
    """Client for streaming audio to linguagap WebSocket API."""

    def __init__(
        self,
        ws_url: str | None = None,
        realtime_factor: float = 0.5,
        sample_rate: int = 16000,
    ):
        """Initialize the streaming client.

        Args:
            ws_url: WebSocket URL. If None, uses LINGUAGAP_WS_URL env var.
            realtime_factor: Speed factor (0.5 = 2x faster than realtime)
            sample_rate: Audio sample rate (default 16kHz)
        """
        self.ws_url = ws_url or os.getenv("LINGUAGAP_WS_URL", "ws://localhost:8000/ws")
        self.realtime_factor = realtime_factor
        self.sample_rate = sample_rate

    async def stream_audio_file(
        self,
        audio_path: str | Path,
        src_lang: str = "auto",
        request_summary: bool = True,
        timeout_sec: float = 300.0,
    ) -> StreamingResult:
        """Stream an audio file through the WebSocket API.

        Args:
            audio_path: Path to WAV audio file
            src_lang: Source language hint ("auto" for detection)
            request_summary: Whether to request a summary at the end
            timeout_sec: Maximum time to wait for processing

        Returns:
            StreamingResult with collected data
        """
        import websockets

        result = StreamingResult()

        # Read audio file
        audio_path = Path(audio_path)
        with wave.open(str(audio_path), "rb") as wav:
            file_sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)
            result.duration_sec = n_frames / file_sample_rate

        # Calculate frame parameters
        frame_duration_ms = 100
        samples_per_frame = int(self.sample_rate * frame_duration_ms / 1000)
        bytes_per_frame = samples_per_frame * 2  # 16-bit audio

        # Generate session token
        session_token = str(uuid.uuid4())

        async with websockets.connect(self.ws_url) as ws:
            # Send config
            config = {
                "type": "config",
                "sample_rate": self.sample_rate,
                "src_lang": src_lang,
                "token": session_token,
            }
            await ws.send(json.dumps(config))

            # Wait for config ack
            try:
                ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ack_data = json.loads(ack)
                if ack_data.get("type") != "config_ack":
                    result.errors.append(f"Unexpected config response: {ack_data}")
            except TimeoutError:
                result.errors.append("Timeout waiting for config acknowledgment")
                return result

            # Start receiver task
            receive_done = asyncio.Event()

            async def receive_messages():
                try:
                    async for message in ws:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "segments":
                            result.segments = data.get("segments", [])
                        elif msg_type == "translation":
                            seg_id = data.get("segment_id")
                            tgt_lang = data.get("tgt_lang")
                            text = data.get("text")
                            if seg_id is not None:
                                if seg_id not in result.translations:
                                    result.translations[seg_id] = {}
                                result.translations[seg_id][tgt_lang] = text
                        elif msg_type == "summary":
                            result.summary = data
                            receive_done.set()
                        elif msg_type == "summary_error":
                            result.errors.append(f"Summary error: {data.get('error')}")
                            receive_done.set()
                        elif msg_type == "error":
                            result.errors.append(data.get("message", str(data)))

                except Exception as e:
                    if "ConnectionClosed" not in str(type(e)):
                        result.errors.append(f"Receive error: {e}")

            receiver = asyncio.create_task(receive_messages())

            # Stream audio frames
            frame_delay = (frame_duration_ms / 1000) * self.realtime_factor
            offset = 0

            while offset < len(audio_data):
                frame = audio_data[offset : offset + bytes_per_frame]
                if len(frame) < bytes_per_frame:
                    # Pad last frame with silence
                    frame = frame + b"\x00" * (bytes_per_frame - len(frame))

                await ws.send(frame)
                offset += bytes_per_frame
                await asyncio.sleep(frame_delay)

            # Wait for processing to complete
            # ASR loop runs every 0.5s and each call takes ~1.6s
            # Worst case timing: audio arrives during ASR call, need to wait for:
            #   - Current ASR to finish (~1.6s)
            #   - Next tick to start (up to 0.5s)
            #   - That ASR to finish (~1.6s)
            # Using 8s provides margin for system load variability
            await asyncio.sleep(8.0)

            # Request summary if desired
            if request_summary:
                await ws.send(json.dumps({"type": "request_summary"}))

                # Wait for summary with timeout
                try:
                    await asyncio.wait_for(receive_done.wait(), timeout=timeout_sec)
                except TimeoutError:
                    result.errors.append("Timeout waiting for summary")

            # Clean up
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

        return result

    async def stream_dialogue(
        self,
        audio_path: str | Path,
        request_summary: bool = True,
    ) -> StreamingResult:
        """Stream a dialogue audio file.

        Convenience method that uses "auto" language detection.

        Args:
            audio_path: Path to synthesized audio
            request_summary: Whether to request summary

        Returns:
            StreamingResult with collected data
        """
        # Use "auto" to test language detection
        return await self.stream_audio_file(
            audio_path=audio_path,
            src_lang="auto",
            request_summary=request_summary,
        )

    async def stream_audio_with_foreign_hint(
        self,
        audio_path: str | Path,
        foreign_lang: str,
        request_summary: bool = True,
        timeout_sec: float = 300.0,
    ) -> StreamingResult:
        """Stream an audio file with a foreign language hint.

        This method provides a foreign_lang hint to the pipeline, which can
        improve language detection accuracy for known bilingual conversations.

        Args:
            audio_path: Path to WAV audio file
            foreign_lang: Foreign language code (e.g., "fa", "ku", "ru")
            request_summary: Whether to request a summary at the end
            timeout_sec: Maximum time to wait for processing

        Returns:
            StreamingResult with collected data
        """
        import websockets

        result = StreamingResult()

        # Read audio file
        audio_path = Path(audio_path)
        with wave.open(str(audio_path), "rb") as wav:
            file_sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)
            result.duration_sec = n_frames / file_sample_rate

        # Calculate frame parameters
        frame_duration_ms = 100
        samples_per_frame = int(self.sample_rate * frame_duration_ms / 1000)
        bytes_per_frame = samples_per_frame * 2  # 16-bit audio

        # Generate session token
        session_token = str(uuid.uuid4())

        async with websockets.connect(self.ws_url) as ws:
            # Send config with foreign language hint
            config = {
                "type": "config",
                "sample_rate": self.sample_rate,
                "src_lang": "auto",
                "foreign_lang": foreign_lang,  # Hint for language detection
                "token": session_token,
            }
            await ws.send(json.dumps(config))

            # Wait for config ack
            try:
                ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ack_data = json.loads(ack)
                if ack_data.get("type") != "config_ack":
                    result.errors.append(f"Unexpected config response: {ack_data}")
            except TimeoutError:
                result.errors.append("Timeout waiting for config acknowledgment")
                return result

            # Start receiver task
            receive_done = asyncio.Event()

            async def receive_messages():
                try:
                    async for message in ws:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "segments":
                            result.segments = data.get("segments", [])
                        elif msg_type == "translation":
                            seg_id = data.get("segment_id")
                            tgt_lang = data.get("tgt_lang")
                            text = data.get("text")
                            if seg_id is not None:
                                if seg_id not in result.translations:
                                    result.translations[seg_id] = {}
                                result.translations[seg_id][tgt_lang] = text
                        elif msg_type == "summary":
                            result.summary = data
                            receive_done.set()
                        elif msg_type == "summary_error":
                            result.errors.append(f"Summary error: {data.get('error')}")
                            receive_done.set()
                        elif msg_type == "error":
                            result.errors.append(data.get("message", str(data)))

                except Exception as e:
                    if "ConnectionClosed" not in str(type(e)):
                        result.errors.append(f"Receive error: {e}")

            receiver = asyncio.create_task(receive_messages())

            # Stream audio frames
            frame_delay = (frame_duration_ms / 1000) * self.realtime_factor
            offset = 0

            while offset < len(audio_data):
                frame = audio_data[offset : offset + bytes_per_frame]
                if len(frame) < bytes_per_frame:
                    # Pad last frame with silence
                    frame = frame + b"\x00" * (bytes_per_frame - len(frame))

                await ws.send(frame)
                offset += bytes_per_frame
                await asyncio.sleep(frame_delay)

            # Wait for processing to complete
            # ASR loop runs every 0.5s and each call takes ~1.6s
            # Worst case timing: audio arrives during ASR call, need to wait for:
            #   - Current ASR to finish (~1.6s)
            #   - Next tick to start (up to 0.5s)
            #   - That ASR to finish (~1.6s)
            # Using 8s provides margin for system load variability
            await asyncio.sleep(8.0)

            # Request summary if desired
            if request_summary:
                await ws.send(json.dumps({"type": "request_summary"}))

                # Wait for summary with timeout
                try:
                    await asyncio.wait_for(receive_done.wait(), timeout=timeout_sec)
                except TimeoutError:
                    result.errors.append("Timeout waiting for summary")

            # Clean up
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

        return result

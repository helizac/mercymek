"""Muse Voice Transcribe realtime client (push-to-talk).

Two audio sources share one session routine, `transcribe_pcm_stream`:
- browser relay (default): the HUD captures the mic, resamples to 24 kHz int16 and sends binary
  WebSocket frames to our server; the server forwards them here. The API key never leaves the server.
- server mic (`MIC_SOURCE=server`): `PushToTalkASR` reads the local microphone with sounddevice.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Callable, Optional

import websockets

from mercy import config

RATE = 24_000
FRAME_BYTES = RATE * 80 // 1000 * 2           # 80 ms of int16 mono
WS_URL = "wss://api.meta.ai/v1/asr/realtime"
KEYWORDS = ["impact", "skid marks", "debris", "shattered glass", "trajectory", "blueprint", "verdict",
            "orbit", "isolate", "rewind", "licence plate", "street view", "enhance", "hatchback", "SUV"]


async def transcribe_pcm_stream(q: "asyncio.Queue[Optional[bytes]]",
                                on_partial: Callable[[str], None] = lambda t: None,
                                encoding: str = "PCM_24KHZ") -> str:
    """Stream PCM chunks from `q` (None = end of speech) and return the final transcript.

    Audio is paced so we never run more than ~1 s ahead of real time (server rejects > 5 s ahead).
    """
    rate = 24_000 if encoding == "PCM_24KHZ" else 16_000
    uri = f"{WS_URL}?sessionId=mercy-{uuid.uuid4()}"
    async with websockets.connect(uri, open_timeout=15, max_size=None) as ws:
        await ws.send(json.dumps({
            "authorization": {"accessToken": f"Bearer {os.environ['MODEL_API_KEY']}"},
            "audioEncoding": encoding,
            "model": config.ASR_MODEL,
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "languageBias": ["English"],
            "keywords": KEYWORDS,
        }))
        ack = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if "sessionId" not in ack:
            raise RuntimeError(f"ASR handshake failed: {ack}")

        final_text, last_partial = "", ""

        async def receiver():
            nonlocal final_text, last_partial
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        continue
                    ev = json.loads(msg)
                    if ev.get("type") == "error":
                        raise RuntimeError(ev.get("message", "ASR error"))
                    if ev.get("type") == "transcript":
                        txt = ev.get("transcript", "")
                        last_partial = txt or last_partial
                        on_partial(txt)
                        if ev.get("final"):
                            final_text = txt
                    # unknown event types are ignored (forward compatibility)
            except websockets.ConnectionClosedOK:
                pass

        rx = asyncio.create_task(receiver())
        t0, sent = time.monotonic(), 0
        buf = b""
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            buf += chunk
            while len(buf) >= FRAME_BYTES:
                frame, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                ahead = sent / (2 * rate) - (time.monotonic() - t0)
                if ahead > 1.0:
                    await asyncio.sleep(ahead - 1.0)
                await ws.send(frame)
                sent += len(frame)
        if buf:
            await ws.send(buf)
        await ws.send(json.dumps({"type": "endStream"}))
        try:
            await asyncio.wait_for(rx, 15)
        except asyncio.TimeoutError:
            rx.cancel()
        return final_text or last_partial


class PushToTalkASR:
    """Server-side microphone (spec section 6). start() opens the mic; stop() ends speech; await result."""

    def __init__(self, on_partial=None):
        self.on_partial = on_partial or (lambda text: None)
        self._q: Optional[asyncio.Queue] = None
        self._stream = None
        self._task: Optional[asyncio.Task] = None

    def start(self):
        import sounddevice as sd
        loop = asyncio.get_running_loop()
        self._q = asyncio.Queue()
        q = self._q

        def on_audio(indata, frames, t, status):
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

        self._stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16",
                                         blocksize=FRAME_BYTES // 2, callback=on_audio)
        self._stream.start()
        self._task = asyncio.create_task(transcribe_pcm_stream(q, self.on_partial))

    async def stop(self) -> str:
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        if self._q:
            self._q.put_nowait(None)
        return await self._task if self._task else ""

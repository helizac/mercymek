"""Spoken replies: ElevenLabs (ELEVEN_MODEL, v4 Turbo by default) over a warm connection, with a phrase cache and a
character budget.

The browser never sees the key: the server registers a phrase, sends {text, url} to the HUD, and streams the audio
from /api/tts/<key> (from the cache when the phrase was spoken before, otherwise straight from ElevenLabs while
writing the cache). A keep-alive ping holds the TLS connection open so first audio arrives in ~150 ms.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re

import httpx

from mercy import config
from mercy.config import log

CACHE = config.ROOT / "data" / "tts_cache"
USAGE = CACHE / "usage.json"
URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"


class TTS:
    def __init__(self):
        self.key = os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice = os.environ.get("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
        self.model = os.environ.get("ELEVEN_MODEL", "eleven_v4_turbo")
        self.budget = int(os.environ.get("TTS_CHAR_BUDGET", "25000"))
        CACHE.mkdir(parents=True, exist_ok=True)
        self.used = json.loads(USAGE.read_text())["chars"] if USAGE.exists() else 0
        self.pending: dict[str, str] = {}
        self.client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=20, headers={"xi-api-key": self.key},
                                            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120))
        return self.client

    async def keep_warm(self):
        """Cheap request every 25 s so the TLS connection is already open when we need to speak (costs no credits)."""
        while self.enabled:
            try:
                client = self._client()
                response = await client.get("https://api.elevenlabs.io/v1/models")
                if response.status_code in (401, 403):
                    # This optional probe needs models_read. A TTS-only key may still
                    # synthesize speech; retrying the same denied probe cannot fix it.
                    log.warning("TTS connection warm-up denied (%s); stopping probes. Speech remains enabled.",
                                response.status_code)
                    return
            except Exception:
                pass
            await asyncio.sleep(25)

    @staticmethod
    def clean(text: str) -> str:
        """Screen text -> speakable text: no object ids, ranges as 'x to y', units and road abbreviations spelled out."""
        text = re.sub(r"\s+#\d+", "", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", r"\1 to \2", text)
        text = re.sub(r"\bkm/h\b", "kilometres per hour", text)
        text = re.sub(r"(\d)\s*m\b", r"\1 metres", text)
        text = re.sub(r"(\d)\s*%", r"\1 percent", text)
        for a, b in (("Blvd", "Boulevard"), ("Ave", "Avenue"), ("Hwy", "Highway"), ("St.", "Street"), (" / ", " or ")):
            text = text.replace(a, b)
        text = re.sub(r"\bsuv\b", "S U V", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:280]

    def register(self, text: str) -> str | None:
        """Returns the audio key, or None when speech is off or the budget is spent (text is still shown)."""
        if not self.enabled or not text:
            return None
        text = self.clean(text)
        k = hashlib.sha1(f"{self.voice}|{self.model}|{text}".encode()).hexdigest()[:16]
        if not (CACHE / f"{k}.mp3").exists() and self.used + len(text) > self.budget:
            log.warning("TTS budget reached (%d chars); replies are text-only", self.used)
            return None
        self.pending[k] = text
        return k

    async def stream(self, k: str):
        """Async byte iterator for /api/tts/<k>: cache hit or live ElevenLabs stream (written to the cache)."""
        f = CACHE / f"{k}.mp3"
        if f.exists():
            yield f.read_bytes()
            return
        text = self.pending.get(k)
        if not text:
            return
        buf = bytearray()
        async with self._client().stream(
                "POST", URL.format(voice=self.voice),
                params={"output_format": "mp3_22050_32",
                        # only the v2.x Flash/Turbo models accept the latency knob (v4 Turbo rejects it)
                        **({"optimize_streaming_latency": 3} if "_v2" in self.model else {})},
                json={"text": text, "model_id": self.model,
                      "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "speed": 1.05}}) as r:
            if r.status_code != 200:
                log.warning("TTS %s: %s", r.status_code, (await r.aread())[:200])
                return
            async for chunk in r.aiter_bytes():
                buf.extend(chunk)
                yield chunk
        f.write_bytes(bytes(buf))
        self.used += len(text)
        USAGE.write_text(json.dumps({"chars": self.used}))


tts = TTS()

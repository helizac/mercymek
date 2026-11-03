"""Step A: four smoke tests against the Meta Model API (Spark text, SAM image, ASR file, Muse Image).

    python scripts/smoke_test.py            # all
    python scripts/smoke_test.py spark sam  # a subset
Costs well under $0.05.
"""
from __future__ import annotations

import base64
import io
import json
import pathlib
import sys
import time
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from mercy import config  # noqa: E402
from mercy.meta_client import meta_client  # noqa: E402

OUT = ROOT / "data" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def t_spark():
    r = meta_client().responses.create(model=config.VISION_MODEL, input="Reply with exactly: MERCY ONLINE",
                                       reasoning={"effort": "minimal"}, max_output_tokens=200)
    print("spark:", repr(r.output_text), r.usage)


def t_sam():
    import cv2
    img = np.full((240, 320, 3), 40, np.uint8)
    cv2.rectangle(img, (90, 120), (230, 190), (40, 40, 220), -1)      # a crude red "car"
    cv2.circle(img, (120, 195), 18, (10, 10, 10), -1); cv2.circle(img, (200, 195), 18, (10, 10, 10), -1)
    ok, buf = cv2.imencode(".jpg", img)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    text = []
    with meta_client().responses.stream(model=config.SAM_MODEL, input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "red rectangle"}, {"type": "input_image", "image_url": url}]}],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"}) as s:
        for ev in s:
            if ev.type == "response.output_text.delta":
                text.append(ev.delta)
    raw = "".join(text)
    (OUT / "sam_image.lines").write_text(raw)
    print("sam:", raw[:200])
    import meta_sam_parser as msp
    p = msp.image_segmentation_format().create_parser()
    p.push(raw + "\n", emit=False)
    res = p.finish(msp.CompletedOutcome()).result
    for m in [r for r in res.records if r.kind == "mask"]:
        b = m.bounds
        print("  mask", m.object_id, m.mask.width, m.mask.height, "bounds", b, "conf", m.confidence)
        assert m.mask.height == round(b.bottom - b.top) and m.mask.width == round(b.right - b.left)


def tts_wav(text: str, path: pathlib.Path):
    """Windows SAPI TTS -> 16 kHz mono PCM WAV (test audio without a microphone)."""
    import subprocess
    tmp = path.with_suffix(".raw.wav")
    ps = (f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{tmp}'); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ac", "1", "-ar", "16000",
                    "-sample_fmt", "s16", str(path)], check=True)
    tmp.unlink(missing_ok=True)


def t_asr():
    import os
    import httpx
    wav = OUT / "cmd.wav"
    tts_wav("Isolate the vehicles and rewind to three seconds before impact.", wav)
    req = {"model": config.ASR_MODEL, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}
    t0 = time.time()
    r = httpx.post(f"{config.META_BASE_URL}/asr/transcribe",
                   headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"},
                   files={"request": (None, json.dumps(req), "application/json"),
                          "audio": ("cmd.wav", wav.read_bytes(), "audio/wav")}, timeout=60)
    print("asr file:", r.status_code, r.text[:500], f"{time.time()-t0:.2f}s")
    # realtime websocket, fed from the same speech resampled to 24 kHz
    import asyncio
    import subprocess
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-ar", "24000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    from mercy.voice.asr import transcribe_pcm_stream

    async def feed():
        q: asyncio.Queue = asyncio.Queue()
        for i in range(0, len(pcm), 3840):
            q.put_nowait(pcm[i:i + 3840])
        q.put_nowait(None)
        return await transcribe_pcm_stream(q, on_partial=lambda t: print("   partial:", t))
    t0 = time.time()
    print("asr realtime final:", repr(asyncio.run(feed())), f"{time.time()-t0:.2f}s")


def t_image():
    r = meta_client().images.generate(model=config.IMAGE_MODEL, prompt="a minimal dark navy tactical grid with one cyan circle",
                                      n=1, size="1024x1024", extra_body={"reasoning_strength": "low", "output_format": "png",
                                      "tool_enablement": {"enable_web_search": False, "enable_image_search": False, "enable_shell": False}})
    (OUT / "image.png").write_bytes(base64.b64decode(r.data[0].b64_json))
    print("image: ok", (OUT / "image.png").stat().st_size, "bytes")


if __name__ == "__main__":
    tests = {"spark": t_spark, "sam": t_sam, "asr": t_asr, "image": t_image}
    for name in (sys.argv[1:] or tests):
        t0 = time.time()
        try:
            tests[name]()
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
        print(f"  ({time.time()-t0:.1f}s)")

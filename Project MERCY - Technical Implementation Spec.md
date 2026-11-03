# Project MERCY — Technical Implementation Spec

## 1. Summary and MVP scope

MERCY turns a single dashcam or CCTV clip into a pre-processed, scrubbable 3D scene (a 4D point cloud) and presents it in a voice-driven cockpit. The winning hackathon strategy: **precompute the heavy work (segmentation, 3D reconstruction, physics) before the demo; let only voice, decision, reasoning and image generation run live.**

This document is written so an AI coding agent can build the project directly: every module lists its inputs, outputs, file formats and sample code. Section 15 contains the instruction block for the agent.

### What the idea claims vs. what the MVP actually builds

| Component | Claim in the idea | Built in the MVP | Stretch goal |
| --- | --- | --- | --- |
| Voice control | Real-time, zero data retention | Muse Voice Transcribe realtime WebSocket, push-to-talk | Always-on listening (ENDPOINTING mode) |
| Command understanding | n/a | Laya intent classification in under 50 ms, falls back to Muse Spark on low confidence | Multi-step command chains |
| Evidence isolation | SAM 3.1 pixel masks | SAM 3.1 video tracking, precomputed masks plus SVG overlays | Live click-to-segment with box/point prompts |
| 4D scene | 4D Gaussian Splatting | Time-indexed point cloud from per-frame depth and camera pose (VGGT or MonST3R), shown in the browser with viser | Offline 4DGS training |
| Physics analytics | Speed, angle, braking distance | Speed, deceleration and impact time from 3D object tracks, metric scale from road geometry | Skid-mark speed formula |
| Fault analysis | Automated official report | Muse Wonder plus web\_search, cited and labelled as an estimate | PDF export |
| Image generation | Tactical blueprint, before/after | Deterministic overhead diagram restyled by Muse Image | Before/after comparison |
| Multi-camera | Multi-angle | Single camera | Second angle sync |

### Success criteria

- 3 prepared cases (Nexar clips) run end to end; the demo fits in 3 minutes.
- Voice-to-visual latency under 1.5 seconds on the Laya path.
- From "analyze" to the verdict panel in under 30 seconds, with cited sources.
- Every numeric output carries an uncertainty range and an "estimate" label.

## 2. System architecture

MERCY has two halves joined by a folder. **Ingest** runs once per case on a GPU and writes everything heavy into `data/cases/<id>/`. The **live demo** runs on the laptop, reads that cache, and only does the light work in real time: voice, decision, reasoning and image generation.

&#91;embedded content: system architecture · offline ingest, case cache, live demo\]

Teal marks the decision path; results from Muse Wonder and Muse Image flow back into StateHub, which is the only thing the browser listens to.

### Journey of a live command

1. Hold Space; the server captures the mic and streams PCM to Muse Voice Transcribe.
2. The final transcript goes to Laya, which returns an intent with a confidence.
3. StateHub applies the intent and pushes a `state` patch to the HUD over the WebSocket.
4. For "who is at fault?", Muse Wonder reasons over the cached measurements and Laya checks the verdict's numbers before it is shown.
5. For "generate blueprint", Muse Image draws the schematic; the HUD labels it AI-generated.

## 3. Technology stack

All Meta models are called through one OpenAI-compatible endpoint (`https://api.meta.ai/v1`, key in `MODEL_API_KEY`). The open-source side (3D reconstruction, Laya, viewer) runs locally or on a GPU machine.

| Layer | Technology | Model ID / package | Role in the project |
| --- | --- | --- | --- |
| Speech to text | [Muse Voice Transcribe](https://dev.meta.ai/docs/speech-to-text) | `muse-voice-transcribe-1.0`, `wss://api.meta.ai/v1/asr/realtime` | Push-to-talk commands in English and Turkish |
| Command intent | [Laya](https://github.com/NandhaKishorM/laya) | `pip install laya` (Router, `english` + `multilingual`) | Single-forward-pass intent classification, report integrity checks |
| Segmentation | [SAM 3.1](https://dev.meta.ai/docs/sam/segmenting) | `sam-3.1` (Responses API), `pip install meta-sam-parser` | Masks for vehicles, pedestrians, glass, debris, plus video tracking |
| Scene understanding | [Muse Spark](https://dev.meta.ai/docs/video-understanding) | `muse-spark-1.3` | Video summary, impact moment, vehicle type and model |
| Deep reasoning | [Muse Wonder](https://dev.meta.ai/docs/models) | `muse-wonder-1.0` (preview) | Fault analysis report with structured output |
| Web research | [Search grounding](https://dev.meta.ai/docs/search-grounding) | `tools=[{"type": "web_search"}]` | Traffic rules, vehicle dimensions, NHTSA data, citations |
| Image generation | [Muse Image](https://dev.meta.ai/docs/image-generation) | `muse-image-1.0` | Stylized blueprint of the overhead diagram |
| 3D/4D reconstruction (A) | [VGGT](https://github.com/facebookresearch/vggt) | `facebook/VGGT-1B` | Per-frame depth, camera pose, point map (seconds) |
| 3D/4D reconstruction (B) | [MonST3R](https://github.com/Junyi42/monst3r) | repo + weights | Time-varying point cloud for moving scenes |
| 3D viewer | [viser](https://pypi.org/project/viser/) | `pip install viser` | Browser 3D scene, time slider, camera control from Python |
| Backend | FastAPI + WebSocket | `fastapi`, `uvicorn`, `websockets` | HUD state broadcast, voice session handling |
| Media | ffmpeg, OpenCV, NumPy, SciPy | n/a | Frame extraction, mask merging, signal smoothing |
| Diagram | matplotlib | n/a | Deterministic, to-scale overhead trajectory plot |

### Why not MASt3R?

MASt3R is designed for static scenes; in a crash clip the vehicles move, which corrupts pose estimation. MonST3R adapts the DUSt3R representation to dynamic scenes and outputs a time-varying point cloud with per-frame camera poses ([source](https://github.com/Junyi42/monst3r)). VGGT infers camera parameters, depth and point maps in one forward pass, typically in under a second ([source](https://huggingface.co/facebook/VGGT-1B)); its Meta AI co-authorship also helps at a Meta hackathon. **Plan A: VGGT** (speed, easy setup). **Plan B: MonST3R** (cleaner geometry on moving objects). License note: the original VGGT checkpoint is non-commercial; use `VGGT-1B-Commercial` for commercial use.

### Model selection rules

- Speech: `muse-voice-transcribe-1.0`. Audio input is not fully supported on Muse Spark 1.3, so do not use Spark for speech.
- Fast tasks such as video understanding and classification: `muse-spark-1.3`.
- Final verdict (long reasoning): `muse-wonder-1.0`. If you have no access, the same request runs on `muse-spark-1.3`; only the model ID changes.
- Standard mode on Model API does not train on your content; do not use the `-contributor` model variants in this project.

## 4. Laya as the System 1 reflex layer

Laya is MERCY's reflex brain: it turns a spoken command into an intent in a single forward pass (about 33 ms on a T4) and audits the large model's output with calibrated probabilities. Muse Wonder and Spark are System 2: slow, deep reasoning. The pairing gives the pitch a strong line: *"Reflexes in Laya, judgment in Muse."*

Laya does not generate text. It returns typed decisions: `choice` (pick one label), `score` (ordinal level) and `noul` (yes/no probability), so there is nothing to parse and nothing to hallucinate ([source](https://github.com/NandhaKishorM/laya)). Its Router detects the language and sends Turkish text to the `laya-multilingual` checkpoint automatically.

### Where it is used (by priority)

1. **Voice command router (required).** Transcript to one of 10 intents. An LLM call takes 1 to 3 seconds; Laya answers in a few hundred ms even on CPU, so the HUD reacts instantly.
2. **Report integrity check (required).** Muse Wonder's fault report is passed to Laya together with the measured evidence summary; `noul` questions score checks such as "is there a fault claim without evidence?" and "does it speculate?". The HUD shows a "Verdict Integrity" indicator.
3. **Evidence prioritization (optional).** A `score` question rates the importance of each event sentence in Muse Spark's scene description.
4. **Command guard (optional).** The `laya.guard_questions()` preset screens voice notes for prompt injection.

### Design rules

- Keep the intent count under 12. Laya's option budget is limited and accuracy drops past about 20 options.
- Do not make Laya extract slots (numbers, object names). Numbers come from regex; object names from fuzzy matching against a fixed concept dictionary.
- Gate with `min_confidence`: if `answer_confidence` falls below the threshold, the command falls back to Muse Spark's structured-output parser. Tune the threshold on your own 30 to 50 command examples; the shipped checkpoints are over-confident out of the box.
- Split compound commands ("isolate, zoom and rewind") into clauses and classify them in one call with `router.predict_batch`.
- Use `Router(preload=True)` at server start so the first request does not pay model load latency.

### Intent schema

```python
INTENT_QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "Which action does the investigator want in the forensic viewer?",
        "criteria": {
            "isolate":   "highlight or segment an object: car, glass, debris, pedestrian, skid marks",
            "seek":      "jump or rewind or fast-forward to a time or to the impact moment",
            "playback":  "play, pause, slow motion, speed change",
            "camera":    "rotate, orbit, top view, driver view, fly around the 3D scene",
            "zoom":      "zoom in or focus on a specific object",
            "trajectory":"show path, braking trajectory, motion trail of a vehicle",
            "analyze":   "compute fault, verdict, physics analysis, who is responsible",
            "diagram":   "generate overhead diagram, blueprint, reconstruction image",
            "note":      "record a spoken note or observation into the case file",
            "reset":     "clear highlights, reset view, start over"
        }
    }
}
```

### Router setup and call

```python
from laya import Router

router = Router(preload=True)          # english + multilingual stay resident

def classify_clauses(clauses: list[str]) -> list[dict]:
    reqs = [{"state": {"request": c}, "questions": INTENT_QUESTION} for c in clauses]
    results = router.predict_batch(reqs, min_confidence=0.55)
    out = []
    for c, r in zip(clauses, results):
        ans = r["answers"]["intent"]
        out.append({
            "clause": c,
            "intent": ans["choice"],
            "confidence": ans.get("answer_confidence", ans["confidence"]),
            "low_confidence": ans.get("low_confidence", False),
            "model": r["routing"]["model"],
        })
    return out
```

### Report integrity questions

```python
INTEGRITY_QUESTIONS = {
    "unsupported_fault": {"type": "noul",
        "instructions": "Does the verdict assign fault without citing a measured value or a cited rule?"},
    "speculation": {"type": "noul",
        "instructions": "Does the verdict state guesses about driver intent or events not visible in the evidence?"},
    "consistency": {"type": "noul",
        "instructions": "Is the verdict consistent with the measured speeds and impact time in the evidence summary?"},
    "severity": {"type": "score", "instructions": "How severe is the collision?",
        "criteria": ["minor contact", "moderate damage", "severe crash"]},
}
# state = {"verdict": verdict_text, "evidence": physics_summary_text}
```

Note: Laya's English checkpoint has a 512-token context. For long reports use `router.predict_long(...)` or send only the `fault_assessment` field.

## 5. Repo layout, setup, environment

The project has two run modes: **ingest** (on a GPU machine, once per case, offline) and **live** (on the demo laptop, no GPU needed). The only contract between them is the set of files in `data/cases/<case_id>/`.

### Directory layout

```text
mercy/
├── pyproject.toml
├── .env.example
├── README.md
├── frontend/                  # static UI (see section 11); served by FastAPI at "/"
│   ├── index.html
│   ├── css/styles.css
│   └── js/ (main, transport, sim, feed, scene3d, timeline, panel, util)
├── mercy/
│   ├── config.py              # env loading, paths, constants
│   ├── meta_client.py         # OpenAI client (base_url=api.meta.ai), retry/backoff
│   ├── schemas.py             # pydantic models: Case, ObjectTrack, Physics, Verdict, Command
│   ├── ingest/
│   │   ├── video.py           # ffmpeg normalize, frame extraction
│   │   ├── scene.py           # Muse Spark video understanding -> scene.json
│   │   ├── segment.py         # SAM 3.1 -> masks/*.lines + overlays.json
│   │   ├── reconstruct.py     # VGGT / MonST3R -> recon.npz
│   │   └── lift.py            # mask + depth -> 3D object tracks
│   ├── analysis/
│   │   ├── physics.py         # speed, acceleration, impact moment, braking distance
│   │   ├── research.py        # Muse Spark + web_search -> research.json
│   │   ├── verdict.py         # Muse Wonder structured report
│   │   └── integrity.py       # Laya integrity check
│   ├── generation/
│   │   ├── diagram.py         # matplotlib overhead diagram
│   │   └── blueprint.py       # Muse Image stylization
│   ├── voice/
│   │   ├── asr.py             # realtime ASR WebSocket client
│   │   └── commands.py        # clause split + Laya + slot extraction + fallback
│   ├── state.py               # SceneState + async event broadcast
│   ├── viewer/viser_scene.py  # 3D scene, time slider, camera presets
│   └── server/app.py          # FastAPI, /ws, static files
├── scripts/
│   ├── ingest_case.py         # python scripts/ingest_case.py --video x.mp4 --case nexar_017
│   └── run_demo.py            # python scripts/run_demo.py --case nexar_017
├── data/cases/                # ingest outputs (not in git)
└── tests/
```

### Case folder contract

| File | Produced by | Contents |
| --- | --- | --- |
| `clip.mp4` | video.py | H.264, faststart, original resolution |
| `frames/000000.jpg` | video.py | 8 fps frames for reconstruction, long edge 518 px |
| `meta.json` | video.py | fps, frame count, width, height, source dataset, `time_of_event` |
| `scene.json` | scene.py | Muse Spark summary: vehicles, events, impact time, weather, road type |
| `masks/<concept>.lines` | segment.py | SAM raw output lines (one line per frame) |
| `overlays.json` | segment.py | frame -> object -> {box, svg\_path, concept, color} |
| `recon.npz` | reconstruct.py | `depth[T,H,W]`, `extrinsic[T,3,4]`, `intrinsic[T,3,3]`, `conf[T,H,W]` |
| `tracks.json` | lift.py | Per-object per-frame 3D center and velocity vector |
| `physics.json` | physics.py | Scale, speeds, impact frame, angles, uncertainty ranges |
| `research.json` | research.py | Rules, vehicle dimensions, citation URLs |
| `verdict.json` | verdict.py | Structured verdict plus Laya integrity scores |
| `diagram.png`, `blueprint.webp` | generation/\* | Overhead diagram, stylized image |

### Setup

```bash
# Python 3.11 recommended (Laya needs 3.10+, MonST3R is tested on 3.11)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install openai websockets sounddevice fastapi "uvicorn[standard]" \
  pydantic python-dotenv numpy scipy opencv-python matplotlib rapidfuzz \
  laya meta-sam-parser viser

# GPU ingest machine only:
uv pip install torch torchvision --torch-backend=auto
uv pip install "vggt @ git+https://github.com/facebookresearch/vggt.git"
# MonST3R (Plan B): git clone --recursive https://github.com/junyi42/monst3r and follow its README

# System: ffmpeg is required
```

### Environment variables (`.env.example`)

```bash
MODEL_API_KEY=            # Meta Model API key (from the dashboard)
META_BASE_URL=https://api.meta.ai/v1
ASR_MODEL=muse-voice-transcribe-1.0
VISION_MODEL=muse-spark-1.3
REASONING_MODEL=muse-wonder-1.0   # use muse-spark-1.3 if you have no access
IMAGE_MODEL=muse-image-1.0
SAM_MODEL=sam-3.1
LAYA_DEVICE=cpu
LAYA_MIN_CONFIDENCE=0.55
VISER_PORT=8080
APP_PORT=8000
CASES_DIR=./data/cases
```

### Meta client

```python
# mercy/meta_client.py
import os
from openai import OpenAI

def meta_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"),
        api_key=os.environ["MODEL_API_KEY"],
        max_retries=3,      # SDK exponential backoff for 429 and 5xx
        timeout=120,
    )
```

## 6. Voice command pipeline

Flow: hold Space, Python streams the microphone as 24 kHz PCM to Meta, release sends `endStream`, a `final: true` transcript arrives, then clause splitting, Laya intent, slot extraction and a `SceneState` update. Target: under 1.5 seconds after the user stops speaking.

The demo runs on a local laptop, so the microphone is read directly from Python (`sounddevice`), not from the browser; the API key never reaches the browser. The HUD only sends `ptt_start` / `ptt_stop` messages.

### ASR protocol summary

- Endpoint: `wss://api.meta.ai/v1/asr/realtime?sessionId=<id>`. Authentication is not an HTTP header: it goes in the **first JSON text frame** after connecting (within 10 seconds).
- Audio: signed 16-bit little-endian mono PCM, `PCM_24KHZ` (preferred) or `PCM_16KHZ`, as binary frames, **paced in real time**. The server rejects streams that run more than 5 seconds ahead.
- In `PUSH_TO_TALK` mode the completion signal is a `transcript` event with `final: true`. In `CUMULATIVE` partial mode each partial replaces the previous one; do not concatenate.
- To finish: drain the queue, send `{"type": "endStream"}`, and keep reading until the server closes with code 1000.
- `languageBias` and `keywords` are set in the handshake only and cannot change mid-session.
- Turkish is among the 25 supported languages; code-switching is supported.
- Price: $0.18 per hour of audio processed; ZDR is priced the same as Standard ([source](https://dev.meta.ai/docs/speech-to-text)). Verify in the dashboard that ZDR is enabled for your account.

### ASR client

```python
# mercy/voice/asr.py
import asyncio, json, os, uuid
import sounddevice as sd
import websockets

RATE = 24_000
FRAME = RATE * 80 // 1000          # 80 ms frames
KEYWORDS = ["impact", "skid marks", "debris", "shattered glass", "trajectory",
            "blueprint", "verdict", "orbit", "çarpışma", "enkaz", "cam kırığı", "yörünge"]

class PushToTalkASR:
    """One key press = one session. run_once() opens the mic; stop() ends it and run_once() returns the final transcript."""

    def __init__(self, on_partial=None):
        self.on_partial = on_partial or (lambda text: None)
        self._stop = asyncio.Event()

    async def run_once(self) -> str:
        self._stop.clear()
        uri = f"wss://api.meta.ai/v1/asr/realtime?sessionId=mercy-{uuid.uuid4()}"
        async with websockets.connect(uri, open_timeout=30) as ws:
            await ws.send(json.dumps({
                "authorization": {"accessToken": f"Bearer {os.environ['MODEL_API_KEY']}"},
                "audioEncoding": "PCM_24KHZ",
                "model": os.environ.get("ASR_MODEL", "muse-voice-transcribe-1.0"),
                "mode": "PUSH_TO_TALK",
                "partialMode": "CUMULATIVE",
                "emitAudioProgress": False,
                "languageBias": ["English", "Turkish"],
                "keywords": KEYWORDS,
            }))
            ack = json.loads(await ws.recv())
            if "sessionId" not in ack:
                raise RuntimeError(f"ASR handshake failed: {ack}")

            loop = asyncio.get_running_loop()
            q: asyncio.Queue[bytes] = asyncio.Queue()
            final_text = ""

            def on_audio(indata, frames, t, status):
                loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

            async def receiver():
                nonlocal final_text
                async for msg in ws:
                    if isinstance(msg, bytes):
                        continue
                    ev = json.loads(msg)
                    if ev.get("type") == "error":
                        raise RuntimeError(ev["message"])
                    if ev.get("type") == "transcript":
                        self.on_partial(ev["transcript"])   # live captions in the HUD
                        if ev.get("final"):
                            final_text = ev["transcript"]
                    # ignore unknown event types (forward compatibility)

            rx = asyncio.create_task(receiver())
            with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16",
                                   blocksize=FRAME, callback=on_audio):
                while not self._stop.is_set():
                    try:
                        await ws.send(await asyncio.wait_for(q.get(), timeout=0.2))
                    except asyncio.TimeoutError:
                        pass
            while not q.empty():
                await ws.send(q.get_nowait())
            await ws.send(json.dumps({"type": "endStream"}))
            await rx
            return final_text

    def stop(self):
        self._stop.set()
```

### Command parsing pipeline

1. **Clause splitting.** Split on conjunctions: `,`, `;`, `and then`, `then`, `and`, `ve`, `sonra`, `ardından`. To avoid breaking noun phrases such as "glass and debris", merge a fragment back into the previous clause if it does not start with a verb or intent keyword.
2. **Intent with Laya.** All clauses in one `predict_batch` call (section 4).
3. **Slot extraction (deterministic).**
   - `seek`: `(\d+(?:[.,]\d+)?)\s*(s|sec|seconds?|saniye)` plus direction (`before|önce` is negative, `after|sonra` is positive) plus anchor (`impact|collision|crash|çarpışma|kaza` maps to `physics.impact_t`). With no anchor, the time is absolute.
   - `isolate` / `zoom` / `trajectory`: `rapidfuzz.process.extractOne` against the concept dictionary (score 70 or higher). Object numbers ("vehicle 2", "araç 2") are extracted by regex as `object_id`.
   - `camera`: keyword to preset (`top|üstten|kuş bakışı` is `top`, `driver|sürücü` is `driver`, `orbit|etrafında|360` is `orbit`, `side|yandan` is `side`).
   - `playback`: `play|oynat`, `pause|durdur`, `slow|yavaş` sets speed 0.25.
4. **Fallback.** If any clause is `low_confidence` or a required slot is missing, the whole transcript goes to Muse Spark structured output (below). Latency rises by 1 to 3 seconds and the HUD shows a "deep parse" indicator.
5. **Execution.** The command list is applied to `SceneState` in order; each step is broadcast to the HUD and viser over WebSocket.

### Concept dictionary

```python
CONCEPTS = {  # short noun phrase sent to SAM -> synonyms
    "car":             ["car", "vehicle", "sedan", "araba", "araç"],
    "truck":           ["truck", "lorry", "kamyon", "tır"],
    "pedestrian":      ["pedestrian", "person", "yaya", "insan"],
    "shattered glass": ["glass", "broken glass", "shattered glass", "cam", "cam kırığı"],
    "debris":          ["debris", "parts", "enkaz", "parça"],
    "traffic light":   ["traffic light", "signal", "trafik ışığı", "lamba"],
    "license plate":   ["plate", "license plate", "plaka"],
}
```

SAM rule: send one concrete noun phrase ("red car"), never a command or question ("segment the car"); comma-separated multi-concept prompts return zero matches ([source](https://dev.meta.ai/docs/sam/segmenting)). Hard concepts like "skid marks" may segment poorly; test them during ingest and drop them from the dictionary if nothing comes back.

### Command schema and Muse Spark fallback

```python
COMMAND_SCHEMA = {
  "type": "object", "additionalProperties": False,
  "required": ["commands"],
  "properties": {"commands": {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "concept", "object_id", "t_seconds", "relative_to_impact",
                 "camera_preset", "speed", "note_text"],
    "properties": {
      "intent": {"type": "string", "enum": ["isolate","seek","playback","camera","zoom",
                 "trajectory","analyze","diagram","note","reset"]},
      "concept": {"type": ["string", "null"]},
      "object_id": {"type": ["integer", "null"]},
      "t_seconds": {"type": ["number", "null"]},
      "relative_to_impact": {"type": "boolean"},
      "camera_preset": {"type": ["string", "null"], "enum": ["top","driver","orbit","side", None]},
      "speed": {"type": ["number", "null"]},
      "note_text": {"type": ["string", "null"]}
    }}}}
}

def deep_parse(client, transcript: str, concepts: list[str]) -> dict:
    r = client.responses.create(
        model=os.environ["VISION_MODEL"],          # muse-spark-1.3
        input=[
          {"role": "system", "content": "Convert the investigator's spoken command into viewer commands. "
             f"Allowed concepts: {concepts}. Negative t_seconds means before."},
          {"role": "user", "content": transcript}],
        text={"format": {"type": "json_schema", "name": "Commands",
                         "schema": COMMAND_SCHEMA, "strict": True}},
        reasoning={"effort": "minimal"},            # for speed; verify the parameter name at /docs/reasoning
    )
    return json.loads(r.output_text)
```

On the Responses API the structured-output parameter is `text.format`; `response_format` belongs to Chat Completions only ([source](https://dev.meta.ai/docs/structured-output)).

## 7. SAM 3.1 evidence segmentation

SAM 3.1 runs once per concept over the whole clip during ingest. In video mode it tracks each object across frames under a **stable object ID**. That ID is the shared key for 3D tracks, overlays and physics.

### API rules to know

- Call: `client.responses.stream(model="sam-3.1", input=[...])`; content is `input_text` (the concept) plus `input_video` (`video_url` or `file_id`), or `input_image`.
- One request = one concept. For several concepts send separate requests (they can run in parallel).
- Output is special-token text in `output_text`, **one line per frame**: `<Nf>id<|box;x1;y1;x2;y2;w;h|><|mask;x=0;y=0;data=H,W,<enc>payload|>,id...` ([source](https://dev.meta.ai/docs/sam/reading-segmentation)).
- Frame indices can skip (no line is emitted for a frame with no visible object); always read the `<Nf>` value, never rely on line order.
- Object IDs are neither dense nor zero-based (a frame can carry `0, 2`); key masks by ID.
- Box coordinates are source pixels with inclusive `x2`/`y2`. The mask raster is box-sized: in the docs example a 60x27 box carries `data=27,60` (height, width). When placing the mask on the full frame, paste it at `(x1, y1)`; in an ingest test, assert `H == y2-y1+1` and `W == x2-x1+1`.
- `metadata`: `{"mask_encoding": "one_bit", "include_confidence": "true"}`. Values are strings; `one_bit` is several times smaller than lossless.
- The mask payload is base85 and can contain characters such as `|`, `>`, `;`. **Do not write your own regex parser**; use `meta-sam-parser` (0.0.6 or later is required for confidence).
- Limits: MP4/MOV with H.264/H.265/MPEG-4 (no WebM/VP9), up to 15,000 frames per request, up to 16 objects per frame in video. Video is billed per frame, with a 50-frame minimum per request.
- Zero matches is not an error: the stream completes with an empty `output_text`.

### Concept list (ingest)

`car`, `truck`, `pedestrian`, `motorcycle`, `shattered glass`, `debris`, `traffic light`, `license plate`. Extra objects named in Muse Spark's `scene.json` (for example "white SUV") are appended automatically.

### Segmentation code

```python
# mercy/ingest/segment.py
import json, pathlib
from concurrent.futures import ThreadPoolExecutor
from mercy.meta_client import meta_client
import meta_sam_parser as msp   # pip install meta-sam-parser

def upload_clip(client, path) -> str:
    with open(path, "rb") as f:
        return client.files.create(purpose="user_data", file=f).id   # synchronous; 200 = ready

def segment_concept(client, file_id: str, concept: str, out_dir: pathlib.Path) -> list[str]:
    lines = []
    with client.responses.stream(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_video", "file_id": file_id}]}],
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    ) as stream:
        for ev in stream:
            if ev.type == "response.output_text.delta":
                lines.append(ev.delta)          # each delta = one frame line
    safe = concept.replace(" ", "_")
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "masks" / f"{safe}.lines").write_text("\n".join(lines))
    return lines

def segment_all(case_dir: pathlib.Path, concepts: list[str]):
    client = meta_client()
    file_id = upload_clip(client, case_dir / "clip.mp4")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda c: segment_concept(client, file_id, c, case_dir), concepts))
```

### From raw lines to overlays and mask arrays

`meta_sam_parser` mirrors the TypeScript package contract: stream parsing, `records_of_kind`, `decode_mask_to_raster`, `decode_mask_to_rle`, `decode_mask_to_svg_path`, `build_object_prompt` ([source](https://dev.meta.ai/docs/sam/client-libraries)). Instruction for the agent: confirm the exact parsing entry points with `python -c "import meta_sam_parser as m; print(dir(m))"` and use the function that parses raw text (the saved `.lines` content).

```python
# Pseudo-code: adapt names to the package's real API
def build_overlays(case_dir, concept, palette) -> dict:
    raw = (case_dir / "masks" / f"{concept.replace(' ', '_')}.lines").read_text()
    result = msp.parse_text(raw, media="video")          # <- verify the real name
    boxes = {(b.frame.frame_index, b.object_id): b for b in msp.records_of_kind(result.records, "box")}
    overlays = {}
    for m in msp.records_of_kind(result.records, "mask"):
        f, oid = m.frame.frame_index, m.object_id
        b = boxes[(f, oid)]
        overlays.setdefault(str(f), {})[f"{concept}:{oid}"] = {
            "concept": concept,
            "object_id": oid,
            "box": [b.left, b.top, b.right, b.bottom],     # right/bottom are half-open
            "confidence": m.confidence,
            "svg_path": msp.decode_mask_to_svg_path(m.mask),  # box-local coordinates
            "offset": [b.left, b.top],                       # HUD applies translate(x1,y1)
            "color": palette[concept],
        }
    return overlays

def full_frame_mask(m, b, W, H):
    import numpy as np
    crop = np.frombuffer(bytes(msp.decode_mask_to_raster(m.mask)), dtype=np.uint8)
    h, w = m.mask.height, m.mask.width                    # verify the field names
    crop = crop.reshape(h, w)
    full = np.zeros((H, W), np.uint8)
    full[b.top:b.top + h, b.left:b.left + w] = crop
    return full
```

In the HUD each object is a `<path d="..." transform="translate(x1,y1)">` inside an SVG whose `viewBox` equals the source video size. The frontend in section 11 already renders this format.

### Live segmentation (the wow moment)

If a concept is requested that is not in the dictionary ("isolate the fire hydrant"), the current frame is sent to SAM as a base64 data URL in `input_image`. It is a single frame, so it returns in a few seconds and the mask appears in the HUD, showing the jury that something not precomputed works live.

### Privacy layer

The `license plate` and `face` masks are used to blur plates and faces in the HUD video (OpenCV during ingest produces `clip_redacted.mp4`). This is an ethical plus and a second use of SAM.

## 8. 4D scene reconstruction

"4D" here means: for every frame, a point cloud in world coordinates plus a camera pose; the time slider selects which frame is shown. The static background (road, buildings) is shown as one accumulated cloud, while moving objects (points inside SAM masks) are drawn frame by frame in accent colors. The user flies freely through the scene and scrubs time back and forth.

### Step 1: trim the clip and extract frames

Nexar clips carry a `time_of_event` label. Trim from 8 seconds before the collision to 4 seconds after; this cuts SAM cost (billed per frame) and GPU memory.

```bash
ffmpeg -ss $((T_EVENT-8)) -t 12 -i raw.mp4 -c:v libx264 -an -movflags +faststart clip.mp4
ffmpeg -i clip.mp4 -vf "fps=8,scale=518:-2" frames/%06d.jpg      # ~96 frames
```

Frame mapping: reconstruction frame `k` is time `t = k / 8`; SAM frames run at the source fps (usually 30). Write both fps values to `meta.json` and map with `sam_frame = round(t * src_fps)`.

### Step 2: VGGT (Plan A)

```python
# mercy/ingest/reconstruct.py -- on the GPU machine
import glob, numpy as np, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def run_vggt(frame_dir: str, out_path: str):
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    paths = sorted(glob.glob(f"{frame_dir}/*.jpg"))
    images = load_and_preprocess_images(paths).to(device)          # [T,3,H,W]
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"].squeeze(0)                               # [T,H,W,1]
    pts = unproject_depth_map_to_point_map(depth, extr.squeeze(0), intr.squeeze(0))
    np.savez_compressed(out_path,
        depth=depth.float().cpu().numpy()[..., 0],
        depth_conf=pred["depth_conf"].squeeze(0).float().cpu().numpy(),
        extrinsic=extr.squeeze(0).float().cpu().numpy(),           # [T,3,4] world->camera (OpenCV)
        intrinsic=intr.squeeze(0).float().cpu().numpy(),           # [T,3,3]
        points=pts.astype(np.float16),                             # [T,H,W,3] world coordinates
        colors=(images.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8))
```

Notes: start with 60 to 100 frames; reduce the count if memory runs out. Record the `H x W` of the image tensor; SAM masks are downscaled to that size with `cv2.resize(..., INTER_NEAREST)` (VGGT preprocessing does not crop 16:9 clips, but verify the shape anyway). Function names come from the VGGT README; if versions differ, the README example wins.

### Step 2B: MonST3R (Plan B)

MonST3R's demo script produces a time-varying point cloud, per-frame pose and intrinsics, and a dynamic/static mask from a moving video, and ships its own viser-based 4D visualizer ([source](https://github.com/Junyi42/monst3r)). Write an adapter that converts its output to the same `recon.npz` contract; the rest of the system does not change. Because of its global optimization it takes minutes per clip, so it runs only during ingest.

### Step 2C: Fallback for CCTV / fixed cameras

If the camera is fixed (CCTV), pose estimation is unnecessary: build a 2.5D point cloud from per-frame monocular depth (for example Depth Anything V2) and fixed intrinsics. The scene is still orbitable.

### Step 3: metric scale

VGGT output has no absolute scale; converting speeds to m/s needs a scale factor `s`. Three methods, in priority order:

1. **Camera height (default).** Fit the road plane to static points with RANSAC; the camera center's distance to the plane is `h_est`. A dashcam is typically 1.2 to 1.5 m high; `s = 1.35 / h_est`, uncertainty about ±12%.
2. **Lane width.** A US highway lane is 12 ft (3.66 m). If lane lines can be measured on the road plane, use this as a cross-check.
3. **Vehicle length.** Muse Spark guesses the vehicle model and web\_search finds its real length (section 9); compare with the 3D box length.

If at least two methods disagree by more than 25%, widen the uncertainty range in the report and show a warning in the HUD.

### Step 4: lift objects to 3D (lift.py)

```python
import numpy as np

def lift_tracks(recon, masks_by_obj, scale):
    """masks_by_obj: {obj_key: {k: HxW bool mask (at recon resolution)}}"""
    pts, conf = recon["points"].astype(np.float32), recon["depth_conf"]
    tracks = {}
    for key, frames in masks_by_obj.items():
        rows = []
        for k, m in sorted(frames.items()):
            sel = m & (conf[k] > np.percentile(conf[k], 30))
            if sel.sum() < 30:
                continue
            p = pts[k][sel] * scale
            c = np.median(p, axis=0)                          # robust center
            lo, hi = np.percentile(p, [5, 95], axis=0)        # robust 3D box
            rows.append({"k": k, "center": c.tolist(), "bbox": [lo.tolist(), hi.tolist()]})
        tracks[key] = rows
    return tracks

def ego_track(extrinsic, scale):
    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    C = -np.einsum("tji,tj->ti", R, t)                        # camera center = -R^T t
    return C * scale
```

### Step 5: viser scene

viser sends point clouds, splines, camera frustums, sliders and buttons from Python to the browser over WebSocket ([source](https://pypi.org/project/viser/)). The HUD embeds this page in an `iframe`.

```python
# mercy/viewer/viser_scene.py
import viser, numpy as np

class Scene4D:
    def __init__(self, case, port=8080):
        self.s = viser.ViserServer(port=port)
        self.s.scene.set_up_direction("-y")                    # OpenCV: y points down
        self.case = case
        self.s.scene.add_point_cloud("/background", case.bg_points, case.bg_colors, point_size=0.02)
        for key, tr in case.tracks.items():                   # neon trajectories (hidden at first)
            h = self.s.scene.add_spline_catmull_rom(f"/traj/{key}",
                    np.array([r["center"] for r in tr]), color=case.color(key), line_width=4.0)
            h.visible = False
        self.slider = self.s.gui.add_slider("Frame", min=0, max=case.T - 1, step=1, initial_value=0)
        self.slider.on_update(lambda _: self.set_frame(self.slider.value))

    def set_frame(self, k: int):
        pts, cols = self.case.dynamic_points(k)               # only in-mask points, in object colors
        self.s.scene.add_point_cloud("/dynamic", pts, cols, point_size=0.035)  # same name = update in place
        self.s.scene.add_camera_frustum("/ego", fov=self.case.fov, aspect=self.case.aspect,
                    scale=0.3, wxyz=self.case.ego_wxyz(k), position=self.case.ego_pos(k))

    def camera_preset(self, name: str, k: int):
        target = self.case.focus_point(k)                     # impact point or selected object
        pos = {"top": target + [0, -25, 0.01], "side": target + [15, -3, 0],
               "driver": self.case.ego_pos(k)}[name]
        for c in self.s.get_clients().values():
            c.camera.position, c.camera.look_at = pos, target
```

The `orbit` preset is a small async loop that rotates 360 degrees around the target in 4 seconds (update the camera position at 30 Hz). Point budget: up to about 300k points for the background at 0.05-unit voxel downsampling and about 20k points per frame for the dynamic cloud; this stays smooth in a laptop browser.

### Stretch: real 4D Gaussian Splatting

If time allows, train an open-source 4DGS implementation offline (using VGGT poses exported in COLMAP format as initialization) and show a few pre-rendered orbit videos in the HUD. Never make the live demo flow depend on it.

## 9. Physics analytics and the fault report

The decision chain has three layers: **(1)** Muse Spark describes what happened in the video, **(2)** deterministic Python computes numbers from 3D tracks, **(3)** Muse Wonder combines the numbers, web sources and scene description into a cited report, and Laya audits it. Never let the LLM compute numbers; it only interprets them.

### 9.1 Scene understanding (scene.py)

The clip is uploaded to the Files API and given to Muse Spark as `input_file` (or `input_video`); the model reads visuals and embedded audio together ([source](https://dev.meta.ai/docs/video-understanding)).

```python
SCENE_SCHEMA = {"type": "object", "additionalProperties": False,
  "required": ["summary", "vehicles", "events", "impact_time_s", "traffic_controls",
               "weather", "road_type", "camera_vehicle_involved"],
  "properties": {
    "summary": {"type": "string"},
    "vehicles": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["label", "color", "body_type", "make_model_guess", "role"],
        "properties": {"label": {"type": "string"}, "color": {"type": "string"},
          "body_type": {"type": "string"}, "make_model_guess": {"type": ["string", "null"]},
          "role": {"type": "string", "enum": ["camera_vehicle", "striking", "struck", "bystander"]}}}},
    "events": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["t_s", "description"],
        "properties": {"t_s": {"type": "number"}, "description": {"type": "string"}}}},
    "impact_time_s": {"type": ["number", "null"]},
    "traffic_controls": {"type": "array", "items": {"type": "string"}},
    "weather": {"type": "string", "enum": ["dry", "wet", "snow", "unknown"]},
    "road_type": {"type": "string"},
    "camera_vehicle_involved": {"type": "boolean"}}}

def understand_scene(client, file_id):
    r = client.responses.create(
        model="muse-spark-1.3",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "You are a traffic forensics analyst. Describe only what is "
             "visible. Give timestamps in seconds from clip start."},
            {"type": "input_file", "file_id": file_id}]}],
        text={"format": {"type": "json_schema", "name": "Scene", "schema": SCENE_SCHEMA, "strict": True}},
        max_output_tokens=4000,      # reasoning shares the output budget
    )
    return json.loads(r.output_text)
```

### 9.2 Triangulating the impact moment

The impact time comes from three independent sources, and the HUD shows all three together:

- The Nexar label (`time_of_event`, minus the trim offset).
- Muse Spark's `impact_time_s`.
- Geometry: the frame where the distance between the two objects' 3D boxes is minimal and at least one object's acceleration reaches its largest negative value.

If they agree within 0.5 seconds, use the geometric value; if not, show a warning and average the label and Spark values.

### 9.3 Formulas

Speed is the central difference of the smoothed center track:

```latex
v_k = \frac{\lVert \mathbf{p}_{k+1} - \mathbf{p}_{k-1} \rVert}{2\,\Delta t}
```

Braking distance and speed from skid length (mu: friction coefficient, g = 9.81 m/s²):

```latex
d_{brake} = \frac{v^2}{2\,\mu\,g} \qquad v_{skid} = \sqrt{2\,\mu\,g\,d}
```

Impact angle and velocity change (delta-V):

```latex
\theta = \arccos\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\lVert \mathbf{v}_1 \rVert \lVert \mathbf{v}_2 \rVert} \qquad \Delta v = \lVert \mathbf{v}_{after} - \mathbf{v}_{before} \rVert
```

Time to collision (TTC) at each frame is the gap divided by the closing speed. Pick mu by weather: dry asphalt 0.7 to 0.8, wet 0.4 to 0.5 (approximate general forensic ranges; the report states them as ranges).

### 9.4 Physics code

```python
# mercy/analysis/physics.py
import numpy as np
from scipy.signal import savgol_filter

def kinematics(centers: np.ndarray, fps: float):
    """centers: [N,3] in meters. Returns speed [N] m/s, acceleration [N] m/s², velocity vectors [N,3]."""
    w = min(9, len(centers) // 2 * 2 - 1)
    p = savgol_filter(centers, window_length=w, polyorder=2, axis=0) if w >= 5 else centers
    vel = np.gradient(p, 1.0 / fps, axis=0)
    speed = np.linalg.norm(vel[:, [0, 2]], axis=1)              # in the road plane (x,z)
    acc = np.gradient(speed, 1.0 / fps)
    return speed, acc, vel

def analyze(tracks, ego, fps, scale_bounds, mu_bounds):
    out = {}
    for name, s in {"low": scale_bounds[0], "mid": scale_bounds[1], "high": scale_bounds[2]}.items():
        # recompute speeds for each scale scenario -> uncertainty range
        ...
    # output per object: pre_impact_speed_mps [low, mid, high], peak_decel, impact_angle_deg,
    # delta_v, closing_speed, ttc_series, braking_distance_m [with mu range], impact_frame
    return out
```

`physics.json` stores every number as `{"value": mid, "range": [low, high], "unit": "km/h", "method": "..."}`. The HUD prints this format directly; the verdict report quotes it.

### 9.5 Web research (research.py)

Muse Spark gets the `web_search` tool and decides itself when to search; `url_citation` annotations on the answer give source URLs ([source](https://dev.meta.ai/docs/search-grounding)). Each research question is a separate request (multi-step research in one request is less reliable):

- Vehicle size: "Overall length in meters of a {make\_model\_guess}" as a scale cross-check.
- Rule: "US traffic law on {scenario: rear-end following distance / right-of-way at intersection / lane change}" (use the general US rule and the Uniform Vehicle Code if the state is unknown).
- NHTSA: crash statistics or safety rating for the scenario type, used as context in the report.

```python
def research(client, question: str) -> dict:
    r = client.responses.create(model="muse-spark-1.3", input=question,
            tools=[{"type": "web_search", "search_context_size": "medium",
                    "user_location": {"type": "approximate", "country": "US"}}],
            include=["web_search_call.results"])
    cites = []
    for item in r.output:
        if item.type == "message":
            for block in item.content:
                for a in (block.annotations or []):
                    if a.type == "url_citation":
                        cites.append({"title": a.title, "url": a.url,
                                      "quote_span": block.text[a.start_index:a.end_index]})
    return {"question": question, "answer": r.output_text, "citations": cites}
```

Research is cached during ingest; only the verdict call runs live. If the internet drops during the demo, the cache is enough.

### 9.6 Fault report (verdict.py)

Instead of combining web search and structured output in one request, use two steps: research (9.5, cached) then the report (below, with `text.format`). This preserves the schema guarantee.

```python
VERDICT_SCHEMA = {"type": "object", "additionalProperties": False,
  "required": ["headline", "timeline", "key_measurements", "fault_assessment",
               "applicable_rules", "overall_confidence", "limitations"],
  "properties": {
    "headline": {"type": "string"},                     # one cold sentence
    "timeline": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["t_s", "event", "evidence"], "properties": {
        "t_s": {"type": "number"}, "event": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}}}}},   # e.g. "physics.car:1.pre_impact_speed"
    "key_measurements": {"type": "array", "items": {"type": "string"}},
    "fault_assessment": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["party", "share_pct_range", "rationale", "citations"], "properties": {
        "party": {"type": "string"},
        "share_pct_range": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}}}}},
    "applicable_rules": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["rule", "source_url"], "properties": {
        "rule": {"type": "string"}, "source_url": {"type": "string"}}}},
    "overall_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "limitations": {"type": "array", "items": {"type": "string"}}}}

SYSTEM = ("You are MERCY, a forensic reconstruction analyst. Use ONLY the provided evidence JSON, "
  "physics measurements and cited sources. Every fault statement must reference a measurement id or "
  "a source URL. Report ranges, never single exact speeds. State limitations of monocular video. "
  "This is an estimate for investigators, not a legal determination. Tone: cold, precise, terse.")

def verdict(client, scene, physics, research, notes):
    r = client.responses.create(
        model=os.environ.get("REASONING_MODEL", "muse-wonder-1.0"),
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": json.dumps({"scene": scene, "physics": physics,
                    "research": research, "investigator_notes": notes})}],
        text={"format": {"type": "json_schema", "name": "Verdict", "schema": VERDICT_SCHEMA, "strict": True}},
        reasoning={"effort": "high"},        # verify the parameter name at /docs/reasoning
        stream=False,
    )
    return json.loads(r.output_text)
```

Once the report arrives, the Laya integrity questions (section 4) run on `{"verdict": fault_assessment text, "evidence": physics summary}`; if `unsupported_fault` exceeds 0.5, the HUD shows a "REVIEW REQUIRED" badge. This shows the jury that the system audits itself.

For demo speed: run the verdict call once during ingest and cache it; live, the "analyze" command starts a real call, and if it has not returned within 20 seconds the cached report is shown with a small "cached" label. Being honest about live versus cached builds trust with the jury.

## 10. Muse Image: visual reconstruction

Rule for image generation: **geometry comes from deterministic code, aesthetics from Muse Image.** First matplotlib draws an accurate, to-scale overhead diagram; then Muse Image's edit endpoint restyles it into a "tactical blueprint" using the diagram as the reference. Muse Image edits change only what you ask and keep the rest intact, so trajectory positions do not drift ([source](https://dev.meta.ai/docs/image-generation)).

### 10.1 Deterministic diagram (diagram.py)

- Axes: road plane (x, z) in meters; 5 m grid; a "camera heading" arrow instead of a north arrow.
- Each object's trajectory in its own color, solid before impact and dashed after, with a timestamp dot every 0.5 seconds.
- Impact point as a cross with a speed label ("38 to 46 km/h").
- Vehicle footprints as rectangles computed from the 3D boxes.
- Output: 1536x1024 PNG, dark background, `diagram.png`.

### 10.2 Blueprint generation (blueprint.py)

```python
import base64
from mercy.meta_client import meta_client

BLUEPRINT_PROMPT = (
  "Restyle this accident diagram as a futuristic forensic tactical blueprint: deep navy background, "
  "cyan grid lines, glowing neon vehicle outlines and trajectories, small HUD-style labels. "
  "Keep every trajectory, vehicle position, label and the impact marker exactly where they are. "
  "Do not add vehicles, people or text that is not in the original.")

def make_blueprint(diagram_path: str, out_path: str):
    client = meta_client()
    with open(diagram_path, "rb") as img:
        r = client.images.edit(
            model="muse-image-1.0", prompt=BLUEPRINT_PROMPT, image=img, n=1,
            extra_body={"reasoning_strength": "low",
                        "output_format": "webp",
                        "tool_enablement": {"enable_web_search": False,
                                            "enable_image_search": False,
                                            "enable_shell": False}})
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(r.data[0].b64_json))
```

Notes:

- Muse Image runs its search tools itself; adding a `web_search` tool returns 400. Because a reference image already exists here, turning the tools off makes it faster.
- The `size` parameter sets the aspect ratio, not exact pixels; the output size may not match exactly. Use `object-fit: contain` in the HUD.
- If `output_format` is not a valid SDK field, send it in `extra_body` (as above).
- Price: $0.01 per successful image; 3 to 5 variations per case is negligible.
- The HUD always shows the image with an "AI ILLUSTRATION - NOT EVIDENCE" watermark. Add the watermark in the HUD with CSS, not through Muse Image (this prevents the model from garbling the text).

### 10.3 Before/after panel (stretch)

Draw two variants from the same diagram, at "impact minus 2 s" and "impact plus 1 s", stylize each separately, and show them in the HUD with a before/after slider. With multi-turn editing on the Responses API (`previous_response_id`) the second image keeps the first's style.

### 10.4 Blind-spot visualization (optional, handle with care)

Generating a moment the camera did not see ("what the impact looked like from the side") is impressive but it is not evidence. Give the prompt only computed positions numerically ("white sedan at 12 m, angle 34 degrees"), label the output as speculative, and never feed it into the verdict.

## 11. Frontend: design system, layout and the UI-to-backend contract

The frontend is already built and ships as `frontend/` (static HTML, CSS and vanilla JS, no build step). It runs on its own in **mock mode** with a synthetic intersection case, and switches to the real backend with `?mode=live` without any UI change. The backend's only job is to serve the folder and speak the contract below.

### Design intent

The look is a calm, dark, chat-style workspace (the same family as claude.ai or chatgpt.com): one accent colour, generous spacing, quiet borders, no neon, no scanlines. The film feeling comes from small details instead: a slowly breathing ring in the logo, a thin vignette on the dashcam view, monospace readouts, a point-cloud reconstruction that fills in as time advances, and a pulse at the impact point.

| Token | Value | Use |
| --- | --- | --- |
| Background / surface | `#121415` / `#1d2123` | Page and cards |
| Accent (teal) | `#7fd1c7` | Vehicle 1, primary actions, focus |
| Warm (amber) | `#e8a76a` | Vehicle 2, warnings |
| Gold | `#f2d28b` | Impact marker, cached-result chip |
| Text | `#e9ecea` / `#a9b2b4` / `#7a8588` | Primary, secondary, tertiary |
| Type | Inter (system fallback), ui-monospace for numbers | Readouts use tabular figures |

### Layout

Top bar (brand, case title, data-source and connection chips), then a left stage and a 380 px right panel. The stage holds the dashcam view with mask and track overlays next to the 4D reconstruction view (Top, Orbit, Follow), a timeline with speed curves, event markers and a draggable playhead, and a pill-shaped command bar with a push-to-talk mic. The right panel has four tabs: Findings (verdict, responsibility bars with ranges, factors, sources, Laya integrity badge), Measurements (value, range bar, method, confidence), Blueprint (Muse Image output with an AI-GENERATED watermark) and Log. Below 1100 px width the panel moves under the stage.

### Files

`index.html`, `css/styles.css`, and `js/` with `util`, `sim` (mock case only), `transport` (mock or WebSocket), `feed` (dashcam view), `scene3d` (canvas 3D, no WebGL dependency), `timeline`, `panel` and `main` (state and render loop).

### Server to client messages

Each frame is JSON, `{"type": …, …payload}` or `{"type": …, "data": …}`.

| `type` | Payload | UI effect |
| --- | --- | --- |
| `hello` | `id, title, duration, fps, impact_t, events[{t,label,kind}], objects[{id,label,color}]` | Loads the case, draws event markers |
| `physics` | `[{id,label,value,lo,hi,unit,method,conf}]` | Measurements tab; every value carries a range and a method |
| `transcript` | `{text, final}` | Streaming caption above the command bar |
| `intent` | `{intent, confidence, ms, router}` | Shows the Laya decision and latency |
| `state` | `{t?, playing?, speed?, selected?, layers?{masks,tracks,hud}}` | Moves the playhead, isolates a vehicle, toggles layers |
| `verdict` | `{status:"running", stage}` then `{status:"ready"\|"cached", data}` | Skeleton while running, then the Findings tab; `cached` shows a gold chip |
| `image` | `{status:"running"}` then `{status:"ready", url, caption, watermark}` | Blueprint tab |
| `error` | `{message}` | One calm line in the conversation strip |

The `verdict.data` object is `{headline, fault[{party,pct,lo,hi}], confidence, factors[{text,weight}], citations[{title,url}], integrity{passed,total,note}, caveats[]}`, which maps one to one onto `VERDICT_SCHEMA` in section 9.6 once the backend adds the display fields.

### Client to server messages

- `cmd {text}`: a typed command; the backend runs it through Laya exactly like a transcript.
- `ptt {down: true|false}`: push-to-talk; the backend owns the microphone, the browser never records audio.
- `ui {patch}`: the user scrubbed or toggled something; the backend stores it as the single source of truth in `StateHub`.

### Intents the UI understands

`play`, `pause`, `restart`, `slow_motion`, `seek_impact`, `seek_brake`, `isolate_1`, `isolate_2`, `show_all`, `show_masks`, `hide_masks`, `show_tracks`, `hide_tracks`, `analyze_fault`, `blueprint`. Laya's `INTENT_QUESTION` options (section 4) should use these exact names; the backend turns each one into a `state`, `verdict` or `image` message.

### Swapping mock for real

1. Serve `frontend/` as static files and open a WebSocket route at `/ws`.
2. Send `hello`, then `physics`, from `data/cases/<id>/`.
3. For real video, set `M.feed.video` to a video element and draw it before the overlays in `feed.js`; for the cloud, call `M.scene.setCloud(rows)` with `[x,y,z,kind]` rows; replace `M.sim.pos(id, t)` with tracks from `physics.json`.
4. Verify with `?mode=live` before touching any UI code. Bugs found in the frontend are fixed there, not worked around in the backend.

## 12. Datasets and demo clip preparation

The main source is Nexar: 1,500 dashcam clips of about 40 seconds each; collision and near-miss clips have the exact event time and alert time labelled, plus lighting, weather and scene type labels ([source](https://arxiv.org/abs/2503.03848)). It is on Hugging Face as `nexar-ai/nexar_collision_prediction` under the Nexar open data license, which restricts unethical use ([source](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)). The ready-made `time_of_event` label makes the "rewind to 3 seconds before impact" command reliable for the demo.

| Dataset | Use | Note |
| --- | --- | --- |
| [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) | Main demo cases | `time_of_event`, weather and scene labels; read the license and cite it in the README |
| DoTA (Detection of Traffic Anomaly) | Extra dashcam anomalies | Temporal anomaly windows; check the license |
| UCF-Crime | CCTV case (optional) | Fixed camera, so use the Step 2C fallback for a 2.5D scene |
| NHTSA data (via web\_search) | Context and statistics | No raw download; use a research call with cited quotes |

### Case selection criteria

Do not process the whole dataset. Skim 15 to 20 candidate clips and pick 3 cases:

1. **Hero case:** daytime, dry road, collision between two vehicles in front of the camera car (third-party view, cleanest geometry for reconstruction).
2. **Camera vehicle involved:** rear-end or lane change; tells the ego speed and braking distance story.
3. **Hard conditions:** night or wet road; shows the system honestly widening its uncertainty range.

Exclude: heavy camera shake, dense rain drops, collisions outside the frame, sources below 15 fps.

### Ingest command

```bash
python scripts/ingest_case.py --video raw/nexar_00417.mp4 --case hero \
    --t-event 23.4 --concepts car truck pedestrian "shattered glass" debris "license plate"
# Order: trim -> frames -> upload -> scene (Spark) -> SAM (parallel) -> VGGT -> scale -> lift
#        -> physics -> research (web_search) -> verdict (cache) -> diagram -> blueprint
```

Every step must be idempotent: skip if the output file exists (`--force` to redo). That lets interrupted work on the GPU machine resume where it stopped. Each case folder is copied to the laptop with `rsync` after ingest.

### Compute environment

- Ingest: one cloud GPU with 24 GB or more VRAM (for example L4/A10/A100) or Kaggle/Colab. VGGT and the optional MonST3R run here.
- Demo: a laptop without a GPU. Laya runs on CPU with preloaded models in a few hundred ms; viser and FastAPI are light.
- Network: Meta API calls (ASR, verdict, live SAM, image) need internet; keep a phone hotspot ready in case the venue Wi-Fi is weak.

## 13. Hackathon timeline and demo script

The plan assumes a 24-hour hackathon; with 36 to 48 hours, give extra time to phases 4 and 6. Golden rule: **have an ugly but end-to-end working build by hour 6**, then polish.

### Build phases

1. **Hours 0-2 · Foundation.** Repo skeleton, `.env`, Meta API key with 4 smoke tests (Spark text, SAM single image, ASR file, Muse Image single image). Install Laya and test the section 4 intent question on 10 sample sentences. Start the GPU machine and install VGGT.
2. **Hours 2-6 · Hero case ingest.** Trim, frames, SAM (3 concepts), VGGT, `recon.npz`. viser with background, dynamic cloud and slider. **Gate: at hour 6 the 3D scene scrubs through time.**
3. **Hours 6-10 · Voice and commands.** ASR client, clause splitting, Laya, slot extraction, `StateHub`. Start with typed commands (`text_command`), then move to voice. **Gate: "rewind to 3 seconds before impact" works by voice.**
4. **Hours 10-15 · Physics and verdict.** Scale, lift, physics, research, verdict, Laya integrity. Sanity-check the numbers by eye (is the vehicle speed plausible?).
5. **Hours 15-19 · HUD.** Connect the frontend (section 11) to the real backend: switch it from mock to live, wire overlays and verdict. Ingest the other 2 cases in the background.
6. **Hours 19-22 · Polish and robustness.** Cache fallbacks, error toasts, orbit animation, optional sound effects, the live SAM wow moment.
7. **Hours 22-24 · Rehearsal.** Rehearse the demo script at least 5 times end to end, once with Wi-Fi off. Record a backup demo video.

### 3-minute demo script

1. **(0:00) Hook.** "Millions of insurance and traffic disputes a year are settled by people staring at a shaky dashcam video. MERCY turns that video into a scene you can walk through." The hero case is open in the cockpit.
2. **(0:20) First command.** Speak: *"Isolate the vehicles and rewind to three seconds before impact."* Intent chips appear within a second (laya, 0.9x), vehicles are outlined, the timeline jumps.
3. **(0:45) 4D.** *"Orbit the scene."* The camera circles the impact point; the background is static while the vehicles move through time. Drag the timeline by hand.
4. **(1:15) Evidence.** *"Show the braking trajectory of car one."* The trajectory draws; the HUD shows the speed range and braking distance. *"Isolate the shattered glass."*
5. **(1:40) Verdict.** *"Analyze fault."* The analysis panel fills: a cold one-sentence headline, measurements, fault share ranges, cited rules, the Laya integrity badge (green).
6. **(2:10) Visual.** *"Generate the overhead blueprint."* The blueprint appears with its watermark.
7. **(2:25) Live evidence and Turkish.** Speak Turkish: *"Trafik ışığını izole et."* A concept that was not precomputed is segmented live on the current frame.
8. **(2:45) Close.** One architecture slide: "Reflexes in Laya, judgment in Muse. The whole Meta model family in one flow." Privacy note: plates are blurred automatically using SAM.

### Jury questions, prepared answers

- **"Is it really 4D Gaussian Splatting?"** No. The MVP is a time-indexed point cloud; 4DGS is an offline stretch goal. Be honest.
- **"How accurate are the speeds?"** Monocular video carries scale uncertainty, which is why every value is given as a range with its method.
- **"Why Laya, isn't an LLM enough?"** An LLM takes 1 to 3 seconds per command; Laya takes tens of ms and gives a calibrated confidence, and hands off to the LLM when unsure.
- **"Is it legally valid?"** No; it is an estimate tool for investigators, and the report says so explicitly.

## 14. Risks, fallbacks, ethics and legal notes

The biggest risk is that 3D reconstruction comes out scattered on moving dashcam footage; that is why the hero case is chosen early and every live step has a cached fallback.

| Risk | Likelihood | Impact | Fallback |
| --- | --- | --- | --- |
| VGGT poses corrupted by moving vehicles | Medium | Scattered 3D scene | Grey out dynamic regions with the SAM mask and redo the pose pass; else MonST3R; else pick a cleaner clip |
| Inconsistent scale estimate | Medium | Nonsense speeds | Cross-check all three methods, widen the range; prioritize the vehicle-length method |
| SAM cannot find a concept (for example skid marks) | High | Empty command result | Drop the concept from the dictionary; show "no evidence detected" in the HUD (that is also a result) |
| ASR mishears in a noisy venue | Medium | Wrong command | Close microphone or headset; `keywords` biasing; keyboard `text_command` fallback |
| Laya misclassifies an intent | Low to medium | Wrong action | `min_confidence` threshold plus Muse Spark fallback; tune the threshold on 30 examples |
| Muse Wonder unavailable or slow | Medium | Verdict delayed | Same request on `muse-spark-1.3`; at a 20 s timeout show the cached report (with a "cached" label) |
| Wi-Fi drops | Medium | Live calls fail | Phone hotspot; all ingest outputs stored locally; backup demo video |
| Rate limit (429) | Low | Delay | SDK `max_retries`; cap ingest parallelism at 4 |
| Too many points in the browser | Low | Stutter | Voxel downsampling, cap dynamic points at 20k per frame |

### Ethics and legal notes

- **Positioning.** MERCY is a research assistant; it makes no legal determination. Every report and HUD footer reads: *"Estimated reconstruction from monocular video. Not a legal determination."*
- **Uncertainty.** No single-point speed is shown; every measurement carries a range and a method.
- **Personal data.** Plates and faces are blurred with SAM masks. The demo uses only openly licensed datasets; state the Nexar license's ethical-use restrictions in the README.
- **Generated images.** Muse Image outputs are not evidence, carry a watermark, and never feed the decision chain.
- **Data retention.** Meta Model API Standard mode does not train on content; verify that ZDR is enabled for speech on your account and avoid `-contributor` models.
- **Bias.** Fault share is computed only from measurements and rules, never from driver type or vehicle brand; the system prompt forbids this explicitly.
- **Geographic restriction.** Muse Wonder is unavailable in some countries (for example South Korea); keep the Spark fallback ready depending on the team's location.

## 15. Instruction block for the AI coding agent

Give the agent this whole document, then paste the block below as the first message. Run the agent module by module; stop at each gate (section 13), verify the output yourself, then let it continue. **The frontend already exists** (`frontend/`, section 11); the agent builds only the backend and must not change the UI contract.

```text
ROLE: You are a senior Python engineer building the BACKEND of "Project MERCY", a voice-controlled
forensic crash-reconstruction cockpit, for a 24-hour hackathon. The attached design document is the spec.
A finished static frontend exists in frontend/ (vanilla HTML/CSS/JS, no build step). It runs in a mock
mode today; your job is to make it run in live mode by implementing the contract in section 11.
Follow the document's module boundaries, file names, data contracts (data/cases/<case_id>/*) and schemas.

HARD RULES
1. Python 3.11. Backend FastAPI + WebSocket. Serve frontend/ at "/" with StaticFiles. Do NOT modify
   frontend/ except to fix a bug you can show; if the contract needs a change, write it in CHANGES.md.
2. All Meta models go through the OpenAI Python SDK with base_url=https://api.meta.ai/v1 and
   api_key=MODEL_API_KEY. Model IDs come from env vars (see section 5).
3. Before writing code against any external API, re-read the cited doc page and match field names
   exactly: SAM (dev.meta.ai/docs/sam/segmenting, /reading-segmentation, /client-libraries),
   ASR (dev.meta.ai/docs/speech-to-text), images (dev.meta.ai/docs/image-generation),
   web search (dev.meta.ai/docs/search-grounding), structured output (dev.meta.ai/docs/structured-output),
   reasoning (dev.meta.ai/docs/reasoning). If a name in this spec disagrees with the docs, the docs win;
   note the change in CHANGES.md.
4. Never write a custom parser for SAM mask payloads; use meta-sam-parser. Introspect its API with dir()
   first and write a 10-line test that decodes one real mask and asserts mask H,W == box height,width.
5. LLMs never compute numbers. Physics is deterministic NumPy/SciPy code. LLMs only describe and judge.
6. Every pipeline step is idempotent (skip if output exists, --force to redo) and logs timing.
7. Every live API call has a timeout and a cached fallback; the UI labels cached results "cached".
8. Keep secrets server-side. The browser never sees MODEL_API_KEY. Mic capture runs in Python.

BUILD ORDER (stop after each step, print how to verify it, wait for confirmation)
A. Skeleton, config, meta_client, 4 smoke tests (spark text, sam single image, asr file, image gen).
B. ingest/video.py + ingest/segment.py + overlays.json for one case.
C. ingest/reconstruct.py (VGGT) + viewer/viser_scene.py with frame slider. Gate: scrubbable 3D scene.
D. voice/asr.py + voice/commands.py (clause split, Laya batch, slots, Muse Spark fallback) + state.py
   + server/app.py implementing /ws and /api/case exactly as in section 11. Gate: the frontend, opened with
   ?mode=live, shows the real case and typed and spoken "rewind to 3 seconds before impact" both work.
E. lift.py, scale estimation, analysis/physics.py -> physics.json with ranges.
F. analysis/research.py, analysis/verdict.py, analysis/integrity.py (Laya); emit the `verdict` WebSocket
   message in the exact shape of section 11 (running -> ready | cached).
G. generation/diagram.py + generation/blueprint.py; emit the `image` message.
H. scripts/ingest_case.py end-to-end, scripts/run_demo.py, README with exact run commands.

TESTS
- Unit: clause splitting (EN+TR), slot regex, kinematics on a synthetic constant-velocity track
  (speed within 2%), impact-frame detection on a synthetic two-body track, overlay offset math.
- Contract: pydantic validation of every JSON file in a case folder and of every WebSocket message.
- Laya: a 30-sentence labelled command set; print accuracy and the confidence threshold chosen.
```

### Acceptance criteria

- [ ] `python scripts/ingest_case.py ...` produces every contract file from an empty case folder.
- [ ] `python scripts/run_demo.py --case hero` opens the cockpit (8000) and viser (8080) with one command.
- [ ] Holding Space and speaking shows live captions; on release the action is applied within 1.5 s.
- [ ] Turkish and English commands both show an intent chip with the path (`laya` or `deep`).
- [ ] Timeline, video and viser show the same frame (0.05 s tolerance).
- [ ] Every value in `physics.json` has `range`, `unit` and `method`.
- [ ] The analysis panel shows clickable citation URLs and the Laya integrity badge.
- [ ] The blueprint carries its watermark; plates are blurred in the redacted video.
- [ ] With the internet off, the demo plays to the end on cache (voice excepted).
- [ ] All unit tests pass.

## Sources

- [Meta Model API: Models](https://dev.meta.ai/docs/models)
- [Speech to text (Muse Voice Transcribe)](https://dev.meta.ai/docs/speech-to-text)
- [Media segmentation (SAM 3.1)](https://dev.meta.ai/docs/media-segmentation)
- [SAM: Segmenting with prompts](https://dev.meta.ai/docs/sam/segmenting)
- [SAM: Reading segmentation output](https://dev.meta.ai/docs/sam/reading-segmentation)
- [SAM: Client libraries](https://dev.meta.ai/docs/sam/client-libraries)
- [Image generation (Muse Image)](https://dev.meta.ai/docs/image-generation)
- [Search grounding](https://dev.meta.ai/docs/search-grounding)
- [Video and audio understanding](https://dev.meta.ai/docs/video-understanding)
- [Structured output](https://dev.meta.ai/docs/structured-output)
- [Laya (GitHub)](https://github.com/NandhaKishorM/laya)
- [VGGT model card](https://huggingface.co/facebook/VGGT-1B)
- [MonST3R (GitHub)](https://github.com/Junyi42/monst3r)
- [viser (PyPI)](https://pypi.org/project/viser/)
- [Nexar Dashcam Collision Prediction Dataset (arXiv)](https://arxiv.org/abs/2503.03848)
- [Nexar dataset (Hugging Face)](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)

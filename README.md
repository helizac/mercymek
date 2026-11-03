# mercymek: voice-driven crash reconstruction

> After a crash it is one driver's word against another's. mercymek turns the dashcam clip into the answer: the key
> instants rebuilt in 3D, fault shares as honest ranges, plates read, the accident statement filled in, the evidence
> sealed. Built on all five Meta Model API surfaces (Muse Voice Transcribe, Spark, Wonder, Image, SAM 3.1) plus VGGT.
> Hackathon pitch, 3-minute video script and judge quick-start: [SUBMISSION.md](SUBMISSION.md).

**Quick start (no GPU, no API key):** `python scripts/pack_demo.py --unpack mercymek-demo-data.zip` (the demo data,
released separately because one case is 370 MB), then `python scripts/run_demo.py --case hero --no-viser --open`.

mercymek turns one dashcam clip into an investigation you talk to. The video is the main screen; say what you want
and the other screens slide in: the key instants as **3D moments** reconstructed from the footage, a **map** of where
it happened,
the **report** with fault shares, and the **evidence** (vehicles, licence plates, files).

Heavy work runs once per case at ingest; live, only voice, intent, reasoning, enhancement and image generation run.

```
ingest (once per case)                                              live (laptop)
clip ─► event time (Spark) ─► trim ─► scene (Spark) ─► SAM 3.1 masks (vehicles, plates, faces, sky)
     ─► VGGT-1B depth + poses (CUDA / XPU / CPU) ─► road plane, metric scale ─► 3D tracks ─► physics (ranges)
     ─► one Muse empty scene + vision review/repair ─► Kaggle SHARP static Gaussian world,    mic ─► transcribe ─► rules / Laya / Spark ─► action
        complete vehicle meshes with observed paint/plates, replayed on tracked paths          StateHub ─► HUD (/ws), viser (:8080)
     ─► RAFT optical-flow slow motion of the impact (×4)
     ─► plates: multi-frame fusion (ECC + RAFT) + Real-ESRGAN + reading
     ─► location: geo-sleuth (OCR, metadata) + Spark judgment + OpenStreetMap
     ─► web research ─► verdict (Wonder watches the clip + measurements) ─► integrity audit ─► diagram ─► blueprint
     ─► European Accident Statement (section 12) ─► evidence seal (signed SHA-256 manifest)
```

## Run

```bash
python scripts/run_demo.py --case hero --open        # http://localhost:8000  (viser on :8080)
```

Deep links: `http://localhost:8000/?case=snow_pileup&view=map&t=8`.

## Talk to it

Hold **Space** (or the mic, top left) and speak; release to run. mercymek answers out loud in short sentences
(ElevenLabs v4 Turbo, ~150 ms to first audio, repeated phrases instantly from a cache). The glow around the mic
follows the voice.

| Say | Happens |
| --- | --- |
| “Isolate the vehicles and rewind to three seconds before impact” | masks isolate both parties, the video jumps |
| “Show the moment before impact in 3D”, “show the impact in 3D”, “show braking in 3D”, “show me 3D” | open that instant with complete vehicle meshes. Replay collision plays at half speed; the timeline scrubs the same replay. Drag to look, wheel/WASD to move, Q/E for height, right-drag to pan. Reset camera returns to the current video pose; Dashcam view follows the source camera explicitly |
| “Driver view” | the moment held still at the dashcam's own viewpoint |
| “Bullet time” | the impact moment in 3D |
| “Replay the impact in super slow motion” | ×4 replay whose in-between frames are reconstructed with RAFT optical flow |
| “What if the camera car had been going 30 km/h?”, “What if it braked a second earlier?”, “…on a wet road?” | avoidance analysis as ranges; the video replays from the hazard at half speed |
| “Objection: the SUV cut in without signalling”, “I argue the camera car was tailgating” | Mercy mode: the judge weighs the argument against the evidence; if it holds, the liability meter moves |
| “Fill in the accident statement” | the European Accident Statement (circumstances and points of impact) from the evidence |
| “Verify the evidence” | re-hashes every file and checks the signature of the evidence seal |
| “Where did this happen?” | OpenStreetMap with the crash location, its uncertainty, heading and evidence |
| “Who is at fault?” | report: the model watches the clip and reads the measurements, rules and plates; fault shares as ranges, integrity audit |
| “Zoom in on the licence plate” | aligned multi-frame fusion + ×4 super-resolution, plate read with unreadable characters marked `?` |
| “Enhance the white car” / “make it sharper” | same enhancement on any tracked object or the zoomed region |
| “Show me the street view” | open street-level photos near the scene (Mapillary with a token, Panoramax otherwise) |
| “How fast was the blue car going?”, “Why is it their fault?” | a short spoken answer from the case facts |
| “Generate the blueprint” | to-scale overhead diagram restyled as an illustration (watermarked, never evidence) |
| “Show the evidence” / “back to the video” | screens |
| “Note that the signal was not visible” | stored and passed to the next verdict |
| “Pause”, “slow motion”, “hide the masks”, “reset” | playback, layers, reset |

Keys: ←/→ ±1 s (Shift: one frame), P play/pause, 1–5 screens, Esc closes the enhancement or the replay, `/` opens a
hidden typed command line.

Layout: everything you read is in the left column (case, screens, time, location, the conversation); the right
stage is 90vh with the 6vh timeline under it.

Settings for voice and street view (`.env`): `ELEVENLABS_API_KEY`, `ELEVEN_VOICE_ID`, `ELEVEN_MODEL=eleven_v4_turbo`,
`TTS_CHAR_BUDGET` (replies become text-only once spent), `MAPILLARY_TOKEN` (optional, free from mapillary.com/dashboard/developers).

## Ingest

The current generated 3D/4D pipeline, Kaggle cache setup, reusable vehicle GLBs, source-quality requirements and
research decisions are documented in [docs/reconstruction.md](docs/reconstruction.md).
The rebuilt scenes include a generated backwards road extension and a **Replay approach** view at a fixed camera
height. The continuation is explicitly marked as generated; its unseen geometry is estimated.

```bash
python scripts/kaggle_gpu.py setup              # one-time private CUDA runtime/model cache
python scripts/rebuild_4d.py --case hero cutin_rearend lanechange night_rain snow_pileup
```

New examples require an open-source provenance/review manifest, native 720p+ video, an ordinary physical collision
and clear unblurred plates. Upstream VGGT and RAFT jobs default to Kaggle. The local commands below also describe
the legacy reconstruction environment retained for reproducing earlier artifacts.

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -e .
uv venv --python 3.11 .venv-recon
uv pip install --python .venv-recon torch torchvision --index-url https://download.pytorch.org/whl/xpu   # or cu124 / cpu
uv pip install --python .venv-recon "vggt @ git+https://github.com/facebookresearch/vggt.git" "open3d==0.19.0" "numpy<2" trimesh opencv-python pillow huggingface_hub python-dotenv transformers
uv pip install --python .venv-recon einops jaxtyping omegaconf plyfile dacite safetensors lightning scikit-video matplotlib e3nn colorspacious "numpy<2"
uv pip install --python .venv-recon openai meta-sam-parser scipy "numpy<2"   # 3D moments: clean plates, vehicle masks
git clone --depth 1 https://github.com/InternRobotics/AnySplat vendor/AnySplat   # feed-forward splats (MIT); weights lhjiang/anysplat
git clone https://github.com/Oldcircle/geo-sleuth vendor/geo-sleuth           # geolocation skill (MIT); needs uv
cp .env.example .env                                                             # MODEL_API_KEY, HF_TOKEN

python scripts/download_nexar.py --split train/positive --limit 100
python scripts/ingest_case.py --video data/raw/nexar/train/positive/00348.mp4 --case hero
python scripts/ingest_case.py --video data/raw/commons/400_SB_Crash.webm --case snow_pileup \
    --title "Hwy 400 SB · Snow pileup" --source "Wikimedia Commons (public domain)"      # event time found automatically
python scripts/ingest_case.py --case hero --force moments location verdict               # redo selected steps
```

Sharper 3D moments with [FastGS](https://github.com/fastgs/FastGS) (per-scene training, CUDA only, so on Kaggle):

```bash
python scripts/fastgs_export.py                  # frames + loss masks + COLMAP cameras -> data/fastgs/mercy_fastgs.zip
python scripts/fastgs_export.py --clip           # whole clip per case, all vehicles masked -> mercy_fastgs_clip.zip
# upload it as a private Kaggle dataset, run kaggle/fastgs_moments.ipynb on a T4, download fastgs_out.zip
python scripts/fastgs_import.py fastgs_out.zip   # replaces the AnySplat world of each moment (kept as *.anysplat.splat)
```

Every step is idempotent. Open3D's TSDF needs NumPy < 2, so the mesh step runs in `.venv-recon` like VGGT.
AnySplat is CUDA-oriented; mercymek runs only its inference, with its three CUDA libraries replaced by small PyTorch
stand-ins (`vendor/anysplat_shims`), so it also runs on Intel Arc (XPU, ~30 s per moment) or CPU. The RAFT slow
motion takes ~100 s per case on the same GPU.

## Cases

| Case | Source | Notes |
| --- | --- | --- |
| `hero` | Nexar 00348 (day, suburban) | blue hatchback crosses into the camera car at a signal |
| `lanechange` | Nexar 00029 (highway) | cut-in near-miss |
| `night_rain` | Nexar 00058 (night, rain) | wet-road ranges widen honestly |
| `snow_pileup` | Wikimedia Commons, public domain (Hwy 400, Toronto) | GPS + speed burned into the video: the location and the speed reconstruction are checked against ground truth |
| `cutin_rearend` | Wikimedia Commons, public domain | readable plates: plate enhancement and reading |
| `london_rearend` | [Dash Cam Owners World](https://www.youtube.com/watch?v=V758npfi-w4), CC BY 3.0 | roundabout sudden-stop rear-end; original-frame plate `LM58 BLZ` |
| `vegas_rearend` | [Fyrokid](https://www.youtube.com/watch?v=-u6YCwnPoL0), CC BY 3.0 | passing witness records sedan/pickup collision; original-frame bystander plate `9296C` |

Nexar blurs plates before release. The two new examples preserve verified readable plates from native video
frames, with exact frame numbers and unmodified crops in their evidence panels. The Vegas plate identifies a
bystander; neither involved vehicle is assigned that plate. Source authorship, license and video links appear
in the Files panel. Source titles are attribution, not a finding about intent or fault.

## What is real and what is estimated

The default 3D/4D view is an AI-generated empty environment reconstructed with SHARP, with class-based complete
vehicle bodies registered to observed video boxes and tracked over time. Type and paint come from the video; plate
text comes from associated OCR, with unknown characters preserved. Road scale, camera height, hidden bodywork and
motion retain uncertainty; deformation is not reconstructed. Labelled collisions with an identified incident pair
use an explicit contact prior at the event time, with a velocity impulse rather than smoothing the impact backwards
through the approach. Near-misses receive no contact prior. Source reprojection errors remain recorded separately.
See [the current reconstruction notes](docs/reconstruction.md) for per-stage provenance and limitations.
The following two bullets describe the retained **legacy** renderer:

- A legacy 3D moment is one video frame placed in 3D at its metric VGGT depth (one camera-facing splat per pixel, depth
  edges kept sharp with an inpainted skirt behind them), in front of the static world that AnySplat reconstructs from
  the frames 2.5 s before to 1 s after (vehicles, sky and everything glued to the camera, such as the hood, dashboard
  reflections and timestamps, removed before fusion; aligned on the same pixels as the VGGT depth and levelled so the
  local road is y = 0). From the dashcam's viewpoint it is the frame itself; leaving that viewpoint reveals the
  limits of the original forward-facing capture. What
  was never seen shows as the sky or road colour.
- The vehicles in a legacy moment are their own pixels from that frame, moved along their tracked (smoothed) paths for the
  1.5 s before it; they may turn at most 25°, because a layer is only the side the camera saw. Behind them, Muse Image
  removed the vehicles from a copy of the frame (the clean plate): only the pixels the vehicles and their shadows
  covered come from it, with depth from the road plane, and the HUD says that background is AI-filled. It is never
  evidence. Without a clean plate (no key, or it does not line up with the frame) the vehicles stay frozen.
- Counterfactuals keep the camera car's measured position when the hazard appeared and its own braking effort, and
  change one thing (speed, reaction, friction); results are ranges over friction and reaction time.
- Mercy-mode arguments only move the meter when the judge finds them supported by the case facts; the ledger feeds
  the next verdict as investigator notes.
- The slow-motion replay's in-between frames are synthesised from optical flow; they are a viewing aid, not evidence.
- The evidence seal proves the files did not change after sealing on this machine; it does not prove the source
  footage itself is authentic.
- Every number is a range with its method. Scale comes from an assumed 1.2–1.5 m camera height.
- Enhancement fuses real frames first, then upsamples; the plate reader marks unreadable characters instead of guessing.
- The blueprint is an illustration and never feeds the verdict.
- Faces are blurred with SAM masks. Plates are not blurred, because identifying the involved vehicles is part of the case.

## Tests

```bash
python -m pytest
python scripts/laya_eval.py      # labelled commands: router accuracy (rules > Laya > deep parse)
```

## Licences

Nexar Collision Prediction (Nexar Open Data License: attribution; no re-identification of people or vehicles);
Wikimedia Commons clips (public domain); geo-sleuth (MIT; use only on footage you may analyse, never to find people);
OpenStreetMap data (ODbL); VGGT-1B weights (non-commercial; use VGGT-1B-Commercial otherwise); Real-ESRGAN (BSD-3);
AnySplat (MIT); FastGS (MIT, with the 3DGS licence it builds on); RAFT weights (torchvision, BSD-3); Spark splat
renderer (MIT); three.js (MIT).

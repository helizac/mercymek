# Changes against the technical spec

Where the spec and the live Meta Model API docs disagreed, the docs won (hard rule 3). Where this build deliberately
departs from the spec, the reason is given.

## API details (docs win)

| Area | Spec | Implemented | Why |
| --- | --- | --- | --- |
| ASR file endpoint | not specified | `POST /v1/asr/transcribe`, multipart `request` (JSON) + `audio`, `audioEncoding: "WAV"` | `PCM_16KHZ` returns `400 Malformed request` on the file endpoint; WAV is the documented value there. Used by `scripts/smoke_test.py`. |
| SAM parsing | `msp.parse_text(...)` (pseudo-code) | `video_segmentation_format().create_parser()` → `push(raw)` → `finish(CompletedOutcome())` | Real entry points of `meta-sam-parser` 0.0.x (introspected with `dir()`). Mask records expose half-open `bounds` and `mask.width/height`; we assert `H == bottom-top` and `W == right-left`. |
| Web search results | `a.title` on results | results arrive as plain dicts; narration messages carry `phase: "commentary"` | Agentic search emits several `message` items; only non-commentary text is the answer. Sources come from `url_citation` annotations first, then pages the model opened, then top search results (each labelled `kind`). |
| Laya `noul` | `ans["probability"]` | `ans["noul"]` | Real field name in laya 0.4.15. Score questions return `probabilities` + `legend`. |
| Reasoning | `reasoning={"effort": ...}` | same | Confirmed at `/docs/reasoning` (`none` is rejected by Spark/Wonder). |

## Deliberate design changes

1. **Intent set.** Section 4 lists 10 generic intents; section 11 lists 15 UI intents and asks Laya to use those
   names. We keep the generic set (plus `layers`, 11 total, under the 12-option budget) because slots make them
   compositional ("rewind to 3 s before impact" = `seek` + slots). The server turns each into `state` / `verdict` /
   `image` messages, so the UI contract is unchanged.
2. **Laya threshold and keyword tier.** On our 45-command EN+TR set (`scripts/laya_eval.py`) the shipped checkpoints
   are 80 % accurate and over-confident, as the spec warns. `LAYA_MIN_CONFIDENCE=0.95` gives 96 % precision on
   accepted commands; below it a small set of high-precision keyword rules is tried before the Muse Spark deep
   parse. Result: 91 % of commands resolve correctly without any LLM call, 3/45 go to Spark.
3. **Integrity questions.** Tested on real and fabricated verdicts, Laya separates *citing evidence* and
   *speculation* but not *numeric consistency*. The audit therefore uses Laya for the first two (positively framed,
   speculation threshold 0.40) and deterministic code for numbers (every km/h figure must sit inside a measured
   range) and for per-share citations. Laya's consistency score is still reported. The state is compacted to fit
   the 512-token context (measurement ids replaced by their ranges).
4. **Road plane / metric scale.** A single global RANSAC plane over a 200 m monocular reconstruction drifted
   (one clip gave a negative camera height). The plane is now fitted per frame on the static road patch just ahead
   of the car; `h_est` is the median camera height over frames and "up" is the mean normal. Spread across frames is
   logged (3-10 % on our cases).
5. **Velocity of other vehicles.** Central differences of monocular object centres are dominated by depth noise.
   Pre-impact velocities use a constant-velocity least-squares fit with MAD outlier rejection; the camera vehicle
   (smooth pose track) uses the same fit. Peak deceleration is clipped to the tyre-friction limit with a warning.
6. **Vehicle-size cross-check** uses lateral (x) extent, which monocular depth measures well, against body length
   (crossing) or width (same direction).
7. **Party selection.** Spark's `striking`/`struck` roles + SAM aliases pick the parties; a near-miss with one
   named party is still camera-vehicle vs. that party even when `camera_vehicle_involved` is false.
8. **Microphone.** Default `MIC_SOURCE=browser`: the HUD captures the mic (AudioWorklet, 24 kHz int16, 80 ms frames)
   and the server relays it to Muse Voice Transcribe, so the key stays server-side and the demo also works when
   the browser is not on the server machine. `MIC_SOURCE=server` keeps the spec's sounddevice path.
9. **Frontend.** Rebuilt from scratch with a film-style look (floating glass cards, AR evidence labels,
   holographic point cloud). It talks only to the live backend (no mock mode); every message type in section 11 is
   kept, plus `case`, `cases`, `overlay` (live SAM), `note` and `status`. The 3D view is three.js (WebGL) so the
   cloud, ego frustum and trajectories render at 60 fps; viser still runs on `:8080` and follows the same state.
10. **Reconstruction hardware.** VGGT-1B runs on an Intel Arc 140V through PyTorch XPU (96 frames in ~110 s);
    the code picks CUDA, XPU or CPU automatically. `--recon mono` (Depth Anything V2, fixed camera) is the Step 2C
    fallback.
11. **Overlays** store simplified polygons (OpenCV contours of the decoded raster) instead of raw SVG paths: same
    information, ~10x smaller, drawn on a canvas.

## v2 (mercymek)

12. **Realistic 4D scene.** The blue point cloud is replaced by (a) a TSDF-fused, vertex-coloured mesh of the static
    world (Open3D 0.19 in `.venv-recon`; 0.20 returns empty meshes with NumPy 2) and (b) per-frame photo-textured depth
    surfaces built on the GPU from `frames4d/` (metric depth PNG + full-resolution video frame + camera→road matrix).
    Sky is removed with SAM 3.1 `sky` masks; moving objects with the SAM vehicle masks. Vehicles are 3D models on
    their tracks; with the live frame on, other vehicles are the real pixels and only the involved ones keep an outline.
13. **Location.** geo-sleuth's `intake.py` (OCR, metadata, optional reverse image search) on keyframes; burned-in GPS
    is read directly; otherwise Muse Spark with web search judges the candidates and OpenStreetMap Nominatim geocodes
    them; geo-sleuth's `osm.py` snaps the point to the nearest mapped road when Overpass is reachable.
14. **Plates and enhancement.** SAM tracks plates; enhancement fuses up to 9 aligned frames (ECC, median) before ×4
    Real-ESRGAN; Muse Spark reads the plate with '?' for unreadable characters. Faces stay blurred; plates do not.
15. **Verdict watches the video.** The clip is attached to the Wonder call next to the measurements, research, plates
    and location.
16. **Router order.** Adding the `view` intent dropped Laya to 70 % on the EN+TR set with confident mistakes, so the
    precise keyword rules now run first, Laya second (threshold 0.95), Spark deep parse last: 93 % correct without any
    LLM call on 57 commands.
17. **UI.** One screen at a time (video, 4D, map, report, evidence), switched by voice; mic top right; no command bar,
    suggestions, disclaimers, model names or status chips. A hidden `/` command line remains for typing.
18. **Event time without labels.** Clips from outside Nexar get their event time from Muse Spark watching the clip.

## v3

19. **Voice out.** Replies are short sentences spoken with ElevenLabs Flash v2.5 (`optimize_streaming_latency=3`,
    warm keep-alive connection: ~130 ms to first audio from here; cached phrases ~10 ms). The key stays on the server;
    the HUD plays `/api/tts/<key>`. A character budget stops speech (text only) before the account limit.
20. **Questions.** Free questions ("how fast was the blue car?") route by rule to a short answer grounded on the case
    facts (Spark, minimal reasoning). "Who is at fault?" reuses the stored analysis unless notes changed or the user
    asks to analyse again, so it answers instantly and costs nothing.
21. **Street level.** Mapillary (needs a free client token) or Panoramax (open, no key). KartaView's API returned
    nothing in tests. Panoramax coverage at our scenes is thin (0–3 photos), which is why Mapillary is supported.
22. **Rules skip Laya.** Clauses a keyword rule already covers no longer run through Laya: ~0.5 s faster replies.
23. **Plain reports.** Verdict prompt limits length; the display strips ids/URLs and rounds numbers.
24. **Layout.** Left column for everything you read; right stage 90vh with a 6vh timeline and a single play button;
    nothing written over the video. English only (ASR language bias English).

## v4

25. **Photoreal 4D (AnySplat).** One feed-forward pass over 32 frames from 3 s before to 1 s after the event gives
    0.4–1.2 M Gaussians of the static world (moving vehicles and sky dropped before AnySplat's voxel fusion). The
    splats are moved into the road frame by a trimmed Umeyama fit on the same pixels as the metric VGGT depth
    (reflection allowed: the road frame is mirrored against OpenCV axes); median point error 0.9–3 m. AnySplat's CUDA
    dependencies (`torch_scatter`, `xformers`, `gsplat`) are only needed for training/rendering, so inference runs on
    Intel Arc with PyTorch stand-ins. Rendered in the browser with Spark (three.js r180); the live frame now draws
    only moving objects (flag in the depth PNG's blue channel), the chase camera sits low near the dashcam's own
    viewpoint, and a free-flight camera (mouse + W/A/S/D/Q/E) was added. Tried and dropped: keeping only near pixels
    per view (did not remove the ripples, lost the overpass).
26. **TTS model.** ElevenLabs v4 Turbo by default (~150 ms first audio; `optimize_streaming_latency` is only sent to v2
    models, v4 rejects it).
27. **Counterfactuals.** "What if…" re-drives the camera car from the hazard onset (first visibility, capped at brake
    onset + 1.5 s) with the driver's own measured braking effort, so "as driven" reproduces the measured impact.
28. **Mercy mode.** Spoken arguments are judged against the case facts; supported ones move a log-odds liability
    meter (bounded likelihood ratios 1.25 / 1.7 / 2.6); the ledger becomes verdict notes.
29. **RAFT.** Slow-motion replays with flow-interpolated frames (RAFT-large, occlusion-aware), and dense residual
    alignment in plate fusion (RAFT-small after ECC) with robust weighting: 2–3.6× sharper (Laplacian variance) on
    motion-blurred plates, unchanged on static ones.
30. **Trust and integration.** Signed SHA-256 evidence seal (Ed25519, key in `data/keys/`), and the European Accident
    Statement's circumstances section filled from the evidence.
31. **Router.** "what was the speed…" no longer triggers playback (questions never match the playback rule).
32. **Splat artefacts.** The ribs on the road were the camera car's own hood, dashboard and its reflection in the
    windscreen: they sit in the same pixels in every view, VGGT/AnySplat place them on the road a few metres ahead,
    and voxel fusion repeated them once per view along the path. `glued_mask` finds such pixels (same local texture in
    the same place while the camera moves; the hood edge fitted as a smooth curve) and they are dropped before fusion,
    with timestamps and logos. Checked by rendering from the input camera poses: AnySplat's own example reproduces
    cleanly through the same export and renderer, so the model, export and viewer were right.
33. **4D camera with splats.** Presets stay low and close to the camera path (orbit 8 m out, 3.4 m up); the viewer
    can no longer look straight down at the splats except in "top view".
34. **Levelled road.** The global road frame drifts over a clip (monocular VGGT; on `lanechange` the camera "sank"
    1.9 m in 4 s), so the splat road sat 0.7–3.9 m below the vehicles, and on `lanechange` it was cut off entirely by
    the height filter. Splats are now levelled per nearest camera so the local road is y = 0 (camera 1.35 m above it,
    the height that sets the metric scale); with splats, the HUD levels the live frame and the driver camera the same way.
35. **3D moments replace the 4D scene.** The key instants (the other party appearing, braking, just before impact, the
    impact; from the physics) are each frozen in 3D and marked with a diamond under the timeline. A moment is its own
    frame as one camera-facing splat per pixel at the metric depth (sharp depth edges, an inpainted skirt behind them,
    hood cropped) in front of the AnySplat world from the frames around it. The viewer drifts around the dashcam's
    viewpoint (scaled to the subject's distance; drag to move, double-click to recentre) on a sky-to-road backdrop.
    Frozen vehicles from AnySplat alone vanished: it gives single-view content ~0.04 opacity. Voice: "show the moment
    before impact in 3D", "driver view", "bullet time" (the impact moment). The ghost car of "what if" went with the 4D
    scene; the what-if replays the video instead.
36. **FastGS on Kaggle.** `scripts/fastgs_export.py` packages every moment (RGBA frames whose alpha is the loss mask,
    COLMAP cameras in the metric road frame with y down, VGGT init points); `kaggle/fastgs_moments.ipynb` installs
    FastGS, fixes the inherited 3DGS alpha bug (`shape[1] == 4`), makes the photometric loss and FastGS's multi-view
    error maps ignore masked pixels, trains each moment and zips the PLYs; `scripts/fastgs_import.py` flips, levels
    and merges them with the photo layer. Export and import were checked by reprojection; the training itself has not
    run yet (needs the Kaggle GPU).
37. **Stale files.** Case-file URLs carry the file's modification time and every response says `Cache-Control:
    no-cache`, so rebuilt splats and updated scripts show up on a normal reload.
38. **Whole-clip FastGS export.** `scripts/fastgs_export.py --clip` packages one scene per case over all 96 frames
    with every vehicle, the sky and the hood masked out of the loss (`data/fastgs/mercy_fastgs_clip.zip`, same notebook).
    It is the input for a car-free static world of the whole road; not trained yet.
39. **Moving vehicles in the 3D moments.** Muse Image removes the vehicles from a copy of each moment's frame
    (`mercy/generation/cleanplate.py`, $0.01, cached as `moments/<id>.k<frame>.clean.png`, aligned to the frame by
    ECC); the background takes only the vehicles' pixels from it, at road-plane depth. Each tracked vehicle is its own
    splat file with its smoothed path over the 1.5 s before the instant (turns clamped to 25°, depth spikes cut); the
    viewer replays it into the instant, holds and loops. World splats in front of the photo from the moment's camera
    are removed (vehicle remnants the masks missed), vehicle depth is fixed with a RANSAC plane, duplicate SAM tracks
    of one car are merged, and splats are stored relative to the moment's camera: the browser packs centres as float16,
    which far from the road frame's origin tore near objects into stripes.

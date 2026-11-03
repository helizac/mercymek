# WildDet3D RGB vehicle placement

The integration runs the [official WildDet3D model](https://github.com/allenai/WildDet3D)
on original source RGB frames and the existing source-associated vehicle boxes. It does not require radar,
LiDAR or an input depth map. Known camera intrinsics are supplied; LingBot predicts depth from RGB internally.

## Execution and saved setup

```bash
python scripts/kaggle_wilddet.py assets
python scripts/kaggle_wilddet.py runtime
python scripts/kaggle_wilddet.py run --case cutin_rearend hero lanechange night_rain snow_pileup
python scripts/evaluate_wilddet.py
```

The first private Kaggle output, `helizac/mercy-wilddet-assets-v1`, saves source revisions, Python dependencies,
the 4.7 GB full model checkpoint and a small LingBot constructor configuration. Its setup completed on
2026-10-31. `helizac/mercy-wilddet-runtime-v1` version 2 saves the compiled CUDA extension and inference wrapper.
Both outputs completed successfully on Tesla T4 (Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8).
The cold setup also supports lazily importing the unused training/evaluation 3D-IoU operator; box-prompt RGB
inference does not require that operator. An additional lightweight setup was GPU-load tested as well.
Runtime building and inference have Internet disabled. Subsequent runs attach these saved outputs; they
do not reinstall packages or download model weights. Credentials stay in the ignored local `.env`.

Source: `allenai/WildDet3D@1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a`.
Checkpoint revision: `allenai/WildDet3D@161edb26239f58d8973045517d9c8b179c515f73`,
`wilddet3d_alldata_all_prompt_v1.0.pt`, SHA-256
`f8b6a9e548f733ba62625a0d2adc4b0f4fdb6007ee11d9927f9c1027010fee57`.

The worker records exact runtime versions and all source patches. The wrapper patches retain the official
architecture and weights: use the saved LingBot config, unwrap every geometric prompt when multiple boxes
share one RGB backbone pass, and reject missing checkpoint parameters. The inference worker also separates
each prompt's postprocessing group, including views of its shared depth/confidence outputs. Upstream's default
postprocessing merges prompts by image ID. Batched versus independent inference is checked on GPU before
continuing a job. No network weights or image features are substituted. Inference returns one result per
existing source identity, so unsupported depth never creates a new vehicle identity.

## Coordinate and consistency checks

vis4d OPENCV boxes use `(cx,cy,cz,width,length,height,qw,qx,qy,qz)` and local +X as their length/front axis.
The renderer uses `(width,height,length)` and heading zero along road +Z. The adapter transforms points and
direction vectors through the calibrated OpenCV-to-road matrix, including its reflection. A camera-frame
quaternion must never be assigned directly to a browser model.

Qualified anchors require finite boxes, valid quaternions, vehicle/class-compatible dimensions, 2D prompt
IoU >= 0.5, projected 3D box IoU >= 0.35, 2D confidence >= 0.3, 3D confidence >= 0.15, body tilt <= 25 degrees,
and ground contact within `max(0.8 m, 0.55 * vehicle height)` of the shared road plane. These are consistency
checks, **not evidence of measured 3D accuracy**. Rejected predictions have a recorded reason.

Every imported batch is bound to source PNG hashes, current masks, intrinsics and full camera poses. Published
anchors also store clip/calibration hashes. Stale results fail explicitly. A full scene rebuild archives stale
predictions before running the new inference step.

## Playback integration

`mercy/ingest/wilddet.py` decodes and gates neural anchors. `vehicle_scene.py` uses accepted nearby anchors
as soft position/heading evidence, plus robust multi-frame dimensions when available. The model's cuboid
front/rear ambiguity is resolved against source appearance. Missing intervals are not extrapolated by the
neural adapter. Source-box fitting remains available for every observed vehicle.

`road_motion.py` uses qualified anchors in its joint trajectory fit. Normal vehicles still have a constant
lane position and body heading, with continuous nonnegative forward speed. Neural frame noise cannot bend
these paths sideways or reverse them. Incident participants keep flexible trajectories. The HUD continues
to identify the scene and motion as estimated. The static environment and manual camera controls are unchanged.

`scripts/rebuild_4d.py` invokes this stage for new cases as well as existing cases. Raw predictions remain in
`data/kaggle-wilddet/output/`; per-case accepted/rejected anchors are in `vehicles/wilddet3d.json`; publication
records supported observation counts and explicit fallback status for each track. The evaluator overlays raw
qualified cuboids on original frames under `data/research/wilddet3d/evaluation/`.

## Validation status

Actual offline Kaggle inference completed on the five original videos. The 15-frame smoke job produced
95 prompted estimates, of which 55 passed the consistency gates. The denser published job sampled each
replay window at 2 FPS: 50 original frames, 309 estimates, 182 qualified anchors, 48 supported source tracks.
These counts describe internal gates, not detection accuracy against human-labelled 3D ground truth.

| Case | Original frames | Estimates | Qualified anchors | Supported tracks |
|---|---:|---:|---:|---:|
| cutin_rearend | 10 | 49 | 34 | 10 |
| hero | 10 | 54 | 38 | 10 |
| lanechange | 10 | 57 | 39 | 10 |
| night_rain | 10 | 56 | 17 | 5 |
| snow_pileup | 10 | 93 | 54 | 13 |

The other 127 estimates were rejected: 60 low confidence, 31 class-incompatible dimensions, 24 incompatible
ground contacts, and 12 implausible dimensions. Night/rain remains the weakest case; unsupported vehicles
continue to use geometric source-box fits. Twenty-nine tracks use stable multi-frame neural dimensions.

GPU batch parity differed by at most `7.16e-7` in centre/dimension parameters, with quaternion dot product
`0.99999994`. Peak allocated GPU memory was approximately 6.23 GB in FP32 eager mode. Checkpoint loading
reported no missing parameters. The denser kernel is `helizac/mercy-wilddet-infer-6de8635057a8`.

All 15 moments were published as `wilddet-rgb-road-vehicles-v4`. The 87 source tracks retain their identities
and source-observation intervals; 82 normal tracks retain fixed lateral position/heading and nonnegative forward motion.
The five source clips and static/extended Gaussian worlds are byte-identical to their previous versions.
Source-box residuals improve on some tracks and worsen on others; this is not a universal accuracy claim.

Python tests: 109 passed. All 16 vehicle model types passed geometry/front-axis checks. Publication audit:
`python scripts/check_wilddet_publication.py --case cutin_rearend hero lanechange night_rain snow_pileup`.
The audit output is `data/wilddet-run.json`; raw outputs and annotated original frames remain under
`data/kaggle-wilddet/output/6de8635057a8/` and `data/research/wilddet3d/evaluation/6de8635057a8/`.

Real Edge/Playwright checks passed all five cases: sampled source-vehicle coverage, no automatic camera drift,
manual look/dolly/walk/height controls, camera independence from timeline changes, reset and explicit following.
`scripts/export_approach.cjs cutin_rearend` also regenerated the actual 720p/30-FPS approach animation using the
published WildDet-supported tracks, checking constant camera height on every frame. Case seals were refreshed
after the export. Browser evidence: `data/research/registration-fix/browser-check.json`.

## Continuous actor lifetime and corrected body directions

Playback no longer interprets `visible_range` (a detector's evidence interval) as a vehicle's existence.
All 87 source actors persist throughout the clip, including before/after detections and during occlusion.
The original evidence interval is retained as `observed_range`; unobserved travel is explicitly estimated.
Actors share the same mesh across every moment in a world, so changing moments neither recreates nor hides them.
Normal frustum culling and occlusion still apply naturally.

`mercy/generation/heading_review.py` reviews incident bodies against the full original video and road direction.
This corrects side-perspective errors: the snow sedan points forward, not -85 degrees; the blue intersection
car approaches from opposite traffic and then turns from 180 to +90 degrees. A per-frame box fit can no longer
flip these body angles. Road/browser forward vectors are checked explicitly for both left and right headings.

`playback_motion.py` integrates the existing video-derived ego speed estimate into continuous camera translation,
anchored at the generated scene's source frame. Camera derivatives are saved and interpolated with a Hermite
curve, including continuation through the last video frames. This replaces interleaved isolated PnP poses and
zero/unaligned fallback translations. It is a bounded camera-motion prior, not measured odometry. Raw camera
evidence, source clips, static worlds and the original Kaggle neural predictions remain unchanged. Cached neural
camera-frame cuboids are transformed into the updated playback frame without rerunning or downloading the model.

Incident paths are fit jointly with continuous velocity controls (maximum 40 m/s, acceleration 10 m/s²).
Normal tracks retain fixed lanes/nonnegative speeds with a hard 6 m/s² acceleration bound. Body direction comes
from the source video and remains meaningful when stopped or sliding. Ego translation is bounded at 8 m/s².
These are physical regularization limits, not measured vehicle speeds or crash deformation recovery.

The real-browser audit now samples the entire clip at 0.1-second intervals, checks every actor's persistent
visibility, and switches through all moments while checking identical mesh UUIDs and continuous world positions.
All five cases passed, including explicit checks of the intersection turn and snow-car direction.

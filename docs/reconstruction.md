# Reviewed empty environments and volumetric 4D replay

The generated reconstruction uses **one empty environment image per case**. Muse Image removes traffic, shadows,
hood and overlays while preserving the scene and viewpoint. A separate Muse Spark vision call examines the original
and generated image. Failed reviews trigger a targeted correction, up to three attempts. Only an accepted image
whose SHA-256 matches its review can be uploaded for reconstruction. Attempts and feedback remain in
`data/cases/<case>/environment/empty.review.json`.

SHARP runs on a Kaggle T4 and reconstructs that image as Gaussian splats. A lower-road RANSAC fit calibrates camera
height and road orientation; collapsed or unsupported geometry is rejected. Wet night road depth failed this check:
that case uses a Muse-reviewed semantic roadway mask and an explicitly recorded planar-road prior instead. Camera
height can use vehicle-class size priors when complete video boxes support them; it remains an estimate. The browser loads the static world once
for all moments of a case. Optional left/anchor/right images are **renders of that same scene**, with unchanged
camera height. Independent image edits are not assumed to have consistent camera poses.

Vehicles are complete parametric meshes selected from eighteen body types. Muse classifies the type, observed paint
and body orientation from several original video crops with full-frame context. The library includes sedan,
hatchback, SUV, crossover, wagon, MPV, fastback, coupe, pickup, van, box truck, semi truck, tanker, dump truck, bus and
motorcycle, a fully domed `rounded_hatchback`, and an upright `offroad_suv` with spare wheel and roof rails.
Hatchback/SUV/MPV roofs and tailgates use curved lofts rather than the sedan's separate boot.
`node scripts/export_vehicle_library.mjs` exports the same meshes as GLBs to
`data/models/vehicles/`. The exporter also refreshes the small checked-in
`mercy/assets/vehicle-fit-points.json` table. Python fitting uses that source asset
without requiring generated GLBs or a local `data/` directory. Paint is independent of overlay annotation colors. Plate text comes only from a plate
explicitly associated with that vehicle; unknown characters remain `?`, and unavailable plates stay blank.

Replay uses every source vehicle detection, including cars without a neural-depth track. Positions are fitted to
source boxes on the shared road plane, with class dimensions as priors. Source front/rear evidence supplies heading;
slowing or stopping does not rotate a car by 180 degrees. Truncated and foreground-occluded edges are containment
constraints, not full vehicle extents. Complete views support the trajectory through partial detections. Duplicate
masks are merged; source-image identity reviews only join detector IDs with high confidence, and ambiguous pairs
remain separate. Detection gaps are interpolated without discarding/restarting the identity. Visibility follows
source entry/exit times, rather than the much shorter span where neural depth happened to work.

The camera uses full source poses and source lens parameters, also used by the vehicle fitter. PnP-calibrated replay
poses are not spliced to unaligned legacy poses outside their window; boundary-motion priors continue them there.
Labelled collisions with an identified incident pair use an explicit event-contact prior; this is recorded separately
from source reprojection residuals. Near-misses do not receive this prior. The video's timeline, play/pause and
scrubbing control vehicles. “Replay collision” plays three seconds before to 1.5 seconds after impact at half speed.

The default camera is manual and has no automatic orbit or sway: drag to look, wheel or W/S to move, A/D to strafe,
Q/E to change height, Shift to move faster, and right-drag to pan. Click the canvas to focus keyboard movement.
Its pose stays fixed when playback time changes. **Reset camera** returns to the current source-video pose.
**Follow video camera** explicitly enables source-pose following; any manual movement leaves following mode.
**Replay approach** remains an optional chase view at the original estimated height. The free camera can move
outside the supported views and turn behind; this reveals holes/unseen surfaces in a single-image environment.

These bodies approximate unseen surfaces. They do not recover exact vehicle dimensions, hidden damage, suspension,
deformation or independently validated collision dynamics. Tracks and impact timestamps retain the limitations of the
upstream monocular reconstruction. Generated environments and estimated bodies are labelled in the viewer and never
used as original photographic evidence.

## Run

Use the existing project Python environment, with `kaggle` installed. Set `KAGGLE_API_TOKEN` in the ignored `.env`.
The credential is read locally by the official API client; uploaded inputs contain only images, reviews and camera
metadata. All datasets and kernels created by these scripts are private.

```bash
python scripts/kaggle_gpu.py setup                  # once; CUDA renderer, extra packages and weights saved as output
python scripts/kaggle_gpu.py status --job setup
python scripts/rebuild_4d.py --case hero cutin_rearend lanechange night_rain snow_pileup
```

The setup kernel consumes the initial download cache; network downloads are skipped for existing source and weights.
The compiled gsplat binary is loaded directly from the setup output. Inference checks Python and Torch compatibility
before loading it. The setup output is attached as an immutable Kaggle dependency, so each case does not reinstall
packages, redownload checkpoints or recompile CUDA. Version 1 of the GPU cache supports prediction only; version 2
adds the reusable CUDA renderer and same-height preview images.

For separate control:

```bash
python scripts/prepare_environments.py --case hero cutin_rearend
python scripts/kaggle_gpu.py run --case hero cutin_rearend
python scripts/kaggle_gpu.py status
python scripts/kaggle_gpu.py fetch
python scripts/import_environments.py data/kaggle/output
```

Accepted image/model results are reused by hash on subsequent rebuilds. Import checks those hashes, preserves the
previous manifest as `moments.legacy.json`, publishes the updated manifest with an atomic rename, and reseals the case.
The original splat files are preserved. The legacy local AnySplat/photo-layer ingest remains available explicitly with
`--moments-backend legacy`; the default is the generated environment pipeline. Upstream VGGT and RAFT work defaults to
Kaggle (`--gpu-backend kaggle`), with a separate reusable motion-model cache, and fails explicitly if its remote job
fails. Existing cached reconstruction/slow-motion products can be reused without running another GPU job.

## New video requirements

New ingest examples require `--source-manifest`: an open license and source URL, exact video hash, native 720p+
resolution, visual confirmation of an ordinary non-graphic physical collision, no deliberate plate blur, and a clear
original-frame plate reading with frame number and confidence at least 0.8. 1080p+ is preferred; resizing a low-resolution
video does not qualify. `scripts/review_source.py` reviews a candidate with Muse; admission checks run before ingest.
Existing Nexar cases retain their source-blurred plates and are not advertised as satisfying this new-source policy.
Rejected candidates stay outside the live catalog. New examples are built in `data/staging/cases` and published
only after `scripts/publish_cases.py` verifies source hashes, readable native-frame plates, all three moments,
persistent actor identities, swept body nonpenetration, event contact within 15cm and intact evidence seals.

The London roundabout example uses source-camera headings to integrate a curved camera path and fits vehicle
motion against the full clip. Adjacent swept-contact intervals share a separating face so their common endpoint
cannot receive contradictory constraints. The Las Vegas example treats the camera as a witness: the incident
pair is the dark sedan and silver pickup. When sparse neural depth misses a visible incident participant,
its real source boxes can seed a class-size/road-plane estimate. This fallback is recorded as estimated, with
wider metric uncertainty; it supplies no fabricated depth measurements. The readable Vegas plate belongs to
the white bystander SUV and is excluded from the collision parties' statement.

Verified plate crops copy original pixels, preserve the original full frame and SHA-256, and take precedence
over generated enhancement/OCR. Plate evidence time is translated through the clip's trim offset. The source
video, license, author and modifications are retained in `source_provenance.json` and the Files panel.

## Research decisions

| Project | What its official implementation does | Application here |
|---|---|---|
| [SHARP](https://github.com/apple-aiml-research/ml-sharp) | Single-image Gaussian reconstruction, nearby novel views, metric representation; OpenCV coordinates | Implemented as the empty-environment backend. Camera-height calibration still bounds scale reliability. |
| [DC-Gaussian](https://github.com/linhanwang/DC-Gaussian) | Dashcam reconstruction with reflection/obstruction decomposition; trains from COLMAP scenes | Relevant for windscreen artifacts. Requires training and a compatible scene export; not reported as tested here. |
| [IDSplat](https://github.com/zenseact/idsplat) | Static background and dynamic instance Gaussians; official Waymo/PandaSet pipelines | Closest long-term representation for instance-specific reconstruction. Its dataset/calibration adapter is not a single-dashcam drop-in. |
| [DrivingForward](https://github.com/fangzhou2000/DrivingForward) | Feed-forward driving reconstruction from sparse surround-view inputs; nuScenes models | Interesting multi-camera direction. Surround-view assumptions and older CUDA/PyTorch environment need a separate adapter. |
| [GaussianCaR](https://github.com/santimontiel/gaussiancar) | Camera–radar fusion into BEV segmentation for vehicles, roads and lanes | Perception research reference; not a vehicle-mesh/scene generator. The current clips have no radar data. |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | Single-object image-to-mesh generation; official default uses about 6 GB VRAM | Potential future individual-vehicle asset backend. Not run or presented as recovered geometry in this implementation. |

Reviewing a repository is distinct from running and validating its model. Only the explicitly implemented backend and
artifact manifests constitute execution results. No claim is made that all surveyed research projects were benchmarked.

## Initial reconstruction verification

- Five reviewed empty images; night/rain needed a second Muse correction.
- Five actual Tesla T4 SHARP outputs, producing 15 published 3D moments in total.
- A subsequent cut-in run loaded the saved CUDA extension and weights with Internet disabled, and exported three views.
- Twelve reusable vehicle GLBs; 78 Python tests passed; all five cases opened in the real browser without JavaScript errors.
- Replay time advanced correctly; play/pause, the shared timeline and the side-view toggle were exercised.
- All five case seals were rebuilt and verified intact. Run summary: `data/reconstruction-run.json`.

Existing VGGT/RAFT products were reused. The newly added remote motion adapter is compiled but has not been exercised
against a newly admitted video in this run. No new source passed admission. The 4D result is an estimated illustration:
vehicle registration can still be inaccurate, especially at night or where occlusion removes ground-contact observations.
It is not a validated rigid-body/contact simulation or a recovered deformation model.

## Backwards road extension and approach replay

`python scripts/extend_approach.py --case cutin_rearend hero` adds a generated approach behind the original world.
The extension distance follows the existing camera path plus a margin (currently approximately 35–55m for the five
demo cases). All cameras keep the same estimated height and heading. The pipeline:

1. Renders the original splats from a physically specified camera moved backwards, on Kaggle. A simple road-plane
   scaffold guides the missing foreground; it is explicitly an approximate layout, not another observed image.
2. Muse uses this render and the original empty scene to complete the approach. A small distant landmark region is
   feathered into the output. Sparse, distorted peripheral splats can be repaired. Muse vision checks traffic removal,
   road continuity, lighting, landmarks and seams, and supplies a road-surface polygon. Failed images are revised.
3. The accepted image is reconstructed with the saved SHARP runtime on Kaggle, with Internet disabled. Overlap depth
   aligns its scale where supported; otherwise a recorded camera-height/road-plane prior supplies scale.
4. The generated road is constrained to the original y=0 plane. Only geometry in the approach strip is added, with
   a short opacity transition at the join. **Every original Gaussian byte is preserved.** This extension stage
   leaves the current vehicle keys unchanged; vehicle registration is refined separately from source detections.

The combined scene is `environment/extended-world.splat`. Images, masks, reviews, camera plans and alignment details
are in `environment/approach/`. The viewer adds **Replay approach**, following the camera vehicle along the
approach into the event at a fixed camera height. The HUD identifies the approach road as generated. Manual camera
navigation permits a 180° rear view, though the forward-facing reconstruction does not recover rear scenery faces.

`scripts/rebuild_4d.py` includes this stage by default; `--no-approach` omits it. Plans, reviewed images, GPU outputs and
uploads are cached by hashes. Geometry generation and guide rendering use the existing Kaggle CUDA cache, so they
do not reinstall libraries or download model weights per image. `--until guide|generate|predict` on the extension
script supports inspecting intermediate results before publishing.

`node scripts/export_approach.cjs cutin_rearend` exports the actual viewer to a 30fps MP4 at half speed, using the
installed browser-test runtime and ffmpeg. It asserts constant camera height for every exported frame and labels
the output as an AI-generated, estimated reconstruction. Neural inference stays on Kaggle.

The unobserved approach is a plausible generated continuation, not recovered evidence of the actual street behind
the dashcam. Its roadside details, lane dimensions and monocular depth remain estimated.

## Vehicle and camera correction verification

`python scripts/refine_vehicles.py --case cutin_rearend hero lanechange night_rain snow_pileup` refreshes the
vehicle/camera manifests while leaving the original and extended static splat files unchanged. It uses cached
multi-frame appearance/heading reviews and identity decisions; geometric fitting runs on CPU without rerunning
the image-to-3D network. Full rebuild/import invokes the same registration stage.

- 114 Python tests pass, including missing-depth vehicles, stopped-car orientation, camera projection conventions,
  occlusion fragments, nested masks, duplicate identities and continuity across the calibrated camera window.
- `node scripts/check_vehicle_models.mjs` checks all 16 finite meshes, local forward axes, distinct hatch/SUV/MPV
  rear silhouettes and heading interpolation across +/-180 degrees. The reusable GLBs were re-exported.
- `node scripts/check_scene_navigation.cjs` exercises actual mouse/keyboard input, stationary camera stability,
  timeline independence, reset/follow behaviour, and source-vehicle coverage in all five browser scenes.
- At cut-in T+0.95, T+4.02 and T+4.56, every eligible source detection has a visible corresponding model. The source
  contains 6, 4 and 6 vehicle detections at those frames; a model inside an interpolated occlusion interval can also
  be present. These checks do not assert that every model is unoccluded at every arbitrary viewpoint.
- The updated cut-in example is a 1280x720, 30fps H.264 export, 271 frames at half speed. All five case seals were
  rebuilt and checked. The report is `data/vehicle-scene-run.json`; per-track residuals and limitations are retained
  in each case's `vehicles/registration.json`.

GaussianCaR's released implementation was reviewed at commit `c0e748b9e95a6a6062579ddebdcd1c1e79500c2e`.
Its `forward` path invokes the radar encoder on `batch["radar_points"]` and fuses radar/camera BEV features;
the dataset uses calibrated camera rigs. Its pretrained fusion model was not run on these radar-free clips.
Local source snapshots are in `data/research/gaussiancar/`. Vehicle positions use source-box geometric fits plus
qualified WildDet3D RGB anchors, not GaussianCaR predictions. Class dimensions and monocular metric depth remain estimates; some partially
occluded/distant and night-time tracks retain sizeable reprojection residuals.

## Joint road-motion stage

Normal traffic now uses a joint fixed-lane / fixed-heading / nonnegative continuous-speed fit, rather than independently
optimizing yaw and lateral position in each frame. Incident participants retain flexible trajectories. Video-reviewed
exceptions handle opposing traffic or actual maneuvers. This is an explicit physical prior; source reprojection can
worsen where class dimensions or camera calibration are wrong, and residuals remain recorded.

`python scripts/refine_vehicles.py --motion-only --case <ids...>` applies this stage from saved unconstrained source
fits. Full publication also invokes it. The estimated-motion label is visible in the HUD. The fixed camera/navigation
controls are unchanged. See [the RGB-only model investigation](vision-only-vehicle-models.md) for alternatives,
actual Kaggle GPU experiments, failure cases and the distinction between motion constraints and 3D accuracy.

## WildDet3D RGB integration

The five cases now publish 15 moments with the WildDet3D backend. Offline Tesla T4 inference processed 50
original frames and 309 source-associated box prompts; 182 passed consistency gates and support 48 tracks.
Twenty-nine tracks also use stable multi-frame dimension estimates. Source identities and observation intervals,
normal-traffic motion constraints and manual camera navigation are retained. Unsupported neural estimates
use the existing source-box geometry. All static/extended Gaussian files and source clips are unchanged.

`python scripts/kaggle_wilddet.py run --case <ids...>` attaches saved private assets/runtime outputs, performs
RGB-only inference, validates source hashes/calibration, updates vehicle fits, and seals the case. The full
`scripts/rebuild_4d.py` path invokes this for new cases too. See [WildDet3D execution and validation](wilddet3d.md)
and `data/wilddet-run.json`. The counts are consistency-gate results, not measured 3D accuracy.

`continuous-world-motion-v1` retains vehicle identities/paths throughout the clip and reuses their meshes across
all moments. Source detection intervals are stored as `observed_range` and describe evidence rather than an actor's
physical lifetime. Leaving that interval never hides a vehicle. Internal detection gaps retain the same model and identity.
Incident body headings are reviewed in the full video, so alongside perspective is not mistaken for a 90-degree
turn. All incident paths use joint bounded continuous velocity/acceleration fits; the camera uses a continuous
video-derived speed prior and Hermite position interpolation. Cached WildDet camera-frame boxes are rebased into
that playback frame without neural inference. Raw calibration evidence and static Gaussian files are preserved.

## Body nonpenetration and dashcam frame synchronization

`joint-source-contact-v1` fits all vehicle positions together using cached source detections and the existing
playback cameras. OSQP enforces fixed lanes for normal traffic, speed/acceleration bounds, and separating planes
for oriented body footprints, including ego. Footprints include mirrors, wheels, bumper trim and plates; checks
against the actual mesh geometry cover all 16 body types. Swept SAT checks the entire linear interpolation
interval, including vehicles that exchange positions between otherwise clear endpoints. Publication fails if
any interval remains unresolved. Corrupt sparse tracking masks no longer count as position evidence; their
physical actors remain in the world. Neither detection gaps nor the contact stage remove actors.

Full vehicle publication and `--motion-only` both invoke this stage. Existing scenes can be refreshed with
`python scripts/refine_contacts.py --case cutin_rearend hero lanechange night_rain snow_pileup`, using saved
predictions without neural inference or new GPU setup. Per-case `vehicles/contacts.json` records the solver run.
`python scripts/audit_contacts.py --case <ids...>` independently audits published intervals and source-image edge
residuals against the snapshots in `data/research/contacts/before/`.

The viewer keeps one native video decoder visible in the 3D comparison window. Playing scenes use the presented
frame's media timestamp, with the same captured pixels. Paused seeks that do not produce a browser frame callback
use the decoded CFR evidence frame after presentation settles. The ingest's redacted copy has constant frame rate.
The 3D camera uses the same interpolated intrinsics and poses as vehicle fitting. **Dashcam view** preserves the
native aspect ratio with letterboxing; the default camera remains manual. Its label distinguishes free navigation
from the source camera. Pending seeks retain the previous frame's scene time until the new frame is available.

`node scripts/check_scene_sync.cjs` verifies seek/play timestamps, decoded source pixels, and world-point projection
through the actual browser camera against independent CPU camera fixtures. `node scripts/check_scene_navigation.cjs`
also checks persistent actor identities and body intersections across the full clip, in all five scenes. Python
regressions cover tunnelling, surface contact, rotation bounds, corrupt masks and joint source fitting.

Nonpenetration is a kinematic constraint, not a recovered rigid-body crash simulation. Absolute monocular positions,
unobserved continuations, deformation, and the generated background remain estimates. Source reprojection residuals
are recorded separately; frame synchronization does not establish pixel-perfect vehicle placement or geometry.

## Impact alignment correction

`joint-source-contact-v2` corrects the visible collision gap in `hero` and `cutin_rearend`. Playback reads the
published road's camera height rather than a stale PnP cache (the hero cache used 1.801m against a 1.35m world).
The PnP cache key also includes a new version and validates height before reuse. Source fitting uses silhouette
points exported from the browser vehicle models by `scripts/export_vehicle_library.mjs`, with horizontal-frustum
clipping so an offscreen roof corner cannot define the top of a partially visible vehicle.

A collision prior requires both collision labels and one identified source-supported incident pair. It constrains
body support gaps to 0.012–0.15m at the exact event timestamp. Native detection cadence is retained around impact.
Normal motion keeps acceleration bounds on each side; a velocity impulse is permitted only at impact. Bounding
acceleration through a crash incorrectly spreads the velocity change into the preceding approach. The fitted
velocity jump is recorded; collision forces, mass, restitution and deformation are not inferred.

Crossing trajectories use interval-specific separation planes. Fixed-lane traffic retains lane ordering; aligned
rear-end participants use longitudinal ordering. Previously published continuations serve as priors when refreshing
contact fits. Per-track post-fit pixel residuals and event provenance are saved in `vehicles/contacts.json`, and
publication fails if the contact gap or continuous nonpenetration check fails. `--dry-run` does not replace the case's
contact report or playback-camera metadata.

Run `python scripts/refine_contacts.py --case hero cutin_rearend` to refresh these scenes from cached inputs.
`node scripts/check_impact_alignment.cjs` checks the rendered bodies at impact and 0.13s later against the
presented video timestamp, and saves overhead screenshots. The existing frame-sync and navigation checks also
remain applicable; acceleration audits distinguish the labelled impulse from bounded motion on either side.

## Persistent traffic and source exit continuity

`observed_range` never gates a persistent body's visibility. In hero, `car:3` ends near T+6.75 but its old
continuation brought it back into the source camera at T+10.95. `joint-source-contact-v3` corrects the continuation
instead of hiding the actor. A narrow fragment at an image edge supports an exit-direction prior after a 0.65s
transition. One-sided slack keeps that estimated exit from overruling observed source positions or physical
nonpenetration. Previously fitted observations have a stronger position prior during refresh. Observed actors
also retain positive camera depth, preventing a bad fit from escaping image residuals behind the camera.

The corrected hero actor remains outside the source camera and can still be inspected by moving the free camera
to its position. All physical identities and full-clip paths remain stored and rendered. Other existing cases keep
their previously accepted paths; the current solver is available to subsequent refinements. Unseen continuations
remain estimates, and individual source residuals and exit-prior violations are recorded separately.

`node scripts/check_actor_presence.mjs` checks persistence through detection boundaries. The actual-browser check
in `scripts/check_actor_presence_browser.cjs` tests the source camera and then moves to the departed vehicle;
it must be present in both the scene graph and the free camera. Artifacts are in `data/research/actor-presence/`.

## Case chooser and body library

The case chooser uses a responsive grid, distinct current-case state, close/Escape and keyboard focus handling.
All five existing demos are shown exactly once. Collision and near-miss badges follow the case impact metadata;
these are the retained demo cases, not newly admitted high-resolution/unblurred-plate sources. A consumed case
deep link no longer overrides the first manual selection. `scripts/check_case_selector.cjs` verifies three
viewport sizes and actual case switching against a test server.

The new rounded hatch has a continuous domed shell and curved tail rather than a rectangular sedan boot.
Off-road SUV geometry has an upright cabin, raised chassis, larger tyres, rear spare and roof rails. Backend
classification, size priors, silhouette points and exported GLBs share all 18 types. The gallery rendered by
`scripts/preview_vehicle_library.cjs` is in `data/research/ui-vehicles/vehicle-library.png`.


## Optional local reconstruction environment

The saved demo needs only the base application environment described in
[README.md](../README.md). The default generated-environment and remote GPU paths
are described above. Local VGGT/mesh execution and the earlier AnySplat renderer
use a separate environment because Open3D requires NumPy below 2.
Choose the Torch index for your hardware; the example uses Intel XPU. Set your
credentials in the ignored `.env` using `.env.example` as a template.

```bash
uv venv --python 3.11 .venv-recon
uv pip install --python .venv-recon torch torchvision --index-url https://download.pytorch.org/whl/xpu   # or cu124 / cpu
uv pip install --python .venv-recon "vggt @ git+https://github.com/facebookresearch/vggt.git" "open3d==0.19.0" "numpy<2" trimesh opencv-python pillow huggingface_hub python-dotenv transformers
uv pip install --python .venv-recon einops jaxtyping omegaconf plyfile dacite safetensors lightning scikit-video matplotlib e3nn colorspacious "numpy<2"
uv pip install --python .venv-recon openai meta-sam-parser scipy "numpy<2"   # 3D moments: clean plates, vehicle masks
git clone --depth 1 https://github.com/InternRobotics/AnySplat vendor/AnySplat   # feed-forward splats (MIT); weights lhjiang/anysplat
git clone https://github.com/Oldcircle/geo-sleuth vendor/geo-sleuth           # geolocation skill (MIT); needs uv
```

Run new-video admission with a reviewed source manifest before ingesting new
footage; see **New video requirements** above. To opt into the earlier renderer,
use `--moments-backend legacy --gpu-backend local` on the ingest command.
The local VGGT/mesh subprocesses use `.venv-recon`; the server stays in `.venv`.

Sharper 3D moments with [FastGS](https://github.com/fastgs/FastGS) (per-scene training, CUDA only, so on Kaggle):

```bash
python scripts/fastgs_export.py                  # frames + loss masks + COLMAP cameras -> data/fastgs/mercy_fastgs.zip
python scripts/fastgs_export.py --clip           # whole clip per case, all vehicles masked -> mercy_fastgs_clip.zip
# upload it as a private Kaggle dataset, run kaggle/fastgs_moments.ipynb on a T4, download fastgs_out.zip
python scripts/fastgs_import.py fastgs_out.zip   # replaces the AnySplat world of each moment (kept as *.anysplat.splat)
```

The earlier renderer replaces three CUDA-oriented dependencies with local
PyTorch compatibility shims (`vendor/anysplat_shims`) for Intel XPU or CPU inference.
Outputs are reused unless the relevant stage is forced.

### Earlier photo-layer renderer

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

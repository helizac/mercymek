"""Import a Kaggle SHARP environment and assemble tracked, volumetric vehicle replay.

Only CPU coordinate conversion happens here. The image-to-Gaussians network runs on Kaggle.
Road coordinates: x right, y up, z forward; browser coordinates negate z.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from mercy.analysis.physics import display_path
from mercy.generation.vehicles import profiles


def read_gaussians(path: Path) -> dict:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("Expected binary little-endian SHARP PLY")
    names, count, in_vertex = [], 0, False
    for line in lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
            if in_vertex:
                count = int(line.split()[2])
        elif in_vertex and line.startswith("property "):
            if not line.startswith("property float "):
                raise ValueError("Unsupported vertex property")
            names.append(line.split()[2])
    data = np.frombuffer(raw, dtype=np.dtype([(name, "<f4") for name in names]), count=count, offset=end)
    stack = lambda ks: np.stack([data[k] for k in ks], axis=1).astype(np.float32)
    return {"pos": stack(["x", "y", "z"]), "scales": np.exp(stack([f"scale_{i}" for i in range(3)])),
            "quat": stack([f"rot_{i}" for i in range(4)]),
            "rgb": np.clip(.5 + .28209479177387814 * stack([f"f_dc_{i}" for i in range(3)]), 0, 1),
            "op": 1 / (1 + np.exp(-np.clip(data["opacity"], -40, 40)))}


def fit_road(points: np.ndarray, focal: float, width: int, height: int, seed=0) -> dict:
    """RANSAC on the lower central roadway. Reject a collapsed/vertical/unsupported road plane."""
    rng = np.random.default_rng(seed)
    P = points[::max(1, len(points) // 100_000)]
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 1) & (P[:, 2] < 100)]
    u = P[:, 0] * focal / P[:, 2] + width / 2
    v = P[:, 1] * focal / P[:, 2] + height / 2
    P = P[(u > width * .28) & (u < width * .78) & (v > height * .65) & (v < height * .92)]
    if len(P) < 100:
        raise ValueError("Insufficient road observations")
    if len(P) > 5000:
        P = P[rng.choice(len(P), 5000, replace=False)]
    best = np.zeros(len(P), bool)
    threshold = max(.04, float(np.median(P[:, 2])) * .008)
    for _ in range(180):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        n /= max(np.linalg.norm(n), 1e-12)
        if n[1] < 0:
            n = -n
        if n[1] < .85:
            continue
        keep = np.abs((P - a) @ n) < threshold
        if keep.sum() > best.sum():
            best = keep
    if best.mean() < .55:
        raise ValueError(f"Road plane unsupported: {best.mean():.1%} inliers")
    road = P[best]
    n = np.linalg.eigh(np.cov(road.T))[1][:, 0]
    if n[1] < 0:
        n = -n
    offset = float(np.median(road @ n))
    if not .15 < offset < 8 or n[1] < .85:
        raise ValueError(f"Invalid road plane distance {offset}, normal {n}")
    forward = np.array([0., 0., 1.]) - n * n[2]
    forward /= np.linalg.norm(forward)
    right = np.cross(n, forward)
    basis = np.stack([right, -n, forward])
    return {"basis": basis.tolist(), "normal_camera": n.tolist(), "offset": offset,
            "inlier_ratio": float(best.mean()), "residual_m_unscaled": float(np.median(np.abs(road @ n - offset))),
            "scale": 1.35 / offset, "scale_method": "assumed dashcam height 1.35m (range 1.2-1.5m)"}


def write_splat(path: Path, pos, scales, quat, rgb, opacity):
    """32-byte .splat format; quaternion wxyz in browser coordinates."""
    dt = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("quat", "u1", 4)])
    out = np.empty(len(pos), dtype=dt)
    out["pos"], out["scale"] = pos, scales
    out["rgba"] = np.clip(np.column_stack([rgb, opacity]) * 255, 0, 255).astype(np.uint8)
    out["quat"] = np.clip(quat * 128 + 128, 0, 255).astype(np.uint8)
    path.write_bytes(out.tobytes())


def transform_gaussians(g, basis, scale, camera_height=1.35):
    # Gaussian orientation must transform with its covariance, including both coordinate reflections.
    A = np.diag([1., 1., -1.]) @ basis
    qs, ss = [], []
    for start in range(0, len(g["pos"]), 100_000):
        sl = slice(start, start + 100_000)
        q = g["quat"][sl]
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        C = (A @ R) * (g["scales"][sl] * scale)[:, None, :]
        values, axes = np.linalg.eigh(C @ C.transpose(0, 2, 1))
        axes[np.linalg.det(axes) < 0, :, 0] *= -1
        qs.append(Rotation.from_matrix(axes).as_quat()[:, [3, 0, 1, 2]])
        ss.append(np.sqrt(np.maximum(values, 1e-10)))
    road_pos = g["pos"] @ basis.T * scale + [0, camera_height, 0]
    return road_pos * [1, 1, -1], np.concatenate(ss), np.concatenate(qs)


def track_keys(rows, to_scene, moment_t, start, end):
    rows = sorted({float(row["t"]): row for row in rows}.values(), key=lambda r: r["t"])
    t = np.array([r["t"] for r in rows])
    P = to_scene(np.array([r["center"] for r in rows]))
    X = display_path(P)
    V = np.gradient(X, t, axis=0)
    speed = np.linalg.norm(V, axis=1)
    moving = speed > 1.0
    if moving.any():
        idx = np.maximum.accumulate(np.where(moving, np.arange(len(t)), 0))
        idx[:np.argmax(moving)] = np.argmax(moving)
        yaw = np.unwrap(np.arctan2(V[idx, 0], V[idx, 1]))
    else:
        yaw = np.zeros(len(t))
    keys = []
    for i in np.where((t >= start) & (t <= end))[0]:
        if not np.isfinite(X[i]).all():
            continue
        keys.append([round(float(t[i] - moment_t), 3), round(float(X[i, 0]), 3),
                     round(float(X[i, 1]), 3), round(float(np.degrees(yaw[i])), 2)])
    # Refuse depth spikes instead of teleporting models; restart after a discontinuity.
    groups = [[]]
    for key in keys:
        if groups[-1]:
            prev = groups[-1][-1]
            if key[0] - prev[0] > .6 or np.hypot(key[1]-prev[1], key[2]-prev[2]) > 45 * (key[0]-prev[0]):
                groups.append([])
        groups[-1].append(key)
    return max(groups, key=lambda group: (any(abs(k[0]) < .2 for k in group), len(group)))


def fit_observed_boxes(keys, key, model, moment_t, overlays, meta, rec, frames, to_scene, linear):
    """Refine metric cuboid positions against observed source boxes, keeping headings from tracks.

    Dimensions remain class priors. Truncated box edges are excluded, and poorly fitting observations are rejected.
    This improves visual registration; it does not independently validate a collision or recover deformation.
    """
    width, height, length = model["dimensions_m"]
    base = np.array([[x, y, z] for x in [-width/2, width/2] for y in [0, height] for z in [-length/2, length/2]])
    refined, errors = [], []
    for entry in keys:
        t = entry[0] + moment_t
        k = int(np.argmin(np.abs(rec["times"] - t)))
        f = int(round(t * meta["src_fps"]))
        obs = next((overlays.get(str(frame), {}).get(key) for frame in [f, f-1, f+1]
                    if overlays.get(str(frame), {}).get(key)), None)
        if not obs:
            continue
        target = np.array(obs["box"], float)
        if target[2] - target[0] < 12 or target[3] - target[1] < 12:
            continue
        M = np.array(frames[k]["M"])
        C = to_scene(M[:3, 3])
        R = linear.T @ M[:3, :3]
        K = rec["intrinsic"][k].astype(float) * np.array([[meta["width"] / rec["depth"].shape[2]],
                                                       [meta["height"] / rec["depth"].shape[1]], [1]])
        yaw = np.radians(entry[3])
        # Road coordinates are left-handed: positive heading points toward +x from +z.
        rot = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        corners = base @ rot.T
        valid = np.array([target[0] > 3, target[1] > 3, target[2] < meta["width"]-3, target[3] < meta["height"]-3])
        if target[3] > meta["height"]*.79 and target[2]-target[0] > meta["width"]*.25:
            valid[3] = False  # the dashboard/hood hides the wheels: this is not a ground-contact edge
        if valid.sum() < 2:
            continue
        norm = np.array([meta["width"], meta["height"], meta["width"], meta["height"]], float)
        def project(xz):
            P = (corners + [xz[0], 0, xz[1]] - C) @ R
            z = np.maximum(P[:, 2], .2)
            u, v = K[0, 0]*P[:, 0]/z + K[0, 2], K[1, 1]*P[:, 1]/z + K[1, 2]
            return np.array([u.min(), v.min(), u.max(), v.max()])
        old = np.array(entry[1:3])
        def residual(xz):
            return np.r_[((project(xz)-target)/norm)[valid], (xz-old)/150]
        result = least_squares(residual, old, loss="soft_l1", f_scale=.02, max_nfev=35,
                               bounds=(old-20, old+20))
        err = float(np.median(np.abs((project(result.x)-target)[valid])))
        if err < max(25., .12 * (target[2]-target[0])):
            refined.append([entry[0], *result.x, entry[3]])
            errors.append(err)
    if len(refined) < 3:
        return keys, {"method": "monocular tracks; box refinement unsupported"}
    refined = np.array(refined)
    # Smooth the fitted observations without bridging a tracking gap.
    P = np.column_stack([refined[:, 1], np.zeros(len(refined)), refined[:, 2]])
    X = display_path(P)
    out = []
    for entry in keys:
        if refined[0, 0] <= entry[0] <= refined[-1, 0]:
            out.append([entry[0], round(float(np.interp(entry[0], refined[:, 0], X[:, 0])), 3),
                        round(float(np.interp(entry[0], refined[:, 0], X[:, 1])), 3), entry[3]])
    return out, {"method": "class-size cuboid fitted to observed video boxes; heading from monocular tracks",
                 "observations": len(refined), "median_box_error_px": round(float(np.median(errors)), 2)}


def estimate_camera_height(case, rec, frames, appearance, impact):
    """Class-height prior from complete, unoccluded vehicle boxes. Never call this a measured camera height."""
    meta = json.loads((case / "meta.json").read_text())
    overlays = json.loads((case / "overlays.json").read_text())
    observations = {}
    for k, time in enumerate(rec["times"]):
        if not max(0, impact-3) <= time <= impact-.5:
            continue
        f = str(round(float(time)*meta["src_fps"]))
        K = rec["intrinsic"][k].astype(float)*np.array([[meta["width"]/rec["depth"].shape[2]],
                                                      [meta["height"]/rec["depth"].shape[1]], [1]])
        R = np.array(frames[k]["M"])[:3,:3]
        for key, obj in overlays.get(f, {}).items():
            profile = appearance.get(key)
            if not profile or profile.get("confidence",0)<.6 or profile["dimensions_m"][1]>2.5:
                continue
            x1,y1,x2,y2 = obj["box"]
            if x1<10 or x2>meta["width"]-10 or y2>meta["height"]*.78 or y2-y1<35:
                continue
            top = R @ [(float(x1+x2)/2-K[0,2])/K[0,0], (y1-K[1,2])/K[1,1], 1]
            bottom = R @ [(float(x1+x2)/2-K[0,2])/K[0,0], (y2-K[1,2])/K[1,1], 1]
            if top[1]-bottom[1] <= .01:
                continue
            height = profile["dimensions_m"][1]*(-bottom[1])/(top[1]-bottom[1])
            if 1.0<height<2.1:
                observations.setdefault(key, []).append(float(height))
    medians = [np.median(v) for v in observations.values() if len(v)>=5]
    if len(medians)<3:
        physics = json.loads((case / "physics.json").read_text())
        primary = next((o["id"] for o in physics["objects"] if o.get("vid")=="v2"), None)
        samples = observations.get(primary, [])
        if len(samples)>=8 and np.median(np.abs(np.asarray(samples)-np.median(samples)))<.15:
            height = float(np.clip(np.median(samples),1.2,1.85))
            return height, {"method":"single visible vehicle-class height prior; not a measured camera height",
                            "vehicle":primary, "observations":len(samples), "range_m":[round(height-.3,2),round(height+.3,2)]}
        return 1.35, {"method":"default dashcam-height prior", "range_m":[1.2,1.5]}
    height = float(np.clip(np.median(medians),1.2,1.85))
    return height, {"method":"vehicle-class height priors and complete video boxes; dimensions not measured",
                    "vehicles":len(medians), "per_vehicle_m":{k:round(float(np.median(v)),3) for k,v in observations.items() if len(v)>=5},
                    "range_m":[round(max(1.0,height-.2),2),round(height+.2,2)]}


def assemble(case: Path, output: Path) -> dict:
    result = json.loads((output / "result.json").read_text())
    env = case / "environment"
    image = env / "empty.png"
    review = json.loads((env / "empty.review.json").read_text())
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    if not review.get("accepted") or result["image_sha256"] != sha or review["image_sha256"] != sha:
        raise ValueError("GPU result does not match the reviewed environment image")
    rec = np.load(case / "recon.npz")
    tracks = json.loads((case / "tracks.json").read_text())
    f4 = json.loads((case / "frames4d.json").read_text())
    physics = json.loads((case / "physics.json").read_text())
    appearance = profiles(case)
    g = read_gaussians(output / "environment.ply")
    valid = (np.isfinite(g["pos"]).all(1) & np.isfinite(g["scales"]).all(1) & (g["op"] > .025)
             & (g["pos"][:, 2] > .3) & (g["pos"][:, 2] < 350))
    g = {key: val[valid] for key, val in g.items()}
    H, W = cv2.imread(str(image)).shape[:2]
    impact = float(physics["impact"]["t"])
    camera_height, height_info = estimate_camera_height(case, rec, f4["frames"], appearance, impact)
    try:
        road = fit_road(g["pos"], result["focal_px"], W, H)
        road["method"] = "RANSAC on predicted road geometry"
    except ValueError as error:
        scene = json.loads((case / "scene.json").read_text())
        if not any(word in json.dumps(scene).lower() for word in ["wet", "rain", "night"]):
            raise
        from mercy.ingest.road_prior import apply
        g, road = apply(case, g, result, np.array(f4["frames"][result["anchor_k"]]["M"])[:3,:3], W, H, str(error))
    road.update(scale=camera_height/road["offset"], camera_height_m=camera_height, height_calibration=height_info)
    road["scale_method"] = f"estimated camera height {camera_height:.3f}m; {height_info['method']}"
    basis, scale = np.asarray(road["basis"]), road["scale"]
    # A uniform image-grid subsample preserves both high-opacity foreground and the far background.
    if len(g["pos"]) > 1_500_000:
        idx = np.linspace(0, len(g["pos"]) - 1, 1_500_000).astype(int)
        g = {key: val[idx] for key, val in g.items()}
    pos, scales, quat = transform_gaussians(g, basis, scale, camera_height)
    splat = env / "world.splat"
    write_splat(splat, pos, scales, quat, g["rgb"], g["op"])
    k = result["anchor_k"]
    M = np.array(f4["frames"][k]["M"])
    # Existing tracks already have camera-height metric scale. The road basis corrects tilt;
    # only the SHARP point cloud needs the newly fitted scale factor.
    def to_scene(P):
        return (np.asarray(P) - M[:3, 3]) @ M[:3, :3] @ basis.T * (camera_height/1.35) + [0, camera_height, 0]
    meta = json.loads((case / "meta.json").read_text())
    start, end = max(0., impact - 3.), min(meta["duration"], impact + 1.5)
    original = np.array([frame["M"] for frame in f4["frames"]])
    fallback = original.copy()
    linear = M[:3, :3] @ basis.T
    fallback[:, :3, :3] = linear.T @ original[:, :3, :3]
    fallback[:, :3, 3] = to_scene(original[:, :3, 3])
    fallback[:, 1, 3] = camera_height
    from mercy.ingest.registration import register
    cameras, camera_supported, camera_errors = register(case, g, road, result, rec, fallback, start, end)
    scene_frames = [{"M": matrix} for matrix in cameras]
    # Match the existing physics assumption: camera to front bumper is 2m, body length prior is 4.7m.
    ego_offset = 4.7/2 - 2.0
    ego_rows = [{"t": float(t), "center": (matrix[:3, 3] - matrix[:3, 2]*ego_offset).tolist()}
                for t, matrix in zip(rec["times"], cameras)]
    all_tracks = dict(tracks["tracks"], ego=ego_rows)
    # Track duplicates are merged in the observation domain, where no depth ambiguity is involved.
    overlays = json.loads((case / "overlays.json").read_text())
    camera_keys = []
    for time, matrix in zip(rec["times"], cameras):
        C, direction = matrix[:3, 3], matrix[:3, 2]
        camera_keys.append([float(time), float(C[0]), float(C[2]), float(np.degrees(np.arctan2(direction[0], direction[2])))])
    selected = []
    from mercy.ingest.lift import box_iou
    involved = {o["id"] for o in physics["objects"] if o.get("vid")}
    ranked = sorted(all_tracks, key=lambda key: (key not in involved, -len(all_tracks[key])))
    for key in ranked:
        if key != "ego" and key not in appearance:
            continue
        duplicate = False
        for previous in selected:
            overlap = [box_iou(objects[key]["box"], objects[previous]["box"]) for objects in overlays.values()
                       if key in objects and previous in objects]
            if len(overlap) >= 3 and np.median(overlap) > .65:
                duplicate = True
                break
        if not duplicate:
            selected.append(key)
    old_path = case / "moments.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
    else:
        from mercy.ingest.moments import pick
        old = {"moments": pick(case)}
    moments = []
    cam = {"pos": [0, camera_height, 0], "fwd": (basis @ [0, 0, 1]).tolist(),
           "fov_y": float(2 * np.arctan(H / (2 * result["focal_px"]))), "crop_bottom": 0,
           "sky": [0.65, .7, .75], "ground": [.23, .24, .25]}
    image_rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    cam["sky"] = (np.median(image_rgb[:max(1,H//8)].reshape(-1,3),axis=0)/255).round(3).tolist()
    cam["ground"] = (np.median(image_rgb[-H//5:,W//4:3*W//4].reshape(-1,3),axis=0)/255).round(3).tolist()
    templates = []
    for key in selected:
        rows = all_tracks[key]
        if len(rows) < 3:
            continue
        keys = track_keys(rows, (lambda p: p) if key == "ego" else to_scene, impact, start, end)
        if len(keys) < 2:
            continue
        model = appearance.get(key) or {"body_type": "sedan", "paint_hex": "#737a80", "paint_name": "unknown",
            "dimensions_m": [1.82, 1.47, 4.7], "plate": {"text": "", "observed": False},
            "geometry": "camera vehicle type/paint not observed; generic approximation"}
        registration = {"method": "dashcam pose; assumed 2m camera-to-front-bumper distance, consistent with physics"}
        if key != "ego":
            keys, registration = fit_observed_boxes(keys, key, model, impact, overlays, meta, rec,
                scene_frames, lambda p: p, np.eye(3))
        templates.append({"key": key, "label": tracks.get("aliases", {}).get(key, key), "model": model,
                         "pivot": [0, 0, 0], "keys": keys, "registration": registration})
    for m in old["moments"]:
        m = {key:value for key,value in m.items() if key not in {"approach","base_file"}}
        t = float(m.get("t_frame", m["t"]))
        vehicles = [{**v,"keys":[[round(key[0]+impact-t,3),*key[1:]] for key in v["keys"]]} for v in templates]
        focus = next((v["keys"][min(range(len(v["keys"])), key=lambda i: abs(v["keys"][i][0]))][1:3]
                      for v in vehicles if v["key"] == next((o["id"] for o in physics["objects"] if o.get("vid") == "v2"), None)), [0, 12])
        moments.append({**m, "file": "environment/world.splat", "origin": [0, 0, 0], "cam": cam,
                        "focus": [focus[0], .9, focus[1]], "count": len(pos), "photo_splats": 0,
                        "t_frame": t, "vehicles": vehicles, "views": 1, "seconds": result["seconds"],
                        "align": {"road_inlier_ratio": road["inlier_ratio"], "scale": road["scale"]},
                        "reconstruction": "generated-environment-3d-vehicles-v1",
                        "replay_start": start, "replay_end": end, "impact_t": impact,
                        "event_kind": physics["impact"].get("kind", "collision"),
                        "camera_keys": camera_keys,
                        "clean": {"method": "Muse Image + independent Muse vision review", "file": "environment/empty.png"}})
    backup = case / "moments.legacy.json"
    if old_path.exists() and not backup.exists():
        backup.write_bytes(old_path.read_bytes())
    info = {"method": "reviewed Muse environment -> Kaggle SHARP -> road calibration + parametric vehicles",
            "device": result["gpu"], "road": road, "anchor_k": k, "moments": moments,
            "limitations": ["unseen environment and vehicle surfaces generated/estimated", "metric scale assumes camera height",
                            "motion inherited from monocular tracks; collision deformation not recovered"]}
    (env / "reconstruction.json").write_text(json.dumps(dict(result, road=road), indent=2))
    for view in output.glob("view-*.png"):
        shutil.copy2(view, env / view.name)
    temp = old_path.with_suffix(".tmp")
    temp.write_text(json.dumps(info, indent=1))
    temp.replace(old_path)
    # Modern playback uses every source detection and full camera poses. Neural-depth tracks remain
    # inputs to the legacy/physics products, rather than a gate that removes visible vehicles.
    from mercy.ingest.vehicle_scene import publish
    from mercy.ingest.wilddet import archive_stale
    archive_stale(case)
    publish(case,appearance=appearance)
    return json.loads(old_path.read_text())

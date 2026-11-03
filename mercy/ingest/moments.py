"""3D moments: the key instants of the case frozen in 3D (Gaussian splats), shown on the timeline.

A moment is one video instant (the other party appearing, braking, just before impact, the impact). AnySplat gets the
frames from 2.5 s before to 1 s after it and gives the static world (vehicles masked in every frame); the moment's own
frame on top of it is the photo layer. The splats are aligned to the road frame and levelled like scene.splat
(mercy.ingest.splat.reconstruct).

The vehicles are separate layers: Muse Image removes them from a copy of the frame (the clean plate,
mercy.generation.cleanplate), the background takes only the pixels they covered from it (depth from the road plane
there), and each vehicle keeps its real pixels in its own splat file with its tracked path over the REPLAY_S before
the instant, so the viewer can replay how they got there. Without a clean plate the vehicles stay frozen in the photo.

Writes moments/<id>.splat, moments/<id>.<vehicle>.splat (three.js axes, relative to the moment's origin) and
moments.json: per moment its time, label, origin,
the dashcam pose at that instant (levelled road frame: x right, y up, z forward, metres), a focus point for the viewer
to circle, and per vehicle its pivot and keys [t - t_moment, dx, dz, dyaw degrees].

Runs in .venv-recon (torch XPU/CUDA/CPU): python -m mercy.ingest.moments --case X [--force]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mercy.analysis.physics import display_path  # noqa: E402
from mercy.config import log  # noqa: E402
from mercy.generation.cleanplate import make_clean_plate  # noqa: E402
from mercy.ingest import lift, segment  # noqa: E402
from mercy.ingest import splat as SP  # noqa: E402  (also puts the AnySplat paths on sys.path)

BEFORE_S, AFTER_S = 2.5, 1.0          # frames used around each moment
MIN_GAP_S = 0.35                      # moments closer than this are the same moment
PHOTO_W = 960                         # the moment's own frame, one splat per pixel at this width
SKY_M = 220.0                         # the sky as a far backdrop
REPLAY_S = 1.5                        # the vehicles replay their paths from this long before the instant
MIN_VEH_PX = 400                      # vehicles smaller than this (at PHOTO_W) stay in the photo
MAX_YAW = 25.0                        # degrees a vehicle layer may turn during its replay
MAX_SPEED = 45.0                      # m/s; faster steps in a track are depth noise


def pick(case: pathlib.Path) -> list[dict]:
    """Key instants from the physics: impact, just before it, braking onset, the other party's first appearance."""
    ph = json.loads((case / "physics.json").read_text())
    meta = json.loads((case / "meta.json").read_text())
    imp = ph.get("impact", {})
    t_imp = float(imp.get("t", meta["time_of_event"]))
    near = imp.get("kind") == "near-miss"
    meas = {m["id"]: m["value"] for m in ph.get("measurements", [])}
    other = next((o["label"] for o in ph.get("objects", []) if o.get("vid") == "v2"), "The other vehicle")
    cand = [("impact", t_imp, "Closest approach" if near else "Impact"),
            ("before", t_imp - 0.6, "Just before the closest approach" if near else "Just before impact")]
    if imp.get("brake_t") is not None:
        cand.append(("brake", float(imp["brake_t"]), "Braking starts"))
    if meas.get("v2.visible_before_impact"):
        cand.append(("hazard", t_imp - float(meas["v2.visible_before_impact"]), f"{other} appears"))
    out = []
    for kind, t, label in cand:                         # in priority order
        if t < 0.5:                                     # "from the first frame on" is not an instant
            continue
        t = min(t, float(meta["duration"]) - 0.05)
        if all(abs(t - m["t"]) >= MIN_GAP_S for m in out):
            out.append({"id": kind, "kind": kind, "t": round(t, 2), "label": label})
    # A passing witness need not brake or see the struck vehicle before contact.
    # Retain a useful third view of the actual aftermath without inventing either event.
    after=t_imp+.8
    if len(out)<3 and not near and after<float(meta['duration'])-.05:
        out.append({'id':'after','kind':'after','t':round(after,2),'label':'After impact'})
    return sorted(out, key=lambda m: m["t"])


def window(times: np.ndarray, t: float) -> tuple[np.ndarray, int]:
    """Recon frames around t (at most MAX_VIEWS) and the position of the moment's own frame among them."""
    k = int(np.argmin(np.abs(times - t)))
    idx = np.where((times >= t - BEFORE_S) & (times <= t + AFTER_S))[0]
    if len(idx) > SP.MAX_VIEWS:
        idx = idx[np.round(np.linspace(0, len(idx) - 1, SP.MAX_VIEWS)).astype(int)]
    idx = np.unique(np.append(idx, k))
    return idx, int(np.where(idx == k)[0][0])


def pose(case: pathlib.Path, k: int) -> tuple[dict, list]:
    """Dashcam pose at recon frame k (levelled) and the focus: the median of nearby vehicle pixels, else 12 m ahead."""
    rec = np.load(case / "recon.npz")
    f4 = json.loads((case / "frames4d.json").read_text())
    s = float(json.loads((case / "tracks.json").read_text())["scale"]["value"])
    M = np.array(f4["frames"][k]["M"], np.float64)
    R, C = M[:3, :3], M[:3, 3].copy()
    lift = C[1] - SP.CAM_H
    C[1] = SP.CAM_H
    fwd = R @ np.array([0, 0, 1.0])
    H = rec["depth"].shape[1]
    K = rec["intrinsic"][k].astype(np.float64)
    cam = {"pos": C.round(3).tolist(), "fwd": fwd.round(4).tolist(), "fov_y": round(float(2 * np.arctan(H / 2 / K[1, 1])), 4)}
    focus = C + 12 * fwd
    focus[1] = 1.0
    if (case / "dynmask.npz").exists():
        dyn = SP.load_dynmask(case)[k]
        d = rec["depth"][k].astype(np.float64) * s
        v, u = np.nonzero(dyn & (d > 2) & (d < 30))
        if len(u) > 50:
            dd = d[v, u]
            P = np.stack([(u - K[0, 2]) / K[0, 0] * dd, (v - K[1, 2]) / K[1, 1] * dd, dd], 1) @ R.T + M[:3, 3]
            focus = np.median(P, 0)
            focus[1] = max(0.6, focus[1] - lift)
    return cam, np.round(focus, 3).tolist()


def read_frame(case: pathlib.Path, k: int) -> np.ndarray:
    """Recon frame k from the clip at full resolution (BGR)."""
    rec = np.load(case / "recon.npz")
    meta = json.loads((case / "meta.json").read_text())
    cap = cv2.VideoCapture(str(case / "clip.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(rec["times"][k]) * meta["src_fps"])))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {k} of the clip")
    return frame


def vehicle_layers(case: pathlib.Path, k: int, W: int, H: int) -> list[tuple[str, np.ndarray]]:
    """Tracked vehicles visible in recon frame k, each with its SAM mask at W x H. When two tracks cover the same
    vehicle (SAM sometimes follows one car under two ids), the longer track keeps it."""
    meta = json.loads((case / "meta.json").read_text())
    tracks = json.loads((case / "tracks.json").read_text())["tracks"]
    concepts = json.loads((case / "concepts.json").read_text())
    f = int(round(float(np.load(case / "recon.npz")["times"][k]) * meta["src_fps"]))
    cand = []
    for key, frames in lift.vehicle_inventory(case, concepts)[0].items():
        r = frames.get(f) or frames.get(f - 1) or frames.get(f + 1)
        if r is None or key not in tracks:
            continue
        m = cv2.resize(segment.full_frame_mask(r, meta["width"], meta["height"]).astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST)
        # SAM leaves cracks along reflections; through them the cleaned road would show inside the body
        m = binary_fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0)
        if m.sum() >= MIN_VEH_PX:
            cand.append((key, m))
    out = []
    for key, m in sorted(cand, key=lambda c: -len(tracks[c[0]])):
        if all((m & o).sum() < 0.5 * min(m.sum(), o.sum()) for _, o in out):
            out.append((key, m))
    return out


def clean_plate(case: pathlib.Path, mid: str, k: int, frame: np.ndarray, vehicles: np.ndarray) -> np.ndarray | None:
    """The frame without its vehicles (Muse Image, cached per moment and frame), aligned to the real frame.
    RGB float at the frame's size, or None (no key, call failed, or it does not line up)."""
    path = case / "moments" / f"{mid}.k{k:03d}.clean.png"
    if not path.exists():
        src = case / "moments" / f"{mid}.k{k:03d}.frame.png"
        cv2.imwrite(str(src), read_frame(case, k))
        ok = make_clean_plate(src, path)
        src.unlink(missing_ok=True)
        if ok is None:
            return None
    H, W = frame.shape[:2]
    clean = cv2.resize(cv2.imread(str(path)), (W, H), interpolation=cv2.INTER_AREA)
    gray = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255  # noqa: E731
    static = ~(cv2.dilate(vehicles.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
    warp = np.eye(2, 3, dtype=np.float32)
    try:                                                    # edits may shift or rescale the image slightly
        cc, warp = cv2.findTransformECC(gray(frame), gray(clean), warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
                                        static.astype(np.uint8) * 255, 5)
    except cv2.error:
        cc = 0.0
    if cc < 0.7:
        log.warning("clean plate %s does not line up with the frame (ecc %.2f); vehicles stay frozen", path.name, cc)
        return None
    clean = cv2.warpAffine(clean, warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def cleaned_region(rgb: np.ndarray, clean: np.ndarray, vehicles: np.ndarray) -> np.ndarray:
    """Pixels to take from the clean plate: the vehicles, a margin, and what changed right next to them (shadows)."""
    box = lambda n: np.ones((n, n), np.uint8)  # noqa: E731
    v = vehicles.astype(np.uint8)
    near = cv2.dilate(v, box(41)) > 0
    changed = np.abs(clean - rgb).mean(-1) > 0.10
    region = (cv2.dilate(v, box(9)) > 0) | (changed & near)
    return cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, box(5)) > 0


def fill_depth(d: np.ndarray, region: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Depth where the vehicles were: the road plane (camera CAM_H above it) where the ray meets the road, else what
    surrounds the hole; whichever is nearer, but always well behind the vehicle that stood there (d there is the
    vehicle's depth), or the background would show through it."""
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    ray = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1) @ R.T
    road = np.where(ray[..., 1] < -1e-3, SP.CAM_H / np.maximum(-ray[..., 1], 1e-3), np.inf)
    around = cv2.inpaint(np.where(region, 0, d).astype(np.float32), region.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    out = d.copy()
    out[region] = np.maximum(np.minimum(road, around), 1.15 * d + 1.0)[region]
    return out


def photo_splats(case: pathlib.Path, k: int, glued: np.ndarray, frame: np.ndarray,
                 layers: list[tuple[str, np.ndarray]] = (), clean: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    """The moment's own video frame (BGR, PHOTO_W wide) as one small camera-facing disc per pixel at its metric depth.

    Seen from the dashcam's pose this is the photo itself; turned a little, it has real parallax. Depth edges stay
    sharp (no rubber sheets between a car and the road behind it) and get a skirt of inpainted background behind them;
    the camera car's own hood and overlays are dropped; the sky is a far backdrop. With a clean plate the vehicle
    layers come out separately. Returns (background, [one splat set per layer], what the photo shows for carve()).
    """
    from scipy.spatial.transform import Rotation
    rec = np.load(case / "recon.npz")
    f4 = json.loads((case / "frames4d.json").read_text())
    s = float(json.loads((case / "tracks.json").read_text())["scale"]["value"])
    T, h, w = rec["depth"].shape
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    K = rec["intrinsic"][k].astype(np.float64) * np.array([[W / w], [H / h], [1]])
    M = np.array(f4["frames"][k]["M"], np.float64)
    R, t = M[:3, :3], M[:3, 3] - np.array([0, M[1, 3] - SP.CAM_H, 0])     # levelled like the rest of the moment
    D = np.diag([1.0, 1.0, -1.0])
    rot = D @ R @ D                                                        # disc axes = camera axes, in three.js axes
    q = Rotation.from_matrix(rot if np.linalg.det(rot) > 0 else rot @ np.diag([1, 1, -1])).as_quat()[[3, 0, 1, 2]]
    box = lambda n: np.ones((n, n), np.uint8)  # noqa: E731
    up = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    near_up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    sky = SP.load_sky(case, (T, h, w))[k]
    sky_h, hood = up(sky), up(glued)

    def discs(mask, depth, colour):
        v, u = np.nonzero(mask)
        z = depth[v, u].astype(np.float64)
        P = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1) @ R.T + t
        fp = z / K[0, 0]                                                   # one pixel's footprint at that depth
        return {"pos": P, "scales": np.stack([0.75 * fp, 0.75 * fp, 0.08 * fp], 1), "quat": np.repeat(q[None], len(P), 0),
                "rgb": colour[v, u].astype(np.float64), "op": np.full(len(P), 0.98)}

    def upsample(d_lo):
        """Smooth depth inside surfaces, but no in-between values across an edge (those would be rubber sheets)."""
        edge = (cv2.dilate(d_lo, box(3)) - cv2.erode(d_lo, box(3))) > 0.07 * d_lo + 0.15
        d = np.where(up(edge), near_up(d_lo), cv2.resize(d_lo, (W, H), interpolation=cv2.INTER_LINEAR))
        return np.minimum(d, 150.0), edge

    d_lo = rec["depth"][k].astype(np.float32) * s
    K_lo = rec["intrinsic"][k].astype(np.float64)
    cleaned = np.zeros((h, w), bool)
    if clean is not None and layers:
        veh = np.any([m for _, m in layers], axis=0)
        region = cleaned_region(rgb, clean, veh)
        rgb_bg = np.where(region[..., None], clean, rgb)
        cleaned = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
        d_bg = fill_depth(d_lo, cleaned, K_lo, R)
        cleaned = cv2.dilate(cleaned.astype(np.uint8), box(5)) > 0
    else:
        layers, rgb_bg, d_bg = [], rgb, d_lo
    view = (~glued & ~sky, np.minimum(d_bg, 150.0), K_lo, R, t, cleaned)  # what the photo shows from this camera

    d, edge = upsample(d_bg)
    d[sky_h] = SKY_M
    keep = ~hood & (d > 1.0)
    # behind the near side of every depth edge, a skirt at the far depth in inpainted background colours: turning
    # the view uncovers that, instead of a black hole
    d_far = cv2.dilate(d_bg, box(7))
    fg = cv2.dilate((edge & (d_bg < d_far - 1.0) & ~sky).astype(np.uint8), box(3)) > 0
    skirt = up(fg) & ~hood & ~sky_h
    bg_rgb = cv2.inpaint((rgb_bg * 255).astype(np.uint8), up(cv2.dilate(fg.astype(np.uint8), box(3))).astype(np.uint8),
                         6, cv2.INPAINT_TELEA).astype(np.float32) / 255
    d_skirt = np.minimum(near_up(d_far), 150.0)
    background = merge(discs(keep, d, rgb_bg), discs(skirt, d_skirt, bg_rgb))

    cars = []
    d_car, _ = upsample(d_lo)
    for _, m in layers:
        sel = m & ~hood & (d_car > 1.0)
        cars.append(discs(sel, vehicle_depth(d_car, sel, K), rgb))
    return background, cars, view


def vehicle_depth(d: np.ndarray, sel: np.ndarray, K: np.ndarray, seed: int = 0) -> np.ndarray:
    """The visible side of a vehicle is close to a plane; monocular depth sometimes puts part of it on the background
    behind (or the road's edge pixels on it). A RANSAC plane through its pixels replaces depths far off that plane."""
    v, u = np.nonzero(sel)
    out = d.copy()
    if len(u) < 50:
        return out
    z = d[v, u].astype(np.float64)
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(u))], 1)
    P = rays * z[:, None]
    rng = np.random.default_rng(seed)
    best, plane = -1, None
    for _ in range(200):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < np.maximum(0.25, 0.04 * z)
        if inl.sum() > best:
            best, plane = inl.sum(), (n, float(n @ a))
    if plane is None or best < 0.4 * len(P):
        return out
    n, c = plane
    zp = c / np.where(np.abs(rays @ n) > 1e-6, rays @ n, np.nan)          # the plane's depth along each pixel ray
    on = np.abs(z - zp) <= np.maximum(1.0, 0.25 * zp)
    zmax = z[on].max() if on.any() else np.median(z)
    off = ~on
    fix = off & np.isfinite(zp) & (zp > 0.5) & (zp < 1.5 * zmax)        # rays near the plane's edge diverge
    z[fix] = zp[fix]
    z[off & ~fix] = np.median(z[~off]) if (~off).any() else np.median(z)
    if on.any():                     # a far, small vehicle gives a poorly conditioned plane; no car is deeper than ~8 m
        mid = float(np.median(z[on]))
        z = np.clip(z, max(z[on].min(), mid - 3.0), min(zmax, mid + 5.0))
    out[v, u] = z
    return out


def carve(world: dict, region: np.ndarray, depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
          cleaned: np.ndarray) -> dict:
    """Drop world splats that, seen from the moment's camera, stand in front of the photo (its background, with a
    clean plate). From that pose the photo is the truth, so they are remnants of vehicles the masks missed in other
    frames (a frozen vehicle used to hide them) or depth noise. Where the vehicles were removed, anything not well
    behind the filled background goes too: remnants there sit about as deep as the guessed background."""
    Pc = (world["pos"] - t) @ R                                            # levelled road -> camera (OpenCV axes)
    z = Pc[:, 2]
    h, w = depth.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K[0, 0] * Pc[:, 0] / z + K[0, 2])
        v = np.floor(K[1, 1] * Pc[:, 1] / z + K[1, 2])
    inb = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui, vi = u[inb].astype(int), v[inb].astype(int)
    drop = np.zeros(len(z), bool)
    d = depth[vi, ui]
    drop[inb] = region[vi, ui] & np.where(cleaned[vi, ui], z[inb] < 1.3 * d + 1.0, z[inb] < 0.88 * d)
    return {key: (val[~drop] if isinstance(val, np.ndarray) and len(val) == len(z) else val) for key, val in world.items()}


def compose(case: pathlib.Path, mid: str, k: int, glued: np.ndarray, world: dict) -> tuple[dict, dict, list[tuple[str, dict]]]:
    """A moment's splats from recon frame k over a static world: (photo + world, photo, [(vehicle key, its layer)]).
    With a clean plate the vehicles come out as layers and the world is carved where they stood; without one they
    stay frozen in the photo and the list is empty."""
    frame = read_frame(case, k)
    frame = cv2.resize(frame, (PHOTO_W, int(round(PHOTO_W * frame.shape[0] / frame.shape[1]))), interpolation=cv2.INTER_AREA)
    layers = vehicle_layers(case, k, frame.shape[1], frame.shape[0])
    clean = clean_plate(case, mid, k, frame, np.any([lm for _, lm in layers], axis=0)) if layers else None
    photo, cars, view = photo_splats(case, k, glued, frame, layers, clean)
    n = len(world["pos"])
    world = carve(world, *view)
    log.info("moment %s: %d world splats in front of the photo removed", mid, n - len(world["pos"]))
    return merge(photo, world), photo, [(key, car) for (key, _), car in zip(layers, cars)]


def vehicle_keys(t, P, t_k: float, span: float = REPLAY_S) -> list[list[float]]:
    """Path keys over [t_k - span, t_k] relative to the vehicle at t_k: [t - t_k, dx, dz, dyaw degrees] in the road
    frame. [] for a vehicle that does not move (or has too short a track)."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return []
    X = display_path(np.asarray(P, float))                                # x, z; outliers repaired, smoothed
    V = np.gradient(X, t, axis=0)
    fast = np.linalg.norm(V, axis=1) > 1.0                                 # heading is noise when (nearly) stopped
    if fast.any():
        idx = np.maximum.accumulate(np.where(fast, np.arange(len(t)), 0))
        idx[: np.argmax(fast)] = np.argmax(fast)
        h = np.unwrap(np.arctan2(V[:, 0], V[:, 1])[idx])
    else:
        h = np.zeros(len(t))
    at = lambda arr, x: float(np.interp(x, t, arr))  # noqa: E731
    xk, zk, hk = at(X[:, 0], t_k), at(X[:, 1], t_k), at(h, t_k)
    # a layer is the side of the vehicle the camera saw: it may turn a little, never show its edge (larger turns in
    # the track are depth noise reversing the path)
    keys = [[round(a - t_k, 3), round(at(X[:, 0], a) - xk, 3), round(at(X[:, 1], a) - zk, 3),
             round(float(np.clip(np.degrees(at(h, a) - hk), -MAX_YAW, MAX_YAW)), 1)]
            for a in t[(t >= t_k - span) & (t < t_k - 1e-3)]] + [[0.0, 0.0, 0.0, 0.0]]
    for i in range(len(keys) - 1, 0, -1):          # a jump faster than any car: the path before it is not this car
        a, b = keys[i - 1], keys[i]
        if np.hypot(b[1] - a[1], b[2] - a[2]) > MAX_SPEED * max(b[0] - a[0], 1e-3):
            keys = keys[i:]
            break
    if keys[0][0] > -0.4 or max(np.hypot(kk[1], kk[2]) for kk in keys) < 0.3:
        return []
    return keys


def backdrop(case: pathlib.Path, k: int, glued: np.ndarray) -> dict:
    """How the viewer frames a moment: above the hood (nothing was ever seen behind it), on a backdrop that fades from
    the sky's colour down to the road's, so what the camera never saw reads as sky or road rather than a hole."""
    rec = np.load(case / "recon.npz")
    img = rec["colors"][k]
    H = img.shape[0]
    rows = glued.mean(1) > 0.5
    crop = float(np.argmax(~rows[::-1]) / H) if rows[-1] else 0.0
    sky = SP.load_sky(case, rec["depth"].shape)[k]
    y0 = int(H * (1 - crop))
    road = img[max(0, y0 - H // 6):y0, img.shape[1] // 4:3 * img.shape[1] // 4].reshape(-1, 3)
    return {"crop_bottom": round(crop, 3),
            "sky": (np.median(img[sky], 0) / 255).round(3).tolist() if sky.sum() > 500 else None,
            "ground": (np.median(road, 0) / 255).round(3).tolist() if len(road) else None}


def merge(a: dict, b: dict) -> dict:
    return {key: np.concatenate([a[key], b[key]]) for key in ("pos", "scales", "quat", "rgb", "op")}


def local(r: dict, origin) -> dict:
    """Splats relative to the moment's origin (its camera). The browser packs splat centres as float16: 150 m from
    the road frame's origin that is a 12 cm grid, which tears a car 2 m from the camera into stripes."""
    return dict(r, pos=r["pos"] - np.asarray(origin, float))


def build(case: pathlib.Path, force: bool = False) -> pathlib.Path:
    out = case / "moments.json"
    if out.exists() and not force:
        return out
    rec = np.load(case / "recon.npz")
    times = rec["times"].astype(float)
    tracks = json.loads((case / "tracks.json").read_text())
    (case / "moments").mkdir(exist_ok=True)
    for f in (case / "moments").glob("*.splat"):                        # incl. vehicle layers of earlier builds
        f.unlink()
    dev = SP.device()
    SP.patch_autocast(dev)
    model = SP.load_model(dev)
    moments = []
    for m in pick(case):
        views, j = window(times, m["t"])
        k = int(views[j])
        world = SP.reconstruct(case, views, model, dev, max_splats=800_000)   # static world around the instant
        glued = SP.glued_mask(rec["colors"][views])
        r, photo, cars = compose(case, m["id"], k, glued, world)
        cam, focus = pose(case, k)
        SP.save(case / "moments" / f"{m['id']}.splat", local(r, cam["pos"]))
        vehicles = []
        for key, car in cars:
            if not len(car["pos"]):
                continue
            fn = f"moments/{m['id']}.{key.replace(':', '-')}.splat"
            SP.save(case / fn, local(car, cam["pos"]))
            rows = tracks["tracks"][key]
            vehicles.append({"key": key, "label": tracks.get("aliases", {}).get(key, key.split(":")[0]), "file": fn,
                             "count": int(len(car["pos"])), "pivot": np.median(car["pos"], 0).round(3).tolist(),
                             "keys": vehicle_keys([rw["t"] for rw in rows], [rw["center"] for rw in rows], float(times[k]))})
        cam.update(backdrop(case, k, glued))
        moments.append({**m, "t_frame": round(float(times[k]), 3), "file": f"moments/{m['id']}.splat", "origin": cam["pos"],
                        "count": int(len(r["pos"])), "photo_splats": int(len(photo["pos"])), "views": len(views),
                        "cam": cam, "focus": focus, "align": world["align"], "seconds": world["seconds"],
                        "clean": {"method": "Muse Image edit", "file": f"moments/{m['id']}.k{k:03d}.clean.png"}
                        if cars else None, "vehicles": vehicles})
        log.info("moment %s at %.2fs: %d splats, %d vehicle layers (%d moving)", m["id"], m["t"], len(r["pos"]),
                 len(vehicles), sum(bool(v["keys"]) for v in vehicles))
    out.write_text(json.dumps({"method": "AnySplat feed-forward 3DGS", "device": dev.type, "replay_s": REPLAY_S,
                               "moments": moments}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--cases-dir", default=str(ROOT / "data" / "cases"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(pathlib.Path(a.cases_dir) / a.case, force=a.force)


if __name__ == "__main__":
    main()

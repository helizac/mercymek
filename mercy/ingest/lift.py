"""Lift SAM masks into 3D with the reconstruction (section 8, steps 3-4).

Outputs (case folder):
  tracks.json   per-object per-frame road-frame center/bbox, ego pose per frame, road frame + scale
  cloud.json + cloud.bin   background cloud (with reveal frame) and per-frame dynamic points for the HUD

Road frame (metres): x = right, y = up (road normal), z = forward (ego travel direction).
Origin = camera-0 centre projected onto the road plane. The HUD maps it to three.js as (x, y, -z).
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

from mercy import config
from mercy.config import log
from mercy.ingest import segment

VEHICLE_CONCEPTS = {"car", "truck", "bus", "motorcycle", "van", "suv", "pickup", "hatchback", "sedan"}
CAM_HEIGHT = (1.2, 1.35, 1.5)          # typical dashcam mounting height range (m)
BODY_LENGTH = {"hatchback": 4.2, "sedan": 4.7, "car": 4.5, "suv": 4.8, "pickup": 5.5, "truck": 6.5,
               "van": 5.0, "bus": 12.0, "motorcycle": 2.1, "semi-truck": 16.0}


# ---------------------------------------------------------------- object inventory
def is_vehicle_concept(concept: str) -> bool:
    words = set(concept.lower().replace("-", " ").split())
    return bool(words & VEHICLE_CONCEPTS) or concept in VEHICLE_CONCEPTS


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def vehicle_inventory(case: pathlib.Path, concepts: list[str]) -> dict[str, dict[int, object]]:
    """All vehicle tracks keyed 'concept:id'. Specific labels ('blue hatchback') are merged into the generic
    track they coincide with (box IoU), so one physical car has one key; the specific label is kept as alias."""
    generic, specific = {}, {}
    for c in concepts:
        if not is_vehicle_concept(c):
            continue
        objs = segment.records_by_object(case, c)
        (generic if c in VEHICLE_CONCEPTS else specific).update(objs)
    aliases: dict[str, str] = {}
    for skey, sframes in specific.items():
        best, best_iou = None, 0.0
        for gkey, gframes in generic.items():
            common = set(sframes) & set(gframes)
            if len(common) < 3:
                continue
            ious = [box_iou(_box(sframes[f]), _box(gframes[f])) for f in common]
            m = float(np.mean(ious))
            if m > best_iou:
                best, best_iou = gkey, m
        if best is not None and best_iou > 0.5:
            merged = dict(generic[best]); merged.update(sframes)      # prefer the specific mask where present
            generic[best] = merged
            aliases[best] = skey.split(":")[0]
        else:
            generic[skey] = sframes
    for k, a in aliases.items():
        log.info("alias %s = %s", k, a)
    return generic, aliases


def _box(r):
    b = r.bounds
    return (b.left, b.top, b.right, b.bottom)


# ---------------------------------------------------------------- geometry helpers
def camera_centers(extr: np.ndarray) -> np.ndarray:
    R, t = extr[:, :, :3], extr[:, :, 3]
    return -np.einsum("tji,tj->ti", R, t)                       # C = -R^T t


def fit_plane(pts: np.ndarray, down_hint: np.ndarray, thr: float, iters: int = 600, seed: int = 0):
    """RANSAC plane n.p + d = 0 constrained to be roughly perpendicular to the camera's down axis."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n @ down_hint) < np.cos(np.radians(30)):
            continue
        d = -n @ s[0]
        inl = np.abs(pts @ n + d) < thr
        c = int(inl.sum())
        if c > best_n:
            best, best_n = inl, c
    P = pts[best]
    mu = P.mean(0)
    n = np.linalg.eigh(np.cov((P - mu).T))[1][:, 0]            # smallest-variance direction
    return n, -n @ mu, best_n / len(pts)


def resize_mask(m: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def vehicle_masks(case, concepts, meta, times, W, H):
    """Per-vehicle masks at reconstruction resolution and recon frames, plus the dilated union of all moving objects.
    Returns (masks {key: {k: HxW bool}}, aliases, union [T,H,W] bool)."""
    W0, H0, fps = meta["width"], meta["height"], meta["src_fps"]
    sam_f = [int(round(t * fps)) for t in times]
    vehicles, aliases = vehicle_inventory(case, concepts)
    for c in ("pedestrian", "motorcycle", "debris"):           # other movers: excluded from the static scene too
        if c in concepts:
            for key, fr in segment.records_by_object(case, c).items():
                vehicles.setdefault(key, fr)
    masks: dict[str, dict[int, np.ndarray]] = {}
    for key, frames in vehicles.items():
        for k, f in enumerate(sam_f):
            r = frames.get(f) or frames.get(f - 1) or frames.get(f + 1)
            if r is None:
                continue
            masks.setdefault(key, {})[k] = resize_mask(segment.full_frame_mask(r, W0, H0), W, H)
    T = len(times)
    any_dyn = np.zeros((T, H, W), bool)
    for key, fr in masks.items():
        for k, m in fr.items():
            any_dyn[k] |= m
    dil = np.stack([cv2.dilate(a.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for a in any_dyn])
    masks = {k: v for k, v in masks.items() if is_vehicle_concept(k.split(":")[0])}
    return masks, aliases, dil


def _sky_frame(job):
    k, recs, W0, H0, W, H = job
    m = np.zeros((H, W), bool)
    for r in recs:
        m |= resize_mask(segment.full_frame_mask(r, W0, H0), W, H)
    return k, m


def sky_masks(case: pathlib.Path, force: bool = False) -> pathlib.Path | None:
    """SAM 3.1 'sky' masks at reconstruction resolution -> skymask.npz (used to keep sky out of the 3D scene)."""
    out = case / "skymask.npz"
    if out.exists() and not force:
        return out
    if not (case / "masks" / "sky.lines").exists():
        return None
    meta = json.loads((case / "meta.json").read_text())
    rec = np.load(case / "recon.npz")
    times = rec["times"].astype(float)
    T, H, W = rec["depth"].shape
    W0, H0, fps = meta["width"], meta["height"], meta["src_fps"]
    frames: dict[int, list] = {}
    for key, fr in segment.records_by_object(case, "sky").items():
        for f, r in fr.items():
            frames.setdefault(f, []).append(r)
    jobs = []
    for k, t in enumerate(times):
        f = int(round(t * fps))
        jobs.append((k, frames.get(f) or frames.get(f - 1) or frames.get(f + 1) or [], W0, H0, W, H))
    sky = np.zeros((T, H, W), bool)
    from concurrent.futures import ProcessPoolExecutor
    import os
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:   # pure-Python mask decoding
        for k, m in ex.map(_sky_frame, jobs):
            sky[k] = m
    np.savez_compressed(out, mask=np.packbits(sky, axis=-1), shape=np.array(sky.shape))
    log.info("sky masks: %.0f%% of pixels", 100 * sky.mean())
    return out


def load_dynmask(case: pathlib.Path) -> np.ndarray | None:
    p = case / "dynmask.npz"
    if not p.exists():
        return None
    z = np.load(p)
    T, H, W = z["shape"]
    return np.unpackbits(z["mask"], axis=-1)[..., :W].astype(bool)


# ---------------------------------------------------------------- main
def lift(case: pathlib.Path, concepts: list[str], force: bool = False) -> dict:
    out_path = case / "tracks.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    meta = json.loads((case / "meta.json").read_text())
    rec = np.load(case / "recon.npz")
    pts = rec["points"].astype(np.float32)            # [T,H,W,3]
    conf = rec["depth_conf"].astype(np.float32)
    cols = rec["colors"]
    extr = rec["extrinsic"]
    times = rec["times"].astype(float)
    method = str(rec["method"])
    T, H, W, _ = pts.shape
    W0, H0, fps = meta["width"], meta["height"], meta["src_fps"]
    sam_f = [int(round(t * fps)) for t in times]

    masks, aliases, dil = vehicle_masks(case, concepts, meta, times, W, H)
    np.savez_compressed(case / "dynmask.npz", mask=np.packbits(dil, axis=-1), shape=np.array(dil.shape))

    # ---- road plane: fitted per frame on the static road patch just ahead of the car (rows 58-84 %, cols 22-78 %).
    # Monocular geometry drifts over a 200 m clip, so one global plane is unreliable; the per-frame camera height
    # above its own local road patch is not. Global "up" = mean of the per-frame normals.
    C = camera_centers(extr)
    rng = np.random.default_rng(0)
    R0 = extr[0, :, :3]
    normals, heights, fracs = [], [], []
    patch = np.zeros((H, W), bool)
    patch[int(H * 0.58):int(H * 0.84), int(W * 0.22):int(W * 0.78)] = True
    for k in range(T):
        sel = patch & ~dil[k]
        c = conf[k]
        sel &= c > np.percentile(c, 40)
        idx = np.flatnonzero(sel)
        if len(idx) < 200:
            continue
        P = pts[k].reshape(-1, 3)[rng.choice(idx, min(1500, len(idx)), replace=False)]
        down_k = extr[k, :, :3].T @ np.array([0, 1.0, 0])
        spread_k = np.median(np.linalg.norm(P - C[k], axis=1))
        try:
            n_k, d_k, frac = fit_plane(P, down_k, thr=0.02 * spread_k, iters=200, seed=k)
        except Exception:
            continue
        if n_k @ C[k] + d_k < 0:
            n_k, d_k = -n_k, -d_k                               # normal points up (towards the camera)
        if frac > 0.4:
            normals.append(n_k); heights.append(n_k @ C[k] + d_k); fracs.append(frac)
    if len(heights) < 3:
        raise RuntimeError("road plane could not be estimated (too few road frames)")
    n = np.mean(normals, axis=0); n /= np.linalg.norm(n)
    h_est = float(np.median(heights))
    inlier_frac = float(np.median(fracs))
    d = h_est - float(n @ C[0])                                 # camera 0 sits h_est above the road
    h_k = np.array(heights)
    s_lo, s_mid, s_hi = (CAM_HEIGHT[0] / h_est, CAM_HEIGHT[1] / h_est, CAM_HEIGHT[2] / h_est)

    # ---- road frame
    up = n / np.linalg.norm(n)
    disp = C[-1] - C[0]
    disp -= (disp @ up) * up
    if np.linalg.norm(disp) > 0.5 * h_est:
        fwd = disp / np.linalg.norm(disp)
    else:                                                       # (nearly) stationary camera
        z0 = R0.T @ np.array([0, 0, 1.0]); z0 -= (z0 @ up) * up
        fwd = z0 / np.linalg.norm(z0)
    right = np.cross(fwd, up)
    origin = C[0] - (C[0] @ up + d) * up
    Rr = np.stack([right, up, fwd])                             # rows: world -> road axes

    def to_road(P):
        return (np.asarray(P) - origin) @ Rr.T * s_mid

    # ---- ego pose per frame
    ego = []
    for k in range(T):
        Rk = extr[k, :, :3]
        f_w = Rk.T @ np.array([0, 0, 1.0]); u_w = Rk.T @ np.array([0, -1.0, 0])
        ego.append({"k": k, "t": round(float(times[k]), 3), "pos": to_road(C[k]).round(3).tolist(),
                    "fwd": (Rr @ f_w).round(4).tolist(), "up": (Rr @ u_w).round(4).tolist()})
    K0 = rec["intrinsic"][0]
    fov_y = float(2 * np.arctan(H / (2 * K0[1, 1])))

    # ---- object tracks
    tracks = {}
    for key, fr in masks.items():
        rows = []
        for k, m in sorted(fr.items()):
            core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if core.sum() < 30:
                core = m
            c = conf[k]
            sel = core & (c > np.percentile(c[core], 30) if core.any() else c > 0)
            if sel.sum() < 30:
                continue
            P = to_road(pts[k][sel])
            dist = np.linalg.norm(P[:, [0, 2]] - np.array(ego[k]["pos"])[[0, 2]], axis=1)
            if np.median(dist) > 90:
                continue
            ctr = np.median(P, 0)
            lo, hi = np.percentile(P, [5, 95], axis=0)
            rows.append({"k": k, "t": round(float(times[k]), 3), "center": ctr.round(3).tolist(),
                         "bbox": [lo.round(3).tolist(), hi.round(3).tolist()], "n": int(sel.sum())})
        if len(rows) >= 3:
            tracks[key] = rows

    out = {
        "method": method, "times": [round(float(t), 3) for t in times], "sam_frames": sam_f,
        "recon_size": [W, H], "fov_y": fov_y, "aspect": W / H,
        "road": {"normal_world": up.tolist(), "d": float(d), "right": right.tolist(), "fwd": fwd.tolist(),
                 "origin": origin.tolist(), "inlier_frac": round(float(inlier_frac), 3)},
        "scale": {"value": s_mid, "range": [s_lo, s_hi], "h_est_units": h_est,
                  "method": "camera height above RANSAC road plane = 1.35 m (1.2-1.5 m)"},
        "ego": ego, "tracks": tracks, "aliases": aliases,
    }
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    log.info("lift: %s, h=%.4f units (spread %.0f%%) -> scale %.2f m/unit, plane inliers %.0f%%, %d tracks",
             method, h_est, 100 * float(np.std(h_k) / max(h_est, 1e-9)), s_mid, 100 * inlier_frac, len(tracks))

    export_cloud(case, pts, conf, cols, dil, masks, to_road, ego)
    return out


def export_cloud(case, pts, conf, cols, dil, masks, to_road, ego, max_bg=260_000, voxel=0.14, dyn_per_obj=2500):
    """cloud.bin layout (little endian, 4-byte aligned), described by cloud.json:
       bg_xyz f32[N*3] | bg_rgb u8[N*3] (+pad) | bg_k u16[N] (+pad) | dyn_xyz f32[M*3] | dyn_rgb u8[M*3] (+pad) | dyn_obj u8[M] (+pad)
    """
    T, H, W, _ = pts.shape
    rng = np.random.default_rng(1)
    bx, bc, bk = [], [], []
    for k in range(T):
        sel = ~dil[k]
        c = conf[k]
        sel &= c > np.percentile(c, 45)
        idx = np.flatnonzero(sel)
        if not len(idx):
            continue
        idx = rng.choice(idx, min(9000, len(idx)), replace=False)
        P = to_road(pts[k].reshape(-1, 3)[idx])
        e = np.array(ego[k]["pos"])
        keep = (np.linalg.norm(P[:, [0, 2]] - e[[0, 2]], axis=1) < 70) & (P[:, 1] > -3) & (P[:, 1] < 25)
        bx.append(P[keep]); bc.append(cols[k].reshape(-1, 3)[idx][keep]); bk.append(np.full(keep.sum(), k))
    bx, bc, bk = np.concatenate(bx), np.concatenate(bc), np.concatenate(bk)
    vox = np.floor(bx / voxel).astype(np.int64)
    _, first = np.unique(vox, axis=0, return_index=True)       # bk is ascending, so this keeps the earliest frame
    if len(first) > max_bg:
        first = rng.choice(first, max_bg, replace=False)
    first.sort()
    bx, bc, bk = bx[first].astype(np.float32), bc[first].astype(np.uint8), bk[first].astype(np.uint16)

    keys = sorted(masks)
    dx, dc, do, frames = [], [], [], []
    start = 0
    for k in range(T):
        cnt = 0
        for oi, key in enumerate(keys):
            m = masks[key].get(k)
            if m is None:
                continue
            c = conf[k]
            sel = m & (c > np.percentile(c[m], 25) if m.any() else c > 0)
            idx = np.flatnonzero(sel)
            if len(idx) < 20:
                continue
            idx = rng.choice(idx, min(dyn_per_obj, len(idx)), replace=False)
            P = to_road(pts[k].reshape(-1, 3)[idx])
            dx.append(P.astype(np.float32)); dc.append(cols[k].reshape(-1, 3)[idx]); do.append(np.full(len(idx), oi, np.uint8))
            cnt += len(idx)
        frames.append({"k": k, "start": start, "count": cnt})
        start += cnt
    dx = np.concatenate(dx) if dx else np.zeros((0, 3), np.float32)
    dc = np.concatenate(dc) if dc else np.zeros((0, 3), np.uint8)
    do = np.concatenate(do) if do else np.zeros((0,), np.uint8)

    parts, layout, off = [], {}, 0
    for name, arr in [("bg_xyz", bx), ("bg_rgb", bc), ("bg_k", bk), ("dyn_xyz", dx), ("dyn_rgb", dc), ("dyn_obj", do)]:
        b = np.ascontiguousarray(arr).tobytes()
        layout[name] = {"offset": off, "bytes": len(b), "dtype": str(arr.dtype)}
        pad = (-len(b)) % 4
        parts.append(b + b"\0" * pad)
        off += len(b) + pad
    (case / "cloud.bin").write_bytes(b"".join(parts))
    lo, hi = np.percentile(bx, [2, 98], axis=0)
    (case / "cloud.json").write_text(json.dumps({
        "n_bg": int(len(bx)), "n_dyn": int(len(dx)), "layout": layout, "frames": frames, "objects": keys,
        "bounds": [lo.round(2).tolist(), hi.round(2).tolist()]}, separators=(",", ":")))
    log.info("cloud: %d background pts, %d dynamic pts (%.1f MB)", len(bx), len(dx), off / 1e6)

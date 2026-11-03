"""Photoreal 3D scene: AnySplat feed-forward Gaussian splatting around the collision.

The frames from a few seconds before to just after impact go through AnySplat in one pass (no per-scene training).
Moving vehicles, the sky and everything glued to the camera (hood, dashboard reflections, timestamp; see glued_mask)
are removed before its voxel fusion, so the splats hold only the static world; vehicles stay as models on their tracks.
The splats are moved into the road frame of tracks.json (x right, y up, z forward, metres) with a similarity transform
fitted on the same static pixels in AnySplat and the metric VGGT depth, then levelled so the local road is y = 0
along the whole path (the global road frame drifts). The result is written as scene.splat (32-byte splats, the widely
used .splat layout) for the browser renderer, already in three.js axes (z negated); scene_splat.json bounds are in the
levelled road frame.

Runs in .venv-recon (torch XPU/CUDA/CPU): python -m mercy.ingest.splat --case X [--force]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "anysplat_shims"))     # torch_scatter / xformers / gsplat stand-ins
sys.path.insert(0, str(ROOT / "vendor" / "AnySplat"))
from mercy.config import log  # noqa: E402
from mercy.ingest.mesh import load_dynmask, load_sky  # noqa: E402

SH_C0 = 0.28209479177387814
WINDOW = (-3.0, 1.0)          # seconds around the event
MAX_VIEWS = 32
NEAR_M = 60.0                 # each view keeps what it saw within this range (far depth is unreliable)
FAR_VIEWS = 3                 # ... except the last views, which also give the far background
MAX_SPLATS = 1_200_000
CAM_H = 1.35                  # camera height above the local road that sets the metric scale (mercy.ingest.lift)


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def patch_autocast(dev):
    """AnySplat hard-codes autocast('cuda'); on XPU that silently disables bf16 and the bf16 input then fails."""
    import torch
    if dev.type == "cuda":
        return
    base = torch.amp.autocast

    class Autocast(base):
        def __init__(self, device_type, *a, **k):
            super().__init__(dev.type if device_type == "cuda" else device_type, *a, **k)

    torch.amp.autocast = Autocast


def masked_voxelize(keep):
    """Replacement for EncoderAnySplat.voxelizaton_with_fusion that drops masked pixels before fusing."""
    import torch
    from torch_scatter import scatter_add, scatter_max

    def fuse(img_feat, pts3d, voxel_size, conf=None):
        k = keep.reshape(-1)
        pts = pts3d.permute(0, 2, 3, 1).flatten(0, 2)[k]
        feats = img_feat.permute(0, 2, 3, 1).flatten(0, 2)[k]
        c = conf.flatten()[k]
        _, inv = torch.unique((pts / voxel_size).round().int(), dim=0, return_inverse=True)
        cmax, _ = scatter_max(c, inv, dim=0)
        e = torch.exp(c - cmax[inv])
        w = (e / (scatter_add(e, inv, dim=0)[inv] + 1e-6)).unsqueeze(-1)
        return scatter_add(pts * w, inv, dim=0), scatter_add(feats * w, inv, dim=0)

    return fuse


def glued_mask(frames: np.ndarray) -> np.ndarray:
    """Pixels that move with the camera (hood, dashboard and its reflection in the windscreen, mounts, timestamp, logos).

    frames: V,H,W,3 uint8 taken while the camera moves. Such pixels keep the same local texture in the same place
    (high patch correlation between views a few frames apart, low temporal deviation). Fused into the world they would
    repeat once per view along the path, which shows up as ribs on the road. The hood is grown up from the bottom edge
    and fitted with a smooth curve (straight lane lines also look glued and must not pull it up); the rest are glued
    components inside the outer band of the frame.
    """
    from scipy.ndimage import median_filter
    c = frames.astype(np.float32) / 255
    g = c.mean(-1)
    V, H, W = g.shape
    mad = np.median(np.abs(c - np.median(c, 0)).mean(-1), 0)
    bl = lambda x: cv2.blur(x, (9, 9))  # noqa: E731
    mu = np.stack([bl(f) for f in g])
    sd = np.sqrt(np.clip(np.stack([bl(f * f) for f in g]) - mu ** 2, 1e-8, None))
    step = 3
    nc = np.median(np.stack([(bl(g[i] * g[i + step]) - mu[i] * mu[i + step]) / (sd[i] * sd[i + step])
                             for i in range(V - step)]), 0)
    tex = np.median(sd, 0)

    ok = ((nc > 0.75) & (mad < 0.08)) | ((mad < 0.015) & (tex < 0.008))   # glued texture, or flat and still
    line = np.full(W, H)
    for x in range(W):
        y, gap = H - 1, 0
        while y >= 0 and gap <= 5:
            gap = 0 if ok[y, x] else gap + 1
            y -= 1
        line[x] = y + gap + 1
    line = median_filter(line, 41, mode="nearest")
    xs = np.arange(W) / W
    rng = np.random.default_rng(0)
    cand = np.where(line > H * 0.6)[0]                     # a hood never reaches the middle of the frame
    best, fit = -1.0, np.full(W, float(H))
    if len(cand) > W * 0.3:
        for _ in range(300):                               # RANSAC quadratic; ties go to the lower curve
            j = rng.choice(cand, 3, replace=False)
            f = np.polyval(np.polyfit(xs[j], line[j], 2), xs)
            inl = (np.abs(f - line) < 5) & (line > H * 0.6)
            score = inl.sum() + 1e-3 * f.mean() / H
            if score > best:
                best, fit = score, np.polyval(np.polyfit(xs[inl], line[inl], 2), xs)
        if best < W * 0.35:
            fit = np.full(W, float(H))
    m = np.arange(H)[:, None] >= np.clip(fit, H * 0.6, H)[None, :] - 2

    still = (nc > 0.8) & (mad < 0.04) & (tex > 0.01)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(still.astype(np.uint8), cv2.MORPH_CLOSE,
                                                                      np.ones((5, 5), np.uint8)))
    bx, by = 0.16 * W, 0.16 * H
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 30 and (x + w <= bx or x >= W - bx or y + h <= by or y >= H - by):
            m |= lab == i
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def select_views(times: np.ndarray, t_event: float) -> np.ndarray:
    idx = np.where((times >= t_event + WINDOW[0]) & (times <= t_event + WINDOW[1]))[0]
    if len(idx) < 8:                                      # short clip or event near an edge: take the nearest frames
        idx = np.sort(np.argsort(np.abs(times - t_event))[:min(len(times), 16)])
    if len(idx) > MAX_VIEWS:
        idx = idx[np.round(np.linspace(0, len(idx) - 1, MAX_VIEWS)).astype(int)]
    return idx


def procrustes(m: np.ndarray, det: float) -> np.ndarray:
    """Closest orthogonal matrix with the given determinant sign (the road frame is mirrored vs OpenCV axes)."""
    u, _, vt = np.linalg.svd(m)
    d = np.sign(det) * np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


def write_splat(path: pathlib.Path, pos, scales, quat_wxyz, rgb, alpha):
    n = len(pos)
    buf = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    buf["p"] = pos
    buf["s"] = scales
    buf["c"][:, :3] = np.clip(rgb * 255, 0, 255)
    buf["c"][:, 3] = np.clip(alpha * 255, 0, 255)
    buf["r"] = np.clip(quat_wxyz * 128 + 128, 0, 255)
    order = np.argsort(-(scales.prod(1) * alpha))          # big, opaque splats first (progressive loading)
    path.write_bytes(buf[order].tobytes())


def load_model(dev):
    from src.model.model.anysplat import AnySplat
    model = AnySplat.from_pretrained("lhjiang/anysplat").to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.distill = False
    return model


def reconstruct(case: pathlib.Path, views: np.ndarray, model, dev, max_splats: int = MAX_SPLATS) -> dict:
    """AnySplat over the given recon frames: the static world (vehicles, sky and camera-glued pixels dropped in every
    view), aligned to the road frame and levelled. Returns the kept splats and the alignment figures."""
    import torch
    from scipy.spatial.transform import Rotation

    rec = np.load(case / "recon.npz")
    f4 = json.loads((case / "frames4d.json").read_text())
    tr = json.loads((case / "tracks.json").read_text())
    s_vggt = float(tr["scale"]["value"])
    T, H, W = rec["depth"].shape
    dyn = load_dynmask(case) if (case / "dynmask.npz").exists() else np.zeros((T, H, W), bool)
    sky = load_sky(case, (T, H, W))
    ker = np.ones((7, 7), np.uint8)
    glued = glued_mask(rec["colors"][views])
    static_drop = np.stack([cv2.dilate((dyn[k] | sky[k]).astype(np.uint8), ker) > 0 for k in views]) | glued[None]
    # far depth is unreliable: only the last views (closest to it) give the distant background
    d_sel = rec["depth"][views].astype(np.float32) * s_vggt
    far = np.zeros(len(views), bool); far[-FAR_VIEWS:] = True
    drop = static_drop | ~((d_sel < NEAR_M) | (far[:, None, None] & (d_sel < 120)))

    t0 = time.time()
    img = torch.from_numpy(rec["colors"][views]).permute(0, 3, 1, 2).float().div(255)[None].to(dev)   # [1,V,3,H,W]
    model.encoder.voxelizaton_with_fusion = masked_voxelize(torch.from_numpy(~drop).to(dev))
    with torch.no_grad():
        enc = model.encoder(img, global_step=0, visualization_dump=None)
    log.info("AnySplat on %s: %d views %dx%d in %.0fs", dev.type, len(views), W, H, time.time() - t0)

    g = enc.gaussians
    mu = g.means[0].float().cpu().numpy().astype(np.float64)
    cov = g.covariances[0].float().cpu().numpy().astype(np.float64)
    dc = g.harmonics[0, :, :, 0].float().cpu().numpy()
    op = g.opacities[0].float().cpu().numpy()
    c2w = enc.pred_context_pose["extrinsic"][0].float().cpu().numpy().astype(np.float64)          # [V,4,4]
    k_any = enc.pred_context_pose["intrinsic"][0].float().cpu().numpy().astype(np.float64)        # normalised
    d_any = enc.depth_dict["depth"][0].float().cpu().numpy().reshape(len(views), H, W)
    del enc, g
    valid = mu[:, 2] > -1e3                                                                      # drop padding

    # similarity transform AnySplat world -> road frame, fitted on the same static pixels seen by both models
    M = np.array([f4["frames"][k]["M"] for k in views], np.float64)
    Rr, Cr = M[:, :3, :3], M[:, :3, 3]
    Ra, Ca = c2w[:, :3, :3], c2w[:, :3, 3]
    d_m = rec["depth"][views].astype(np.float64) * s_vggt
    conf = rec["depth_conf"][views]
    ok = (~drop) & (d_m > 2) & (d_m < 35) & (d_any > 1e-6) & (conf > np.percentile(conf, 30))
    src, dst = [], []
    rng = np.random.default_rng(0)
    for i, k in enumerate(views):
        v, u = np.nonzero(ok[i])
        if len(u) > 6000:
            j = rng.choice(len(u), 6000, replace=False); u, v = u[j], v[j]
        Kv = rec["intrinsic"][k].astype(np.float64)
        dm = d_m[i, v, u]
        dst.append(np.stack([(u - Kv[0, 2]) / Kv[0, 0] * dm, (v - Kv[1, 2]) / Kv[1, 1] * dm, dm], 1) @ Rr[i].T + Cr[i])
        Ka = k_any[i] * np.array([[W], [H], [1]])
        da = d_any[i, v, u]
        src.append(np.stack([(u - Ka[0, 2]) / Ka[0, 0] * da, (v - Ka[1, 2]) / Ka[1, 1] * da, da], 1) @ Ra[i].T + Ca[i])
    src, dst = np.concatenate(src), np.concatenate(dst)
    w = np.ones(len(src), bool)
    for _ in range(4):                                                       # trimmed Umeyama (reflection allowed)
        a, b = src[w], dst[w]
        ma, mb = a.mean(0), b.mean(0)
        Q = procrustes((b - mb).T @ (a - ma), np.linalg.det(Rr[0]))
        sc = float(np.sum((b - mb) * ((a - ma) @ Q.T)) / np.sum((a - ma) ** 2))
        tvec = mb - sc * ma @ Q.T
        res = np.linalg.norm(src * sc @ Q.T + tvec - dst, axis=1)
        w = res < np.percentile(res, 75)
    rot_err = np.degrees([np.linalg.norm(Rotation.from_matrix(Rr[i].T @ Q @ Ra[i]).as_rotvec()) for i in range(len(views))])
    pos_err = np.linalg.norm(Cr - (sc * (Ca @ Q.T) + tvec), axis=1)
    geo_err = float(np.median(res))
    log.info("alignment: scale %.3f, point error median %.2f m, camera rotation %.1f deg, centre %.2f m",
             sc, geo_err, np.median(rot_err), np.median(pos_err))

    P = sc * (mu @ Q.T) + tvec
    D = np.diag([1.0, 1.0, -1.0])                                  # road z forward -> three.js -z forward
    C = (sc ** 2) * (D @ Q @ cov @ Q.T @ D)
    lam, vec = np.linalg.eigh(C)
    scales = np.sqrt(np.clip(lam, 1e-10, None))
    vec[np.linalg.det(vec) < 0, :, 0] *= -1
    quat = Rotation.from_matrix(vec).as_quat()[:, [3, 0, 1, 2]]
    rgb = np.clip(0.5 + SH_C0 * dc, 0, 1)

    # level the road: monocular drift tilts the global road frame over a clip (camera height there drifts by metres),
    # while the scale makes the camera 1.35 m above its own local road. Shift each splat by its nearest camera's
    # drift so the local road is y = 0 everywhere, where the vehicles are drawn.
    d2 = np.linalg.norm(P[:, None, [0, 2]] - Cr[None, :, [0, 2]], axis=2)
    near, dist = np.argmin(d2, axis=1), np.min(d2, axis=1)
    PL = P.copy()
    PL[:, 1] -= Cr[near, 1] - CAM_H

    # keep the world near the path: no sky remnants, no giant blobs
    keep_g = valid & (op > 0.02) & (scales.max(1) < 4) & (dist < 70) & (PL[:, 1] > -1.5) & (PL[:, 1] < 25)
    for i in range(len(views)):                                              # nothing in the box just ahead of the hood
        pc = (P - Cr[i]) @ Rr[i]                                              # road -> camera (OpenCV axes)
        keep_g &= ~((pc[:, 2] > 0) & (pc[:, 2] < 3.2) & (np.abs(pc[:, 0]) < 1.3) & (pc[:, 1] > 0.15) & (PL[:, 1] > 0.25))
    idx = np.where(keep_g)[0]
    if len(idx) > max_splats:
        idx = idx[np.argsort(-op[idx])[:max_splats]]
    return {"pos": PL[idx], "scales": scales[idx], "quat": quat[idx], "rgb": rgb[idx], "op": op[idx],
            "seconds": round(time.time() - t0, 1),
            "align": {"scale": round(sc, 4), "rot_err_deg": round(float(np.median(rot_err)), 2),
                      "centre_err_m": round(float(np.median(pos_err)), 3), "point_err_m": round(geo_err, 3)}}


def save(path: pathlib.Path, r: dict):
    """Write reconstruct() output as a .splat file in three.js axes (z negated)."""
    write_splat(path, (r["pos"] * [1, 1, -1]).astype(np.float32), r["scales"].astype(np.float32), r["quat"], r["rgb"], r["op"])


def build(case: pathlib.Path, force: bool = False) -> pathlib.Path:
    """Static world around the event as scene.splat (+ scene_splat.json)."""
    out = case / "scene.splat"
    if out.exists() and not force:
        return out
    meta = json.loads((case / "meta.json").read_text())
    times = np.load(case / "recon.npz")["times"].astype(float)
    views = select_views(times, float(meta.get("time_of_event") or times[len(times) // 2]))
    dev = device()
    patch_autocast(dev)
    r = reconstruct(case, views, load_model(dev), dev)
    save(out, r)
    lo, hi = np.percentile(r["pos"], 2, axis=0), np.percentile(r["pos"], 98, axis=0)
    info = {"file": "scene.splat", "count": int(len(r["pos"])), "method": "AnySplat feed-forward 3DGS",
            "views": [int(k) for k in views], "t_range": [round(float(times[views[0]]), 2), round(float(times[views[-1]]), 2)],
            "bounds": [lo.round(2).tolist(), hi.round(2).tolist()], "align": r["align"],
            "seconds": r["seconds"], "device": dev.type, "level": "local road at y = 0 (camera 1.35 m above it)"}
    (case / "scene_splat.json").write_text(json.dumps(info, indent=1))
    log.info("scene.splat: %d splats (%.0f MB)", len(r["pos"]), out.stat().st_size / 1e6)
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

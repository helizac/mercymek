"""Realistic 3D/4D scene from the reconstruction.

1. scene.glb: TSDF fusion of the VGGT depth maps into one coloured surface mesh of the static world (road, kerbs,
   buildings, trees, poles, signs). Moving objects, the car's own hood, sky and low-confidence pixels are excluded.
2. frames4d/: for every reconstruction frame, its metric depth map (PNG, centimetres in R*256+G, B=255 on moving
   objects) and the full-resolution
   video frame (JPEG), plus the camera->road matrix. The HUD turns these into a photo-textured 3D surface on the GPU,
   so at time t you see the real scene exactly as the camera saw it, in 3D (the fourth dimension).

Coordinates: road frame (x right, y up, z forward, metres), the same frame as tracks.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mercy.config import log  # noqa: E402

TEX_W = 1024


def load_dynmask(case: pathlib.Path):
    """Union of moving-object masks per recon frame, written by lift.py (dynmask.npz)."""
    z = np.load(case / "dynmask.npz")
    T, H, W = z["shape"]
    return np.unpackbits(z["mask"], axis=-1)[..., :W].astype(bool)


def load_sky(case: pathlib.Path, shape):
    p = case / "skymask.npz"
    if not p.exists():
        return np.zeros(shape, bool)
    z = np.load(p)
    return np.unpackbits(z["mask"], axis=-1)[..., :shape[2]].astype(bool)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def sky_mask(height: np.ndarray, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Sky and far haze: too high above the road, or sky-coloured and well above it, or beyond 60 m."""
    H = height.shape[0]
    top = height[: max(2, H // 12)]
    cand = rgb[: max(2, H // 12)][top > 5]
    m = (height > 16) | (height < -3) | (depth > 60)
    if len(cand) > 50:
        sky = np.median(cand.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(rgb.astype(np.float32) - sky, axis=-1)
        m |= (dist < 34) & (height > 3)
    return m


def prepare(case: pathlib.Path):
    rec = np.load(case / "recon.npz")
    tr = json.loads((case / "tracks.json").read_text())
    s = float(tr["scale"]["value"])
    road = tr["road"]
    Rr = np.array([road["right"], road["normal_world"], road["fwd"]], np.float64)
    origin_u = np.array(road["origin"], np.float64)
    up = np.array(road["normal_world"], np.float64)
    pts = rec["points"].astype(np.float32)
    height = ((pts - origin_u) @ up) * s                              # metres above the road plane
    return rec, tr, s, Rr, origin_u, height


def build(case: pathlib.Path, voxel: float = 0.06, max_depth: float = 40.0,
          target_tris: int = 1_200_000, force: bool = False) -> pathlib.Path:
    out = case / "scene.glb"
    if out.exists() and not force:
        return out
    import open3d as o3d
    import trimesh

    rec, tr, s, Rr, origin_u, height = prepare(case)
    depth = rec["depth"].astype(np.float32) * s
    conf = rec["depth_conf"].astype(np.float32)
    cols = rec["colors"]
    extr = rec["extrinsic"].astype(np.float64)
    K = rec["intrinsic"].astype(np.float64)
    T, H, W = depth.shape
    dyn = load_dynmask(case)
    sky = load_sky(case, depth.shape)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 6, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for k in range(T):
        d = depth[k].copy()
        bad = dyn[k] | sky[k] | (conf[k] < np.percentile(conf[k], 15)) | (d < 2.2) | (d > max_depth) | \
            sky_mask(height[k], cols[k], d)
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        d[bad] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(cols[k])), o3d.geometry.Image(np.ascontiguousarray(d)),
            depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[k, 0, 0], K[k, 1, 1], K[k, 0, 2], K[k, 1, 2])
        E = np.eye(4)
        E[:3, :3] = extr[k, :, :3]
        E[:3, 3] = extr[k, :, 3] * s                                 # world->camera, translation in metres
        vol.integrate(rgbd, intr, E)
    mesh = vol.extract_triangle_mesh()
    log.info("tsdf: %d vertices, %d triangles", len(mesh.vertices), len(mesh.triangles))
    labels, counts, _ = mesh.cluster_connected_triangles()             # drop floating fragments
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < 300)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > target_tris:
        mesh = mesh.simplify_quadric_decimation(target_tris)

    V = (np.asarray(mesh.vertices) - origin_u * s) @ Rr.T              # world metres -> road frame
    F = np.asarray(mesh.triangles)
    if np.linalg.det(Rr) < 0:
        F = F[:, ::-1]
    lin = (srgb_to_linear(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255) * 255).round().astype(np.uint8)
    tm = trimesh.Trimesh(vertices=V.astype(np.float32), faces=F,
                         vertex_colors=np.c_[lin, np.full(len(lin), 255, np.uint8)], process=False)
    tm.export(out)
    lo, hi = V.min(0), V.max(0)
    (case / "scene_mesh.json").write_text(json.dumps({"vertices": len(V), "triangles": len(F), "voxel_m": voxel,
                                                      "bounds": [lo.round(2).tolist(), hi.round(2).tolist()]}))
    log.info("scene.glb: %d vertices, %d triangles (%.1f MB)", len(V), len(F), out.stat().st_size / 1e6)
    return out


def frames4d(case: pathlib.Path, force: bool = False) -> pathlib.Path:
    out = case / "frames4d.json"
    if out.exists() and not force:
        return out
    rec, tr, s, Rr, origin_u, height = prepare(case)
    depth = rec["depth"].astype(np.float32) * s
    conf = rec["depth_conf"].astype(np.float32)
    cols = rec["colors"]
    extr = rec["extrinsic"].astype(np.float64)
    K = rec["intrinsic"].astype(np.float64)
    times = rec["times"].astype(float)
    T, H, W = depth.shape
    sky = load_sky(case, depth.shape)
    dyn = load_dynmask(case) if (case / "dynmask.npz").exists() else np.zeros(depth.shape, bool)
    fdir = case / "frames4d"
    fdir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(case / "clip.mp4"))
    frames = []
    for k in range(T):
        d = depth[k].copy()
        bad = sky[k] | (conf[k] < np.percentile(conf[k], 8)) | (d < 1.6) | sky_mask(height[k], cols[k], d)
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        d[bad] = 0
        cm = np.clip(np.round(d * 100), 0, 65535).astype(np.uint16)
        png = np.zeros((H, W, 3), np.uint8)
        png[..., 2] = cm >> 8                                         # OpenCV writes BGR: R channel = high byte
        png[..., 1] = cm & 255
        png[..., 0] = np.where(cv2.dilate(dyn[k].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, 255, 0)  # B: moving object
        cv2.imwrite(str(fdir / f"d_{k:03d}.png"), png)
        cap.set(cv2.CAP_PROP_POS_MSEC, times[k] * 1000)
        ok, img = cap.read()
        if not ok:
            img = cv2.cvtColor(cols[k], cv2.COLOR_RGB2BGR)
        th = int(round(TEX_W * img.shape[0] / img.shape[1] / 2) * 2)
        cv2.imwrite(str(fdir / f"c_{k:03d}.jpg"), cv2.resize(img, (TEX_W, th), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        R, t = extr[k, :, :3], extr[k, :, 3]
        M = np.eye(4)
        M[:3, :3] = Rr @ R.T                                          # camera (metres, OpenCV axes) -> road frame
        M[:3, 3] = -s * Rr @ (R.T @ t + origin_u)
        frames.append({"k": k, "t": round(float(times[k]), 3), "depth": f"frames4d/d_{k:03d}.png",
                       "color": f"frames4d/c_{k:03d}.jpg", "M": M.round(6).tolist(),
                       "K": [K[k, 0, 0], K[k, 1, 1], K[k, 0, 2], K[k, 1, 2]]})
    cap.release()
    out.write_text(json.dumps({"grid": [W, H], "frames": frames}))
    log.info("frames4d: %d frames (%dx%d depth, %d px textures)", T, W, H, TEX_W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--cases-dir", default=str(ROOT / "data" / "cases"))
    ap.add_argument("--voxel", type=float, default=0.06)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--frames-only", action="store_true")
    a = ap.parse_args()
    case = pathlib.Path(a.cases_dir) / a.case
    if not a.frames_only:
        build(case, voxel=a.voxel, force=a.force)
    frames4d(case, force=a.force)


if __name__ == "__main__":
    main()

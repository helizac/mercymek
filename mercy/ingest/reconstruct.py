"""VGGT (Plan A) -> recon.npz (section 8, step 2). Runs on CUDA, Intel XPU, or CPU.

Run inside the reconstruction environment (torch + vggt):
    .venv-recon/Scripts/python -m mercy.ingest.reconstruct --case hero
Plan C fallback (no VGGT): `--method mono` builds a fixed-camera 2.5D cloud from Depth Anything V2.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def pick_device():
    import torch
    if torch.cuda.is_available():
        dev = "cuda"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev, dtype = "xpu", torch.bfloat16
    else:
        dev, dtype = "cpu", torch.bfloat16
    return dev, dtype


def run_vggt(case: pathlib.Path, max_frames: int = 96, model_id: str = "facebook/VGGT-1B") -> dict:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    meta = json.loads((case / "meta.json").read_text())
    paths = sorted(glob.glob(str(case / "frames" / "*.jpg")))
    idx = list(range(len(paths)))
    stride = 1
    while len(idx[::stride]) > max_frames:
        stride += 1
    device, dtype = pick_device()
    print(f"[vggt] device={device} dtype={dtype} frames={len(paths)} stride={stride}", flush=True)
    loading_at = time.monotonic()
    model = VGGT.from_pretrained(model_id).to(device).eval()
    print(f"[vggt] weights loaded in {time.monotonic()-loading_at:.1f}s", flush=True)

    while True:
        sel = idx[::stride]
        sel_paths = [paths[i] for i in sel]
        try:
            t0 = time.time()
            images = load_and_preprocess_images(sel_paths).to(device)        # [T,3,H,W]
            with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=device != "cpu"):
                pred = model(images)
            break
        except (RuntimeError, MemoryError) as e:
            if "memory" not in str(e).lower() and not isinstance(e, MemoryError) or stride > 6:
                raise
            stride += 1
            print(f"[vggt] out of memory, retrying with stride {stride}", flush=True)
            if device == "xpu":
                torch.xpu.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
    print(f"[vggt] forward {time.time() - t0:.1f}s on {len(sel)} frames, image {tuple(images.shape[-2:])}", flush=True)

    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"].squeeze(0)                                        # [T,H,W,1]
    extr_np = extr.squeeze(0).float().cpu().numpy()
    intr_np = intr.squeeze(0).float().cpu().numpy()
    depth_np = depth.float().cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth_np, extr_np, intr_np)      # [T,H,W,3] world (camera 0)
    times = np.array(sel, dtype=np.float32) / float(meta["recon_fps"])
    out = dict(
        depth=depth_np[..., 0].astype(np.float16),
        depth_conf=pred["depth_conf"].squeeze(0).float().cpu().numpy().astype(np.float16),
        extrinsic=extr_np, intrinsic=intr_np,
        points=pts.astype(np.float16),
        colors=(images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8),
        times=times, frame_index=np.array(sel, dtype=np.int32),
        method=np.array("vggt"))
    return out


def run_mono(case: pathlib.Path, max_frames: int = 96) -> dict:
    """Step 2C fallback: per-frame relative depth (Depth Anything V2 small), fixed camera, fixed intrinsics."""
    import torch
    from PIL import Image
    from transformers import pipeline

    meta = json.loads((case / "meta.json").read_text())
    paths = sorted(glob.glob(str(case / "frames" / "*.jpg")))[:max_frames]
    device, _ = pick_device()
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    deps, cols = [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        d = np.asarray(pipe(im)["predicted_depth"].squeeze(), dtype=np.float32)
        d = np.asarray(Image.fromarray(d).resize(im.size))
        deps.append(10.0 / np.clip(d / d.max() * 10, 0.3, None))           # inverse relative depth -> pseudo-depth
        cols.append(np.asarray(im))
    D = np.stack(deps); T, H, W = D.shape
    f = 0.9 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.stack([(xs - W / 2) / f, (ys - H / 2) / f, np.ones_like(xs, dtype=np.float32)], -1)
    pts = rays[None] * D[..., None]
    E = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))])[None], (T, 1, 1)).astype(np.float32)
    return dict(depth=D.astype(np.float16), depth_conf=np.ones_like(D, np.float16) * 2,
                extrinsic=E, intrinsic=np.tile(K[None], (T, 1, 1)), points=pts.astype(np.float16),
                colors=np.stack(cols).astype(np.uint8),
                times=np.arange(T, dtype=np.float32) / float(meta["recon_fps"]),
                frame_index=np.arange(T, dtype=np.int32), method=np.array("mono"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--cases-dir", default=str(ROOT / "data" / "cases"))
    ap.add_argument("--method", choices=["vggt", "mono"], default="vggt")
    ap.add_argument("--max-frames", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    case = pathlib.Path(a.cases_dir) / a.case
    out = case / "recon.npz"
    if out.exists() and not a.force:
        print("[recon] exists, skip"); return
    t0 = time.time()
    res = run_vggt(case, a.max_frames) if a.method == "vggt" else run_mono(case, a.max_frames)
    np.savez_compressed(out, **res)
    print(f"[recon] {a.method} -> {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

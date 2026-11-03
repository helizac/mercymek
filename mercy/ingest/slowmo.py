"""Super slow motion of the impact: RAFT optical-flow frame interpolation (torchvision, pretrained).

For each pair of source frames around the impact, RAFT-large estimates forward and backward flow; in-between frames
at t = 1/4, 1/2, 3/4 are rendered by warping both neighbours to time t (linear flow approximation) and blending them
with occlusion weights from forward-backward consistency. The result plays at the source frame rate, four times
slower, with real motion between the frames instead of repeated frames.

Runs in .venv-recon (torch XPU/CUDA/CPU): python -m mercy.ingest.slowmo --case X [--force]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mercy.config import log  # noqa: E402

FACTOR = 4
WINDOW = (-1.5, 1.0)
OUT_W = 960


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def read_frames(clip: pathlib.Path, t0: float, t1: float):
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0) * 1000)
    frames, times = [], []
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, f = cap.read()
        if not ok or t > t1:
            break
        h = int(round(OUT_W * f.shape[0] / f.shape[1] / 8) * 8)
        frames.append(cv2.resize(f, (OUT_W, h), interpolation=cv2.INTER_AREA))
        times.append(t)
    cap.release()
    return frames, times, fps


def warp(img, flow):
    """Backward warp: sample img at x + flow(x)."""
    import torch
    import torch.nn.functional as F
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
    gx = (xx[None] + flow[:, 0]) / (W - 1) * 2 - 1
    gy = (yy[None] + flow[:, 1]) / (H - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


def build(case: pathlib.Path, force: bool = False) -> pathlib.Path | None:
    out = case / "slowmo.mp4"
    if out.exists() and not force:
        return out
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    meta = json.loads((case / "meta.json").read_text())
    t_imp = float(json.loads((case / "physics.json").read_text(encoding="utf-8"))["impact"]["t"])
    t0, t1 = max(0.0, t_imp + WINDOW[0]), min(float(meta["duration"]), t_imp + WINDOW[1])
    frames, times, fps = read_frames(case / "clip.mp4", t0, t1)
    if len(frames) < 3:
        log.warning("slowmo: too few frames")
        return None
    dev = device()
    t_start = time.time()
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(dev).eval()
    tf = weights.transforms()

    def tens(f):
        return torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(dev)

    ff = shutil.which("ffmpeg")
    H, W = frames[0].shape[:2]
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    with torch.no_grad():
        prev = tens(frames[0])
        for k in range(1, len(frames)):
            cur = tens(frames[k])
            a, b = tf(prev, cur)
            f01 = model(a, b)[-1]
            f10 = model(b, a)[-1]
            # occlusion: forward-backward consistency (pixels whose round trip does not return are occluded)
            occ0 = torch.exp(-((f01 + warp(f10, f01)) ** 2).sum(1, keepdim=True) / 4.0)
            occ1 = torch.exp(-((f10 + warp(f01, f10)) ** 2).sum(1, keepdim=True) / 4.0)
            proc.stdin.write(frames[k - 1].tobytes())
            for i in range(1, FACTOR):
                t = i / FACTOR
                ft0 = -(1 - t) * t * f01 + t * t * f10
                ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
                w0 = (1 - t) * warp(occ0, ft0) + 1e-4
                w1 = t * warp(occ1, ft1) + 1e-4
                mid = (w0 * warp(prev, ft0) + w1 * warp(cur, ft1)) / (w0 + w1)
                img = (mid[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                proc.stdin.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR).tobytes())
            prev = cur
        proc.stdin.write(frames[-1].tobytes())
    proc.stdin.close()
    proc.wait()
    info = {"file": "slowmo.mp4", "t0": round(times[0], 3), "t1": round(times[-1], 3), "factor": FACTOR,
            "fps": round(fps, 3), "frames_in": len(frames), "frames_out": (len(frames) - 1) * FACTOR + 1,
            "method": "RAFT-large optical flow, bidirectional warping with occlusion weights",
            "seconds": round(time.time() - t_start, 1), "device": dev.type}
    (case / "slowmo.json").write_text(json.dumps(info, indent=1))
    log.info("slowmo: %d -> %d frames on %s in %.0fs", info["frames_in"], info["frames_out"], dev.type, info["seconds"])
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

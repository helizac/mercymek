"""Package the 3D moments for FastGS training on Kaggle (CUDA; this PC only runs inference).

For every moment of every case (moments.json): the frames around it as RGBA PNGs whose alpha is the loss mask
(the static world in every frame; the vehicles only in the moment's own frame, so they train frozen at that instant;
never the sky or the camera car's hood), a COLMAP text model with the VGGT cameras in the metric road frame (y flipped
down, see FLIP), and initial points from the metric VGGT depth. FastGS output comes back through
scripts/fastgs_import.py.

    python scripts/fastgs_export.py                    # all cases -> data/fastgs/mercy_fastgs.zip
    python scripts/fastgs_export.py --case hero --width 960
    python scripts/fastgs_export.py --clip             # whole clip per case, every vehicle masked -> mercy_fastgs_clip.zip
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import zipfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config  # noqa: E402
from mercy.config import log  # noqa: E402
from mercy.ingest import moments as MO  # noqa: E402
from mercy.ingest import splat as SP  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

OUT = config.ROOT / "data" / "fastgs"
POINTS = 150_000
# The road frame (x right, y up, z forward) is mirrored against OpenCV camera axes; COLMAP needs proper rotations, so
# training happens in the road frame with y pointing down. fastgs_import.py flips it back.
FLIP = np.diag([1.0, -1.0, 1.0, 1.0])


def export_moment(case: pathlib.Path, m: dict, width: int, dst: pathlib.Path, points: int = POINTS) -> dict:
    rec = np.load(case / "recon.npz")
    meta = json.loads((case / "meta.json").read_text())
    f4 = json.loads((case / "frames4d.json").read_text())
    s = float(json.loads((case / "tracks.json").read_text())["scale"]["value"])
    times = rec["times"].astype(float)
    T, h, w = rec["depth"].shape
    if m["id"] == "clip":                      # static world of the whole clip: no frame keeps its vehicles
        views, j = np.arange(T), -1
    else:
        views, j = MO.window(times, m["t"])
    dyn = SP.load_dynmask(case) if (case / "dynmask.npz").exists() else np.zeros((T, h, w), bool)
    sky = SP.load_sky(case, (T, h, w))
    glued = SP.glued_mask(rec["colors"][views])
    ker = np.ones((7, 7), np.uint8)

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(case / "clip.mp4"))
    cams, imgs, pts, cols = [], [], [], []
    rng = np.random.default_rng(0)
    for i, k in enumerate(views):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(times[k] * meta["src_fps"])))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{case.name}: cannot read frame {k}")
        W = min(width, frame.shape[1])
        H = int(round(W * frame.shape[0] / frame.shape[1]))
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        lo_drop = sky[k] | glued if i == j else (cv2.dilate((dyn[k] | sky[k]).astype(np.uint8), ker) > 0) | glued
        alpha = cv2.resize((~lo_drop).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        name = f"{i:03d}_k{k:03d}.png"
        cv2.imwrite(str(dst / "images" / name), np.dstack([img, alpha]))
        K = rec["intrinsic"][k].astype(np.float64) * np.array([[W / w], [H / h], [1]])
        M = FLIP @ np.array(f4["frames"][k]["M"], np.float64)     # camera (OpenCV axes) -> training world, metres
        Rw = M[:3, :3].T                                            # world -> camera
        tw = -Rw @ M[:3, 3]
        q = Rotation.from_matrix(Rw).as_quat()                      # x y z w
        cams.append(f"{i + 1} PINHOLE {W} {H} {K[0, 0]:.4f} {K[1, 1]:.4f} {K[0, 2]:.4f} {K[1, 2]:.4f}")
        imgs.append(f"{i + 1} {q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f} {i + 1} {name}\n")
        # initial points: metric VGGT depth of the pixels this frame trains on
        d = rec["depth"][k].astype(np.float64) * s
        v, u = np.nonzero(~lo_drop & (d > 1) & (d < 80) & (rec["depth_conf"][k] > np.percentile(rec["depth_conf"][k], 30)))
        take = rng.choice(len(u), min(len(u), points // len(views) * (4 if i == j else 1)), replace=False)
        u, v = u[take], v[take]
        Kl = rec["intrinsic"][k].astype(np.float64)
        dd = d[v, u]
        pts.append(np.stack([(u - Kl[0, 2]) / Kl[0, 0] * dd, (v - Kl[1, 2]) / Kl[1, 1] * dd, dd], 1) @ M[:3, :3].T + M[:3, 3])
        cols.append(rec["colors"][k][v, u])
    cap.release()
    P, C = np.concatenate(pts), np.concatenate(cols)
    (dst / "sparse" / "0" / "cameras.txt").write_text("# PINHOLE per frame (VGGT intrinsics)\n" + "\n".join(cams) + "\n")
    (dst / "sparse" / "0" / "images.txt").write_text("# road frame with y down (x right, y down, z forward, metres)\n" + "".join(imgs))
    (dst / "sparse" / "0" / "points3D.txt").write_text(
        "".join(f"{n + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n" for n, (p, c) in enumerate(zip(P, C))))
    info = {"case": case.name, "moment": m["id"], "t": m["t"], "label": m["label"], "views": [int(k) for k in views],
            "moment_view": int(views[j]) if j >= 0 else None, "width": W, "height": H, "points": int(len(P))}
    (dst / "moment.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="*", help="case ids (default: every case with moments)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--clip", action="store_true", help="one scene per case over the whole clip instead of per moment")
    a = ap.parse_args()
    cases = [config.case_dir(c) for c in a.case] if a.case else \
        sorted(p.parent for p in config.CASES_DIR.glob("*/moments.json"))
    stage = OUT / ("mercy_fastgs_clip" if a.clip else "mercy_fastgs")
    shutil.rmtree(stage, ignore_errors=True)
    done = []
    for case in cases:
        ms = [{"id": "clip", "t": 0.0, "label": "whole clip"}] if a.clip else             json.loads((case / "moments.json").read_text())["moments"]
        for m in ms:
            done.append(export_moment(case, m, a.width, stage / case.name / m["id"], POINTS * (2 if a.clip else 1)))
            log.info("fastgs export: %s/%s (%d frames)", case.name, m["id"], len(done[-1]["views"]))
    (stage / "index.json").write_text(json.dumps(done, indent=1))
    zp = OUT / f"{stage.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:          # PNGs are already compressed
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage).as_posix())
    log.info("%d scenes -> %s (%.0f MB)", len(done), zp, zp.stat().st_size / 1e6)


if __name__ == "__main__":
    main()

"""Make one reviewed Muse empty environment per case; no local GPU work.

python scripts/prepare_environments.py --case hero cutin_rearend
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from mercy.generation.cleanplate import verified_clean_plate


def prepare(case: pathlib.Path) -> dict:
    meta = json.loads((case / "meta.json").read_text())
    scene = json.loads((case / "scene.json").read_text())
    rec = np.load(case / "recon.npz")
    # A single common environment for all moments. Keep some distance from the collision.
    t = max(0, float(scene.get("impact_time_s") or meta["time_of_event"]) - 1.0)
    k = int(np.argmin(np.abs(rec["times"] - t)))
    dst = case / "environment"
    dst.mkdir(exist_ok=True)
    frame = dst / "source.png"
    cap = cv2.VideoCapture(str(case / "clip.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(rec["times"][k]) * meta["src_fps"])))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {case.name} anchor frame")
    cv2.imwrite(str(frame), image)
    clean = verified_clean_plate(frame, dst / "empty.png")
    clean_image = cv2.imread(str(clean))
    K = rec["intrinsic"][k].astype(float)
    info = {"case": case.name, "anchor_k": k, "anchor_t": float(rec["times"][k]),
            "image": "empty.png", "focal_px": float(K[0, 0] * clean_image.shape[1] / rec["depth"].shape[2]),
            "camera_height_m": 1.35, "generated": True, "review": "empty.review.json"}
    (dst / "input.json").write_text(json.dumps(info, indent=2))
    print(f"{case.name}: reviewed empty environment -> {clean}", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.case:
        prepare(config.case_dir(name))


if __name__ == "__main__":
    main()

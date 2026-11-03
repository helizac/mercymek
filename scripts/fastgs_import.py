"""Bring FastGS results from Kaggle back into the 3D moments.

The notebook (kaggle/fastgs_moments.ipynb) returns <case>/<moment>/point_cloud.ply (standard 3DGS layout) trained in
the road frame with y down (scripts/fastgs_export.py). Each one is flipped back, levelled like the AnySplat world
(local road at y = 0) and merged with the moment's photo layer; the AnySplat version is kept as <id>.anysplat.splat.

    python scripts/fastgs_import.py fastgs_out.zip        # or a folder
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config  # noqa: E402
from mercy.config import log  # noqa: E402
from mercy.ingest import moments as MO  # noqa: E402
from mercy.ingest import splat as SP  # noqa: E402

FLIP = np.diag([1.0, -1.0, 1.0])                 # training world (y down) <-> road frame (y up)


def read_ply(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Binary little-endian PLY with float vertex properties (what 3DGS/FastGS save)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in head:
        raise ValueError(f"{path}: not a binary little-endian PLY")
    n = next(int(l.split()[2]) for l in head if l.startswith("element vertex"))
    names = [l.split()[2] for l in head if l.startswith("property float")]
    data = np.frombuffer(raw[end:end + 4 * n * len(names)], "<f4").reshape(n, len(names))
    return {k: data[:, i] for i, k in enumerate(names)}


def gaussians(ply: dict[str, np.ndarray]) -> dict:
    """3DGS parameters -> road frame (y up) splats: positions, three.js-axis scales + quaternions, colour, opacity."""
    from scipy.spatial.transform import Rotation
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64) @ FLIP
    rgb = np.clip(0.5 + SP.SH_C0 * np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    op = 1 / (1 + np.exp(-ply["opacity"].astype(np.float64)))
    s = np.exp(np.stack([ply[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    q = np.stack([ply[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)          # w x y z
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]] / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
    G = np.diag([1.0, -1.0, -1.0])                 # training world -> three.js axes (y up, z back) = (z flip) . FLIP
    C = G @ (R * s[:, None, :] ** 2) @ np.swapaxes(R, 1, 2) @ G
    lam, vec = np.linalg.eigh(C)
    vec[np.linalg.det(vec) < 0, :, 0] *= -1
    quat = Rotation.from_matrix(vec).as_quat()[:, [3, 0, 1, 2]]
    return {"pos": xyz, "scales": np.sqrt(np.clip(lam, 1e-10, None)), "quat": quat, "rgb": rgb, "op": op}


def level(case: pathlib.Path, views: list[int], g: dict) -> dict:
    """Same levelling and clean-up as mercy.ingest.splat.reconstruct: local road at y = 0, the world near the path."""
    f4 = json.loads((case / "frames4d.json").read_text())
    Cr = np.array([np.array(f4["frames"][k]["M"])[:3, 3] for k in views])
    P = g["pos"]
    d2 = np.linalg.norm(P[:, None, [0, 2]] - Cr[None, :, [0, 2]], axis=2)
    near, dist = np.argmin(d2, 1), np.min(d2, 1)
    PL = P.copy()
    PL[:, 1] -= Cr[near, 1] - SP.CAM_H
    keep = (g["op"] > 0.02) & (g["scales"].max(1) < 4) & (dist < 70) & (PL[:, 1] > -1.5) & (PL[:, 1] < 25)
    idx = np.where(keep)[0]
    if len(idx) > 900_000:
        idx = idx[np.argsort(-g["op"][idx])[:900_000]]
    return {"pos": PL[idx], "scales": g["scales"][idx], "quat": g["quat"][idx], "rgb": g["rgb"][idx], "op": g["op"][idx]}


def import_dir(root: pathlib.Path) -> set[str]:
    done = set()
    for ply in sorted(root.rglob("point_cloud.ply")):
        info_p = next((p / "moment.json" for p in ply.parents if (p / "moment.json").exists()), None)
        if info_p is None:
            log.warning("%s: no moment.json next to it, skipped", ply)
            continue
        info = json.loads(info_p.read_text())
        case = config.case_dir(info["case"])
        mj = json.loads((case / "moments.json").read_text())
        m = next((x for x in mj["moments"] if x["id"] == info["moment"]), None)
        if m is None:
            log.warning("%s/%s: no such moment any more, skipped", info["case"], info["moment"])
            continue
        world = level(case, info["views"], gaussians(read_ply(ply)))
        rec = np.load(case / "recon.npz")
        # same photo layer, clean plate (cached) and vehicle layers as the AnySplat build; only the world differs
        r, photo, _ = MO.compose(case, m["id"], info["moment_view"], SP.glued_mask(rec["colors"][info["views"]]), world)
        out = case / m["file"]
        backup = out.with_name(out.stem + ".anysplat.splat")
        if out.exists() and not backup.exists():
            out.replace(backup)
        SP.save(out, MO.local(r, m.get("origin", [0, 0, 0])))
        log_p = info_p.parent / "train_log.json"
        m.update({"method": "FastGS", "count": int(len(r["pos"])), "fastgs_splats": int(len(world["pos"])),
                  "fastgs": json.loads(log_p.read_text()) if log_p.exists() else {}})
        (case / "moments.json").write_text(json.dumps(mj, indent=1))
        log.info("%s/%s: FastGS world %d splats + photo %d", info["case"], info["moment"], len(world["pos"]), len(photo["pos"]))
        done.add(info["case"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path, help="the notebook's output zip, or a folder")
    a = ap.parse_args()
    if a.src.is_dir():
        cases = import_dir(a.src)
    else:
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(a.src) as z:
            z.extractall(tmp)
            cases = import_dir(pathlib.Path(tmp))
    from mercy.analysis import seal
    for c in sorted(cases):                         # the moments changed: seal the case again
        seal.seal(config.case_dir(c))
    log.info("FastGS moments imported for %s", ", ".join(sorted(cases)) or "no case")


if __name__ == "__main__":
    main()

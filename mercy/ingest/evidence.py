"""Privacy layer + evidence stills (section 7, privacy layer).

clip_redacted.mp4   license plates and faces blurred with their SAM masks (the HUD plays this file)
evidence/*.jpg      object crops (from the redacted video), mask silhouettes, key stills around impact
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import cv2
import numpy as np

from mercy.config import log
from mercy.ingest import segment

PRIVACY_CONCEPTS = ["face"]          # plates stay readable: identifying the involved vehicles is part of the case


def redact(case: pathlib.Path, force: bool = False) -> pathlib.Path:
    out = case / "clip_redacted.mp4"
    if out.exists() and not force:
        return out
    meta = json.loads((case / "meta.json").read_text())
    W, H, fps = meta["width"], meta["height"], meta["src_fps"]
    by_frame: dict[int, list] = {}
    for c in PRIVACY_CONCEPTS:
        for key, frames in segment.records_by_object(case, c).items():
            for f, r in frames.items():
                by_frame.setdefault(f, []).append(r)
    cap = cv2.VideoCapture(str(case / "clip.mp4"))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    f = n_blur = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for r in by_frame.get(f, []):
            x1, y1 = int(r.bounds.left), int(r.bounds.top)
            crop = segment.mask_crop(r)
            h, w = crop.shape
            pad = 6
            X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
            X2, Y2 = min(W, x1 + w + pad), min(H, y1 + h + pad)
            m = np.zeros((Y2 - Y1, X2 - X1), np.uint8)
            hh, ww = min(h, H - y1), min(w, W - x1)
            m[y1 - Y1:y1 - Y1 + hh, x1 - X1:x1 - X1 + ww] = crop[:hh, :ww]
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            roi = img[Y1:Y2, X1:X2]
            k = max(9, (max(w, h) // 3) | 1)
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            roi[m > 0] = blurred[m > 0]
            n_blur += 1
        ff.stdin.write(img.tobytes())
        f += 1
    cap.release()
    ff.stdin.close(); ff.wait()
    log.info("redacted %d regions over %d frames -> %s", n_blur, f, out.name)
    return out


def _frame(clip: pathlib.Path, f: int):
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def stills(case: pathlib.Path, force: bool = False) -> dict:
    ev = case / "evidence"
    idx_path = ev / "index.json"
    if idx_path.exists() and not force:
        return json.loads(idx_path.read_text())
    ev.mkdir(exist_ok=True)
    meta = json.loads((case / "meta.json").read_text())
    ph = json.loads((case / "physics.json").read_text())
    overlays = json.loads((case / "overlays.json").read_text())
    clip = case / "clip_redacted.mp4"
    fps, imp = meta["src_fps"], ph["impact"]["t"]
    items = []
    for name, dt in [("pre_impact", -1.5), ("impact", 0.0), ("post_impact", 1.0)]:
        img = _frame(clip, int(round((imp + dt) * fps)))
        if img is not None:
            cv2.imwrite(str(ev / f"{name}.jpg"), cv2.resize(img, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            items.append({"file": f"evidence/{name}.jpg", "kind": "still", "t": round(imp + dt, 2),
                          "label": name.replace("_", " ")})
    for o in ph["objects"]:
        key = o.get("sam_key")
        if not key:
            continue
        # best view: largest mask area before impact (fallback: any frame)
        cands = [(int(f), d[key]) for f, d in overlays.items() if key in d]
        pre = [c for c in cands if c[0] / fps <= imp] or cands
        if not pre:
            continue
        f, o2 = max(pre, key=lambda c: c[1].get("area", 0))
        img = _frame(clip, f)
        if img is None:
            continue
        x1, y1, x2, y2 = map(int, o2["box"])
        pw, ph_ = int((x2 - x1) * 0.25) + 8, int((y2 - y1) * 0.25) + 8
        X1, Y1, X2, Y2 = max(0, x1 - pw), max(0, y1 - ph_), min(img.shape[1], x2 + pw), min(img.shape[0], y2 + ph_)
        crop = img[Y1:Y2, X1:X2]
        s = 360 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        safe = key.replace(":", "_").replace(" ", "_")
        cv2.imwrite(str(ev / f"{safe}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # silhouette: white mask on transparent, same crop
        sil = np.zeros((Y2 - Y1, X2 - X1, 4), np.uint8)
        for poly in o2["poly"]:
            pts = (np.array(poly).reshape(-1, 2) - [X1, Y1]).astype(np.int32)
            cv2.fillPoly(sil, [pts], (235, 240, 245, 255))
        sil = cv2.resize(sil, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(ev / f"{safe}_mask.png"), sil)
        items.append({"file": f"evidence/{safe}.jpg", "mask": f"evidence/{safe}_mask.png", "kind": "object",
                      "object": o["id"], "t": round(f / fps, 2), "label": o["label"]})
    idx = {"items": items}
    idx_path.write_text(json.dumps(idx, indent=1))
    log.info("evidence: %d stills/crops", len(items))
    return idx

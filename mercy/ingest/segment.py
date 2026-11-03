"""SAM 3.1 video segmentation -> masks/<concept>.lines + overlays.json (section 7).

Payloads are decoded only through meta-sam-parser; we never parse mask payloads ourselves.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import meta_sam_parser as msp
import numpy as np

from mercy import config
from mercy.config import log
from mercy.ingest.scene import _file_id
from mercy.meta_client import meta_client

DEFAULT_CONCEPTS = ["car", "truck", "pedestrian", "motorcycle", "shattered glass", "debris",
                    "traffic light", "license plate"]


def safe(concept: str) -> str:
    return concept.replace(" ", "_")


# ---------- API ----------
def _stream_lines(content: list) -> str:
    chunks = []
    with meta_client().responses.stream(
            model=config.SAM_MODEL,
            input=[{"type": "message", "role": "user", "content": content}],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"}) as stream:
        for ev in stream:
            if ev.type == "response.output_text.delta":
                chunks.append(ev.delta)
    return "".join(chunks)


def segment_concept(file_id: str, concept: str, case: pathlib.Path, force: bool = False) -> str:
    path = case / "masks" / f"{safe(concept)}.lines"
    if path.exists() and not force:
        return path.read_text()
    raw = _stream_lines([{"type": "input_text", "text": concept}, {"type": "input_video", "file_id": file_id}])
    path.parent.mkdir(exist_ok=True)
    path.write_text(raw)
    n = sum(1 for ln in raw.splitlines() if ln.strip())
    log.info("SAM %-16s %4d frame lines", concept, n)
    return raw


def segment_image(img_bgr: np.ndarray, concept: str) -> str:
    """Single-frame segmentation for live 'isolate <new concept>' (section 7, live wow moment)."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return _stream_lines([{"type": "input_text", "text": concept}, {"type": "input_image", "image_url": url}])


def segment_all(case: pathlib.Path, concepts: list[str], force: bool = False) -> dict[str, str]:
    file_id = _file_id(case)
    with ThreadPoolExecutor(max_workers=4) as ex:   # cap parallelism at 4 (rate limits)
        raws = list(ex.map(lambda c: segment_concept(file_id, c, case, force), concepts))
    return dict(zip(concepts, raws))


# ---------- parsing ----------
def parse(raw: str, media: str = "video"):
    fmt = msp.video_segmentation_format() if media == "video" else msp.image_segmentation_format()
    p = fmt.create_parser()
    p.push(raw if raw.endswith("\n") else raw + "\n", emit=False)
    res = p.finish(msp.CompletedOutcome()).result
    for d in res.diagnostics:
        log.debug("sam diag %s: %s", d.code, d.message)
    return [r for r in res.records if r.kind == "mask"]


def mask_crop(rec) -> np.ndarray:
    """Box-sized uint8 0/1 raster for one mask record. Very large masks (sky, road) can arrive at reduced resolution;
    they are scaled to their box."""
    m = rec.mask
    crop = np.frombuffer(msp.decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    b = rec.bounds
    bh, bw = round(b.bottom - b.top), round(b.right - b.left)
    if (m.height, m.width) != (bh, bw):
        crop = cv2.resize(crop, (max(1, bw), max(1, bh)), interpolation=cv2.INTER_NEAREST)
    return crop


def full_frame_mask(rec, W: int, H: int) -> np.ndarray:
    crop = mask_crop(rec)
    x1, y1 = int(rec.bounds.left), int(rec.bounds.top)
    full = np.zeros((H, W), np.uint8)
    h, w = crop.shape
    x2, y2 = min(W, x1 + w), min(H, y1 + h)
    full[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]
    return full


def polygons(crop: np.ndarray, offset: tuple[float, float], eps: float = 1.6, min_area: float = 12) -> list[list[float]]:
    """Simplified outer contours in source pixels: [[x0,y0,x1,y1,...], ...] (compact for the HUD)."""
    cs, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        c[:, 0] += offset[0]; c[:, 1] += offset[1]
        out.append([round(v, 1) for v in c.reshape(-1).tolist()])
    return out


def records_by_object(case: pathlib.Path, concept: str) -> dict[str, dict[int, object]]:
    """{object_key: {sam_frame: mask_record}} for one concept."""
    path = case / "masks" / f"{safe(concept)}.lines"
    if not path.exists():
        return {}
    out: dict[str, dict[int, object]] = {}
    for r in parse(path.read_text()):
        f = r.frame.frame_index if r.frame else 0
        out.setdefault(f"{concept}:{r.object_id}", {})[f] = r
    return out


def _overlay_chunk(jobs: list) -> list:
    """Worker: decode a batch of mask records into HUD overlay entries."""
    out = []
    for concept, key, f, r in jobs:
        crop = mask_crop(r)
        b = r.bounds
        out.append((f, key, {
            "concept": concept, "object_id": key,
            "box": [b.left, b.top, b.right, b.bottom],
            "confidence": None if r.confidence is None else round(r.confidence, 3),
            "poly": polygons(crop, (b.left, b.top)),
            "area": int(crop.sum()),
            "color": config.CONCEPT_COLORS.get(concept, "#cfd5d8"),
        }))
    return out


def build_overlays(case: pathlib.Path, concepts: list[str], force: bool = False) -> dict:
    out_path = case / "overlays.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    overlays: dict[str, dict] = {}
    stats, jobs = {}, []
    for concept in concepts:
        objs = records_by_object(case, concept)
        stats[concept] = len(objs)
        for key, frames in objs.items():
            for f, r in frames.items():
                jobs.append((concept, key, f, r))
    # the parser decodes (and re-encodes to verify) every mask in pure Python: spread it over the CPU cores
    chunks = [jobs[i:i + 200] for i in range(0, len(jobs), 200)]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_overlay_chunk, chunks):
            for f, key, entry in part:
                overlays.setdefault(str(f), {})[key] = entry
    overlays = dict(sorted(overlays.items(), key=lambda kv: int(kv[0])))
    out_path.write_text(json.dumps(overlays, separators=(",", ":")))
    log.info("overlays: %d frames, objects per concept %s", len(overlays), stats)
    return overlays

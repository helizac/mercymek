"""Zoom & enhance: multi-frame fusion + Real-ESRGAN super-resolution, optional plate reading.

Detail comes first from the video itself: up to 9 neighbouring frames of the same region are aligned (ECC affine,
then RAFT optical flow for the non-rigid residual) and fused with robust per-pixel weights, which removes compression
noise and recovers sub-pixel detail. Real-ESRGAN then upsamples x4.
The reading step is told to mark unreadable characters with '?' instead of guessing.
"""
from __future__ import annotations

import json
import hashlib
import os
import pathlib
import threading
import urllib.request

import cv2
import numpy as np

from mercy import config
from mercy.config import log

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS = config.ROOT / "data" / "models" / "RealESRGAN_x4plus.pth"
_model, _lock = None, threading.Lock()


def _cpu_threads():
    """Small live crops are slower with a large CPU thread pool (notably on Windows)."""
    import torch
    limit = max(1, int(os.environ.get("MERCY_INFERENCE_THREADS", "4")))
    if torch.get_num_threads() > limit:
        torch.set_num_threads(limit)

PLATE_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["legible", "text", "confidence", "region", "notes"],
                "properties": {"legible": {"type": "boolean"}, "text": {"type": "string"},
                               "confidence": {"type": "number"}, "region": {"type": "string"},
                               "notes": {"type": "string"}}}


def esrgan():
    """Real-ESRGAN x4 (spandrel), loaded once; CPU is fine for crops of a few hundred pixels."""
    global _model
    with _lock:
        if _model is None:
            import spandrel
            import torch
            if not WEIGHTS.exists():
                WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
                log.info("downloading Real-ESRGAN weights…")
                urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
            m = spandrel.ModelLoader().load_from_file(str(WEIGHTS))
            m.model.eval()
            torch.set_grad_enabled(False)
            _model = m
        return _model


def super_resolve(img: np.ndarray, max_side: int = 1200) -> np.ndarray:
    import torch
    _cpu_threads()
    m = esrgan()
    h, w = img.shape[:2]
    if max(h, w) * 4 > max_side * 2:                     # keep the network input small (speed)
        s = max_side * 2 / (max(h, w) * 4)
        img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255
    with torch.no_grad():
        y = m(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    out = cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    if max(out.shape[:2]) > max_side:
        s = max_side / max(out.shape[:2])
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return out


def _read_frames(clip: pathlib.Path, f0: int, f1: int) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
    out = {}
    for f in range(max(0, f0), f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        out[f] = img
    cap.release()
    return out


_raft = None


def raft():
    """RAFT-small optical flow (torchvision, pretrained), loaded once; None when torchvision is unavailable."""
    global _raft
    if _raft is None:
        try:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            w = Raft_Small_Weights.DEFAULT
            _raft = (raft_small(weights=w).eval(), w.transforms())
        except Exception as e:                                   # no torchvision / no download: ECC only
            log.warning("RAFT unavailable (%s); fusion uses ECC alignment only", e)
            _raft = False
    return _raft or None


def refine(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Dense residual alignment of img onto ref (both BGR, same size) after the global ECC warp: RAFT flow
    corrects what an affine cannot (perspective change across the plate, rolling shutter, small parallax)."""
    import torch
    import torch.nn.functional as F
    _cpu_threads()
    m = raft()
    if m is None:
        return img
    model, tf = m
    H, W = ref.shape[:2]
    Hp, Wp = max(128, -(-H // 8) * 8), max(128, -(-W // 8) * 8)
    padr = lambda a: cv2.copyMakeBorder(a, 0, Hp - H, 0, Wp - W, cv2.BORDER_REFLECT)  # noqa: E731
    ten = lambda a: torch.from_numpy(cv2.cvtColor(padr(a), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255  # noqa: E731
    a, b = ten(ref), ten(img)
    with torch.no_grad():
        flow = model(*tf(a, b), num_flow_updates=12)[-1]               # ref pixel x matches img pixel x + flow(x)
    yy, xx = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    grid = torch.stack([(xx + flow[0, 0]) / (Wp - 1) * 2 - 1, (yy + flow[0, 1]) / (Hp - 1) * 2 - 1], -1)[None]
    out = F.grid_sample(b, grid, mode="bicubic", padding_mode="border", align_corners=True)
    out = (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)[:H, :W]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def fuse(crops: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Align every crop to the middle one at 2x (ECC affine, then RAFT dense residual) and fuse them with
    per-pixel weights that fall off with the distance to the median, so misaligned pixels cannot blur the result."""
    ref = crops[len(crops) // 2]
    H, W = ref.shape[:2]
    up = lambda im: cv2.resize(im, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)  # noqa: E731
    R = up(ref)
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32)
    stack = [R.astype(np.float32)]
    for c in crops:
        if c is ref:
            continue
        C = up(cv2.resize(c, (W, H)))
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(Rg, cv2.cvtColor(C, cv2.COLOR_BGR2GRAY).astype(np.float32), warp,
                                           cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4),
                                           None, 3)
            A = cv2.warpAffine(C, warp, (W * 2, H * 2), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)
        except cv2.error:
            continue                                     # frame could not be aligned: leave it out
        try:
            A = refine(R, A)
        except Exception as e:
            log.debug("RAFT refine skipped: %s", e)
        stack.append(A.astype(np.float32))
    X = np.stack(stack)
    med = np.median(X, axis=0)
    w = np.exp(-((X - med) ** 2).mean(-1, keepdims=True) / (2 * 10.0 ** 2))
    return np.clip((w * X).sum(0) / (w.sum(0) + 1e-6), 0, 255).astype(np.uint8), len(stack)


def enhance(clip: pathlib.Path, fps: float, t: float, box_at, out_dir: pathlib.Path, name: str,
            n: int = 9, pad: float = 0.35) -> dict:
    """box_at(frame) -> [x1,y1,x2,y2] in source pixels (tracks the object across frames) or None.
    Writes <name>_raw.png / _fused.png / _enhanced.png; returns their paths and the frame count used."""
    f_mid = int(round(t * fps))
    frames = _read_frames(clip, f_mid - n // 2, f_mid + n // 2)
    crops = []
    for f, img in sorted(frames.items()):
        b = box_at(f)
        if b is None:
            continue
        x1, y1, x2, y2 = b
        pw, ph = (x2 - x1) * pad + 4, (y2 - y1) * pad + 4
        X1, Y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        X2, Y2 = int(min(img.shape[1], x2 + pw)), int(min(img.shape[0], y2 + ph))
        if X2 - X1 >= 8 and Y2 - Y1 >= 8:
            crops.append(img[Y1:Y2, X1:X2])
    if not crops:
        raise ValueError("object not visible around this moment")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cache the actual input pixels, rather than the requested time or a file's
    # modification date. It survives a demo ZIP round trip and invalidates when
    # a crop changes. Hash the outputs too so edited cache files are not trusted.
    digest = hashlib.sha256(b"mercy-enhance-v1:ecc-affine:raft-small:RealESRGAN-x4plus")
    for crop in crops:
        digest.update(str(crop.shape).encode())
        digest.update(crop.tobytes())
    cache = out_dir / f"enhance_cache_{digest.hexdigest()}.json"
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            paths = {k: (out_dir / v).resolve() for k, v in saved["paths"].items()}
            if (set(paths) == {"raw", "fused", "enhanced"}
                    and all(p.parent == out_dir.resolve() and p.is_file()
                            and hashlib.sha256(p.read_bytes()).hexdigest() == saved["hashes"][k]
                            for k, p in paths.items())):
                return {"paths": paths, "frames_used": saved["frames_used"],
                        "source_px": saved["source_px"], "output_px": saved["output_px"], "cache_hit": True}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    raw = crops[len(crops) // 2]
    fused, used = fuse(crops) if len(crops) >= 3 else (cv2.resize(raw, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 1)
    sr = super_resolve(fused)
    paths = {}
    for k, im in (("raw", cv2.resize(raw, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)),
                  ("fused", cv2.resize(fused, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)),
                  ("enhanced", sr)):
        p = out_dir / f"{name}_{k}.png"
        cv2.imwrite(str(p), im)
        paths[k] = p
    result = {"paths": paths, "frames_used": used, "source_px": [raw.shape[1], raw.shape[0]],
              "output_px": [sr.shape[1], sr.shape[0]]}
    saved = {**result, "paths": {k:p.name for k,p in paths.items()},
             "hashes": {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
             "input_sha256": digest.hexdigest(), "method": "ECC affine + RAFT residual alignment + Real-ESRGAN x4"}
    temporary = cache.with_suffix(".json.part")
    temporary.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    temporary.replace(cache)
    return result


def read_plate(img_path: pathlib.Path, timeout: float = 30) -> dict:
    """Read a plate from the enhanced crop, with the faithful multi-frame fusion beside it as the reference."""
    from mercy.meta_client import data_url, structured
    fused = img_path.with_name(img_path.name.replace("_enhanced", "_fused"))
    content = [{"type": "input_text", "text":
        "Two crops of the same vehicle licence plate from dashcam footage: image 1 is the faithful multi-frame fusion, "
        "image 2 the same after AI super-resolution (sharper, but it can invent strokes). Trust image 1 where they "
        "disagree. Read the plate exactly, including any separator dot and regional prefix (e.g. a Chinese province "
        "character). Use '?' for any character you cannot read with certainty; never guess. If the plate is mirrored "
        "(a reflection) read it un-mirrored and say so in notes. legible=false if fewer than half the characters are "
        "readable. region: issuing country/province/state if identifiable, else ''. confidence 0-1."}]
    if fused.exists():
        content.append({"type": "input_image", "image_url": data_url(fused)})
    content.append({"type": "input_image", "image_url": data_url(img_path)})
    last = None
    for tokens in (5000, 9000):                 # reasoning shares the output budget; retry once with more room
        try:
            return structured(config.VISION_MODEL, [{"type": "message", "role": "user", "content": content}],
                              PLATE_SCHEMA, "Plate", effort="low", timeout=timeout, max_output_tokens=tokens)
        except (json.JSONDecodeError, ValueError) as e:
            last = e
    raise last


def box_tracker(overlays: dict, key: str | None, fallback: list[float]):
    """box_at(frame) from SAM overlays for one object key, falling back to the nearest known box."""
    frames = sorted(int(f) for f, d in overlays.items() if key and key in d)

    def at(f):
        if not frames:
            return fallback
        g = min(frames, key=lambda x: abs(x - f))
        return overlays[str(g)][key]["box"] if abs(g - f) <= 3 else None
    return at


def verified_source_plate(case: pathlib.Path, meta: dict, overlays: dict, vehicle_keys: list[str]):
    """Keep a reviewed original-frame reading separate from generated detail."""
    provenance = case / 'source_provenance.json'
    audit = json.loads(provenance.read_text()).get('original_plate_audit') if provenance.exists() else None
    if not audit:
        return None
    if not audit.get('text') or '?' in audit['text'] or audit.get('confidence', 0) < .8 or audit.get('enhanced') is not False:
        raise ValueError('Original plate audit must contain a readable, unenhanced source reading')
    source = (config.ROOT / audit['evidence_file']).resolve()
    if not source.is_relative_to((config.ROOT / 'data').resolve()):
        raise ValueError('Original plate evidence must be inside the data directory')
    if hashlib.sha256(source.read_bytes()).hexdigest() != audit['evidence_sha256']:
        raise ValueError('Original plate evidence hash changed')
    image = cv2.imread(str(source))
    if image is None or image.shape[:2] != (meta['height'], meta['width']):
        raise ValueError('Original plate evidence dimensions do not match the source')
    t = audit['frame'] / audit.get('source_fps', meta['src_fps']) - meta['trim_start']
    if not 0 <= t < meta['duration']:
        raise ValueError('The retained video window omits its verified original plate frame')
    x1,y1,x2,y2 = map(int, audit['box'])
    if not (0 <= x1 < x2 <= meta['width'] and 0 <= y1 < y2 <= meta['height']):
        raise ValueError('Original plate box is outside the source frame')
    frame = round(t * meta['src_fps'])
    cx,cy = (x1+x2)/2,(y1+y2)/2
    owners = []
    for key in vehicle_keys:
        box = overlays.get(str(frame), {}).get(key, {}).get('box')
        if box and box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            owners.append(((box[2]-box[0])*(box[3]-box[1]), key))
    owner = min(owners)[1] if owners else None
    directory = case / 'evidence'; directory.mkdir(exist_ok=True)
    output = directory / 'plate_verified_original.png'
    cv2.imwrite(str(output), image[y1:y2,x1:x2])
    original = directory / 'plate_verified_frame.png'
    original.write_bytes(source.read_bytes())
    return {'key': f'source_plate:{audit["frame"]}', 'vehicle': owner, 't': round(t, 4),
            'box': [x1,y1,x2,y2], 'images': {'raw': f'evidence/{output.name}'},
            'frames_used': 1, 'source_px': [x2-x1,y2-y1], 'legible': True,
            'text': audit['text'], 'confidence': audit['confidence'], 'region': audit.get('region', ''),
            'notes': 'Read from the reviewed original native frame; no fusion or super-resolution.',
            'source_verified': True, 'original_frame': audit['frame'],
            'original_frame_image': f'evidence/{original.name}',
            'original_frame_sha256': audit['evidence_sha256']}


def plates_for_case(case: pathlib.Path, max_plates: int = 6, force: bool = False) -> dict:
    """Ingest step: enhance and read the clearest view of each tracked plate; link plates to vehicles."""
    out = case / "plates.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    meta = json.loads((case / "meta.json").read_text())
    if meta.get("plates_blurred_at_source"):
        res = {"blurred_at_source": True, "plates": [],
               "note": "The dataset provider blurred licence plates before release; plates cannot be recovered."}
        out.write_text(json.dumps(res, indent=1)); return res
    overlays = json.loads((case / "overlays.json").read_text())
    ph = json.loads((case / "physics.json").read_text())
    vehicle_keys = [o["id"] for o in ph["objects"] if o["id"] != "ego"]
    best: dict[str, tuple[int, dict]] = {}
    for f, objs in overlays.items():
        for k, o in objs.items():
            if o["concept"] == "license plate" and o.get("area", 0) > best.get(k, (0, {"area": 0}))[1].get("area", 0):
                best[k] = (int(f), o)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][1]["area"])[:max_plates]
    native = verified_source_plate(case, meta, overlays, vehicle_keys)
    plates = [native] if native else []
    for key, (f, o) in ranked:
        # link to the vehicle whose box contains the plate centre in the same frame
        cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
        owner = None
        for vk in vehicle_keys:
            vb = overlays.get(str(f), {}).get(vk, {}).get("box")
            if vb and vb[0] <= cx <= vb[2] and vb[1] <= cy <= vb[3]:
                owner = vk; break
        try:
            e = enhance(case / "clip.mp4", meta["src_fps"], f / meta["src_fps"], box_tracker(overlays, key, o["box"]),
                        case / "evidence", f"plate_{key.replace(':', '_').replace(' ', '_')}", pad=0.25)
            reading = read_plate(e["paths"]["enhanced"])
        except Exception as ex:
            log.warning("plate %s: %s", key, ex); continue
        plates.append({"key": key, "vehicle": owner, "t": round(f / meta["src_fps"], 2), "box": o["box"],
                       "images": {k: f"evidence/{p.name}" for k, p in e["paths"].items()},
                       "frames_used": e["frames_used"], "source_px": e["source_px"], **reading})
        log.info("plate %s (vehicle %s): %s legible=%s conf=%.2f", key, owner, reading["text"], reading["legible"],
                 reading["confidence"])
    res = {"blurred_at_source": False, "plates": plates}
    out.write_text(json.dumps(res, indent=1))
    return res

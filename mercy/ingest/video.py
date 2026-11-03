"""Trim the raw clip around the labelled event, normalise it, extract reconstruction frames, write meta.json."""
from __future__ import annotations

import csv
import json
import pathlib
import shutil
import subprocess

import cv2

from mercy import config
from mercy.config import log


def probe(path: pathlib.Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    info["duration"] = info["frames"] / info["fps"]
    return info


def nexar_labels(video: pathlib.Path) -> dict:
    """Look up time_of_event and condition labels from the sibling metadata.csv (Nexar layout)."""
    meta = video.parent / "metadata.csv"
    if not meta.exists():
        return {}
    for r in csv.DictReader(open(meta, newline="", encoding="utf-8")):
        if r.get("file_name") == video.name:
            return r
    return {}


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


EVENT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["event_time_s", "kind", "description"],
                "properties": {"event_time_s": {"type": "number"},
                               "kind": {"type": "string", "enum": ["collision", "near-miss", "none"]},
                               "description": {"type": "string"}}}


def detect_event(video: pathlib.Path, out: pathlib.Path) -> float:
    """Clips without a dataset label: Muse Spark watches the whole clip and returns the moment of first contact."""
    from mercy import config as C
    from mercy.meta_client import structured, upload_file
    full = out / "source_full.mp4"
    if not full.exists():
        _ff("-i", video, "-t", "180", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-vf", "scale=-2:720",
            "-an", "-movflags", "+faststart", full)
    fid = upload_file(full)
    r = structured(C.VISION_MODEL, [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "This is dashcam or traffic-camera footage. Give the time in seconds from the start "
         "of the video of the first physical contact between vehicles (or the closest approach if there is no contact)."},
        {"type": "input_file", "file_id": fid}]}], EVENT_SCHEMA, "Event", effort="low", max_output_tokens=3000)
    (out / "event_detect.json").write_text(json.dumps(r, indent=1))
    log.info("event detected at %.2fs (%s): %s", r["event_time_s"], r["kind"], r["description"][:120])
    return float(r["event_time_s"])


def prepare(video: pathlib.Path, out: pathlib.Path, case_id: str, t_event: float | None = None,
            title: str | None = None, force: bool = False, source: str = "user",
            before_s: float | None = None, after_s: float | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    labels = nexar_labels(video)
    if t_event is None:
        t_event = float(labels["time_of_event"]) if labels else detect_event(video, out)
    src = probe(video)
    before = config.TRIM_BEFORE_S if before_s is None else before_s
    after = config.TRIM_AFTER_S if after_s is None else after_s
    if before < 0 or after < 0 or before + after <= 0:
        raise ValueError("The event window must have nonnegative durations and a positive total length")
    start = max(0.0, t_event - before)
    length = min(before + after, src["duration"] - start)

    clip = out / "clip.mp4"
    # Re-encode (frame-accurate trim), H.264 + faststart, original resolution, keep source fps.
    _ff("-ss", f"{start:.3f}", "-i", video, "-t", f"{length:.3f}", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", clip)
    info = probe(clip)

    frames = out / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir()
    _ff("-i", clip, "-vf", f"fps={config.RECON_FPS},scale={config.RECON_LONG_EDGE}:-2", "-q:v", "2",
        "-start_number", "0", frames / "%06d.jpg")
    n_recon = len(list(frames.glob("*.jpg")))

    meta = {
        "case_id": case_id,
        "title": title or f"Nexar {video.stem} · {labels.get('scene', 'road')} · {labels.get('weather', '')}".strip(" ·"),
        "source": "nexar-ai/nexar_collision_prediction" if labels else source,
        "plates_blurred_at_source": bool(labels),            # Nexar blurs plates before release
        "source_clip": video.name,
        "src_fps": round(info["fps"], 3),
        "recon_fps": config.RECON_FPS,
        "width": info["width"], "height": info["height"],
        "duration": round(info["duration"], 3),
        "trim_start": round(start, 3),
        "time_of_event": round(t_event - start, 3),
        "n_src_frames": info["frames"],
        "n_recon_frames": n_recon,
        "light": labels.get("light_conditions", "unknown"),
        "weather": labels.get("weather", "unknown"),
        "scene": labels.get("scene", "unknown"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("clip %.1fs @%.1ffps %dx%d, %d recon frames, event at %.2fs", meta["duration"], meta["src_fps"],
             meta["width"], meta["height"], n_recon, meta["time_of_event"])
    return meta


def poster(out: pathlib.Path, t: float, name: str = "poster.jpg"):
    """Still frame at time t (used for evidence cards and live SAM)."""
    cap = cv2.VideoCapture(str(out / "clip.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out / name), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return ok


def read_frame(clip: pathlib.Path, t: float):
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, f = cap.read()
    cap.release()
    return f if ok else None

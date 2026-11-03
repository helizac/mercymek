"""Muse Image clean plate: a moment's frame with the vehicles removed, so they can move on their own in the 3D moment.
Only the pixels the vehicles covered are taken from it (mercy.ingest.moments); everything else stays the real frame.
Output is never evidence: the HUD says the background behind the vehicles is AI-filled."""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib

from mercy import config
from mercy.config import log
from mercy.meta_client import data_url, has_key, meta_client, structured

CLEAN_PROMPT = (
    "Remove every vehicle from this dashcam photo (all cars, vans, trucks, buses and motorcycles, including their "
    "shadows) and fill in the road surface, lane markings, crosswalk lines, curbs and background that would be behind "
    "them. Keep everything else exactly the same: same camera position, angle, lens and framing, same lighting, sky, "
    "trees, poles, buildings, and the dashboard and overlay text. Photorealistic, not stylized.")

REVIEW_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["vehicles_remaining", "vehicle_shadows_remaining", "scene_preserved", "road_plausible", "feedback"],
    "properties": {"vehicles_remaining": {"type": "boolean"}, "vehicle_shadows_remaining": {"type": "boolean"},
                   "scene_preserved": {"type": "boolean"}, "road_plausible": {"type": "boolean"},
                   "feedback": {"type": "string"}}}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_clean_plate(frame: pathlib.Path, candidate: pathlib.Path) -> dict:
    """Ask Muse vision to look at both images. Image generation itself has no vision-review endpoint."""
    result = structured(config.VISION_MODEL, [{"role": "user", "content": [
        {"type": "input_text", "text": (
            "Image 1 is the original dashcam scene. Image 2 should be the SAME scene with EVERY vehicle and its "
            "shadow removed, including distant traffic, parked vehicles and the visible camera-car hood. Inspect "
            "image 2 carefully, including edges and far background. Are any cars, trucks, vans, buses, motorcycles, "
            "vehicle fragments or vehicle shadows still present? Are road markings and road surface plausible? "
            "Are viewpoint, perspective, camera height, overpasses, buildings, poles and lighting preserved? "
            "Be strict. Give concrete feedback specifying locations of defects. Dashboard text may be removed.")},
        {"type": "input_image", "image_url": data_url(frame)},
        {"type": "input_image", "image_url": data_url(candidate)}]}],
        REVIEW_SCHEMA, "clean_plate_review", timeout=120)
    result["accepted"] = (not result["vehicles_remaining"] and not result["vehicle_shadows_remaining"]
                          and result["scene_preserved"] and result["road_plausible"])
    result["review_model"] = config.VISION_MODEL
    return result


def verified_clean_plate(frame: pathlib.Path, out: pathlib.Path, attempts: int = 3) -> pathlib.Path:
    """Generate one empty-scene image, review it, and repair defects; persist a hash-bound review trail.

    Never publish an unreviewed/failed image as an accepted environment. Cached success requires identical inputs.
    """
    report_path = out.with_suffix(".review.json")
    source_hash = digest(frame)
    if out.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get("accepted") and report.get("source_sha256") == source_hash
                and report.get("image_sha256") == digest(out)):
            return out
    if not has_key():
        raise RuntimeError("MODEL_API_KEY is required for Muse empty-scene generation and review")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"source_sha256": source_hash, "image_model": config.IMAGE_MODEL, "attempts": [], "accepted": False}
    prompt = CLEAN_PROMPT + (
        " Also remove the visible camera vehicle's hood, mirrors and any dashboard or timestamps, extending the "
        "road naturally to the bottom edge. This is ONE full empty environment image for 3D reconstruction. "
        "Preserve the exact viewpoint and camera height. No vehicles anywhere, including distant or parked ones.")
    previous = frame
    for i in range(attempts):
        candidate = out.with_name(f"{out.stem}.attempt-{i + 1}.png")
        with previous.open("rb") as img:
            r = meta_client().with_options(timeout=300).images.edit(
                model=config.IMAGE_MODEL, image=img, prompt=prompt, n=1, size="1280x720",
                extra_body={"reasoning_strength": "high", "output_format": "png",
                            "tool_enablement": {"enable_web_search": False, "enable_image_search": False,
                                                "enable_shell": False}})
        candidate.write_bytes(base64.b64decode(r.data[0].b64_json))
        review = review_clean_plate(frame, candidate)
        report["attempts"].append({"file": candidate.name, "sha256": digest(candidate), **review})
        if review["accepted"]:
            out.write_bytes(candidate.read_bytes())
            report.update(accepted=True, image_sha256=digest(out))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("empty scene %s attempt %d: %s", out.name, i + 1, review["feedback"])
        if report["accepted"]:
            return out
        previous = candidate
        prompt = ("Correct this empty environment image. " + review["feedback"] +
                  " Remove ALL vehicles, their remnants and shadows. Preserve camera height, framing, architecture "
                  "and road perspective. Photorealistic continuous road, no new traffic or objects.")
    raise RuntimeError(f"Empty scene failed Muse review after {attempts} attempts; see {report_path}")


def make_clean_plate(frame: pathlib.Path, out: pathlib.Path, timeout: float = 240) -> pathlib.Path | None:
    """One Muse Image edit ($0.01). None when there is no key or the call fails (the moment then keeps its vehicles)."""
    if not has_key():
        return None
    try:
        with open(frame, "rb") as img:
            r = meta_client().with_options(timeout=timeout).images.edit(
                model=config.IMAGE_MODEL, prompt=CLEAN_PROMPT, image=img, n=1, size="1280x720",
                extra_body={"reasoning_strength": "high", "output_format": "png",
                            "tool_enablement": {"enable_web_search": False, "enable_image_search": False,
                                                "enable_shell": False}})
    except Exception as e:                                   # network, quota, refusal
        log.warning("clean plate failed for %s: %s", frame.name, e)
        return None
    out.write_bytes(base64.b64decode(r.data[0].b64_json))
    log.info("clean plate -> %s (%d KB)", out.name, out.stat().st_size // 1024)
    return out

"""Muse Spark video understanding -> scene.json (section 9.1)."""
from __future__ import annotations

import json
import pathlib

from mercy import config
from mercy.config import log
from mercy.meta_client import structured, upload_file

SCENE_SCHEMA = {"type": "object", "additionalProperties": False,
  "required": ["summary", "vehicles", "events", "impact_time_s", "traffic_controls",
               "weather", "road_type", "camera_vehicle_involved"],
  "properties": {
    "summary": {"type": "string"},
    "vehicles": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["label", "color", "body_type", "make_model_guess", "role"],
        "properties": {"label": {"type": "string"}, "color": {"type": "string"},
          "body_type": {"type": "string"}, "make_model_guess": {"type": ["string", "null"]},
          "role": {"type": "string", "enum": ["camera_vehicle", "striking", "struck", "bystander"]}}}},
    "events": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["t_s", "description"],
        "properties": {"t_s": {"type": "number"}, "description": {"type": "string"}}}},
    "impact_time_s": {"type": ["number", "null"]},
    "traffic_controls": {"type": "array", "items": {"type": "string"}},
    "weather": {"type": "string", "enum": ["dry", "wet", "snow", "unknown"]},
    "road_type": {"type": "string"},
    "camera_vehicle_involved": {"type": "boolean"}}}

PROMPT = ("You are a traffic forensics analyst reviewing dashcam footage recorded from the camera vehicle. "
          "Describe only what is visible. Give timestamps in seconds from clip start (the clip is {dur:.1f} s long). "
          "List every vehicle that matters to the incident (the camera vehicle first, as role camera_vehicle), "
          "with a short distinctive label such as 'blue hatchback' that could be used to segment it in the video. "
          "Events: 4 to 8 key moments (approach, braking, lane change, impact, aftermath). "
          "impact_time_s is the moment of first contact, or null if there is no contact (near miss).")


def understand_scene(case: pathlib.Path, meta: dict, force: bool = False) -> dict:
    out = case / "scene.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    file_id = _file_id(case)
    scene = structured(
        config.VISION_MODEL,
        [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": PROMPT.format(dur=meta["duration"])},
            {"type": "input_file", "file_id": file_id}]}],
        SCENE_SCHEMA, "Scene", effort="low", max_output_tokens=6000)
    out.write_text(json.dumps(scene, indent=2))
    log.info("scene: %s", scene["summary"][:160])
    return scene


def _file_id(case: pathlib.Path) -> str:
    """Upload clip.mp4 once per case and remember the file id."""
    p = case / ".file_id"
    if p.exists():
        return p.read_text().strip()
    fid = upload_file(case / "clip.mp4")
    p.write_text(fid)
    return fid

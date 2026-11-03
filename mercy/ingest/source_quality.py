"""Admission requirements for NEW demonstration videos. Existing cases remain reproducible."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mercy.ingest.video import probe

OPEN_LICENSES = {"CC0", "Public domain", "CC BY 4.0", "CC BY-SA 4.0", "CC BY 3.0", "CC BY-SA 3.0"}


def validate(video: Path, manifest: Path) -> dict:
    info = json.loads(manifest.read_text(encoding="utf-8"))
    problems = []
    actual = probe(video)
    if min(actual["width"], actual["height"]) < 720:
        problems.append("source must be native 720p or higher")
    if info.get("license") not in OPEN_LICENSES or not info.get("license_url") or not info.get("source_url"):
        problems.append("document an open license and its original source URL")
    if info.get("sha256") != hashlib.sha256(video.read_bytes()).hexdigest():
        problems.append("manifest does not match the source video")
    review = info.get("review", {})
    if review.get("event_kind") != "collision" or not review.get("ordinary_vehicle_collision"):
        problems.append("requires an ordinary physical vehicle collision")
    if review.get("graphic_content") is not False:
        problems.append("requires non-graphic footage")
    if review.get("plates_blurred") is not False:
        problems.append("plates must be unblurred in the original video")
    readings = review.get("readable_plates", [])
    if not readings or not any(p.get("text") and "?" not in p["text"] and p.get("confidence", 0) >= .8
                               and p.get("frame") is not None for p in readings):
        problems.append("requires a clear original-frame plate reading with frame number and confidence >= 0.8")
    if not review.get("model") or not review.get("notes"):
        problems.append("requires a recorded visual review")
    if problems:
        raise ValueError("Video does not meet the demo requirements: " + "; ".join(problems))
    return dict(info, video=actual)

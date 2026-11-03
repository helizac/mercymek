"""Review a licensed HD candidate before admitting it as a new example.

python scripts/review_source.py data/raw/commons/highway_rearend.webm
Expects a sibling .source.json with original source, license and SHA-256.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from mercy.meta_client import structured, upload_file
from mercy.ingest.source_quality import validate

SCHEMA = {"type": "object", "additionalProperties": False,
 "required": ["event_kind", "event_time_s", "ordinary_vehicle_collision", "graphic_content", "plates_blurred", "readable_plates", "notes"],
 "properties": {"event_kind": {"type": "string", "enum": ["collision", "near-miss", "none"]},
  "event_time_s": {"type": "number"}, "ordinary_vehicle_collision": {"type": "boolean"},
  "graphic_content": {"type": "boolean"}, "plates_blurred": {"type": "boolean"}, "notes": {"type": "string"},
  "readable_plates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
    "required": ["text", "frame", "confidence", "vehicle", "box"], "properties": {
     "text": {"type": "string"}, "frame": {"type": "integer"}, "confidence": {"type": "number"},
     "vehicle": {"type": "string"}, "box": {"type": "array", "items": {"type": "integer"}}}}}}}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path)
args = parser.parse_args()
path = args.video.with_suffix(".source.json")
info = json.loads(path.read_text())
if info["sha256"] != hashlib.sha256(args.video.read_bytes()).hexdigest():
    raise ValueError("Source hash mismatch")
if not info.get("review"):
    upload = args.video.with_suffix(".review.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.video), "-c:v", "libx264",
                    "-crf", "18", "-preset", "fast", "-an", str(upload)], check=True)
    import cv2
    cap = cv2.VideoCapture(str(args.video)); fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    fid = upload_file(upload)
    review = structured(config.VISION_MODEL, [{"role": "user", "content": [
        {"type": "input_text", "text": f"Review this ORIGINAL {w}x{h} dashcam video ({fps:.3f}fps) as a demo candidate. "
         "Require an ordinary physical car/van/truck collision with no graphic injury, not a near miss or spectacular "
         "stunt. Determine whether plates are deliberately blurred/pixelated anywhere. Find plates that are clearly "
         "readable in the ORIGINAL frames; no enhancement and NEVER invent characters. If unsure use '?' and low "
         "confidence. For each readable plate give exact text including regional characters, frame number (seconds "
         f"times {fps:.3f}), vehicle description and [x1,y1,x2,y2] box in original pixels. "
         "An empty array is correct if none is genuinely clear. Give first-contact event time in seconds."},
        {"type": "input_file", "file_id": fid}]}], SCHEMA, "source_review", timeout=180)
    review["model"] = config.VISION_MODEL
    info["review"] = review
    path.write_text(json.dumps(info, indent=2))
try:
    validate(args.video, path)
    print(f"ACCEPTED: {args.video}; event {info['review']['event_time_s']}s")
except ValueError as error:
    print(f"REJECTED: {error}")
    sys.exit(1)

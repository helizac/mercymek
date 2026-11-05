"""Env loading, paths and constants shared by ingest and the live demo."""
from __future__ import annotations

import logging
import os
import pathlib
import re
import time
from contextlib import contextmanager

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

META_BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")
ASR_MODEL = os.environ.get("ASR_MODEL", "muse-voice-transcribe-1.0")
VISION_MODEL = os.environ.get("VISION_MODEL", "muse-spark-1.3")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "muse-wonder-1.0")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "muse-image-1.0")
SAM_MODEL = os.environ.get("SAM_MODEL", "sam-3.1")
LAYA_DEVICE = os.environ.get("LAYA_DEVICE", "cpu")
LAYA_MIN_CONFIDENCE = float(os.environ.get("LAYA_MIN_CONFIDENCE", "0.95"))
VISER_PORT = int(os.environ.get("VISER_PORT", "8080"))
APP_PORT = int(os.environ.get("APP_PORT", "8000"))
MIC_SOURCE = os.environ.get("MIC_SOURCE", "browser")
CASES_DIR = (ROOT / os.environ.get("CASES_DIR", "./data/cases")).resolve()
RAW_DIR = ROOT / "data" / "raw"
FRONTEND_DIR = ROOT / "frontend"

# Reconstruction sampling (section 8): 8 fps, long edge 518 px.
RECON_FPS = 8
RECON_LONG_EDGE = 518
# Trim window around the labelled event (section 8, step 1).
TRIM_BEFORE_S = 8.0
TRIM_AFTER_S = 4.0

# Object palette: teal for vehicle 1, amber for vehicle 2 (section 11), then a cool series.
PALETTE = ["#7fd1c7", "#e8a76a", "#8fb3ff", "#d98cff", "#f2d28b", "#9be38f", "#ff8f8f", "#cfd5d8"]
CONCEPT_COLORS = {
    "car": "#7fd1c7", "truck": "#8fb3ff", "pedestrian": "#ff6b6b", "motorcycle": "#d98cff",
    "shattered glass": "#e9f6ff", "debris": "#f2d28b", "traffic light": "#9be38f",
    "license plate": "#cfd5d8", "bus": "#8fb3ff",
}

DISCLAIMER = "Estimated reconstruction from monocular video. Not a legal determination."

log = logging.getLogger("mercy")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def case_dir(case_id: str) -> pathlib.Path:
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case_id):
        raise ValueError("Case ID must be a simple directory name")
    root = CASES_DIR.resolve()
    target = (root / case_id).resolve()
    if target.parent != root:
        raise ValueError("Case must stay inside CASES_DIR")
    return target


@contextmanager
def timed(step: str):
    """Log how long a pipeline step took (rule 6: every step logs timing)."""
    t0 = time.perf_counter()
    log.info("▶ %s", step)
    try:
        yield
    finally:
        log.info("✓ %s (%.1fs)", step, time.perf_counter() - t0)

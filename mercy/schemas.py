"""Pydantic models for every case-folder JSON file and every WebSocket message (contract tests use these)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------- case folder ----------
class Meta(_M):
    case_id: str
    title: str
    source: str
    source_clip: str
    src_fps: float
    recon_fps: float
    width: int
    height: int
    duration: float
    trim_start: float
    time_of_event: float          # seconds from clip start (label minus trim offset)
    n_src_frames: int
    n_recon_frames: int
    light: str = "unknown"
    weather: str = "unknown"
    scene: str = "unknown"


class SceneVehicle(_M):
    label: str
    color: str
    body_type: str
    make_model_guess: Optional[str]
    role: Literal["camera_vehicle", "striking", "struck", "bystander"]


class SceneEvent(_M):
    t_s: float
    description: str


class Scene(_M):
    summary: str
    vehicles: list[SceneVehicle]
    events: list[SceneEvent]
    impact_time_s: Optional[float]
    traffic_controls: list[str]
    weather: Literal["dry", "wet", "snow", "unknown"]
    road_type: str
    camera_vehicle_involved: bool


class Overlay(_M):
    concept: str
    object_id: str
    box: list[float]              # [x1, y1, x2, y2] half-open, source pixels
    confidence: Optional[float]
    poly: list[list[float]]       # simplified polygons, source pixels
    color: str


class Measurement(_M):
    id: str
    label: str
    value: float
    range: list[float]
    unit: str
    method: str
    conf: Literal["high", "medium", "low"]
    estimate: bool = True


class TrackRow(_M):
    k: int
    t: float
    center: list[float]
    bbox: list[list[float]]
    speed: Optional[float] = None


class ObjectInfo(_M):
    id: str
    label: str
    color: str
    concept: str
    role: str = "other"
    sam_key: Optional[str] = None


class Physics(_M):
    scale: dict[str, Any]
    impact: dict[str, Any]
    measurements: list[Measurement]
    series: dict[str, Any]
    objects: list[ObjectInfo]
    warnings: list[str] = []


class Citation(_M):
    title: str
    url: str
    quote_span: str = ""


class ResearchItem(_M):
    key: str
    question: str
    answer: str
    citations: list[Citation]


class FaultShare(_M):
    party: str
    share_pct_range: list[int]
    rationale: str
    citations: list[str]


class VerdictRaw(_M):
    headline: str
    timeline: list[dict]
    key_measurements: list[str]
    fault_assessment: list[FaultShare]
    applicable_rules: list[dict]
    overall_confidence: Literal["low", "medium", "high"]
    limitations: list[str]


class Integrity(_M):
    passed: int
    total: int
    note: str
    review_required: bool = False
    scores: dict[str, Any] = {}


class VerdictFile(_M):
    raw: VerdictRaw
    display: dict[str, Any]
    integrity: Integrity
    model: str
    created: float


# ---------- WebSocket: server -> client ----------
class Hello(_M):
    id: str
    title: str
    duration: float
    fps: float
    impact_t: float
    events: list[dict]
    objects: list[dict]


class PhysicsRow(_M):
    id: str
    label: str
    value: float
    lo: float
    hi: float
    unit: str
    method: str
    conf: str


class TranscriptMsg(_M):
    text: str
    final: bool


class IntentMsg(_M):
    intent: str
    confidence: float
    ms: float
    router: str
    clause: str = ""
    slots: dict = {}


class StateMsg(_M):
    t: Optional[float] = None
    playing: Optional[bool] = None
    speed: Optional[float] = None
    selected: Optional[str] = None
    layers: Optional[dict[str, bool]] = None
    camera: Optional[str] = None
    focus: Optional[str] = None
    trajectory: Optional[list[str]] = None
    isolate: Optional[list[str]] = None


class VerdictMsg(_M):
    status: Literal["running", "ready", "cached"]
    stage: Optional[str] = None
    data: Optional[dict] = None


class ImageMsg(_M):
    status: Literal["running", "ready", "cached", "failed"]
    url: Optional[str] = None
    caption: Optional[str] = None
    watermark: Optional[str] = None


class ErrorMsg(_M):
    message: str


SERVER_MESSAGES = {"hello": Hello, "transcript": TranscriptMsg, "intent": IntentMsg, "state": StateMsg,
                   "verdict": VerdictMsg, "image": ImageMsg, "error": ErrorMsg}


class Command(_M):
    intent: str
    concept: Optional[str] = None
    object_id: Optional[int] = None
    t_seconds: Optional[float] = None
    relative_to_impact: bool = False
    camera_preset: Optional[str] = None
    speed: Optional[float] = None
    note_text: Optional[str] = None
    layer: Optional[str] = None
    visible: Optional[bool] = None
    clause: str = ""
    confidence: float = 1.0
    router: str = "laya"
    ms: float = 0.0
    playing: Optional[bool] = Field(default=None)

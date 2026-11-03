"""SceneState + async broadcast hub: the single source of truth the HUD and viser listen to."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from mercy.config import log


@dataclass
class SceneState:
    case_id: str = ""
    t: float = 0.0
    playing: bool = False
    speed: float = 1.0
    selected: Optional[str] = None
    isolate: list[str] = field(default_factory=list)
    trajectory: list[str] = field(default_factory=list)
    layers: dict[str, bool] = field(default_factory=lambda: {"masks": True, "tracks": True, "hud": True, "cloud": True})
    camera: str = "orbit"
    focus: Optional[str] = None
    zoom: Optional[str] = None
    panel: Optional[str] = None
    view: str = "video"                  # video | scene (3D moments) | map | report | evidence
    moment: Optional[str] = None         # the 3D moment on screen (mercy.ingest.moments id)
    street: bool = False                 # street-level imagery open on the map screen


class StateHub:
    def __init__(self):
        self.state = SceneState()
        self.clients: set = set()
        self.listeners: list[Callable[[dict], None]] = []
        self._lock = asyncio.Lock()

    def snapshot(self) -> dict:
        return asdict(self.state)

    def apply(self, patch: dict) -> dict:
        """Apply a partial update; returns the patch actually applied (for broadcasting)."""
        clean = {}
        for k, v in patch.items():
            if k == "layers" and isinstance(v, dict):
                self.state.layers.update(v); clean["layers"] = dict(v)
            elif hasattr(self.state, k):
                setattr(self.state, k, v); clean[k] = v
        for fn in self.listeners:
            try:
                fn(clean)
            except Exception as e:  # a viewer failure must not break the HUD
                log.debug("listener error: %s", e)
        return clean

    async def send(self, ws, type_: str, data: Any):
        try:
            await ws.send_text(json.dumps({"type": type_, "data": data}, default=str))
        except Exception:
            self.clients.discard(ws)

    async def broadcast(self, type_: str, data: Any, exclude=None):
        msg = json.dumps({"type": type_, "data": data}, default=str)
        dead = []
        for ws in list(self.clients):
            if ws is exclude:
                continue
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def patch(self, p: dict, exclude=None):
        clean = self.apply(p)
        if clean:
            await self.broadcast("state", clean, exclude=exclude)

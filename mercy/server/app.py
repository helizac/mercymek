"""FastAPI server: serves the HUD, case files, and the /ws contract (section 11).

Server -> client: hello, case, physics, transcript, intent, state, verdict, image, overlay, note, error, status
Client -> server: cmd {text}, ptt {down}, ui {patch}, load_case {id}; binary frames = 24 kHz int16 mic PCM (browser relay)
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mercy import __version__, config
from mercy.config import log
from mercy.state import StateHub
from mercy.voice.tts import tts

hub = StateHub()


@asynccontextmanager
async def lifespan(_app):
    await _startup()
    yield


app = FastAPI(title="mercymek", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def revalidate(request, call_next):
    """Browsers must revalidate (ETag, cheap 304) so rebuilt case files and updated scripts show up on reload."""
    resp = await call_next(request)
    resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


VERDICT_CACHE_AFTER_S = 20.0
CASE_FILES = {"clip_redacted.mp4", "clip.mp4", "overlays.json", "cloud.bin", "cloud.json", "tracks.json", "physics.json",
              "scene.json", "meta.json", "research.json", "verdict.json", "diagram.png", "blueprint.webp", "poster.jpg",
              "notes.json", "concepts.json", "scene.glb", "scene_mesh.json", "location.json", "plates.json", "frames4d.json", "streetview.json",
              "scene.splat", "scene_splat.json", "slowmo.mp4", "slowmo.json", "moments.json",
              "claims.json", "seal.json", "statement.json", "source_provenance.json"}
CASE_DIRS = {"evidence", "live", "frames", "geo", "frames4d", "moments", "environment", "vehicles"}
MOMENT_WORDS = [("before", r"\b(before|prior)\b"), ("impact", r"\b(impact|crash|collision|hit|closest)\b"),
                ("brake", r"\bbrak"), ("hazard", r"\b(appear|first seen|hazard|cut)")]


# ------------------------------------------------------------------ case data
class Case:
    def __init__(self, cid: str):
        self.id = cid
        self.dir = config.case_dir(cid)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(cid)
        self.meta = self._j("meta.json")
        self.scene = self._j("scene.json", {})
        self.physics = self._j("physics.json", {"measurements": [], "objects": [], "impact": {}, "series": {}})
        self.tracks = self._j("tracks.json", {})
        self.research = self._j("research.json", {"items": []})
        self.evidence = self._j("evidence/index.json", {"items": []})
        self.concepts = self._j("concepts.json", [])
        self.notes: list[dict] = self._j("notes.json", [])
        self.location = self._j("location.json")
        self.plates = self._j("plates.json", {"plates": []})
        self.streetview = self._j("streetview.json", {"items": []})
        imp = self.physics.get("impact", {})
        self.impact_t = float(imp.get("t", self.meta["time_of_event"]))
        self.brake_t = imp.get("brake_t")
        self.objects = self.physics.get("objects", [])
        for o in self.objects:                       # title-cased labels: keep acronyms upper case
            o["label"] = re.sub(r"\bSuv\b", "SUV", o["label"])
        self.involved = self.physics.get("involved", [])
        self.aliases = self.tracks.get("aliases", {})
        # concepts with at least one detection (from the saved SAM lines)
        self.detected = {}
        for c in self.concepts:
            p = self.dir / "masks" / f"{c.replace(' ', '_')}.lines"
            self.detected[c] = p.exists() and p.stat().st_size > 0
        self._overlay_keys: Optional[dict[str, list[str]]] = None

    def _j(self, name, default=None):
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    @property
    def overlay_keys(self) -> dict[str, list[str]]:
        """concept -> SAM object keys present in overlays.json (loaded lazily, it is large)."""
        if self._overlay_keys is None:
            keys: dict[str, set] = {}
            ov = self._j("overlays.json", {})
            for objs in ov.values():
                for k, o in objs.items():
                    keys.setdefault(o["concept"], set()).add(k)
            self._overlay_keys = {c: sorted(v) for c, v in keys.items()}
        return self._overlay_keys

    def overlays(self) -> dict:
        if getattr(self, "_ov", None) is None:
            self._ov = self._j("overlays.json", {})
        return self._ov

    def url(self, name: str) -> Optional[str]:
        p = self.dir / name
        # the modification time in the URL: a rebuilt file (e.g. scene.splat) is never served from a stale browser cache
        return f"/cases/{self.id}/{name}?v={int(p.stat().st_mtime)}" if p.exists() else None

    def labels(self) -> dict[str, str]:
        """Specific spoken labels -> object key ('blue hatchback' -> car:7). Generic concept names are left to the
        concept dictionary, otherwise 'car one' would fuzzy-match an arbitrary 'Car #10'."""
        from mercy.voice.commands import CONCEPTS
        generic = {s for syn in CONCEPTS.values() for s in syn} | set(CONCEPTS)
        out = {}
        involved = set(self.involved)
        for k, a in sorted(self.aliases.items(), key=lambda kv: kv[0] not in involved):   # involved party wins a shared alias
            if a.lower() not in generic:
                out.setdefault(a.lower(), k)
        return out

    def events(self) -> list[dict]:
        ev = [{"t": round(float(e["t_s"]), 2), "label": e["description"], "kind": "scene"}
              for e in self.scene.get("events", [])]
        near = self.physics.get("impact", {}).get("kind") == "near-miss"
        ev.append({"t": self.impact_t, "label": "Near-miss detected" if near else "Impact detected", "kind": "impact"})
        if self.brake_t is not None:
            ev.append({"t": round(self.brake_t, 2), "label": "Brake detected", "kind": "brake"})
        for o in self.objects:
            if o.get("vid") == "v2":
                s = self.physics["series"]["objects"].get(o["id"])
                if s and s["t"]:
                    ev.append({"t": s["t"][0], "label": f"{o['label']} detected", "kind": "detect"})
        return sorted(ev, key=lambda e: e["t"])

    def physics_rows(self) -> list[dict]:
        return [{"id": m["id"], "label": m["label"], "value": m["value"], "lo": m["range"][0], "hi": m["range"][1],
                 "unit": m["unit"], "method": m["method"], "conf": m["conf"]} for m in self.physics["measurements"]]

    def hello(self) -> dict:
        return {"id": self.id, "title": self.meta["title"], "duration": self.meta["duration"],
                "fps": self.meta["src_fps"], "impact_t": self.impact_t, "events": self.events(),
                "objects": [{"id": o["id"], "label": o["label"], "color": o["color"]} for o in self.objects]}

    def files(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.parent.name not in ("frames", "frames4d", "live"):
                rel = p.relative_to(self.dir).as_posix()
                out.append({"name": rel, "bytes": p.stat().st_size,
                            "url": self.url(rel) if (rel in CASE_FILES or rel.split("/")[0] in CASE_DIRS) else None})
        frames = sorted((self.dir / "frames").glob("*.jpg"))
        if frames:
            out.append({"name": f"frames/ ({len(frames)} × jpg)", "bytes": sum(f.stat().st_size for f in frames),
                        "url": self.url(f"frames/{frames[len(frames) // 2].name}")})
        return out

    def claims(self) -> dict:
        from mercy.analysis import claims
        st = claims.load(self.dir)
        return {"meter": st["meter"], "start": st["start"], "ledger": st["ledger"][-8:]}

    def moments(self) -> list[dict]:
        """Key instants frozen in 3D (mercy.ingest.moments), each with the URL of its splat file and of its vehicles'."""
        return [dict(m, url=self.url(m["file"]),
                     vehicles=[dict(v, url=self.url(v["file"]) if v.get("file") else None)
                               for v in m.get("vehicles", []) if v.get("model") or (v.get("file") and (self.dir / v["file"]).exists())])
                for m in (self._j("moments.json") or {}).get("moments", []) if (self.dir / m["file"]).exists()]

    def moment_for(self, text: str, t: float) -> Optional[dict]:
        """The moment a sentence names ("before impact", "braking"...), else the one nearest to t."""
        ms = self.moments()
        for mid, pat in MOMENT_WORDS:
            if re.search(pat, text or "", re.I):
                hit = next((m for m in ms if m["id"] == mid), None)
                if hit:
                    return hit
        return min(ms, key=lambda m: abs(m["t"] - t)) if ms else None

    def seal_summary(self) -> Optional[dict]:
        s = self._j("seal.json")
        return {"root": s["root"], "created": s["created"], "files": len(s["files"]),
                "source_sha256": s.get("source_sha256")} if s else None

    def bundle(self) -> dict:
        ev_items = [dict(i, url=self.url(i["file"]), mask_url=self.url(i["mask"]) if i.get("mask") else None)
                    for i in self.evidence["items"]]
        cached = self._j("verdict.json")
        return {
            "id": self.id, "meta": self.meta, "scene": self.scene, "impact_t": self.impact_t, "brake_t": self.brake_t,
            "duration": self.meta["duration"], "fps": self.meta["src_fps"],
            "urls": {"video": self.url("clip_redacted.mp4") or self.url("clip.mp4"), "overlays": self.url("overlays.json"),
                     "cloud": self.url("cloud.json"), "cloud_bin": self.url("cloud.bin"), "tracks": self.url("tracks.json"),
                     "diagram": self.url("diagram.png"), "blueprint": self.url("blueprint.webp"),
                     "poster": self.url("evidence/impact.jpg"), "slowmo": self.url("slowmo.mp4")},
            "moments": self.moments(), "slowmo": self._j("slowmo.json"), "location": self.location, "plates": self.plates,
            "streetview": self.streetview, "mapillary_token": os.environ.get("MAPILLARY_TOKEN") or None,
            "objects": self.objects, "involved": self.involved, "aliases": self.aliases,
            "hide_concepts": sorted(set(self.aliases.values())),
            "events": self.events(), "measurements": self.physics_rows(), "series": self.physics.get("series", {}),
            "impact": self.physics.get("impact", {}), "scale": self.physics.get("scale", {}),
            "warnings": self.physics.get("warnings", []), "weather_mu": self.physics.get("weather_mu", {}),
            "evidence": ev_items, "files": self.files(), "detected": self.detected,
            "research": [{"key": i["key"], "answer": i["answer"][:900], "citations": i["citations"][:5]}
                         for i in self.research["items"]],
            "verdict_cached": bool(cached), "notes": self.notes,
            "claims": self.claims(), "statement": self._j("statement.json"), "seal": self.seal_summary(),
            "recon": {"method": self.tracks.get("method"), "fov_y": self.tracks.get("fov_y"),
                      "aspect": self.tracks.get("aspect")},
        }


def list_cases() -> list[dict]:
    out = []
    if config.CASES_DIR.exists():
        for d in sorted(config.CASES_DIR.iterdir()):
            m = d / "meta.json"
            if m.exists() and (d / "physics.json").exists():
                meta = json.loads(m.read_text())
                impact = json.loads((d / "physics.json").read_text()).get("impact", {})
                out.append({"id": d.name, "title": meta["title"], "light": meta.get("light"), "weather": meta.get("weather"),
                            "event_kind": impact.get("kind"),
                            "scene": meta.get("scene"), "duration": meta["duration"],
                            "poster": f"/cases/{d.name}/evidence/impact.jpg" if (d / "evidence/impact.jpg").exists() else None})
    order = {"hero": 0}
    return sorted(out, key=lambda c: (order.get(c["id"], 1), c["id"]))


# ------------------------------------------------------------------ runtime
class Runtime:
    def __init__(self):
        self.case: Optional[Case] = None
        self.parser = None
        self.laya_ready = threading.Event()
        self.asr_sessions: dict[int, dict] = {}
        self.verdict_task: Optional[asyncio.Task] = None
        self.image_task: Optional[asyncio.Task] = None
        self.viser = None

    def load(self, cid: str):
        self.case = Case(cid)
        st = hub.state
        st.case_id, st.t, st.playing, st.speed = cid, 0.0, False, 1.0
        st.selected, st.isolate, st.trajectory, st.zoom, st.panel, st.focus = None, [], [], None, None, None
        st.camera, st.view, st.moment = "orbit", "video", None
        if self.viser:
            try:
                self.viser.load_case(self.case)
            except Exception as e:
                log.warning("viser load failed: %s", e)

    def preload_laya(self):
        def _go():
            try:
                from mercy.voice.commands import CommandParser
                p = CommandParser()
                p.parse("go to impact", allow_deep=False)      # warm the forward pass
                self.parser = p
                log.info("Laya router ready")
            except Exception as e:
                log.error("Laya unavailable: %s", e)
            finally:
                self.laya_ready.set()
        threading.Thread(target=_go, daemon=True).start()


rt = Runtime()


async def send_case(ws: Optional[WebSocket] = None):
    c = rt.case
    if c is None:
        return
    msgs = [("cases", list_cases()), ("hello", c.hello()), ("case", c.bundle()), ("physics", c.physics_rows()),
            ("state", hub.snapshot()), ("status", status())]
    for t, d in msgs:
        if ws is not None:
            await hub.send(ws, t, d)
        else:
            await hub.broadcast(t, d)


def status() -> dict:
    return {"meta_key": bool(os.environ.get("MODEL_API_KEY")), "laya": rt.parser is not None,
            "laya_loading": not rt.laya_ready.is_set(), "mic": config.MIC_SOURCE,
            "models": {"asr": config.ASR_MODEL, "vision": config.VISION_MODEL, "reasoning": config.REASONING_MODEL,
                       "image": config.IMAGE_MODEL, "sam": config.SAM_MODEL},
            "viser": f"http://localhost:{config.VISER_PORT}" if rt.viser else None}


# ------------------------------------------------------------------ speech
async def say(text: str | None):
    """Show the reply and speak it (ElevenLabs); text-only when speech is off or the budget is spent."""
    if not text:
        return
    k = tts.register(text)
    await hub.broadcast("say", {"text": text, "url": f"/api/tts/{k}" if k else None})


def spell(plate: str) -> str:
    """Plate characters spoken one by one; '?' becomes 'unclear'."""
    return ", ".join("unclear" if ch == "?" else ch for ch in plate.replace(" ", "").replace("·", ""))


def fault_sentence(d: dict) -> str:
    f = d.get("fault") or []
    if not f:
        return d.get("headline", "")
    s = f"{f[0]['party']} is mostly at fault, about {f[0]['pct']} percent"
    if len(f) > 1:
        s += f". {f[1]['party']}, about {f[1]['pct']} percent"
    return s + "."


def obj_label(key: str) -> str:
    if key == "ego":
        return "the camera car"
    o = next((o for o in rt.case.objects if o["id"] == key), None)
    return (o["label"].split(" #")[0] if o else key).lower().replace("suv", "SUV")


ASK_SYSTEM = ("You are mercymek, a crash investigator speaking out loud. Answer the question in at most two short "
              "sentences (35 words max) using only the case facts given. Plain words, no ids, no markdown. Give speeds "
              "as ranges. If the facts do not answer it, say what is unknown in one sentence.")


def case_facts(c) -> str:
    m = {x["id"]: x for x in c.physics.get("measurements", [])}
    pick = lambda k, lab: f"{lab}: {m[k]['range'][0]:.0f}-{m[k]['range'][1]:.0f} {m[k]['unit']}" if k in m else None  # noqa: E731
    facts = [f"Summary: {c.scene.get('summary', '')}",
             f"Event: {c.physics.get('impact', {}).get('kind', 'collision')} at {c.impact_t:.1f} s",
             *(x for x in [pick("v1.pre_impact_speed", "Camera car speed before impact"),
                           pick("v2.pre_impact_speed", "Other vehicle speed before impact"),
                           pick("v1.brake_onset", "Camera car braking began, seconds before impact"),
                           pick("v2.visible_before_impact", "Other vehicle visible, seconds before impact"),
                           pick("impact.closing_speed", "Closing speed")] if x)]
    v = c._j("verdict.json")
    if v:
        facts.append("Fault: " + "; ".join(f"{f['party']} {f['lo']}-{f['hi']}%" for f in v["display"]["fault"]))
        facts.append("Finding: " + v["display"]["headline"])
    if c.location and c.location.get("place_name"):
        facts.append(f"Location: {c.location['place_name']} (confidence {c.location.get('confidence')})")
    for pl in c.plates.get("plates", []):
        if pl.get("legible"):
            owner = next((o["label"] for o in c.objects if o["id"] == pl.get("vehicle")), pl.get("vehicle") or "a vehicle")
            facts.append(f"Plate of {owner}: {pl['text']}")
    if c.plates.get("blurred_at_source"):
        facts.append("Plates were blurred by the data provider.")
    return "\n".join(facts)


async def answer(question: str):
    from mercy.meta_client import meta_client
    if not os.environ.get("MODEL_API_KEY"):
        await say("I can't answer questions without the model connection.")
        return
    try:
        r = await asyncio.wait_for(asyncio.to_thread(
            meta_client().responses.create, model=config.VISION_MODEL, reasoning={"effort": "minimal"},
            max_output_tokens=600, input=[{"role": "system", "content": ASK_SYSTEM},
                                          {"role": "user", "content": case_facts(rt.case) + "\n\nQuestion: " + question}]), 20)
        await say(r.output_text.strip())
    except Exception as e:
        log.warning("answer failed: %s", e)
        await say("Sorry, I couldn't work that out just now.")


# ------------------------------------------------------------------ command execution
def resolve_objects(cmd: dict) -> list[str]:
    c = rt.case
    vids = {o.get("vid"): o["id"] for o in c.objects if o.get("vid")}
    if cmd.get("sam_key"):
        return [cmd["sam_key"]]
    oid = cmd.get("object_id")
    if oid == 0:
        return [k for k in c.involved]
    if oid == 1:
        return [vids.get("v1", "ego")]
    if oid == 2 and "v2" in vids:
        return [vids["v2"]]
    if isinstance(oid, int) and oid >= 3:
        others = [o["id"] for o in c.objects if o["role"] == "other"]
        if len(others) >= oid - 2:
            return [others[oid - 3]]
    concept = cmd.get("concept")
    if concept:
        if concept == "car":
            return [vids["v2"]] if "v2" in vids else c.overlay_keys.get("car", [])[:3]
        return c.overlay_keys.get(concept, [])
    return [k for k in c.involved if k != "ego"][:1]


async def execute(cmd: dict, ws: Optional[WebSocket]):
    c, st = rt.case, hub.state
    intent = cmd["intent"]
    await hub.broadcast("intent", {"intent": intent, "confidence": cmd.get("confidence", 1.0), "ms": cmd.get("ms", 0),
                                   "router": cmd.get("router", "laya"), "clause": cmd.get("clause", ""),
                                   "slots": {k: v for k, v in cmd.items() if k not in
                                             ("intent", "confidence", "ms", "router", "clause")}})
    if intent == "seek":
        anchor, off = cmd.get("anchor", "impact"), float(cmd.get("t_seconds") or 0)
        base = {"impact": c.impact_t, "brake": c.brake_t if c.brake_t is not None else c.impact_t - 1.5,
                "current": st.t, "start": 0.0, "absolute": 0.0}.get(anchor, c.impact_t)
        t = max(0.0, min(c.meta["duration"] - 0.05, base + off))
        await hub.patch({"t": round(t, 3), "playing": False})
        n = abs(off)
        num = f"{n:g} second{'s' if n != 1 else ''}"
        return {"impact": f"{num} {'before' if off < 0 else 'after'} impact." if off else "At the impact.",
                "brake": "Braking starts here.", "start": "From the start.",
                "current": f"{num} {'back' if off < 0 else 'forward'}."}.get(anchor, f"At {t:.1f} seconds.")
    elif intent == "playback":
        p = {"playing": cmd.get("playing", True)}
        if cmd.get("speed"):
            p["speed"] = cmd["speed"]
        await hub.patch(p)
        return "Slow motion." if (cmd.get("speed") or 1) < 1 else ("Playing." if p["playing"] else "Paused.")
    elif intent == "view":
        v = cmd.get("view") or "video"
        if v == "scene":
            m = c.moment_for(cmd.get("clause", ""), st.t)
            if not m:
                return "There are no 3D moments for this case."
            await hub.patch({"view": "scene", "moment": m["id"], "t": m["t"], "playing": False})
            return f"{m['label']}, in 3D."
        await hub.patch({"view": "map" if v == "street" else v, "street": v == "street"})
        loc = c.location or {}
        if v == "map":
            short = ", ".join(x.strip() for x in loc.get("place_name", "").split(",")[:2])
            return f"It happened on {short}." if short else "The location is unknown."
        if v == "street":
            n = len(c.streetview.get("items", []))
            return f"{n} street-level photos near the scene." if n else "No open street-level photos near this place."
        return {"report": "Here is the report.", "evidence": "Here is the evidence.",
                "video": "Back to the video."}.get(v)
    elif intent == "zoom" and (cmd.get("enhance") or cmd.get("concept") == "license plate"):
        start_enhance(cmd)
    elif intent in ("isolate", "zoom", "trajectory"):
        if cmd.get("live") or (not cmd.get("sam_key") and cmd.get("concept") and cmd["concept"] not in ("car",) and not c.overlay_keys.get(cmd["concept"])):
            await live_segment(cmd.get("concept") or cmd.get("clause", ""))
            return
        keys = resolve_objects(cmd)
        if not keys:
            await hub.broadcast("error", {"message": f"No evidence detected for '{cmd.get('concept')}'. That is also a result."})
            return
        names = " and ".join(obj_label(k) for k in keys[:2])
        if intent == "isolate":
            await hub.patch({"isolate": keys, "selected": keys[0], "layers": {"masks": True}})
            return f"Isolating {names}."
        elif intent == "zoom":
            await hub.patch({"zoom": None if keys[0] == "ego" else keys[0], "selected": keys[0],
                             **({"camera": "driver"} if keys[0] == "ego" else {})})
            return f"Zooming in on {names}."
        else:
            await hub.patch({"trajectory": keys, "selected": keys[0], "layers": {"tracks": True}})
            return f"Path of {names}."
    elif intent == "camera":
        m = c.moment_for(cmd.get("clause", ""), st.t)
        if not m:
            return "There are no 3D moments for this case."
        if cmd.get("camera_preset") == "approach":
            m = next((x for x in c.moments() if x.get("approach") and x["id"] == "impact"), None) \
                or next((x for x in c.moments() if x.get("approach")), None)
            if not m:
                return "This case has no approach replay."
            await hub.patch({"camera": "approach", "view": "scene", "moment": m["id"], "t": m["replay_start"],
                             "playing": True, "speed": 0.5})
            return "The approach, replayed through the reconstructed street."
        preset = "driver" if cmd.get("camera_preset") == "driver" else "orbit"   # a moment drifts, or holds the dashcam's view
        await hub.patch({"camera": preset, "view": "scene", "moment": m["id"], "t": m["t"], "playing": False})
        return "From the driver's seat." if preset == "driver" else f"{m['label']}, in 3D."
    elif intent == "layers":
        if cmd.get("layer"):
            await hub.patch({"layers": {cmd["layer"]: bool(cmd.get("visible", True))}})
    elif intent == "analyze":
        await hub.patch({"view": "report"})
        cached = c._j("verdict.json")
        again = re.search(r"again|re-?analy|re-?run|update", cmd.get("clause", ""), re.I)
        new_notes = cached and any(n["at"] > cached.get("created", 0) for n in c.notes)
        if cached and not again and not new_notes:     # the analysis already exists: show it, no new model call
            await hub.broadcast("verdict", {"status": "ready", "data": cached["display"]})
            return fault_sentence(cached["display"])
        start_verdict()
        return "Analysing the video. One moment."
    elif intent == "diagram":
        await hub.patch({"view": "report"})
        start_image()
        return "Drawing the blueprint."
    elif intent == "ask":
        asyncio.create_task(answer(cmd.get("question") or cmd.get("clause", "")))
    elif intent == "note":
        note = {"t": round(st.t, 2), "text": cmd.get("note_text") or cmd.get("clause", ""), "at": time.time()}
        c.notes.append(note)
        (c.dir / "notes.json").write_text(json.dumps(c.notes, indent=1))
        await hub.broadcast("note", note)
        return "Noted."
    elif intent == "whatif":
        from mercy.analysis import whatif
        road = (c.location or {}).get("road") or {}
        limit = road.get("maxspeed") if isinstance(road.get("maxspeed"), (int, float)) else None
        res = whatif.run(c.physics, cmd.get("question") or cmd.get("clause", ""), limit)
        await hub.broadcast("whatif", res)
        await hub.patch({"view": "video", "t": max(0.0, res["t_hazard"] - 0.5), "playing": True, "speed": 0.5})
        return res["say"]
    elif intent == "replay":
        if cmd.get("mode") == "bullet":
            m = c.moment_for("impact", c.impact_t)
            if m:
                await hub.patch({"t": m["t"], "playing": False, "view": "scene", "moment": m["id"], "camera": "orbit"})
                return "Time frozen at the impact, in 3D."
        info = c._j("slowmo.json")
        if info and c.url("slowmo.mp4"):
            await hub.patch({"view": "video", "playing": False})
            await hub.broadcast("replay", {**info, "url": c.url("slowmo.mp4"), "impact_t": c.impact_t})
            return "The impact at quarter speed, with the frames in between reconstructed."
        await hub.patch({"t": max(0.0, c.impact_t - 2), "speed": 0.25, "playing": True, "view": "video"})
        return "Slow motion from two seconds before the impact."
    elif intent == "claim":
        text = re.sub(r"^(objection|for the record|counter.?point)\s*(,|:|and)?\s*", "", cmd.get("claim") or cmd.get("clause", ""),
                      flags=re.I).strip()
        asyncio.create_task(claim_flow(text[:1].upper() + text[1:]))
    elif intent == "verify":
        from mercy.analysis import seal
        v = await asyncio.to_thread(seal.verify, c.dir)
        await hub.broadcast("seal", v)
        if not v.get("sealed"):
            return "This case has not been sealed yet."
        if v["intact"]:
            return f"All {v['files']} files match the seal. Nothing has changed since it was sealed."
        bad = v["changed"] + v["missing"]
        return f"{len(bad)} sealed file{'s' if len(bad) != 1 else ''} changed since sealing: {', '.join(bad[:3])}."
    elif intent == "statement":
        from mercy.analysis import statement
        await hub.patch({"view": "report"})
        try:
            st = c._j("statement.json") or await asyncio.to_thread(statement.build, c.dir)
        except Exception as e:
            log.warning("statement failed: %s", e)
            return "I couldn't fill in the accident statement just now."
        await hub.broadcast("statement", st)
        a, b = st["A"], st["B"]
        part = lambda v: (f"box {', '.join(map(str, v['circumstances']))}, {v['ticked'][0]}" if v["circumstances"]  # noqa: E731
                          else "no box ticked")
        return f"Accident statement filled. Vehicle A, {part(a)}. Vehicle B, {part(b)}."
    elif intent == "reset":
        await hub.patch({"selected": None, "isolate": [], "trajectory": [], "zoom": None, "camera": "orbit", "moment": None,
                         "panel": None, "focus": None, "view": "video", "street": False,
                         "layers": {"masks": True, "tracks": True, "hud": True, "cloud": True}})
        await hub.broadcast("overlay", None)
        await hub.broadcast("whatif", None)
        return "Reset."


async def run_text(text: str, ws: Optional[WebSocket], typed: bool):
    if rt.case is None:
        return
    if typed:
        await hub.broadcast("transcript", {"text": text, "final": True, "typed": True})
    if not rt.laya_ready.is_set():
        await hub.broadcast("status", status() | {"note": "Laya is loading…"})
        await asyncio.to_thread(rt.laya_ready.wait, 120)
    if rt.parser is None:
        await hub.broadcast("error", {"message": "Command router unavailable (Laya failed to load)."})
        return
    loop = asyncio.get_running_loop()

    def on_deep():          # called from the parser thread only when the Spark fallback actually starts
        asyncio.run_coroutine_threadsafe(hub.broadcast("status", {"deep": True}), loop)

    cmds, info = await asyncio.to_thread(rt.parser.parse, text, rt.case.labels(),
                                         [k for k, v in rt.case.detected.items() if v],
                                         bool(os.environ.get("MODEL_API_KEY")), on_deep)
    if not cmds:
        await hub.broadcast("error", {"message": "Not sure what you meant."})
        await say("Sorry, I didn't catch that.")
        return
    replies = []
    for cmd in cmds:
        r = await execute(cmd, ws)
        if r:
            replies.append(r)
    await say(" ".join(replies[-2:]))


# ------------------------------------------------------------------ verdict / image / live SAM
async def claim_flow(text: str):
    from mercy.analysis import claims
    c = rt.case
    if not os.environ.get("MODEL_API_KEY"):
        await say("I can't weigh arguments without the model connection.")
        return
    await hub.broadcast("meter", {"status": "weighing", "claim": text})
    try:
        r = await asyncio.to_thread(claims.judge, c.dir, case_facts(c), text)
    except Exception as e:
        log.warning("claim failed: %s", e)
        await hub.broadcast("meter", {"status": "error"})
        await say("Sorry, I couldn't weigh that just now.")
        return
    await hub.broadcast("meter", {"status": "ready", **r})
    e = r["entry"]
    tail = ""
    if e["moved"] and r["meter"]:
        top = max(r["meter"], key=lambda m: m["pct"])
        tail = f" Liability now {top['party'].split(' #')[0]}, {top['pct']:.0f} percent."
    elif not r["meter"]:
        tail = " Ask who is at fault first, so I have liability to adjust."
    await say(e["reply"] + tail)


def start_verdict():
    if rt.verdict_task and not rt.verdict_task.done():
        return
    rt.verdict_task = asyncio.create_task(verdict_flow())


async def verdict_flow():
    from mercy.analysis import verdict as V
    c = rt.case
    cached = c._j("verdict.json")
    await hub.broadcast("verdict", {"status": "running", "stage": "Loading measurements, tracks and cached web research…"})
    await asyncio.sleep(0.4)
    live_ok = bool(os.environ.get("MODEL_API_KEY")) and (c.dir / "research.json").exists()
    if not live_ok:
        if cached:
            await hub.broadcast("verdict", {"status": "cached", "data": cached["display"], "model": cached.get("model")})
        else:
            await hub.broadcast("error", {"message": "No cached verdict and no API access."})
        return
    await hub.broadcast("verdict", {"status": "running", "stage": "Watching the video and weighing the measurements…"})
    from mercy.analysis import claims
    notes = [n["text"] for n in c.notes] + claims.as_notes(c.dir)
    job = asyncio.create_task(asyncio.to_thread(V.run, c.dir, notes, True, "medium", 60.0, False))
    t0 = time.time()
    sent_cached = False
    while not job.done():
        await asyncio.sleep(0.5)
        el = time.time() - t0
        if el > 6 and int(el * 2) % 12 == 0:
            await hub.broadcast("verdict", {"status": "running", "stage": f"Analysing · {el:.0f}s"})
        if el > VERDICT_CACHE_AFTER_S and cached and not sent_cached:
            sent_cached = True
            await hub.broadcast("verdict", {"status": "cached", "data": cached["display"], "model": cached.get("model"),
                                            "note": "live call still running"})
    try:
        res = job.result()
        await hub.broadcast("verdict", {"status": "running", "stage": "Auditing the findings…"})
        await asyncio.sleep(0.2)
        await hub.broadcast("verdict", {"status": "ready", "data": res["display"], "model": res["model"],
                                        "latency_s": res["latency_s"]})
        await say(fault_sentence(res["display"]))
    except Exception as e:
        log.warning("live verdict failed: %s", e)
        if cached and not sent_cached:
            await hub.broadcast("verdict", {"status": "cached", "data": cached["display"], "model": cached.get("model")})
        elif not cached:
            await hub.broadcast("error", {"message": f"Verdict failed: {e}"})


def start_image():
    if rt.image_task and not rt.image_task.done():
        return
    rt.image_task = asyncio.create_task(image_flow())


async def image_flow():
    from mercy.generation import blueprint, diagram
    c = rt.case
    await hub.broadcast("image", {"status": "running", "stage": "Drawing to-scale overhead diagram…"})
    live = c.dir / "live"; live.mkdir(exist_ok=True)
    stamp = int(time.time())
    d_path = live / f"diagram_{stamp}.png"
    try:
        await asyncio.to_thread(diagram.draw, c.dir, d_path, None, True)
    except Exception as e:
        log.warning("diagram failed: %s", e); d_path = c.dir / "diagram.png"
    await hub.broadcast("image", {"status": "running", "stage": "Rendering the blueprint…",
                                  "diagram": f"/cases/{c.id}/{d_path.relative_to(c.dir).as_posix()}"})
    cap = "Restyled from the to-scale overhead diagram; geometry from the 3D tracks."
    if os.environ.get("MODEL_API_KEY"):
        try:
            out = live / f"blueprint_{stamp}.webp"
            await asyncio.wait_for(asyncio.to_thread(blueprint.make_blueprint, d_path, out, 60), 75)
            await hub.broadcast("image", {"status": "ready", "url": f"/cases/{c.id}/live/{out.name}", "caption": cap,
                                          "watermark": "AI ILLUSTRATION · NOT EVIDENCE",
                                          "diagram": f"/cases/{c.id}/{d_path.relative_to(c.dir).as_posix()}"})
            return
        except Exception as e:
            log.warning("live blueprint failed: %s", e)
    url = c.url("blueprint.webp") or c.url("diagram.png")
    await hub.broadcast("image", {"status": "cached", "url": url, "caption": cap + " (cached)",
                                  "watermark": "AI ILLUSTRATION · NOT EVIDENCE", "diagram": c.url("diagram.png")})


def start_enhance(cmd: dict):
    asyncio.create_task(enhance_flow(cmd))


def _enhance_target(cmd: dict):
    """(object key, label, is_plate) for 'zoom in / enhance' at the current time."""
    c, st = rt.case, hub.state
    ov = c.overlays()
    f = int(round(st.t * c.meta["src_fps"]))
    win = range(f - 3, f + 4)
    near = lambda key: any(key in ov.get(str(g), {}) for g in win)  # noqa: E731
    if cmd.get("concept") == "license plate":
        want = None
        if cmd.get("sam_key") or cmd.get("object_id") is not None:
            want = (resolve_objects({k: v for k, v in cmd.items() if k != "concept"}) or [None])[0]
        owner = {p["key"]: p.get("vehicle") for p in c.plates.get("plates", [])}
        keys = list({k for g in win for k, o in ov.get(str(g), {}).items() if o["concept"] == "license plate"})
        pref = want or st.selected
        area = lambda k: max((ov.get(str(g), {}).get(k, {}).get("area", 0) for g in win), default=0)  # noqa: E731
        # the requested / selected vehicle's plate first, then the largest plate in view
        keys.sort(key=lambda k: (pref is not None and owner.get(k) != pref, -area(k)))
        return (keys[0] if keys else None), "licence plate", True
    key = st.zoom or st.selected
    if cmd.get("sam_key") or cmd.get("object_id") is not None or cmd.get("concept"):
        key = (resolve_objects(cmd) or [key])[0]
    if key and key != "ego" and near(key):
        return key, next((o["label"] for o in c.objects if o["id"] == key), key), False
    return None, "frame", False


async def enhance_flow(cmd: dict):
    from mercy.analysis import enhance as E
    c, st = rt.case, hub.state
    await hub.patch({"playing": False})
    key, label, is_plate = _enhance_target(cmd)
    if is_plate and key is None:
        msg = ("Licence plates in this footage were blurred by the data provider before release."
               if c.meta.get("plates_blurred_at_source") else "No licence plate is visible at this moment.")
        await hub.broadcast("enhance", {"status": "failed", "message": msg})
        await say(msg)
        return
    await hub.broadcast("enhance", {"status": "running", "label": label})
    if key:
        await hub.patch({"zoom": key})
    ov = c.overlays()
    W, H = c.meta["width"], c.meta["height"]
    stamp = int(time.time() * 1000)
    try:
        e = await asyncio.to_thread(E.enhance, c.dir / "clip.mp4", c.meta["src_fps"], st.t,
                                    E.box_tracker(ov, key, [W * 0.3, H * 0.3, W * 0.7, H * 0.7]), c.dir / "live",
                                    f"enh_{stamp}", 9, 0.25 if is_plate else 0.12)
    except Exception as ex:
        await hub.broadcast("enhance", {"status": "failed", "message": f"Could not enhance: {ex}"})
        return
    msg = {"status": "ready", "label": label, "t": round(st.t, 2),
           "cached": bool(e.get("cache_hit")),
           "images": {k: f"/cases/{c.id}/live/{p.name}" for k, p in e["paths"].items()},
           "frames_used": e["frames_used"], "source_px": e["source_px"], "output_px": e["output_px"], "plate": None}
    if is_plate and os.environ.get("MODEL_API_KEY"):
        await hub.broadcast("enhance", dict(msg, status="reading"))
        try:
            msg["plate"] = await asyncio.wait_for(asyncio.to_thread(E.read_plate, e["paths"]["enhanced"]), 40)
        except Exception as ex:
            log.warning("plate read failed: %s", ex)
    await hub.broadcast("enhance", msg)
    pl = msg.get("plate")
    if is_plate and pl:
        await say(f"The plate reads {spell(pl['text'])}." if pl.get("legible") and pl.get("text")
                  else "The plate is not readable, even after enhancement.")
    elif not is_plate:
        await say(f"Enhanced {label}.")


async def live_segment(concept: str):
    """Isolate a concept that was not precomputed: SAM 3.1 on the current frame (section 7, the wow moment)."""
    from mercy.ingest import segment
    from mercy.ingest.video import read_frame
    c, st = rt.case, hub.state
    concept = concept.strip() or "object"
    await hub.patch({"playing": False})
    await hub.broadcast("status", {"note": f"Looking for “{concept}” in this frame…"})
    if not os.environ.get("MODEL_API_KEY"):
        await hub.broadcast("error", {"message": "Live segmentation needs API access."}); return
    img = read_frame(c.dir / "clip.mp4", st.t)
    if img is None:
        return
    t0 = time.time()
    try:
        raw = await asyncio.wait_for(asyncio.to_thread(segment.segment_image, img, concept), 30)
        recs = segment.parse(raw, "image")
    except Exception as e:
        await hub.broadcast("error", {"message": f"Live segmentation failed: {e}"}); return
    objs = []
    for r in recs:
        crop = segment.mask_crop(r)
        b = r.bounds
        objs.append({"key": f"live:{concept}:{r.object_id}", "box": [b.left, b.top, b.right, b.bottom],
                     "poly": segment.polygons(crop, (b.left, b.top)), "confidence": r.confidence})
    if not objs:
        await hub.broadcast("error", {"message": f"SAM found no “{concept}” in this frame. No evidence detected."})
        return
    await hub.broadcast("overlay", {"t": st.t, "concept": concept, "objects": objs, "color": "#ff5a5f",
                                    "ms": round((time.time() - t0) * 1000)})


# ------------------------------------------------------------------ voice
async def ptt_start(ws: WebSocket):
    from mercy.voice.asr import PushToTalkASR, transcribe_pcm_stream
    if not os.environ.get("MODEL_API_KEY"):
        await hub.send(ws, "error", {"message": "Voice needs MODEL_API_KEY. Type the command instead."}); return
    loop = asyncio.get_running_loop()

    def partial(text):
        loop.call_soon_threadsafe(asyncio.ensure_future, hub.broadcast("transcript", {"text": text, "final": False}))

    sess: dict = {"t0": time.time()}
    if config.MIC_SOURCE == "server":
        asr = PushToTalkASR(on_partial=partial)
        asr.start()
        sess["server"] = asr
    else:
        q: asyncio.Queue = asyncio.Queue()
        sess["q"] = q
        sess["task"] = asyncio.create_task(transcribe_pcm_stream(q, partial))
    rt.asr_sessions[id(ws)] = sess


async def ptt_stop(ws: WebSocket):
    sess = rt.asr_sessions.pop(id(ws), None)
    if not sess:
        return
    t_release = time.time()
    try:
        if "server" in sess:
            text = await sess["server"].stop()
        else:
            sess["q"].put_nowait(None)
            text = await asyncio.wait_for(sess["task"], 20)
    except Exception as e:
        await hub.broadcast("error", {"message": f"Voice transcription failed: {e}"}); return
    text = (text or "").strip()
    await hub.broadcast("transcript", {"text": text or "…", "final": True,
                                       "asr_ms": round((time.time() - t_release) * 1000)})
    if text:
        await run_text(text, ws, typed=False)


# ------------------------------------------------------------------ routes
async def _startup():
    cases = list_cases()
    want = os.environ.get("MERCY_CASE")
    if cases:
        rt.load(want if want and any(c["id"] == want for c in cases) else cases[0]["id"])
    rt.preload_laya()
    asyncio.create_task(tts.keep_warm())
    if os.environ.get("MERCY_VISER", "1") == "1":
        try:
            from mercy.viewer.viser_scene import Scene4D
            rt.viser = Scene4D(port=config.VISER_PORT)
            if rt.case:
                rt.viser.load_case(rt.case)
            hub.listeners.append(rt.viser.on_state)
        except Exception as e:
            log.warning("viser disabled: %s", e)


@app.get("/api/tts/{k}")
async def api_tts(k: str):
    if not re.fullmatch(r"[0-9a-f]{16}", k):
        raise HTTPException(404)
    return StreamingResponse(tts.stream(k), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/cases")
def api_cases():
    return list_cases()


@app.get("/api/case/{cid}")
def api_case(cid: str):
    try:
        return Case(cid).bundle()
    except (FileNotFoundError, ValueError):
        raise HTTPException(404)


@app.get("/api/status")
def api_status():
    return status()


@app.get("/cases/{cid}/{path:path}")
def case_file(cid: str, path: str):
    try:
        d = config.case_dir(cid).resolve()
    except ValueError:
        raise HTTPException(404)
    p = (d / path).resolve()
    if d not in p.parents or not p.is_file():
        raise HTTPException(404)
    rel = p.relative_to(d).as_posix()
    if rel not in CASE_FILES and rel.split("/")[0] not in CASE_DIRS:
        raise HTTPException(403)
    return FileResponse(p)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # Browsers send Origin; keep API-key-backed commands on this application's
    # own origin. Native local WebSocket clients may omit the browser header.
    origin = ws.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
            allowed = parsed.scheme in {"http", "https"} and parsed.netloc.lower() == ws.headers.get("host", "").lower()
        except ValueError:
            allowed = False
        if not allowed:
            await ws.close(code=1008)
            return
    await ws.accept()
    hub.clients.add(ws)
    await send_case(ws)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                sess = rt.asr_sessions.get(id(ws))
                if sess and "q" in sess:
                    sess["q"].put_nowait(msg["bytes"])
                continue
            try:
                m = json.loads(msg.get("text") or "{}")
            except json.JSONDecodeError:
                continue
            t = m.get("type")
            if t == "cmd" and m.get("text", "").strip():
                asyncio.create_task(run_text(m["text"].strip(), ws, typed=True))
            elif t == "ptt":
                await (ptt_start(ws) if m.get("down") else ptt_stop(ws))
            elif t == "ui":
                patch = m.get("patch") or {}
                await hub.patch(patch, exclude=ws)
            elif t == "load_case":
                try:
                    rt.load(m["id"])
                    await send_case()
                except (FileNotFoundError, ValueError):
                    await hub.send(ws, "error", {"message": f"Unknown case {m.get('id')}"})
            elif t == "analyze":
                start_verdict()
            elif t == "blueprint":
                start_image()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)
        sess = rt.asr_sessions.pop(id(ws), None)
        if sess and "q" in sess:
            sess["q"].put_nowait(None)


@app.get("/")
def root():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")

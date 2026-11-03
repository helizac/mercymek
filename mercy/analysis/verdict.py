"""Muse Wonder structured fault report (section 9.6) + display mapping for the HUD verdict message."""
from __future__ import annotations

import json
import re
import pathlib
import time

from mercy import config
from mercy.analysis.physics import summary_text
from mercy.config import log
from mercy.meta_client import structured

VERDICT_SCHEMA = {"type": "object", "additionalProperties": False,
  "required": ["headline", "timeline", "key_measurements", "fault_assessment",
               "applicable_rules", "overall_confidence", "limitations"],
  "properties": {
    "headline": {"type": "string"},
    "timeline": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["t_s", "event", "evidence"], "properties": {
        "t_s": {"type": "number"}, "event": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}}}}},
    "key_measurements": {"type": "array", "items": {"type": "string"}},
    "fault_assessment": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["party", "share_pct_range", "rationale", "citations"], "properties": {
        "party": {"type": "string"},
        "share_pct_range": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}}}}},
    "applicable_rules": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["rule", "source_url"], "properties": {
        "rule": {"type": "string"}, "source_url": {"type": "string"}}}},
    "overall_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "limitations": {"type": "array", "items": {"type": "string"}}}}

SYSTEM = ("You are MERCY, a forensic reconstruction analyst. Use ONLY the provided evidence JSON, "
  "physics measurements and cited sources. Every fault statement must reference a measurement id "
  "(e.g. physics.v1.pre_impact_speed) or a source URL. Report ranges, never single exact speeds. State limitations "
  "of monocular video. Never use driver demographics, vehicle brand or price as a factor. "
  "This is an estimate for investigators, not a legal determination. Tone: cold, precise, terse. "
  "Write for a busy investigator: short, plain, meaningful sentences, no filler. The headline is one sentence of at "
  "most 14 words without measurement ids. Name each party by its label exactly as given "
  "(e.g. 'Blue Hatchback #7', 'Camera vehicle'). Fault share ranges across parties should straddle 100% in total. "
  "Each rationale: at most 2 sentences / 30 words, citing ids in parentheses. key_measurements: at most 4 items, each "
  "under 12 words. timeline: at most 5 events, each under 10 words. limitations: at most 3, each under 15 words. "
  "applicable_rules: at most 3, each paraphrased in under 20 words, with the research source URL it came from. "
  "Never assume facts the scene description does not state (signal colours, driver intent, distraction): if fault "
  "depends on an unseen signal state, say so in limitations and widen the fault ranges instead.")
LOW_QUALITY = ("gumtree.", "reddit.", "quora.", "facebook.", "pinterest.", "tiktok.")
CITE_ORDER = {"rule": 0, "statistics": 1, "reaction": 2, "vehicle": 3}


def build_input(scene: dict, physics: dict, research: dict, notes: list[str]) -> str:
    phys = {"impact": physics["impact"], "measurements": [
        {k: m[k] for k in ("id", "label", "value", "range", "unit", "method", "conf")} for m in physics["measurements"]],
        "parties": [{"id": o["id"], "label": o["label"], "role": o["role"], "measurement_prefix": o.get("vid")}
                    for o in physics["objects"] if o["role"] != "other"],
        "warnings": physics.get("warnings", [])}
    res = [{"topic": i["key"], "answer": i["answer"][:1500],
            "sources": [{"title": c["title"], "url": c["url"]} for c in i["citations"][:6]]} for i in research["items"]]
    return json.dumps({"scene": scene, "physics": phys, "research": res, "investigator_notes": notes})


def run_verdict(scene, physics, research, notes, effort="high", timeout=None, model=None, file_id=None,
                extra: dict | None = None) -> dict:
    """The model watches the clip itself (file_id) and reads the measured evidence alongside it."""
    model = model or config.REASONING_MODEL
    text = build_input(scene, physics, research, notes)
    if extra:
        text = text[:-1] + ", " + json.dumps(extra)[1:]
    content = [{"type": "input_text", "text": "Watch the clip and assess fault. Evidence JSON follows.\n" + text}]
    if file_id:
        content.append({"type": "input_file", "file_id": file_id})
    raw = structured(model, [{"role": "system", "content": SYSTEM},
                             {"type": "message", "role": "user", "content": content}],
                     VERDICT_SCHEMA, "Verdict", effort=effort, timeout=timeout, max_output_tokens=16000)
    return raw


def plain(text: str) -> str:
    """Display text for people: no measurement ids or URLs in brackets, whole numbers, 'Suv' -> 'SUV'."""
    t = re.sub(r"\s*\((?:[^()]*?(?:physics\.|v[12]\.|impact\.|scale\.|https?://)[^()]*)\)", "", text)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\b\d+\.\d+\b", lambda m: f"{float(m.group()):.1f}" if float(m.group()) < 10 else f"{round(float(m.group()))}", t)
    t = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1–\2", t)
    t = re.sub(r"\bSuv\b", "SUV", t)
    return re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s{2,}", " ", t)).strip()


def display(raw: dict, research: dict, integrity: dict) -> dict:
    """Map VERDICT_SCHEMA onto the HUD's verdict.data shape (section 11)."""
    fault = []
    for f in raw["fault_assessment"]:
        r = (f["share_pct_range"] + [0, 0])[:2]
        lo, hi = sorted(r)
        party = plain(f["party"].split(" — ", 1)[-1] if " — " in f["party"] else f["party"])   # drop "car:7 — " id prefix
        fault.append({"party": party,"pct": round((lo + hi) / 2), "lo": lo, "hi": hi, "rationale": plain(f["rationale"]),
                      "citations": f["citations"]})
    fault.sort(key=lambda x: -x["pct"])
    urls = {}
    for rule in raw["applicable_rules"]:
        if rule["source_url"].startswith("http"):
            urls.setdefault(rule["source_url"], rule["rule"][:90])
    for it in sorted(research.get("items", []), key=lambda i: CITE_ORDER.get(i["key"], 9)):
        for c in it["citations"]:
            urls.setdefault(c["url"], f"{c['title']} · {it['key']}")
    citations = [{"title": t, "url": u} for u, t in urls.items() if not any(b in u for b in LOW_QUALITY)][:10]
    factors = [{"text": plain(m), "weight": "high" if i < 2 else "medium" if i < 4 else "low"}
               for i, m in enumerate(raw["key_measurements"][:6])]
    return {"headline": plain(raw["headline"]), "fault": fault, "confidence": raw["overall_confidence"], "factors": factors,
            "citations": citations, "rules": raw["applicable_rules"], "timeline": raw["timeline"],
            "integrity": {k: integrity[k] for k in ("passed", "total", "note", "review_required", "checks")
                          if k in integrity},
            "caveats": [plain(x) for x in raw["limitations"][:5]]}


def run(case: pathlib.Path, notes: list[str] | None = None, force: bool = False, effort: str = "high",
        timeout: float | None = None, write: bool = True) -> dict:
    out = case / "verdict.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    from mercy.analysis import integrity as integ
    scene = json.loads((case / "scene.json").read_text())
    physics = json.loads((case / "physics.json").read_text())
    research = json.loads((case / "research.json").read_text())
    extra = {}
    for name in ("plates.json", "location.json"):
        if (case / name).exists():
            d = json.loads((case / name).read_text(encoding="utf-8"))
            extra[name.split(".")[0]] = ([{k: p.get(k) for k in ("vehicle", "text", "legible", "confidence")}
                                          for p in d.get("plates", [])] if name == "plates.json"
                                         else {k: d.get(k) for k in ("place_name", "radius_m", "confidence", "road")})
    try:
        from mercy.ingest.scene import _file_id
        file_id = _file_id(case)
    except Exception as e:
        log.warning("clip upload failed (%s); verdict from evidence only", e); file_id = None
    t0 = time.time()
    model = config.REASONING_MODEL
    try:
        raw = run_verdict(scene, physics, research, notes or [], effort=effort, timeout=timeout, file_id=file_id,
                          extra=extra)
    except Exception as e:
        if timeout is not None:
            raise
        log.warning("verdict on %s failed (%s); retrying on %s", model, e, config.VISION_MODEL)
        model = config.VISION_MODEL
        raw = run_verdict(scene, physics, research, notes or [], effort=effort, model=model, file_id=file_id, extra=extra)
    integrity = integ.check(raw, physics)
    res = {"raw": raw, "display": display(raw, research, integrity), "integrity": integrity, "model": model,
           "created": time.time(), "latency_s": round(time.time() - t0, 1), "notes": notes or []}
    if write:
        out.write_text(json.dumps(res, indent=1))
    log.info("verdict (%s, %.1fs): %s | integrity %d/%d", model, res["latency_s"], raw["headline"],
             integrity["passed"], integrity["total"])
    return res


__all__ = ["run", "display", "VERDICT_SCHEMA", "summary_text"]

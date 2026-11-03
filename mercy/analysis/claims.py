"""Mercy mode: the investigator argues, the judge weighs each argument against the evidence.

Each spoken claim is checked against the case facts by the vision model (short, cheap call). A supported claim moves
the liability meter of the party it concerns by a bounded likelihood ratio in log-odds; a contradicted or
unverifiable claim does not move it, and the judge says why. The ledger (claims.json) is passed to the next verdict
as investigator notes, so arguments that hold up end up in the report.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

from mercy import config
from mercy.meta_client import structured

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["assessment", "party", "direction", "strength", "reply"],
          "properties": {
              "assessment": {"type": "string", "enum": ["supported", "contradicted", "unverifiable"]},
              "party": {"type": "string"},
              "direction": {"type": "string", "enum": ["more_at_fault", "less_at_fault", "no_change"]},
              "strength": {"type": "string", "enum": ["weak", "moderate", "strong"]},
              "reply": {"type": "string"}}}

SYSTEM = ("You are the judge in a crash investigation. The investigator states a claim. Decide ONLY from the case facts "
          "whether the evidence supports it, contradicts it, or cannot tell. 'party' is the exact party label from the "
          "facts whose fault the claim concerns. 'direction' is how a supported claim changes that party's fault. "
          "Strength: strong only if a measurement directly shows it. Reply out loud in at most 2 short sentences "
          "(30 words), plain words, no ids, cold and precise, citing the decisive fact.")

LR = {"weak": 1.25, "moderate": 1.7, "strong": 2.6}
CLAMP = (3.0, 97.0)


def logit(p: float) -> float:
    p = min(max(p, 0.5), 99.5) / 100
    return math.log(p / (1 - p))


def sigm(x: float) -> float:
    return 100 / (1 + math.exp(-x))


def meter_start(verdict: dict | None) -> list[dict]:
    """Two-party meter from the verdict's fault shares (midpoints, renormalised to 100)."""
    fault = (verdict or {}).get("display", {}).get("fault") or []
    if len(fault) < 2:
        return []
    a, b = fault[0], fault[1]
    tot = max(1, a["pct"] + b["pct"])
    pa = round(100 * a["pct"] / tot, 1)
    return [{"party": a["party"], "pct": pa}, {"party": b["party"], "pct": round(100 - pa, 1)}]


def load(case: pathlib.Path) -> dict:
    p = case / "claims.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    v = json.loads((case / "verdict.json").read_text()) if (case / "verdict.json").exists() else None
    return {"meter": meter_start(v), "start": meter_start(v), "ledger": []}


def match_party(name: str, meter: list[dict]) -> int | None:
    n = name.lower()
    for i, m in enumerate(meter):
        if m["party"].lower() in n or n in m["party"].lower():
            return i
    words = set(n.replace("#", " ").split())
    best = max(range(len(meter)), key=lambda i: len(words & set(meter[i]["party"].lower().replace("#", " ").split())),
               default=None)
    if best is not None and words & set(meter[best]["party"].lower().replace("#", " ").split()) - {"the", "car", "vehicle"}:
        return best
    return None


def judge(case: pathlib.Path, facts: str, claim: str, timeout: float = 25) -> dict:
    state = load(case)
    meter = state["meter"]
    parties = ", ".join(m["party"] for m in meter) or "unknown"
    r = structured(config.VISION_MODEL, [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": f"Case facts:\n{facts}\nParties: {parties}\n\nClaim: {claim}"}],
                   SCHEMA, "Ruling", effort="low", max_output_tokens=1500, timeout=timeout)
    before = [m["pct"] for m in meter]
    i = match_party(r["party"], meter) if meter else None
    moved = 0.0
    if meter and i is not None and r["assessment"] == "supported" and r["direction"] != "no_change":
        step = math.log(LR[r["strength"]]) * (1 if r["direction"] == "more_at_fault" else -1)
        new = min(max(sigm(logit(meter[i]["pct"]) + step), CLAMP[0]), CLAMP[1])
        moved = round(new - meter[i]["pct"], 1)
        meter[i]["pct"] = round(new, 1)
        meter[1 - i]["pct"] = round(100 - new, 1)
    entry = {"claim": claim, **r, "moved": moved, "before": before, "after": [m["pct"] for m in meter], "at": time.time()}
    state["ledger"].append(entry)
    (case / "claims.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    return {"meter": meter, "start": state["start"], "entry": entry, "ledger": state["ledger"][-8:]}


def as_notes(case: pathlib.Path) -> list[str]:
    """Supported arguments become investigator notes for the next verdict."""
    st = load(case)
    return [f"Investigator argument ({e['assessment']}, {e['strength']}): {e['claim']}" for e in st["ledger"]
            if e["assessment"] == "supported"]

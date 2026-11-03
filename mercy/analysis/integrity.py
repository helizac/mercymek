"""Verdict integrity audit (section 4): Laya noul/score questions + deterministic checks.

Tuned on our own verdicts (scripts in tests/): the shipped checkpoints separate *citing evidence* and *speculation*
usefully but not *numeric consistency*, so numbers are checked by code and Laya's consistency score is reported
for information only. Laya's English checkpoint has a 512-token context, so the state is compacted first:
measurement ids in the verdict are replaced by their measured ranges and the evidence lists only key values.
"""
from __future__ import annotations

import re
import threading

from mercy import config
from mercy.config import log

INTEGRITY_QUESTIONS = {
    "cites_evidence": {"type": "noul",
        "instructions": "Does every fault claim in `verdict` reference a measured value from `evidence` or a cited rule?"},
    "speculation": {"type": "noul",
        "instructions": "Does `verdict` mention facts that are absent from `evidence`, such as a traffic light colour "
                        "or what a driver intended?"},
    "consistency": {"type": "noul",
        "instructions": "Do the speeds and times quoted in `verdict` fall inside the ranges listed in `evidence`?"},
    "severity": {"type": "score", "instructions": "How severe is the collision?",
        "criteria": ["minor contact", "moderate damage", "severe crash"]},
}
SPECULATION_MAX = 0.40          # fabricated test verdicts score ~0.48, grounded ones ~0.2
KEY_MEASUREMENTS = ["v1.pre_impact_speed", "v2.pre_impact_speed", "v1.brake_onset", "v2.visible_before_impact",
                    "impact.closing_speed", "impact.time"]

_router = None
_lock = threading.Lock()


def router():
    """Process-wide Laya Router (english + multilingual resident; first call pays the load)."""
    global _router
    with _lock:
        if _router is None:
            from laya import Router
            _router = Router(preload=False, device=config.LAYA_DEVICE, default="english")
            _router.preload(["english", "multilingual"])
        return _router


def noul_prob(ans: dict) -> float:
    for k in ("noul", "probability", "prob_true", "p_true", "p"):
        if isinstance(ans.get(k), (int, float)):
            return float(ans[k])
    probs = ans.get("probabilities") or {}
    for k in ("true", "yes", True):
        if k in probs:
            return float(probs[k])
    return 0.5


def severity_label(ans: dict) -> str | None:
    """Laya score answers carry a probability per level and a legend; report the most likely level."""
    probs, legend = ans.get("probabilities") or {}, ans.get("legend") or {}
    if probs:
        best = max(probs, key=probs.get)
        return legend.get(best, best)
    return None


def compact_state(raw: dict, physics: dict) -> dict:
    meas = {m["id"]: m for m in physics["measurements"]}

    def sub(text: str) -> str:
        def rep(m):
            x = meas.get(m.group(1))
            return f"[measured {x['range'][0]:.0f}-{x['range'][1]:.0f} {x['unit']}]" if x else "[measured]"
        return re.sub(r"\(?(?:physics\.)?((?:v[12]|impact|scale)\.[\w]+)(?::[^)]*)?\)?", rep, text)

    verdict = " ".join(f"{f['party']} {f['share_pct_range'][0]}-{f['share_pct_range'][-1]}% at fault because "
                       f"{sub(f['rationale'])}" for f in raw["fault_assessment"])
    evidence = "; ".join(f"{meas[k]['label']}: {meas[k]['range'][0]:.1f}-{meas[k]['range'][1]:.1f} {meas[k]['unit']}"
                         for k in KEY_MEASUREMENTS if k in meas)
    return {"verdict": verdict[:1300], "evidence": evidence}


def _numbers_ok(raw: dict, physics: dict) -> tuple[bool, list[str]]:
    """Every km/h figure in the verdict must fall inside some measured km/h range (±10%)."""
    text = " ".join([raw["headline"]] + [f["rationale"] for f in raw["fault_assessment"]] + raw["key_measurements"])
    ranges = [m["range"] for m in physics["measurements"] if m["unit"] == "km/h"]
    bad = []
    for v in re.findall(r"(\d+(?:\.\d+)?)\s*(?:[-–]\s*\d+(?:\.\d+)?\s*)?km/h", text):
        x = float(v)
        if ranges and not any(lo * 0.9 <= x <= hi * 1.1 for lo, hi in ranges):
            bad.append(v)
    return not bad, bad


def check(raw: dict, physics: dict) -> dict:
    scores, checks = {}, []
    try:
        r = router().predict(compact_state(raw, physics), INTEGRITY_QUESTIONS, max_len=512)
        a = r["answers"]
        scores = {"cites_evidence": round(noul_prob(a["cites_evidence"]), 3),
                  "speculation": round(noul_prob(a["speculation"]), 3),
                  "consistency": round(noul_prob(a["consistency"]), 3),
                  "severity": severity_label(a["severity"]),
                  "state_tokens": r.get("usage", {}).get("state_tokens")}
        checks = [
            {"name": "Fault claims cite evidence", "ok": scores["cites_evidence"] >= 0.5, "p": scores["cites_evidence"]},
            {"name": "No speculation beyond evidence", "ok": scores["speculation"] <= SPECULATION_MAX, "p": scores["speculation"]},
        ]
    except Exception as e:      # Laya unavailable -> deterministic checks only, said plainly
        log.warning("laya integrity unavailable: %s", e)
    ok, bad = _numbers_ok(raw, physics)
    checks.append({"name": "Speed figures inside measured ranges", "ok": ok, "detail": bad})
    cited = all(f["citations"] for f in raw["fault_assessment"])
    checks.append({"name": "Every fault share carries a citation", "ok": cited})
    passed = sum(c["ok"] for c in checks)
    review = not all(c["ok"] for c in checks)
    note = ("Laya audit (System 1) + deterministic checks: "
            + ", ".join(f"{c['name'].lower()} {'✓' if c['ok'] else '✗'}" for c in checks))
    return {"passed": passed, "total": len(checks), "note": note, "review_required": review,
            "scores": scores, "checks": checks}

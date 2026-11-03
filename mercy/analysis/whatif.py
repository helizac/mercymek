"""Counterfactuals ("what if"): avoidance analysis on the measured case.

From the moment the other vehicle first became visible, the camera car is re-driven along its own measured path
with a different speed, reaction time or braking, holding everything else as measured (a conservative, standard
avoidance analysis: same position when the hazard appeared, same conflict point). Every answer is a range over the
friction and reaction-time ranges, plus the counterfactual trajectory of the camera car ("ghost").
"""
from __future__ import annotations

import math
import re
from typing import Optional

G = 9.81
ALERT_REACTION = (0.7, 1.0)     # s, an alert driver who reacts immediately
TYPICAL_REACTION = (1.0, 2.0)   # s, the range physics.py already uses for stopping distances


def _m(physics: dict, mid: str) -> Optional[dict]:
    return next((m for m in physics.get("measurements", []) if m["id"] == mid), None)


def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            u = (x - xs[i - 1]) / max(1e-9, xs[i] - xs[i - 1])
            return ys[i - 1] + u * (ys[i] - ys[i - 1])
    return ys[-1]


def parse(clause: str) -> dict:
    """Spoken counterfactual -> parameters. Unknown pieces stay as measured."""
    lc = clause.lower()
    p: dict = {}
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(km/?h|kph|kilomet|k\b|mph|miles)", lc)
    if m:
        v = float(m.group(1).replace(",", "."))
        p["speed_kmh"] = v * 1.609 if m.group(2).startswith(("mph", "miles")) else v
    elif re.search(r"speed limit|limit", lc):
        p["speed_kmh"] = "limit"
    m = re.search(r"(\d+(?:[.,]\d+)?|half a|one|two|a)\s*(?:s\b|sec\w*|second\w*)\s*(earlier|sooner|faster)", lc)
    if m:
        p["earlier_s"] = {"half a": 0.5, "one": 1.0, "a": 1.0, "two": 2.0}.get(m.group(1)) or float(m.group(1).replace(",", "."))
    elif re.search(r"immediately|instantly|right away|paying attention|alert|reacted (in time|faster|quicker)", lc):
        p["reaction"] = "alert"
    if re.search(r"\bwet|rain", lc):
        p["mu"] = (0.4, 0.55)
    elif re.search(r"\bsnow|\bice|icy", lc):
        p["mu"] = (0.1, 0.3)
    elif re.search(r"\bdry\b", lc):
        p["mu"] = (0.7, 0.8)
    if re.search(r"(had not|hadn't|didn't|did not|never) (brake|braked)|no brak", lc):
        p["no_brake"] = True
    return p


def path_arclength(xz: list) -> list[float]:
    s = [0.0]
    for a, b in zip(xz, xz[1:]):
        s.append(s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return s


def simulate(v: float, tr: float, a: float, D: float, no_brake: bool = False) -> dict:
    """Constant speed v (m/s) for tr seconds, then braking at a (m/s²). D = distance to the conflict point."""
    if no_brake or a <= 0:
        return {"stops": False, "margin": -D, "impact_v": v, "t_arrive": D / max(v, 0.1)}
    stop = v * tr + v * v / (2 * a)
    if stop <= D:
        return {"stops": True, "margin": D - stop, "impact_v": 0.0, "t_arrive": None}
    db = D - v * tr
    if db <= 0:
        return {"stops": False, "margin": D - stop, "impact_v": v, "t_arrive": D / max(v, 0.1)}
    vi = math.sqrt(max(0.0, v * v - 2 * a * db))
    return {"stops": False, "margin": D - stop, "impact_v": vi, "t_arrive": tr + (v - vi) / a}


def run(physics: dict, clause: str, limit_kmh: Optional[float] = None) -> dict:
    p = parse(clause)
    ego = physics["series"]["objects"]["ego"]
    t_imp = float(physics["impact"]["t"])
    vis = _m(physics, "v2.visible_before_impact")
    brake = _m(physics, "v1.brake_onset")
    # hazard onset: first visibility, but a vehicle seen for the whole clip is not a hazard yet; cap the
    # perception-reaction time at 1.5 s before the measured brake onset
    before = vis["value"] if vis else 2.0
    if brake:
        before = min(before, brake["value"] + 1.5)
    t_vis = t_imp - min(before, 4.0)
    ts, xz, sp = ego["t"], ego["xz"], ego["speed_kmh"]
    s = path_arclength(xz)
    s_vis, s_imp = _interp(ts, s, t_vis), _interp(ts, s, t_imp)
    D = max(0.5, s_imp - s_vis)
    near = [v for t, v in zip(ts, sp) if t_vis - 0.5 <= t <= t_vis + 0.3] or [float(_interp(ts, sp, t_vis))]
    v_actual = float(sorted(near)[len(near) // 2])
    actual_tr = max(0.3, (t_imp - t_vis) - brake["value"]) if brake else 1.5

    v_kmh = p.get("speed_kmh", v_actual)
    if v_kmh == "limit":
        v_kmh = limit_kmh or 50.0
    mu = p.get("mu") or tuple(physics.get("weather_mu", {}).get("mu", (0.7, 0.8)))
    # the driver's own braking effort, calibrated so that "as driven" reproduces the measured impact speed
    pre = _m(physics, "v1.pre_impact_speed")
    v_hit = min(pre["value"] if pre else v_actual, float(_interp(ts, sp, t_imp - 0.2))) / 3.6
    v0 = v_actual / 3.6
    a_eff = (v0 * v0 - v_hit * v_hit) / (2 * max(1.0, D - v0 * actual_tr))
    hard = bool(re.search(r"hard|full|emergency|slam", clause.lower()))
    a_eff = max(0.1, a_eff)
    a_lo = min(mu[0] * G, a_eff * 0.85) if not hard else max(mu[0] * G, min(a_eff, mu[1] * G))
    a_hi = min(mu[1] * G, a_eff * 1.15) if not hard else max(mu[1] * G, a_eff)        # never weaker than measured
    note = ""
    if "earlier_s" in p:
        trs = (max(0.2, actual_tr - p["earlier_s"]),) * 2
    elif p.get("reaction") == "alert":
        trs = (min(actual_tr, ALERT_REACTION[0]), min(actual_tr, ALERT_REACTION[1]))
        if actual_tr <= ALERT_REACTION[0]:
            note = f"The camera car already braked {actual_tr:.1f} seconds after the other vehicle appeared. "
    else:
        trs = (actual_tr, actual_tr)
    v = v_kmh / 3.6
    base = simulate(v0, actual_tr, a_eff, D)                      # as driven
    # best and worst case over braking and reaction ranges
    best = simulate(v, trs[0], a_hi, D, p.get("no_brake", False))
    worst = simulate(v, trs[1], a_lo, D, p.get("no_brake", False))
    if best["stops"] and worst["stops"]:
        verdict = "avoided"
    elif best["stops"] or (worst["impact_v"] * 3.6 < 8):
        verdict = "borderline"
    else:
        verdict = "not avoided"
    real = {"speed_kmh": round(v_actual), "reaction_s": round(actual_tr, 2), "distance_m": round(D, 1),
            "decel_ms2": round(a_eff, 1), "impact_kmh": round(base["impact_v"] * 3.6)}

    # ghost: the camera car under the counterfactual (median friction / reaction), sampled like the series
    a_mid, tr_mid = (a_lo + a_hi) / 2, (trs[0] + trs[1]) / 2
    gt, gxz, pos, vel, prev = [], [], s_vis, v, t_vis
    for t in ts:
        if t < t_vis:
            gt.append(t); gxz.append(xz[ts.index(t)]); continue
        dt = t - prev; prev = t
        if not p.get("no_brake") and t - t_vis > tr_mid:
            vel = max(0.0, vel - a_mid * dt)
        pos += vel * dt
        gt.append(t)
        gxz.append([round(_interp(s, [q[0] for q in xz], min(pos, s[-1])), 3),
                    round(_interp(s, [q[1] for q in xz], min(pos, s[-1])), 3)])

    change = []
    if "speed_kmh" in p:
        change.append(f"at {v_kmh:.0f} km/h instead of {v_actual:.0f}")
    if "earlier_s" in p:
        change.append(f"reacting {p['earlier_s']:g} s earlier")
    elif p.get("reaction") == "alert":
        change.append("reacting immediately")
    if p.get("no_brake"):
        change.append("without braking")
    if hard:
        change.append("braking as hard as the road allows")
    if p.get("mu"):
        change.append("on a " + ("wet" if p["mu"][0] == 0.4 else "snowy" if p["mu"][0] == 0.1 else "dry") + " road")
    what = ", ".join(change) or "as driven"
    miss = physics["impact"].get("kind") == "near-miss"
    point = "the point of closest approach" if miss else "the impact point"
    if verdict == "avoided":
        say = (f"{what.capitalize()}, the camera car stops {rng(max(0, worst['margin']), best['margin'])} metres before "
               f"{point}. " + ("It never gets close." if miss else "The crash is avoided."))
    elif verdict == "borderline":
        say = (f"{what.capitalize()}, it is borderline: the camera car either stops just short or reaches {point} at up to "
               f"{worst['impact_v'] * 3.6:.0f} kilometres per hour.")
    else:
        lo, hi = sorted([best["impact_v"] * 3.6, worst["impact_v"] * 3.6])
        say = (f"{what.capitalize()}, the camera car still reaches {point} at {rng(lo, hi)} kilometres per hour, "
               f"instead of about {base['impact_v'] * 3.6:.0f}.")
    if note and abs((best["impact_v"] + worst["impact_v"]) / 2 - base["impact_v"]) * 3.6 < 4:
        say = "A faster reaction would not change the outcome."
    say = note + say
    return {"params": p, "what": what, "verdict": verdict, "say": say, "real": real,
            "reaction_s": [round(trs[0], 2), round(trs[1], 2)], "mu": list(mu), "speed_kmh": round(v_kmh, 1),
            "impact_kmh": sorted([round(best["impact_v"] * 3.6, 1), round(worst["impact_v"] * 3.6, 1)]),
            "margin_m": sorted([round(best["margin"], 1), round(worst["margin"], 1)]),
            "t_hazard": round(t_vis, 2), "ghost": {"t": gt, "xz": gxz},
            "method": "avoidance analysis from first visibility of the other vehicle along the measured path; "
                      "stopping = v·t_r + v²/(2·μ·g)"}


def rng(lo: float, hi: float) -> str:
    lo, hi = sorted([round(lo), round(hi)])
    return f"about {lo}" if hi - lo < 2 else f"{lo} to {hi}"

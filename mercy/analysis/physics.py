"""Deterministic physics from 3D tracks (section 9.2-9.4). LLMs never compute these numbers.

physics.json stores every number as {value, range, unit, method, conf}; series feed the HUD timeline.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.signal import medfilt, savgol_filter

from mercy import config
from mercy.config import log
from mercy.ingest.lift import BODY_LENGTH, box_iou

G = 9.81
MU = {"dry": (0.7, 0.8), "wet": (0.4, 0.5), "snow": (0.15, 0.3), "unknown": (0.5, 0.8)}
PRT = (1.0, 1.5, 2.0)          # perception-reaction time range (s), common forensic planning values
EGO_FRONT = 2.0                # camera (windshield) to front bumper (m)


def kinematics(centers: np.ndarray, t: np.ndarray, window: int = 9):
    """centers [N,3] metres at times t. Returns speed [N] m/s, acc [N] m/s², vel [N,3], smoothed positions."""
    centers = np.asarray(centers, float); t = np.asarray(t, float)
    n = len(centers)
    w = min(window, n if n % 2 else n - 1)
    p = savgol_filter(centers, window_length=w, polyorder=2, axis=0) if w >= 5 else centers
    vel = np.gradient(p, t, axis=0)
    speed = np.linalg.norm(vel[:, [0, 2]], axis=1)             # road plane (x, z)
    sp_s = savgol_filter(speed, window_length=w, polyorder=2) if w >= 5 else speed
    acc = np.gradient(sp_s, t)
    return speed, acc, vel, p


def display_path(P: np.ndarray) -> np.ndarray:
    """Road-plane path for drawing only: median-filter outlier repair (monocular depth spikes), then Savitzky-Golay."""
    X = np.asarray(P, float)[:, [0, 2]].copy()
    n = len(X)
    if n < 5:
        return X
    k = 5 if n >= 5 else 3
    med = np.stack([medfilt(X[:, j], k) for j in range(2)], 1)
    r = np.linalg.norm(X - med, axis=1)
    bad = r > 3 * (np.median(r) + 1e-6)
    X[bad] = med[bad]
    w = min(11, n if n % 2 else n - 1)
    return savgol_filter(X, w, 2, axis=0) if w >= 5 else X


def robust_velocity(t, P):
    """Least-squares constant-velocity fit in the road plane with MAD outlier rejection.
    Returns (velocity [vx, vz] m/s, residual spread m)."""
    t = np.asarray(t, float); X = np.asarray(P, float)[:, [0, 2]]
    keep = np.ones(len(t), bool)
    for _ in range(2):
        A = np.vstack([t[keep], np.ones(keep.sum())]).T
        coef, *_ = np.linalg.lstsq(A, X[keep], rcond=None)
        r = np.linalg.norm(X - (np.outer(t, coef[0]) + coef[1]), axis=1)
        mad = np.median(r[keep]) + 1e-6
        nk = r < 3 * mad
        if nk.sum() < 3 or (nk == keep).all():
            break
        keep = nk
    return coef[0], float(np.median(r[keep]))


def rect_point_dist(lo, hi, p) -> float:
    dx = max(lo[0] - p[0], 0, p[0] - hi[0]); dz = max(lo[1] - p[1], 0, p[1] - hi[1])
    return float(np.hypot(dx, dz))


def rect_rect_dist(alo, ahi, blo, bhi) -> float:
    dx = max(blo[0] - ahi[0], alo[0] - bhi[0], 0); dz = max(blo[1] - ahi[1], alo[1] - bhi[1], 0)
    return float(np.hypot(dx, dz))


def M(id, label, value, lo, hi, unit, method, conf, **kw):
    lo, hi = min(lo, hi), max(lo, hi)
    return {"id": id, "label": label, "value": round(float(value), 2), "range": [round(float(lo), 2), round(float(hi), 2)],
            "unit": unit, "method": method, "conf": conf, "estimate": True, **kw}


def _window(t, a, b):
    return (t >= a) & (t <= b)


def choose_objects(tracks: dict, overlays: dict, meta: dict, scene: dict, impact_label: float, aliases: dict):
    """Pick the involved vehicle(s): V1 = camera vehicle when involved, V2 = the vehicle that dominates the
    frame around the labelled impact (2D mask area x proximity to the bottom centre), preferring a Spark-labelled one."""
    W, H, fps = meta["width"], meta["height"], meta["src_fps"]
    f0, f1 = int((impact_label - 1.5) * fps), int((impact_label + 0.3) * fps)
    score: dict[str, float] = {}
    pair_overlap: dict[tuple, float] = {}
    for f in range(max(0, f0), f1 + 1):
        objs = {k: o for k, o in overlays.get(str(f), {}).items() if k in tracks}
        for k, o in objs.items():
            x1, y1, x2, y2 = o["box"]
            cx, by = (x1 + x2) / 2 / W, y2 / H
            s = o.get("area", (x2 - x1) * (y2 - y1)) / (W * H) * (1.0 - abs(cx - 0.5)) * (0.5 + by)
            score[k] = max(score.get(k, 0), s)
        ks = list(objs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                v = box_iou(objs[ks[i]]["box"], objs[ks[j]]["box"])
                if v > 0:
                    key = tuple(sorted((ks[i], ks[j])))
                    pair_overlap[key] = max(pair_overlap.get(key, 0), v)
    for k, a in aliases.items():                   # Spark named it -> it matters
        if k in score:
            score[k] *= 1.5
    ranked = sorted(score, key=score.get, reverse=True)
    # Spark's roles decide the parties: one striking/struck vehicle => conflict with the camera vehicle
    # (a near-miss can have camera_vehicle_involved=false and still be ego vs. that vehicle).
    parties = [v["label"] for v in scene.get("vehicles", []) if v["role"] in ("striking", "struck")]
    named=[]
    for party in parties:
        candidates=[k for k,a in aliases.items() if a==party and k in tracks and k not in named]
        if candidates:named.append(max(candidates,key=lambda k:score.get(k,0)))
    if not scene.get("camera_vehicle_involved", True) and len(parties) >= 2 and len(named) >= 2:
        return named[:2], ranked
    if scene.get("camera_vehicle_involved", True) or len(parties) <= 1 or not pair_overlap:
        other = named[:1] or ranked[:1]
        return ["ego"] + other, [k for k in ranked if k not in other] + other
    pair = max(pair_overlap, key=lambda p: pair_overlap[p] + score.get(p[0], 0) + score.get(p[1], 0))
    return list(pair), ranked


def analyze(case: pathlib.Path, force: bool = False) -> dict:
    out_path = case / "physics.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    meta = json.loads((case / "meta.json").read_text())
    scene = json.loads((case / "scene.json").read_text())
    from mercy.ingest.incident_tracks import ensure
    tr = ensure(case)
    overlays = json.loads((case / "overlays.json").read_text())
    t_all = np.array(tr["times"])
    s_mid = tr["scale"]["value"]; s_lo, s_hi = tr["scale"]["range"]
    k_lo, k_hi = s_lo / s_mid, s_hi / s_mid
    weather = scene.get("weather", "unknown")
    if meta.get("weather", "").lower() in ("rain", "snow"):
        weather = "wet" if meta["weather"].lower() == "rain" else "snow"
    mu_lo, mu_hi = MU.get(weather, MU["unknown"])
    warnings = []

    impact_label = float(meta["time_of_event"])
    impact_spark = scene.get("impact_time_s")
    involved, ranked = choose_objects(tr["tracks"], overlays, meta, scene, impact_label, tr.get("aliases", {}))
    estimated=[key for key in involved if key in tr.get('track_seeds',{})]
    if estimated:
        k_lo,k_hi=min(k_lo,.5),max(k_hi,1.5)
        warnings.append('Sparse neural depth missed an incident party; '+', '.join(estimated)+
                        ' uses source-box/class-size geometry, not measured depth. Metric uncertainty is widened.')

    # ---- per-object kinematics
    def obj_series(key):
        if key == "ego":
            t = t_all; P = np.array([e["pos"] for e in tr["ego"]]); win = 9
            lo = P[:, [0, 2]] - 0.9; hi = P[:, [0, 2]] + 0.9
            lo[:, 1] = P[:, 2] - 2.6; hi[:, 1] = P[:, 2] + EGO_FRONT          # footprint around the camera
        else:
            rows = tr["tracks"][key]
            t = np.array([r["t"] for r in rows]); P = np.array([r["center"] for r in rows]); win = 9
            lo = np.array([[r["bbox"][0][0], r["bbox"][0][2]] for r in rows])
            hi = np.array([[r["bbox"][1][0], r["bbox"][1][2]] for r in rows])
        speed, acc, vel, Ps = kinematics(P, t, win)
        disp = Ps[:, [0, 2]] if key == "ego" else display_path(P)
        if key == "ego":   # acceleration from a longer window: pose jitter dominates short windows
            sp = savgol_filter(speed, min(15, len(t) // 2 * 2 - 1), 2) if len(t) >= 9 else speed
            acc = np.gradient(sp, t)
        resid = np.linalg.norm((P - Ps)[:, [0, 2]], axis=1)
        noise = float(np.sqrt(2) * np.median(resid) / max(np.median(np.diff(t)), 1e-3) / 3) if len(t) > 3 else 2.0
        return {"t": t, "P": Ps, "raw": P, "disp": disp, "speed": speed, "acc": acc, "vel": vel, "lo": lo, "hi": hi, "noise": noise}

    S = {k: obj_series(k) for k in set(involved) | set(ranked[:6]) | {"ego"} if k == "ego" or k in tr["tracks"]}
    a, b = involved[0], involved[1] if len(involved) > 1 else None

    # ---- geometric impact time: min footprint gap between the two parties (+ ego peak decel)
    impact_geo, gap_series = None, []
    if b is not None:
        A, B = S[a], S[b]
        common = np.intersect1d(np.round(A["t"], 3), np.round(B["t"], 3))
        for tt in common:
            i = int(np.argmin(abs(A["t"] - tt))); j = int(np.argmin(abs(B["t"] - tt)))
            if a == "ego":
                front = A["P"][i][[0, 2]] + np.array([0, EGO_FRONT])
                g = rect_point_dist(B["lo"][j], B["hi"][j], front)
            else:
                g = rect_rect_dist(A["lo"][i], A["hi"][i], B["lo"][j], B["hi"][j])
            gap_series.append((float(tt), g))
        if gap_series:
            g = np.array(gap_series)
            near = g[np.abs(g[:, 0] - impact_label) < 2.0]
            if len(near):
                impact_geo = float(near[np.argmin(near[:, 1]), 0])
    cands = [x for x in (impact_label, impact_spark) if x is not None]
    if impact_geo is not None and all(abs(impact_geo - c) <= 0.5 for c in cands):
        impact_t, impact_method = impact_geo, "geometry (min gap), agrees with label and Muse Spark within 0.5 s"
    else:
        impact_t = float(np.mean(cands))
        impact_method = "mean of dataset label and Muse Spark (geometry disagreed by > 0.5 s)"
        if impact_geo is not None:
            warnings.append(f"Impact-time sources disagree: label {impact_label:.2f}s, Spark {impact_spark}, geometry {impact_geo:.2f}s")
    spread = [x for x in (impact_label, impact_spark, impact_geo) if x is not None]

    # ---- measurements
    ms = []
    labels = {"ego": "Camera vehicle"}
    for k in S:
        if k != "ego":
            labels[k] = tr.get("aliases", {}).get(k, k.split(":")[0]).title() + f" #{k.split(':')[1]}"

    def pre_impact(key):
        s = S[key]; w = _window(s["t"], impact_t - 1.2, impact_t - 0.2)
        if w.sum() < 3:
            w = _window(s["t"], impact_t - 2.0, impact_t - 0.1)
        return s, w

    vids = {a: "v1", b: "v2"} if b else {a: "v1"}
    pre_v, pre_vel = {}, {}
    for key, vid in vids.items():
        s, w = pre_impact(key)
        if w.sum() < 3:
            continue
        vel, resid = robust_velocity(s["t"][w], s["P"][w] if key == "ego" else s["raw"][w])
        v = float(np.linalg.norm(vel))
        e = (0.4 if key == "ego" else 1.5) + resid / max(1.0, s["t"][w][-1] - s["t"][w][0])
        pre_v[key], pre_vel[key] = v, vel
        conf = "medium" if key == "ego" else "low"
        ms.append(M(f"{vid}.pre_impact_speed", f"{labels[key]} · speed before impact", v * 3.6,
                    (v * k_lo - e) * 3.6, (v * k_hi + e) * 3.6, "km/h",
                    "robust linear fit of the 3D track over [impact-1.2 s, impact-0.2 s]; scale from camera height",
                    conf))
    brake_t = None
    if "ego" in S:
        s = S["ego"]
        w3 = _window(s["t"], impact_t - 3.0, impact_t - 0.3)
        if w3.sum() >= 3:
            amin = float(np.min(s["acc"][w3]))
            limit = min(1.05, mu_hi * 1.3) * G           # ABS on dry asphalt reaches ~1 g; nothing brakes harder
            if -amin > limit:
                warnings.append(f"Camera-vehicle deceleration estimate {-amin:.1f} m/s² exceeds the friction limit; "
                                "pose jitter suspected, value clipped.")
                ms.append(M("v1.peak_decel", "Camera vehicle · peak deceleration (pre-impact)", mu_hi * G,
                            mu_lo * G * 0.6, mu_hi * G, "m/s²",
                            f"raw estimate {-amin:.1f} m/s² exceeded μ·g; clipped to the friction limit", "low"))
            else:
                ms.append(M("v1.peak_decel", "Camera vehicle · peak deceleration (pre-impact)", max(0.0, -amin),
                            max(0.0, -amin * k_lo - 1.5), min(limit, -amin * k_hi + 1.5), "m/s²",
                            "minimum of smoothed d(speed)/dt in [impact-3 s, impact-0.3 s]", "low"))
        idx = np.flatnonzero(_window(s["t"], impact_t - 4.0, impact_t + 0.05))
        onset = None
        for j in range(len(idx) - 2):     # sustained: three consecutive frames below -3 m/s²
            if all(s["acc"][idx[j + q]] < -3.0 for q in range(3)):
                onset = idx[j]; break
        if onset is not None:
            brake_t = float(s["t"][onset])
            ms.append(M("v1.brake_onset", "Camera vehicle · braking began before impact", impact_t - brake_t,
                        max(0, impact_t - brake_t - 0.25), impact_t - brake_t + 0.25, "s",
                        "first of 3 consecutive frames with deceleration > 3 m/s² within 4 s before impact", "low"))
        else:
            ms.append(M("v1.brake_onset", "Camera vehicle · braking began before impact", 0, 0, 0.25, "s",
                        "no sustained deceleration > 3 m/s² detected within 4 s before impact", "low"))
        if "ego" in pre_v:
            v = pre_v["ego"]
            d_mid = v ** 2 / (2 * np.mean([mu_lo, mu_hi]) * G)
            d_lo = (v * k_lo) ** 2 / (2 * mu_hi * G); d_hi = (v * k_hi) ** 2 / (2 * mu_lo * G)
            ms.append(M("v1.braking_distance", "Camera vehicle · braking distance needed", d_mid, d_lo, d_hi, "m",
                        f"v²/(2·μ·g), μ {mu_lo}-{mu_hi} ({weather} asphalt)", "medium"))
            ms.append(M("v1.stopping_distance", "Camera vehicle · stopping distance incl. reaction", v * PRT[1] + d_mid,
                        v * k_lo * PRT[0] + d_lo, v * k_hi * PRT[2] + d_hi, "m",
                        f"v·t_r + v²/(2·μ·g), reaction {PRT[0]}-{PRT[2]} s", "medium"))
            wa = _window(s["t"], impact_t + 0.4, impact_t + 1.4)
            if wa.sum() >= 3:
                va, _ = robust_velocity(s["t"][wa], s["P"][wa])
                dv = float(np.linalg.norm(va - pre_vel["ego"]))
                ms.append(M("v1.delta_v", "Camera vehicle · velocity change (ΔV)", dv * 3.6, dv * k_lo * 3.6 * 0.7,
                            dv * k_hi * 3.6 * 1.3, "km/h", "|v_after - v_before| from linear fits of the camera pose track", "low"))
    if b is not None and a in pre_vel and b in pre_vel:
        va, vb = pre_vel[a], pre_vel[b]
        rel = float(np.linalg.norm(va - vb))
        if np.linalg.norm(va) > 0.5 and np.linalg.norm(vb) > 0.5:
            ang = float(np.degrees(np.arccos(np.clip(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb), -1, 1))))
            ms.append(M("impact.angle", "Angle between headings at impact", ang, max(0, ang - 20), min(180, ang + 20),
                        "°", "arccos of pre-impact velocity directions (road plane)", "low"))
        ms.append(M("impact.closing_speed", "Closing speed at impact", rel * 3.6, max(0, rel * k_lo - 2) * 3.6,
                    (rel * k_hi + 2) * 3.6, "km/h", "|v1 - v2| from pre-impact linear fits", "low"))
        sb = S[b]
        first_t = float(sb["t"][0])
        if first_t < impact_t:
            ms.append(M("v2.visible_before_impact", f"{labels[b]} · visible before impact", impact_t - first_t,
                        max(0, impact_t - first_t - 0.25), impact_t - first_t + 0.13, "s",
                        "first SAM 3.1 detection of the vehicle to impact", "medium"))
        if gap_series and rel > 0.5:
            g = np.array(gap_series)
            pre = g[(g[:, 0] < impact_t - 0.5) & (g[:, 0] >= impact_t - 2.0)]
            if len(pre):
                gap1 = float(np.median(pre[:, 1]))
                tt = float(np.median(pre[:, 0]))
                ttc = gap1 / rel
                ms.append(M("impact.ttc", f"Time-to-collision at t={tt:.1f}s", ttc, ttc * 0.6, ttc * 1.4, "s",
                            "median footprint gap in [impact-2 s, impact-0.5 s] / closing speed", "low"))
    ms.append(M("impact.time", "Impact moment (clip time)", impact_t, min(spread), max(spread), "s", impact_method,
                "high" if impact_geo is not None and abs(impact_geo - impact_label) < 0.5 else "medium"))
    ms.append(M("scale.camera_height", "Scale: camera height assumption", 1.35, 1.2, 1.5, "m", tr["scale"]["method"],
                "medium", scale_m_per_unit=round(s_mid, 3)))

    # vehicle-length cross-check: visible extent of the other vehicle vs. a typical body length
    if b is not None and b in tr["tracks"]:
        rows = tr["tracks"][b]
        # lateral (x) extent is well conditioned in monocular depth; depth-axis extent is not
        ext = float(np.percentile([r["bbox"][1][0] - r["bbox"][0][0] for r in rows], 75))
        body = next((v["body_type"].lower() for v in scene["vehicles"] if v["role"] != "camera_vehicle"), "car")
        crossing = any(m_["id"] == "impact.angle" and 45 <= m_["value"] <= 135 for m_ in ms)
        L = next((v for k2, v in BODY_LENGTH.items() if k2 in body), 4.5) if crossing else 1.8
        what = f"typical {body} length {L} m" if crossing else "typical car width 1.8 m"
        ratio = L / ext if ext > 0.3 else None
        if ratio is not None:
            ms.append(M("scale.vehicle_size_check", "Scale cross-check: reference size / measured", ratio, ratio * 0.8,
                        ratio * 1.25, "×", f"{what} / lateral 3D extent {ext:.1f} m (1.0 = scales agree)", "low"))
            if ratio < 0.75 or ratio > 1.33:
                warnings.append(f"Vehicle-length cross-check differs from camera-height scale (ratio {ratio:.2f}); ranges widened.")
                for m_ in ms:
                    if m_["unit"] in ("km/h", "m") and m_["id"] != "scale.camera_height":
                        m_["range"] = [round(m_["range"][0] * 0.85, 2), round(m_["range"][1] * 1.15, 2)]
                        m_["conf"] = "low"

    for m_ in ms:          # physical floor: no negative speeds/distances
        if m_["unit"] in ("km/h", "m", "s", "m/s²"):
            m_["range"][0] = max(0.0, m_["range"][0])

    # ---- series for the HUD (timeline speed curves, top-down paths)
    series = {"gap": [[round(x, 3), round(y, 2)] for x, y in gap_series], "objects": {}}
    for k, s in S.items():
        series["objects"][k] = {"t": np.round(s["t"], 3).tolist(), "speed_kmh": np.round(s["speed"] * 3.6, 1).tolist(),
                                "acc": np.round(s["acc"], 2).tolist(),
                                "xz": np.round(s["disp"], 2).tolist()}
    colors = {a: config.PALETTE[0]}
    if b:
        colors[b] = config.PALETTE[1]
    objects = []
    for i, k in enumerate([x for x in involved] + [x for x in ranked if x not in involved][:6]):
        if k not in S:
            continue
        role = "camera_vehicle" if k == "ego" else ("involved" if k in involved else "other")
        objects.append({"id": k, "label": labels[k], "color": colors.get(k, config.PALETTE[2 + i % 5]),
                        "concept": "camera vehicle" if k == "ego" else k.split(":")[0], "role": role,
                        "sam_key": None if k == "ego" else k, "vid": vids.get(k)})
    fp = None
    if b and b in S and a in S:
        i = int(np.argmin(abs(S[a]["t"] - impact_t))); j = int(np.argmin(abs(S[b]["t"] - impact_t)))
        pa = S[a]["P"][i][[0, 2]] + (np.array([0, EGO_FRONT]) if a == "ego" else 0)
        fp = ((pa + S[b]["P"][j][[0, 2]]) / 2).round(2).tolist()
    out = {
        "scale": tr["scale"] | {"k_range": [k_lo, k_hi]},
        "impact": {"t": round(impact_t, 3), "label_t": impact_label, "spark_t": impact_spark,
                   "kind": "collision" if impact_spark is not None else "near-miss",
                   "geometry_t": impact_geo, "method": impact_method, "point_xz": fp, "brake_t": brake_t},
        "measurements": ms, "series": series, "objects": objects, "involved": involved,
        "weather_mu": {"weather": weather, "mu": [mu_lo, mu_hi]}, "warnings": warnings,
    }
    out_path.write_text(json.dumps(out, indent=1))
    log.info("physics: impact %.2fs (%s), involved %s, %d measurements, warnings=%d",
             impact_t, impact_method.split(",")[0], involved, len(ms), len(warnings))
    return out


def summary_text(phys: dict) -> str:
    """Compact evidence summary for the verdict prompt and the Laya integrity check."""
    lines = [f"impact at {phys['impact']['t']:.2f}s ({phys['impact']['method']})"]
    for m in phys["measurements"]:
        lines.append(f"{m['id']}: {m['value']} {m['unit']} (range {m['range'][0]}-{m['range'][1]}, {m['conf']})")
    return "; ".join(lines)

"""Deterministic, to-scale overhead diagram (section 10.1). Geometry comes from code, never from the image model."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

BG, GRID, TEXT = "#0b0f14", "#1d2833", "#c9d3dc"
FONT = dict(family="monospace")


def _footprint(ax, fwd, right, heading, color, L=4.5, W=1.85, alpha=0.35):
    """Vehicle rectangle centred at (fwd, right), rotated to its heading (plot axes: x = forward, y = right)."""
    ang = np.degrees(np.arctan2(heading[1], heading[0])) if np.linalg.norm(heading) > 1e-3 else 0.0
    ax.add_patch(Rectangle((fwd - L / 2, right - W / 2), L, W, angle=ang, rotation_point="center",
                           facecolor=color, edgecolor=color, alpha=alpha, lw=1.5))


def draw(case: pathlib.Path, out: pathlib.Path | None = None, t_at: float | None = None, force: bool = False) -> pathlib.Path:
    out = out or case / "diagram.png"
    if out.exists() and not force:
        return out
    ph = json.loads((case / "physics.json").read_text())
    imp = ph["impact"]["t"]
    t_at = imp if t_at is None else t_at
    near = ph["impact"].get("kind") == "near-miss"
    meas = {m["id"]: m for m in ph["measurements"]}
    series = ph["series"]["objects"]
    parties = [o for o in ph["objects"] if o["role"] != "other"]
    others = [o for o in ph["objects"] if o["role"] == "other"][:2]

    fig = plt.figure(figsize=(15.36, 10.24), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.05, 0.07, 0.91, 0.85], facecolor=BG)
    if ph["impact"].get("point_xz"):
        center = np.array(ph["impact"]["point_xz"], float)              # [right, forward]
    else:
        e = series["ego"]; center = np.array(e["xz"][int(np.argmin(np.abs(np.array(e["t"]) - imp)))], float)
    span_f, span_r = 33.0, 22.0

    for o in others + parties:
        s = series.get(o["id"])
        if not s:
            continue
        t = np.array(s["t"]); xz = np.array(s["xz"])
        win = (t >= imp - 4) & (t <= imp + 2)
        if win.sum() < 2:
            continue
        t, xz = t[win], xz[win]
        main = o in parties
        lw, a = (3.2, 1.0) if main else (1.2, 0.35)
        pre, post = t <= imp + 1e-6, t >= imp - 1e-6
        ax.plot(xz[pre, 1], xz[pre, 0], color=o["color"], lw=lw, alpha=a, solid_capstyle="round")
        ax.plot(xz[post, 1], xz[post, 0], color=o["color"], lw=lw, alpha=a, ls=(0, (4, 3)))
        if not main:
            continue
        for tt in np.arange(np.ceil((imp - 4) * 2) / 2, imp + 2.01, 0.5):     # timestamp dot every 0.5 s
            i = int(np.argmin(np.abs(t - tt)))
            if abs(t[i] - tt) < 0.13:
                ax.plot(xz[i, 1], xz[i, 0], "o", color=o["color"], ms=4.5)
                if abs(round(tt - imp) - (tt - imp)) < 1e-6 and tt != imp:
                    ax.annotate(f"{tt - imp:+.0f}s", (xz[i, 1], xz[i, 0]), xytext=(0, 9), textcoords="offset points",
                                color=TEXT, fontsize=8, alpha=0.75, ha="center", **FONT)
        i = int(np.argmin(np.abs(t - t_at)))
        hd = xz[min(i + 2, len(xz) - 1)] - xz[max(i - 2, 0)]
        _footprint(ax, xz[i, 1], xz[i, 0], hd[::-1], o["color"])
        v = meas.get(f"{o.get('vid')}.pre_impact_speed")
        lbl = o["label"] + (f"\n{v['range'][0]:.0f}–{v['range'][1]:.0f} km/h before impact" if v else "")
        j = 0                                                            # label at the start of the drawn path
        ax.annotate(lbl, (xz[j, 1], xz[j, 0]), xytext=(0, -30 if o.get("vid") == "v1" else 22),
                    textcoords="offset points", color=o["color"], fontsize=11, ha="left", va="center", **FONT)
    # impact marker with a leader to an uncluttered corner
    ax.plot(center[1], center[0], marker="x", color="#f2d28b", ms=24, mew=3, zorder=5)
    cs = meas.get("impact.closing_speed")
    txt = ("CLOSEST APPROACH" if near else "IMPACT") + f"  t = {imp:.2f} s" + \
          (f"\nclosing {cs['range'][0]:.0f}–{cs['range'][1]:.0f} km/h" if cs else "")
    ax.annotate(txt, (center[1], center[0]), xytext=(center[1] + 7, center[0] - 12), color="#f2d28b", fontsize=12,
                weight="bold", arrowprops=dict(arrowstyle="-", color="#f2d28b", lw=1, alpha=0.7), **FONT)
    # camera heading arrow
    ax.add_patch(FancyArrow(center[1] - span_f * 0.92, center[0] - span_r * 0.8, 6, 0, width=0.4, color=TEXT, alpha=0.7))
    ax.text(center[1] - span_f * 0.92, center[0] - span_r * 0.68, "camera heading", color=TEXT, fontsize=9, **FONT)
    ax.set_xlim(center[1] - span_f, center[1] + span_f)
    ax.set_ylim(center[0] + span_r, center[0] - span_r)                 # driver's right side is down
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(np.ceil((center[1] - span_f) / 5) * 5, center[1] + span_f, 5))
    ax.set_yticks(np.arange(np.ceil((center[0] - span_r) / 5) * 5, center[0] + span_r, 5))
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors="#53606b", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlabel("forward (m)", color="#53606b", **FONT)
    ax.set_ylabel("right (m)", color="#53606b", **FONT)
    fig.text(0.05, 0.95, "OVERHEAD RECONSTRUCTION · to scale · 5 m grid", color=TEXT, fontsize=13, **FONT)
    fig.text(0.96, 0.95, "ESTIMATE", color="#e8a76a", fontsize=11, ha="right", **FONT)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out

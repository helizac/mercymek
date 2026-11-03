"""v4: counterfactuals, Mercy-mode meter, evidence seal, new voice intents, RAFT-refined fusion."""
import json
import math

import numpy as np
import pytest

from mercy.analysis import claims, enhance, seal, whatif
from mercy.voice import commands as C


def physics(kind="collision"):
    """Camera car at 72 km/h, hazard 2 s before impact, brakes 1.5 s before impact down to 36 km/h."""
    ts = [i / 8 for i in range(81)]
    sp, xz, x = [], [], 0.0
    for t in ts:
        v = 72.0 if t < 8.5 else max(36.0, 72.0 - (t - 8.5) * 24)
        sp.append(v); xz.append([0.0, x]); x += v / 3.6 / 8
    return {"impact": {"t": 10.0, "kind": kind}, "weather_mu": {"mu": [0.7, 0.8]},
            "measurements": [{"id": "v2.visible_before_impact", "value": 2.0}, {"id": "v1.brake_onset", "value": 1.5},
                             {"id": "v1.pre_impact_speed", "value": 40.0}],
            "series": {"objects": {"ego": {"t": ts, "speed_kmh": sp, "xz": xz}}}}


def test_whatif_parse():
    assert whatif.parse("what if it had been going 30 km/h")["speed_kmh"] == 30
    assert whatif.parse("what if it reacted immediately")["reaction"] == "alert"
    assert whatif.parse("what if it braked one second earlier")["earlier_s"] == 1.0
    assert whatif.parse("what if the road was wet")["mu"] == (0.4, 0.55)


def test_whatif_physics():
    p = physics()
    slow = whatif.run(p, "what if it had been going 20 km/h")
    assert slow["verdict"] == "avoided" and slow["margin_m"][0] > 0
    asis = whatif.run(p, "what if")                       # as driven: reproduces a crash near the measured speed
    assert asis["verdict"] == "not avoided" and abs(asis["real"]["impact_kmh"] - asis["impact_kmh"][1]) < 15
    fast = whatif.run(p, "what if it had been going 100 km/h")
    assert fast["impact_kmh"][0] > asis["impact_kmh"][0]
    assert len(fast["ghost"]["t"]) == len(fast["ghost"]["xz"]) == 81


def test_stopping_formula():
    r = whatif.simulate(20.0, 1.0, 7.0, 100.0)
    assert r["stops"] and math.isclose(r["margin"], 100 - (20 + 400 / 14), rel_tol=1e-9)
    r = whatif.simulate(20.0, 1.0, 7.0, 30.0)
    assert not r["stops"] and math.isclose(r["impact_v"], math.sqrt(400 - 2 * 7 * 10), rel_tol=1e-9)


def test_meter_logodds():
    meter = [{"party": "White SUV #4", "pct": 70.0}, {"party": "Camera vehicle", "pct": 30.0}]
    up = claims.sigm(claims.logit(70.0) + math.log(claims.LR["moderate"]))
    assert 70 < up < 85
    assert claims.match_party("the camera vehicle", meter) == 1
    assert claims.match_party("White SUV", meter) == 0
    assert claims.meter_start({"display": {"fault": [{"party": "A", "pct": 72}, {"party": "B", "pct": 38}]}})[0]["pct"] == \
        pytest.approx(65.5, abs=.1)


def test_seal_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(seal, "KEY", tmp_path / "keys" / "k.pem")
    case = tmp_path / "case"
    case.mkdir()
    (case / "physics.json").write_text(json.dumps({"v": 1}))
    (case / "clip.mp4").write_bytes(b"\x00" * 1000)
    s = seal.seal(case)
    assert seal.verify(case)["intact"] and len(s["files"]) == 2
    (case / "physics.json").write_text(json.dumps({"v": 2}))
    v = seal.verify(case)
    assert not v["intact"] and v["changed"] == ["physics.json"] and v["signature_valid"]


@pytest.mark.parametrize("text,intent", [
    ("what if the camera car had been going 30 km/h", "whatif"),
    ("would it have stopped if it braked earlier", "whatif"),
    ("Objection, the SUV never signalled", "claim"),
    ("I argue the camera car was tailgating", "claim"),
    ("replay the impact in super slow motion", "replay"),
    ("bullet time", "replay"),
    ("verify the evidence", "verify"),
    ("fill in the accident statement", "statement"),
    ("what was the speed of the SUV", "ask"),
    ("slow motion", "playback"),
    ("let me walk around the scene", "camera"),
    ("show the moment before impact in 3D", "view"),
    ("show braking in 3D", "view"),
])
def test_new_rules(text, intent):
    assert C.rule_intent(text) == intent


def test_free_and_bullet_camera():
    assert C.camera_slots("free camera") == {"camera_preset": "free"}
    assert C.slots_for("replay", "bullet time", {}) == {"mode": "bullet"}
    assert C.slots_for("replay", "replay the impact", {}) == {"mode": "slowmo"}


def test_glued_mask_finds_hood_and_overlay():
    import cv2
    from mercy.ingest.splat import glued_mask
    rng = np.random.default_rng(1)
    world = cv2.GaussianBlur((rng.random((200, 900, 3)) * 255).astype(np.uint8), (0, 0), 2)
    hood = cv2.GaussianBlur((rng.random((40, 300, 3)) * 120).astype(np.uint8), (0, 0), 3)
    frames = []
    for i in range(24):                                   # the world slides past, hood and timestamp stay put
        f = world[:, i * 12:i * 12 + 300].copy()
        f[160:] = hood
        cv2.putText(f, "2025-01-23", (190, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        frames.append(f)
    m = glued_mask(np.stack(frames))
    assert m[170:, :].mean() > 0.95                       # hood
    assert m[2:18, 195:290].mean() > 0.3                  # timestamp
    assert m[30:140, 40:260].mean() < 0.02                # the moving world is kept


def test_moments_pick(tmp_path):
    from mercy.ingest import moments
    (tmp_path / "meta.json").write_text(json.dumps({"time_of_event": 8.0, "duration": 12.0}))
    ph = {"impact": {"t": 8.0, "kind": "collision", "brake_t": 7.5},
          "measurements": [{"id": "v2.visible_before_impact", "value": 3.0}],
          "objects": [{"vid": "v2", "label": "White SUV #4"}]}
    (tmp_path / "physics.json").write_text(json.dumps(ph))
    got = [(m["id"], m["t"]) for m in moments.pick(tmp_path)]
    assert got == [("hazard", 5.0), ("before", 7.4), ("impact", 8.0)]       # braking 0.1 s after "before": same moment
    ph["measurements"][0]["value"] = 8.0                                       # visible from the first frame: no moment
    (tmp_path / "physics.json").write_text(json.dumps(ph))
    assert "hazard" not in [m["id"] for m in moments.pick(tmp_path)]


def test_vehicle_keys():
    from mercy.ingest.moments import vehicle_keys
    t = np.arange(0, 4.01, 0.125)
    P = np.stack([np.zeros_like(t), np.full_like(t, 0.6), 10 + 8.0 * t], 1)      # straight ahead at 8 m/s
    k = vehicle_keys(t, P, 3.0, span=1.5)
    assert k[-1] == [0.0, 0.0, 0.0, 0.0] and k[0][0] == -1.5
    assert abs(k[0][2] + 12.0) < 0.3 and abs(k[0][1]) < 0.05 and abs(k[0][3]) < 1    # 1.5 s back: 12 m behind
    assert vehicle_keys(t, P * [1, 1, 0] + [0, 0, 30], 3.0) == []                    # parked: no replay
    J = P.copy(); J[t < 2.2, 2] -= 40                                                # depth spike: another object
    assert vehicle_keys(t, J, 3.0, span=1.5)[0][0] > -0.9
    turn = np.stack([8 * np.sin(t), np.full_like(t, 0.6), 8 * np.cos(t)], 1)      # sharp turn: yaw is clamped
    assert max(abs(r[3]) for r in vehicle_keys(t, turn, 3.0, span=1.5)) <= 25.0


def test_moment_for_words():
    from types import SimpleNamespace
    from mercy.server.app import Case
    ms = [{"id": "hazard", "t": 5.0}, {"id": "before", "t": 7.4}, {"id": "impact", "t": 8.0}]
    stub = SimpleNamespace(moments=lambda: ms)
    pick = lambda text, t=0.0: Case.moment_for(stub, text, t)["id"]  # noqa: E731
    assert pick("show the moment just before impact in 3D") == "before"
    assert pick("show the crash in 3D") == "impact"
    assert pick("show the white SUV in 3D", 5.2) == "hazard"                 # "white" is not "hit": nearest instead
    assert pick("show me 3D", 7.9) == "impact"


def test_fusion_keeps_shape():
    rng = np.random.default_rng(0)
    import cv2
    base = cv2.GaussianBlur((rng.random((60, 130, 3)) * 255).astype(np.uint8), (0, 0), 2.5)
    crops = [base[10:50, 18 + i:108 + i] for i in range(-2, 3)]       # the same texture, shifted by whole pixels
    out, n = enhance.fuse(crops)
    assert out.shape == (80, 180, 3) and n >= 4
    ref = cv2.resize(crops[2], (180, 80), interpolation=cv2.INTER_CUBIC).astype(float)
    assert np.abs(out.astype(float) - ref).mean() < 12                    # aligned, not smeared

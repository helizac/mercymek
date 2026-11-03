"""v2: screen switching and enhance commands, GPS overlay parsing, multi-frame fusion."""
import numpy as np
import pytest

from mercy.analysis import enhance, geolocate
from mercy.voice import commands as C


@pytest.mark.parametrize("text,intent,slots", [
    ("where did this happen", "view", {"view": "map"}),
    ("haritayı aç", "view", {"view": "map"}),
    ("show me the 3D scene", "view", {"view": "scene"}),
    ("go back to the video", "view", {"view": "video"}),
    ("kanıtları göster", "view", {"view": "evidence"}),
    ("zoom in on the licence plate", "zoom", {"concept": "license plate", "enhance": True}),
    ("plakayı netleştir", "zoom", {"concept": "license plate", "enhance": True}),
    ("enhance the image", "zoom", {"enhance": True}),
    ("orbit the scene", "camera", {"camera_preset": "orbit"}),
    ("go back to the start", "seek", {"anchor": "start"}),
])
def test_rules_and_slots(text, intent, slots):
    assert C.rule_intent(text) == intent
    got = C.slots_for(intent, text, {})
    for k, v in slots.items():
        assert got.get(k) == v, (text, got)


def test_gps_overlay_regex():
    line = "GARMIN 15/01/2026 12:24:03 43.73877 -79.52705 83 KM/H"
    m = geolocate.GPS_RE.search(line)
    assert (float(m.group(1)), float(m.group(2))) == (43.73877, -79.52705)
    assert abs(geolocate.haversine((43.73897, -79.52710), (43.73758, -79.52674)) - 157) < 5
    assert 160 < geolocate.bearing((43.73897, -79.52710), (43.73758, -79.52674)) < 200      # southbound


def test_fusion_reduces_noise():
    rng = np.random.default_rng(0)
    clean = np.zeros((40, 100, 3), np.uint8)
    clean[10:30, 10:90] = 200
    crops = [np.clip(clean.astype(int) + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8) for _ in range(9)]
    fused, used = enhance.fuse(crops)
    ref = np.kron(clean, np.ones((2, 2, 1), np.uint8))
    assert used >= 7
    assert np.abs(fused.astype(int) - ref).mean() < np.abs(np.kron(crops[4], np.ones((2, 2, 1), np.uint8)).astype(int) - ref).mean()


def test_box_tracker_falls_back():
    ov = {"10": {"license plate:0": {"box": [1, 2, 3, 4]}}}
    at = enhance.box_tracker(ov, "license plate:0", [0, 0, 9, 9])
    assert at(11) == [1, 2, 3, 4] and at(40) is None
    assert enhance.box_tracker(ov, None, [0, 0, 9, 9])(5) == [0, 0, 9, 9]

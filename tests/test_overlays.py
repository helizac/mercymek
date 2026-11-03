"""SAM mask decoding through meta-sam-parser and overlay offset math."""
import pathlib

import numpy as np
import pytest

from mercy.ingest import segment

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_polygon_offset():
    crop = np.zeros((20, 30), np.uint8); crop[5:15, 10:20] = 1
    polys = segment.polygons(crop, (100, 200), eps=0.5, min_area=1)
    xs, ys = np.array(polys[0][0::2]), np.array(polys[0][1::2])
    assert xs.min() == 110 and xs.max() == 119 and ys.min() == 205 and ys.max() == 214


@pytest.mark.skipif(not (ROOT / "data/smoke/sam_image.lines").exists(), reason="run scripts/smoke_test.py sam first")
def test_real_mask_matches_box():
    recs = segment.parse((ROOT / "data/smoke/sam_image.lines").read_text(), "image")
    assert recs
    for r in recs:
        crop = segment.mask_crop(r)                       # asserts H == y2-y1+1 and W == x2-x1+1 internally
        assert crop.shape == (r.mask.height, r.mask.width) and crop.max() == 1


def test_case_overlays_have_frames():
    for case in (ROOT / "data/cases").glob("*/overlays.json"):
        import json
        ov = json.loads(case.read_text())
        f, objs = next(iter(ov.items()))
        o = next(iter(objs.values()))
        assert len(o["box"]) == 4 and o["box"][2] > o["box"][0] and isinstance(o["poly"], list)

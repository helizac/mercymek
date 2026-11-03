"""Contract tests: every JSON file in each case folder and every WebSocket message validates against mercy.schemas."""
import json
import pathlib

import pytest

from mercy import schemas as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = [p for p in (ROOT / "data/cases").glob("*") if (p / "physics.json").exists()]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_files(case):
    S.Meta(**json.loads((case / "meta.json").read_text()))
    S.Scene(**json.loads((case / "scene.json").read_text()))
    ph = S.Physics(**json.loads((case / "physics.json").read_text()))
    for m in ph.measurements:                     # every value carries range, unit and method
        assert len(m.range) == 2 and m.range[0] <= m.range[1] and m.unit and m.method
    tr = json.loads((case / "tracks.json").read_text())
    for rows in tr["tracks"].values():
        for r in rows:
            S.TrackRow(**r)
    if (case / "research.json").exists():
        for it in json.loads((case / "research.json").read_text())["items"]:
            S.ResearchItem(**it)
    if (case / "verdict.json").exists():
        S.VerdictFile(**json.loads((case / "verdict.json").read_text()))


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_ws_messages(case):
    from mercy.server.app import Case
    c = Case(case.name)
    S.Hello(**c.hello())
    for row in c.physics_rows():
        S.PhysicsRow(**row)
    S.StateMsg(t=1.0, playing=False, layers={"masks": True}, isolate=["car:1"])
    S.VerdictMsg(status="running", stage="x")
    if (case / "verdict.json").exists():
        S.VerdictMsg(status="cached", data=json.loads((case / "verdict.json").read_text())["display"])
    S.ImageMsg(status="ready", url="/x.webp", caption="c", watermark="AI ILLUSTRATION · NOT EVIDENCE")
    S.IntentMsg(intent="seek", confidence=0.9, ms=30, router="laya:english")

"""Clause splitting (EN + TR) and deterministic slot extraction. No model needed."""
import pytest

from mercy.voice import commands as C


@pytest.mark.parametrize("text,n", [
    ("isolate the vehicles and rewind to three seconds before impact", 2),
    ("isolate the glass and debris", 1),                         # noun phrase stays together
    ("orbit the scene, then zoom in on vehicle two", 2),
    ("araçları izole et ve çarpışmadan 3 saniye önceye git", 2),
    ("durdur sonra üstten göster", 2),
    ("who is at fault?", 1),
    ("fill in the accident statement and verify the evidence", 2),
    ("show the report and fill in the accident statement", 2),
])
def test_split(text, n):
    assert len(C.split_clauses(text)) == n


@pytest.mark.parametrize("text,anchor,t", [
    ("rewind to three seconds before impact", "impact", -3.0),
    ("go to 2.5 seconds after the collision", "impact", 2.5),
    ("jump to the impact", "impact", 0.0),
    ("go to 5 seconds", "absolute", 5.0),
    ("rewind 2 seconds", "current", -2.0),
    ("çarpışmadan 3 saniye önceye git", "impact", -3.0),
    ("go to first braking", "brake", 0.0),
    ("back to the start", "start", 0.0),
])
def test_seek_slots(text, anchor, t):
    s = C.seek_slots(text)
    assert s["anchor"] == anchor and s["t_seconds"] == pytest.approx(t)


def test_object_slots():
    labels = {"blue hatchback": "car:7"}
    assert C.object_slots("isolate vehicle 2", labels)["object_id"] == 2
    assert C.object_slots("zoom in on car one", labels)["object_id"] == 1
    assert C.object_slots("isolate the vehicles", labels)["object_id"] == 0
    assert C.object_slots("highlight the blue hatchback", labels)["sam_key"] == "car:7"
    assert C.object_slots("isolate the shattered glass", labels)["concept"] == "shattered glass"
    assert C.object_slots("trafik ışığını izole et", {})["concept"] == "traffic light"
    s = C.object_slots("isolate the fire hydrant", {})
    assert s.get("live") and "hydrant" in s["concept"]


def test_camera_playback_layers():
    assert C.camera_slots("show me the top view")["camera_preset"] == "top"
    assert C.camera_slots("orbit the scene")["camera_preset"] == "orbit"
    assert C.camera_slots("üstten göster")["camera_preset"] == "top"
    assert C.playback_slots("slow motion please") == {"speed": 0.25, "playing": True}
    assert C.playback_slots("pause")["playing"] is False
    assert C.layer_slots("hide the masks") == {"layer": "masks", "visible": False}
    assert C.note_text("note that the light was red") == "the light was red"


@pytest.mark.parametrize("text", ["replay the approach", "show me the lead-up", "play the approach", "fly the run up"])
def test_approach_replay_is_a_rule(text):
    assert C.rule_intent(text) == "camera"
    assert C.camera_slots(text) == {"camera_preset": "approach"}


def test_approach_rule_leaves_other_replays_alone():
    assert C.rule_intent("replay the impact in super slow motion") == "replay"
    assert C.rule_intent("show the impact in 3D") == "view"


def test_short_spoken_vehicle_name_resolves_saved_object():
    labels = {"silver bmw sedan": "car:3", "dark car ahead": "car:1"}
    slots = C.zoom_slots("Zoom in on the silver BMW", labels)
    assert slots["sam_key"] == "car:3"
    assert not slots.get("live")


def test_named_zoom_applies_zoom_without_new_segmentation(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from mercy.server import app as server

    async def exercise():
        patches = []
        async def patch(value):
            patches.append(value)
        async def broadcast(*args):
            pass
        async def unexpected_segment(*args):
            pytest.fail("A saved named vehicle must use its existing object key")
        monkeypatch.setattr(server.rt, "case", SimpleNamespace(
            objects=[{"id": "car:3", "label": "Silver BMW sedan"}], overlay_keys={}))
        monkeypatch.setattr(server.hub, "patch", patch)
        monkeypatch.setattr(server.hub, "broadcast", broadcast)
        monkeypatch.setattr(server, "live_segment", unexpected_segment)
        await server.execute({"intent": "zoom", **C.zoom_slots(
            "Zoom in on the silver BMW", {"silver bmw sedan": "car:3"})}, None)
        assert patches == [{"zoom": "car:3", "selected": "car:3"}]
    asyncio.run(exercise())


def test_generic_car_does_not_match_a_bystanders_alias():
    slots = C.zoom_slots("Zoom in on the car", {"dark car ahead": "car:1"})
    assert slots["concept"] == "car"
    assert "sam_key" not in slots

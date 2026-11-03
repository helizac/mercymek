"""European Accident Statement (the insurers' joint report form), pre-filled from the evidence.

Section 12 of the form ("circumstances") is 17 tick boxes per involved vehicle; the model ticks them from the
scene description, the measurements and the verdict, and only when the footage shows it.
"""
from __future__ import annotations

import json
import pathlib

from mercy import config
from mercy.meta_client import structured

CIRCUMSTANCES = [
    "parked / stationary", "leaving a parking space / opening a door", "entering a parking space",
    "emerging from a car park, private ground or track", "entering a car park, private ground or track",
    "entering a roundabout", "circulating in a roundabout",
    "striking the rear of the other vehicle while going in the same direction and in the same lane",
    "going in the same direction but in a different lane", "changing lanes", "overtaking", "turning to the right",
    "turning to the left", "reversing", "encroaching on a lane reserved for traffic in the opposite direction",
    "coming from the right (at a junction)", "had not observed a sign or a red light"]
IMPACT_POINTS = ["front", "front-left", "front-right", "left side", "right side", "rear-left", "rear-right", "rear",
                 "unknown"]

_veh = {"type": "object", "additionalProperties": False,
        "required": ["label", "circumstances", "initial_impact", "plate"],
        "properties": {"label": {"type": "string"},
                       "circumstances": {"type": "array", "items": {"type": "integer"}},
                       "initial_impact": {"type": "string", "enum": IMPACT_POINTS},
                       "plate": {"type": "string"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["A", "B", "remarks"],
          "properties": {"A": _veh, "B": _veh, "remarks": {"type": "string"}}}

SYSTEM = ("Fill section 12 of the European Accident Statement from crash evidence. Use the supplied form_parties "
          "mapping for vehicles A and B. A witness camera or bystander is never an involved party. "
          "Only copy the plate belonging to that mapped party; unknown plates stay empty. Circumstances are numbered 1-17:\n" +
          "\n".join(f"{i + 1}. {c}" for i, c in enumerate(CIRCUMSTANCES)) +
          "\nTick a box only if the evidence shows it; unknown stays unticked. Box 17 only if a sign or red light is "
          "visible in the evidence. 'plate' is the plate text if read, else ''. Remarks: one plain sentence, under 25 "
          "words, about what the form cannot capture.")


def build(case: pathlib.Path, force: bool = False) -> dict:
    out = case / "statement.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    j = lambda n, d=None: json.loads((case / n).read_text(encoding="utf-8")) if (case / n).exists() else d  # noqa: E731
    scene, physics, verdict = j("scene.json", {}), j("physics.json", {}), j("verdict.json")
    plates, loc, meta = j("plates.json", {}), j("location.json", {}), j("meta.json", {})
    m = {x["id"]: x for x in physics.get("measurements", [])}
    involved=sorted((o for o in physics.get('objects',[]) if o.get('vid')),key=lambda o:o['vid'])
    form_parties={letter:{'id':o['id'],'label':o['label'],
                         'plate':next((p.get('text','') for p in plates.get('plates',[])
                                       if p.get('vehicle')==o['id'] and p.get('legible')), '')}
                  for letter,o in zip(('A','B'),involved)}
    facts = {"scene": {k: scene.get(k) for k in ("summary", "vehicles", "road", "signals", "light", "weather")
                       if k in scene},
             "impact": physics.get("impact"), "impact_angle_deg": m.get("impact.angle", {}).get("range"),
             "speeds_kmh": {k: m[k]["range"] for k in ("v1.pre_impact_speed", "v2.pre_impact_speed") if k in m},
             "form_parties":form_parties,
             "verdict": (verdict or {}).get("display", {}).get("headline")}
    r = structured(config.VISION_MODEL, [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": json.dumps(facts)}],
                   SCHEMA, "AccidentStatement", effort="low", max_output_tokens=3000)
    for v in ("A", "B"):
        if v in form_parties:
            r[v]['label']=form_parties[v]['label'];r[v]['plate']=form_parties[v]['plate']
        r[v]["circumstances"] = sorted({c for c in r[v]["circumstances"] if 1 <= c <= 17})
        r[v]["ticked"] = [CIRCUMSTANCES[c - 1] for c in r[v]["circumstances"]]
    r["date"] = meta.get("recorded") or ""
    r["place"] = (loc or {}).get("place_name") or ""
    r["form"] = "European Accident Statement, section 12 (circumstances) and 10 (point of initial impact)"
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return r

"""Transcript -> viewer commands (section 6): clause split, Laya intent batch, deterministic slots, Spark fallback.

Laya never extracts slots: numbers come from regex, objects from fuzzy matching against a fixed dictionary.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from rapidfuzz import fuzz, process

from mercy import config
from mercy.config import log

INTENT_QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "Which action does the investigator want in the forensic viewer?",
        "criteria": {
            "isolate":    "highlight, isolate, segment or show only an object: car, vehicles, glass, debris, pedestrian, "
                          "traffic light (izole et, araçları göster)",
            "seek":       "move the video to a moment: go to or jump to the impact, rewind, go back to the start, "
                          "seconds before or after the crash (git, atla, geri sar, anına git)",
            "playback":   "play, pause, stop, freeze, resume, slow motion, playback speed (oynat, durdur, yavaşlat)",
            "camera":     "change the 3D viewpoint: orbit, rotate, top view, bird's eye, driver view, first person, side "
                          "view, fly around (üstten, kuş bakışı, sürücü görünümü)",
            "zoom":       "zoom in, enhance, sharpen, improve quality, magnify, read the licence plate, close-up of one "
                          "object (yakınlaştır, netleştir, kaliteyi artır, plakayı oku)",
            "trajectory": "show the path, braking trajectory or motion trail of a vehicle (yörünge)",
            "analyze":    "decide fault: verdict, who is at fault, who is responsible, who caused the crash, analyze "
                          "(kimin hatası, analiz et)",
            "diagram":    "generate or create an image: overhead diagram, blueprint, reconstruction image (çiz, oluştur)",
            "note":       "record, note or remember an observation in the case file (not al, kaydet)",
            "reset":      "reset or clear the view and highlights, close panels, start over (sıfırla, temizle)",
            "layers":     "show or hide an overlay layer: masks, tracks, labels, point cloud (maskeleri gizle)",
            "view":       "switch screen: open the map, where did this happen, show the 3D or 4D scene, open the "
                          "report, show the evidence, back to the video (haritayı aç, nerede oldu, 3B sahneye geç, "
                          "raporu aç, kanıtları göster, videoya dön)",
        },
    }
}
INTENTS = list(INTENT_QUESTION["intent"]["criteria"]) + ["ask", "whatif", "replay", "claim", "verify", "statement"]    # rules-only intents

CONCEPTS = {  # short noun phrase sent to SAM -> synonyms
    "car":             ["car", "cars", "vehicle", "vehicles", "sedan", "hatchback", "araba", "araç", "araçlar"],
    "truck":           ["truck", "lorry", "semi", "kamyon", "tır"],
    "pedestrian":      ["pedestrian", "person", "people", "yaya", "insan"],
    "motorcycle":      ["motorcycle", "motorbike", "bike", "motosiklet"],
    "shattered glass": ["glass", "broken glass", "shattered glass", "cam", "cam kırığı", "kırık cam"],
    "debris":          ["debris", "parts", "fragments", "enkaz", "parça"],
    "traffic light":   ["traffic light", "signal", "light", "trafik ışığı", "ışık", "lamba"],
    "license plate":   ["plate", "license plate", "plaka"],
}

SPLIT_RE = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\bve\b|\bsonra\b|\bardından\b|\bdaha sonra\b)\s*", re.I)
VERBS = ("isolate highlight show hide segment rewind go jump seek play pause stop slow speed rotate orbit top view zoom "
         "focus trace draw analyze analyse who compute generate create make note record remember reset clear start verify fill read replay "
         "freeze resume fast back move turn fly enhance find track open close give display what izole göster gizle git "
         "atla oynat durdur yavaş döndür yakınlaş yakınlaştır analiz oluştur çiz kaydet not sıfırla temizle geri ileri "
         "kim bul aç kapat başlat devam").split()
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "half": 0.5, "a": 1, "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6,
             "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10, "yarım": 0.5, "first": 1, "second": 2, "third": 3}
ORDINALS = {"one": 1, "first": 1, "1": 1, "bir": 1, "birinci": 1, "two": 2, "second": 2, "2": 2, "iki": 2,
            "ikinci": 2, "three": 3, "third": 3, "3": 3}


def split_clauses(text: str) -> list[str]:
    parts = [p.strip(" .!?") for p in SPLIT_RE.split(text) if p and p.strip(" .!?")]
    out: list[str] = []
    for p in parts:
        words = p.lower().split()
        # English clauses start with the verb; Turkish ones end with it ("üstten göster")
        has_verb = bool(words) and any(words[0].startswith(v) or words[-1].startswith(v) for v in VERBS)
        if out and not has_verb:
            out[-1] = f"{out[-1]} and {p}"          # "glass and debris" stays one clause
        else:
            out.append(p)
    return out or [text.strip()]


# ---------------------------------------------------------------- slots
def _num(s: str) -> Optional[float]:
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return NUM_WORDS.get(s.lower())


def seek_slots(c: str) -> dict:
    lc = c.lower()
    m = re.search(r"(\d+(?:[.,]\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|half|a|bir|iki|üç|dört|beş|"
                  r"altı|yedi|sekiz|dokuz|on|yarım)\s*(?:s\b|sec\b|secs\b|second|seconds|saniye|sn\b)", lc)
    t = _num(m.group(1)) if m else None
    anchor_impact = bool(re.search(r"impact|collision|crash|contact|hit|çarpış|kaza|temas", lc))
    anchor_brake = bool(re.search(r"brak|fren", lc))
    before = bool(re.search(r"before|prior|earlier|önce|evvel", lc))
    after = bool(re.search(r"after|later|sonra", lc))
    if re.search(r"\b(start|beginning|baş|başa)\b", lc) and t is None:
        return {"anchor": "start", "t_seconds": 0.0}
    if anchor_impact or anchor_brake:
        off = (t or 0.0) * (-1 if before else 1 if after else (-1 if t else 0))
        return {"anchor": "impact" if anchor_impact else "brake", "t_seconds": off, "relative_to_impact": anchor_impact}
    if t is not None:
        if re.search(r"rewind|back|geri", lc):
            return {"anchor": "current", "t_seconds": -t}
        if re.search(r"forward|skip|ileri", lc):
            return {"anchor": "current", "t_seconds": t}
        return {"anchor": "absolute", "t_seconds": t}
    if re.search(r"rewind|geri sar", lc):
        return {"anchor": "start", "t_seconds": 0.0}
    return {}


def object_slots(c: str, extra_labels: dict[str, str]) -> dict:
    """Object reference: vehicle number / ego / plural vehicles / concept via fuzzy match (score >= 70)."""
    lc = c.lower()
    out: dict = {}
    m = re.search(r"(?:vehicle|car|araç|araba|v|party|number|#)\s*(?:number\s*)?(\d|one|two|three|first|second|third|"
                  r"bir|iki|birinci|ikinci)\b", lc)
    m2 = re.search(r"\b(first|second|third|birinci|ikinci)\s+(?:vehicle|car|araç)", lc)
    if m or m2:
        out["object_id"] = ORDINALS.get((m or m2).group(1))
    if re.search(r"\b(camera car|camera vehicle|our car|my car|ego|dashcam car|kendi aracım|kamera aracı)\b", lc):
        out["object_id"] = 1
    if re.search(r"\b(vehicles|cars|both|araçlar|arabalar|ikisi)\b", lc) and "object_id" not in out:
        out["object_id"] = 0                      # 0 = all involved vehicles
    # Compare the spoken noun phrase with scene aliases. Including the command
    # verb can hide a shorter name such as "silver BMW" within "silver BMW sedan".
    words = re.sub(r"\b(isolate|highlight|show|segment|zoom|in|on|to|the|focus|track|trace|trajectory|path|of|izole|et|"
                   r"göster|yakınlaş|yörünge|please|me|all|every|braking)\b", " ", lc)
    words = " ".join(words.split())
    syn = {s: k for k, v in CONCEPTS.items() for s in v}
    if extra_labels and words and words not in syn and "object_id" not in out:
        hit = process.extractOne(words, list(extra_labels), scorer=fuzz.partial_ratio)
        if hit and hit[1] >= 85:
            out["concept"] = hit[0]; out["sam_key"] = extra_labels[hit[0]]
            return out
    if words:
        hit = process.extractOne(words, list(syn), scorer=fuzz.WRatio)
        if hit and hit[1] >= 70:
            out["concept"] = syn[hit[0]]
        elif "object_id" not in out and len(words) >= 3:
            out["concept"] = words               # unknown noun phrase -> live SAM on the current frame
            out["live"] = True
    return out


def camera_slots(c: str) -> dict:
    lc = c.lower()
    for pat, preset in [(r"approach|lead.?up|run.?up", "approach"),
                        (r"\bfree\b|walk|\bfly\b|explore|roam|look around|serbest|dolaş|gez", "free"),
                        (r"top|above|overhead|bird|üstten|kuş bakışı|yukarı", "top"),
                        (r"driver|first.person|dashcam view|sürücü|cockpit", "driver"),
                        (r"orbit|360|around|circle|etrafında|döndür|spin", "orbit"),
                        (r"side|yandan|profile", "side"), (r"follow|chase|takip", "follow")]:
        if re.search(pat, lc):
            return {"camera_preset": preset}
    return {}


def playback_slots(c: str) -> dict:
    lc = c.lower()
    if re.search(r"slow|yavaş|ağır", lc):
        return {"speed": 0.25, "playing": True}
    if re.search(r"half|yarım", lc):
        return {"speed": 0.5, "playing": True}
    if re.search(r"double|fast|hızlı|2x|twice", lc):
        return {"speed": 2.0, "playing": True}
    if re.search(r"normal|real.?time", lc):
        return {"speed": 1.0, "playing": True}
    if re.search(r"pause|stop|freeze|hold|durdur|dur\b|beklet", lc):
        return {"playing": False}
    return {"playing": True}


def layer_slots(c: str) -> dict:
    lc = c.lower()
    layer = next((n for p, n in [(r"mask|maske|segment", "masks"), (r"track|trail|path|iz", "tracks"),
                                 (r"label|readout|hud|etiket", "hud"), (r"cloud|point|bulut", "cloud")]
                  if re.search(p, lc)), None)
    vis = not re.search(r"hide|remove|off|turn off|gizle|kapat|kaldır", lc)
    return {"layer": layer, "visible": vis} if layer else {}


VIEWS = [(r"street ?view|street.level|mapillary|panoramax", "street"),
         (r"map|location|where|harita|konum|nerede|adres", "map"),
         (r"\b(3d|4d|three.?d|four.?d|scene|reconstruction|sahne|model|üç boyut|3b|4b)\b", "scene"),
         (r"report|verdict|findings|rapor|karar|blueprint|diagram", "report"),
         (r"evidence|files|dossier|plates|kanıt|dosya|delil", "evidence"),
         (r"video|footage|camera|kamera|görüntü", "video")]


def view_slots(c: str) -> dict:
    lc = c.lower()
    for pat, v in VIEWS:
        if re.search(pat, lc):
            return {"view": v}
    return {}


def zoom_slots(c: str, labels: dict[str, str]) -> dict:
    lc = c.lower()
    s = object_slots(c, labels)
    if re.search(r"plate|plaka|registration|number plate", lc):
        s.update({"concept": "license plate", "enhance": True}); s.pop("live", None)
    if re.search(r"enhance|sharpen|quality|clear|read|netleş|kalite|oku|iyileştir|büyüt", lc):
        s["enhance"] = True
        if s.get("live") and not re.search(r"isolate|segment", lc):
            s.pop("live"); s.pop("concept", None)          # "enhance this" is not a new concept to segment
    return s


def note_text(c: str) -> str:
    return re.sub(r"^(please\s+)?(note|record|remember|log|add a note|not al|kaydet|not)\s*(that|:)?\s*", "", c,
                  flags=re.I).strip() or c


def slots_for(intent: str, clause: str, labels: dict[str, str]) -> dict:
    if intent == "seek":
        return seek_slots(clause)
    if intent == "zoom":
        return zoom_slots(clause, labels)
    if intent in ("isolate", "trajectory"):
        return object_slots(clause, labels)
    if intent == "view":
        return view_slots(clause)
    if intent == "camera":
        return camera_slots(clause)
    if intent == "playback":
        return playback_slots(clause)
    if intent == "layers":
        return layer_slots(clause)
    if intent == "note":
        return {"note_text": note_text(clause)}
    if intent == "ask":
        return {"question": clause}
    if intent == "whatif":
        return {"question": clause}
    if intent == "claim":
        return {"claim": clause}
    if intent == "replay":
        return {"mode": "bullet" if re.search(r"bullet|freeze the (crash|impact)|frozen", clause, re.I) else "slowmo"}
    return {}


# High-precision keyword rules: consulted only when Laya is unsure, before paying for the Spark deep parse.
RULES = [
    # the approach replay: the lead-up to the crash flown through the reconstructed street ("Replay approach" button)
    ("camera", r"\b(replay|play|run|show|watch|fly)\b.*\b(approach|lead.?up|run.?up)\b"),
    # mercy mode: the investigator argues a point; the judge weighs it against the evidence
    ("claim", r"^(objection|i (claim|argue|submit|contend|object|maintain)|my (client|argument|claim)|for the record|"
              r"in (his|her|their) defen[cs]e|consider that|the evidence shows|counter.?point)"),
    ("whatif", r"\bwhat if\b|\bwhat would have happened|\bwould (it|the crash|they|the car) (have|still)|counterfactual|"
               r"\bhad (it|they|the \w+( \w+)?) (been|braked|reacted|slowed|driven)"),
    ("verify", r"\bverify\b|tamper|chain of custody|(evidence|footage|video|files?) (is |are )?(intact|authentic|untouched|"
               r"original|genuine)|integrity of the (evidence|footage)|\bseal\b"),
    ("statement", r"accident (statement|report form)|insurance (form|report)|claim form|european (accident )?"
                  r"(statement|form)|fill (in|out) the (form|report|statement)"),
    ("replay", r"super.?slow|bullet.?time|replay (the )?(impact|crash|collision|moment)|frame by frame|"
               r"freeze the (crash|impact)|slow.?mo(tion)? replay"),
    ("view", r"\b(street ?view|street.level|map|location|where (did|was|is)|harita|konum|nerede|kanıt|videoya|rapor|sahne)|"
             r"\b(open|show|go|switch|take me|back)\b.*\b(report|3d|4d|evidence|video)\b|"
             r"\b(open|show|go to|switch to)\b.*\b(the )?scene\b"),
    ("zoom", r"\b(plate|plaka|enhance|sharpen|netleş|kalite)"),
    ("analyze", r"\b(fault|blame|responsib|verdict|who caused|kimin hata|kusur|analiz)"),
    ("diagram", r"\b(blueprint|diagram|schematic|overhead image|reconstruction image|şema|çizim)"),
    ("note", r"^(please\s+)?(note|record|remember|log)\b|not al|kaydet"),
    ("trajectory", r"\b(trajectory|path|trail|yörünge|rota)\b"),
    ("zoom", r"\b(zoom|enhance|magnify|close.?up|yakınlaş)"),
    ("layers", r"\b(hide|show|turn (on|off)|gizle|göster)\b.*\b(mask|track|label|cloud|maske|etiket|bulut)"),
    ("camera", r"\b(orbit|top view|bird|driver view|first.person|side view|fly around|walk around|free camera|fly through|explore the scene|look around|üstten|kuş bakışı|yandan|sürücü)"),
    ("seek", r"\b(rewind|go to|jump|skip|seconds? (before|after)|back to the (start|beginning)|saniye|anına|geri sar|"
             r"başa|\bgit\b|atla)"),
    ("playback", r"^(?!(what|how|why|when|which|who|was|were|did|is|are)\b).*\b(play|pause|stop|freeze|resume|slow|speed|"
                 r"oynat|durdur|yavaş|devam)\b"),
    ("reset", r"\b(reset|clear|start over|sıfırla|temizle)\b"),
    ("isolate", r"\b(isolate|highlight|segment|show only|izole|vurgula)|araç\w*\s+göster"),
    # a question none of the above covers is a question about the case: answered briefly, out loud
    ("ask", r"^(what|why|how|when|which|who|was|were|did|does|do|is|are|can|could|would|should|tell me|explain|"
            r"describe|summari[sz]e)\b|\?$"),
]


def rule_intent(clause: str) -> Optional[str]:
    lc = clause.lower()
    for intent, pat in RULES:
        if re.search(pat, lc):
            return intent
    return None


def missing_slot(intent: str, s: dict) -> bool:
    return ((intent == "seek" and "t_seconds" not in s) or (intent == "camera" and "camera_preset" not in s)
            or (intent == "layers" and "layer" not in s) or (intent == "view" and "view" not in s))


# ---------------------------------------------------------------- Laya + fallback
COMMAND_SCHEMA = {
  "type": "object", "additionalProperties": False, "required": ["commands"],
  "properties": {"commands": {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "concept", "object_id", "t_seconds", "relative_to_impact", "camera_preset", "speed",
                 "note_text", "view", "enhance"],
    "properties": {
      "intent": {"type": "string", "enum": INTENTS},
      "concept": {"type": ["string", "null"]},
      "object_id": {"type": ["integer", "null"]},
      "t_seconds": {"type": ["number", "null"]},
      "relative_to_impact": {"type": "boolean"},
      "camera_preset": {"type": ["string", "null"], "enum": ["top", "driver", "orbit", "side", "follow", "free", "bullet", None]},
      "speed": {"type": ["number", "null"]},
      "note_text": {"type": ["string", "null"]},
      "view": {"type": ["string", "null"], "enum": ["video", "scene", "map", "report", "evidence", None]},
      "enhance": {"type": "boolean"}}}}}}


def deep_parse(transcript: str, concepts: list[str], timeout: float = 10.0) -> list[dict]:
    from mercy.meta_client import structured
    r = structured(config.VISION_MODEL, [
        {"role": "system", "content": "Convert the investigator's spoken command (English or Turkish) into forensic "
         f"viewer commands. Allowed concepts: {concepts}. Vehicle 1 is the camera car, vehicle 2 the other party; "
         "object_id 0 means all involved vehicles. Negative t_seconds means before; relative_to_impact=true when the "
         "time is measured from the impact. playback: speed 0.25 for slow motion, speed null and t null for pause. "
         "view switches screens (video, scene = 3D/4D, map = where it happened, report, evidence). enhance=true when "
         "the user wants a sharper / higher-quality / readable image (e.g. of a licence plate: concept 'license plate')."},
        {"role": "user", "content": transcript}], COMMAND_SCHEMA, "Commands", effort="minimal", timeout=timeout)
    return r["commands"]


class CommandParser:
    def __init__(self, router=None):
        self._router = router

    @property
    def router(self):
        if self._router is None:
            from mercy.analysis.integrity import router
            self._router = router()
        return self._router

    def parse(self, transcript: str, labels: dict[str, str] | None = None, concepts: list[str] | None = None,
              allow_deep: bool = True, on_deep=None) -> tuple[list[dict], dict]:
        """Returns (commands, info). Each command: {intent, clause, confidence, router, ms, **slots}."""
        labels = labels or {}
        clauses = split_clauses(transcript)
        t0 = time.perf_counter()
        # precise keyword rules win, so Laya only runs on the clauses they do not cover (saves ~0.5 s per command)
        need = [c for c in clauses if not rule_intent(c)]
        got = iter(self.router.predict_batch([{"state": {"request": c}, "questions": INTENT_QUESTION} for c in need],
                                             min_confidence=config.LAYA_MIN_CONFIDENCE) if need else [])
        ruled = {"answers": {"intent": {"choice": None, "answer_confidence": 1.0}}, "routing": {"model": "rules"}}
        results = [ruled if rule_intent(c) else next(got) for c in clauses]
        ms = (time.perf_counter() - t0) * 1000
        cmds, need_deep = [], False
        for c, r in zip(clauses, results):
            ans = r["answers"]["intent"]
            conf = float(ans.get("answer_confidence", ans.get("confidence", 0)))
            intent = ans.get("choice")
            router = f"laya:{r.get('routing', {}).get('model', '?')}"
            low = bool(ans.get("low_confidence")) or ans.get("abstention") == "abstained" or conf < config.LAYA_MIN_CONFIDENCE
            rule = rule_intent(c)
            if rule:                           # precise keyword rules win; Laya decides everything they do not cover
                intent, router, low = rule, "rules", False
            s = slots_for(intent, c, labels) if intent else {}
            if low or intent is None or missing_slot(intent, s):
                need_deep = True
            cmds.append({"intent": intent, "clause": c, "confidence": round(conf, 3), "router": router,
                         "ms": round(ms / len(clauses), 1), **s})
        info = {"clauses": clauses, "laya_ms": round(ms, 1), "deep": False}
        if need_deep and allow_deep:
            if on_deep:
                on_deep()
            t1 = time.perf_counter()
            try:
                deep = deep_parse(transcript, (concepts or list(CONCEPTS)) + list(labels))
                dms = (time.perf_counter() - t1) * 1000
                info["deep"] = True
                out = []
                for d in deep:
                    s = {k: v for k, v in d.items() if v is not None and k != "intent"}
                    s.pop("relative_to_impact", None)
                    if d["intent"] == "seek" and d.get("t_seconds") is not None:
                        s["anchor"] = "impact" if d.get("relative_to_impact") else "absolute"
                    if d["intent"] == "playback":
                        s["playing"] = not (d.get("speed") is None and d.get("t_seconds") is None
                                            and re.search(r"pause|stop|freeze|durdur", transcript, re.I))
                    if d["intent"] in ("isolate", "zoom", "trajectory") and d.get("concept"):
                        hit = process.extractOne(d["concept"].lower(), list(labels), scorer=fuzz.partial_ratio) if labels else None
                        if hit and hit[1] >= 85:
                            s["sam_key"] = labels[hit[0]]
                        elif d["concept"] not in CONCEPTS:
                            s["live"] = True
                    if d["intent"] == "camera" and d.get("camera_preset"):
                        s["camera_preset"] = d["camera_preset"]
                    out.append({"intent": d["intent"], "clause": transcript, "confidence": 1.0,
                                "router": "deep:muse-spark", "ms": round(dms, 1), **s})
                if out:
                    return out, info
            except Exception as e:      # Spark unreachable -> keep Laya's best guess, flagged
                log.warning("deep parse failed: %s", e)
                info["deep_error"] = str(e)
        return [c for c in cmds if c["intent"]], info

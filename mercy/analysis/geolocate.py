"""Where did it happen? geo-sleuth evidence + Muse Spark judgment + OpenStreetMap verification -> location.json.

1. Keyframes go through geo-sleuth's intake.py (metadata, OCR, optional reverse image search).
2. Burned-in GPS (dashcam overlays) is read straight from the OCR: the strongest evidence there is.
3. Otherwise Muse Spark looks at the keyframes + OCR text with web search and proposes candidates (geo-sleuth's
   "the model judges only among the top few"), then OpenStreetMap Nominatim geocodes the best one.
4. geo-sleuth osm.py snaps the point to the nearest mapped road (Overpass; skipped if the servers are busy).
geo-sleuth (MIT): https://github.com/Oldcircle/geo-sleuth. Use only on footage you may analyse; it locates the
scene, never people.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2

from mercy import config
from mercy.config import log

SKILL = config.ROOT / "vendor" / "geo-sleuth" / "skills" / "geo-sleuth"
UA = "mercymek/1.0 (forensic demo; contact via repo)"
GPS_RE = re.compile(r"(-?\d{1,2}\.\d{4,6})[,\s]+(-?\d{1,3}\.\d{4,6})")

LOC_SCHEMA = {"type": "object", "additionalProperties": False,
  "required": ["place_name", "geocode_query", "lat", "lon", "radius_m", "confidence", "camera_heading_deg",
               "evidence", "alternatives"],
  "properties": {
    "place_name": {"type": "string"}, "geocode_query": {"type": "string"},
    "lat": {"type": ["number", "null"]}, "lon": {"type": ["number", "null"]},
    "radius_m": {"type": "number"}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "camera_heading_deg": {"type": ["number", "null"]},
    "evidence": {"type": "array", "items": {"type": "string"}},
    "alternatives": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["place_name", "lat", "lon"], "properties": {"place_name": {"type": "string"},
        "lat": {"type": ["number", "null"]}, "lon": {"type": ["number", "null"]}}}}}}


def _http_json(url: str, timeout: float = 20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def nominatim_search(q: str):
    time.sleep(1.1)                                           # Nominatim usage policy: max 1 request/second
    r = _http_json("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": q, "format": "jsonv2", "limit": 1}))
    return (float(r[0]["lat"]), float(r[0]["lon"]), r[0]["display_name"]) if r else None


def nominatim_reverse(lat: float, lon: float):
    time.sleep(1.1)
    try:
        r = _http_json("https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
            {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 17}))
        return r.get("display_name"), r.get("address", {})
    except Exception:
        return None, {}


def haversine(a, b):
    R = 6371000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    return 2 * R * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def bearing(a, b):
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def intake(img: pathlib.Path, out: pathlib.Path, rev: bool = False) -> list[dict]:
    """geo-sleuth intake.py (OCR + metadata [+ reverse image search]); returns OCR items."""
    cmd = ["uv", "run", str(SKILL / "scripts" / "intake.py"), str(img), "--out-dir", str(out)] + ([] if rev else ["--no-rev"])
    subprocess.run(cmd, cwd=SKILL, capture_output=True, timeout=900)
    p = out / "ocr.json"
    return json.loads(p.read_text(encoding="utf-8")).get("items", []) if p.exists() else []


def snap_to_road(lat: float, lon: float, out: pathlib.Path) -> dict | None:
    """geo-sleuth osm.py geom: nearest mapped road within ~250 m (Overpass; optional)."""
    d = 0.0025
    bbox = f"{lat - d},{lon - d * 1.4},{lat + d},{lon + d * 1.4}"
    gj = out / "roads.geojson"
    try:
        subprocess.run(["uv", "run", str(SKILL / "scripts" / "osm.py"), "geom", '["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|motorway_link|trunk_link|primary_link)$"]',
                        "--bbox", bbox, "--out", str(gj)], cwd=SKILL, capture_output=True, timeout=120)
        feats = json.loads(gj.read_text(encoding="utf-8")).get("features", [])
    except Exception as e:
        log.info("osm snap skipped: %s", e); return None
    best = None
    for f in feats:
        g = f.get("geometry", {})
        lines = [g["coordinates"]] if g.get("type") == "LineString" else g.get("coordinates", []) if g.get("type") == "MultiLineString" else []
        for line in lines:
            for (x1, y1), (x2, y2) in zip(line, line[1:]):
                # project onto the segment in a local metric frame
                kx = 111320 * math.cos(math.radians(lat))
                ax, ay, bx, by = (x1 - lon) * kx, (y1 - lat) * 110540, (x2 - lon) * kx, (y2 - lat) * 110540
                vx, vy = bx - ax, by - ay
                u = max(0, min(1, -(ax * vx + ay * vy) / (vx * vx + vy * vy + 1e-9)))
                px, py = ax + u * vx, ay + u * vy
                dist = math.hypot(px, py)
                if best is None or dist < best["dist_m"]:
                    p = f.get("properties", {})
                    best = {"lat": lat + py / 110540, "lon": lon + px / kx, "dist_m": round(dist, 1),
                            "road": p.get("name") or p.get("ref") or p.get("highway"), "highway": p.get("highway"),
                            "ref": p.get("ref")}
    return best


def keyframes(case: pathlib.Path, meta: dict, out: pathlib.Path) -> list[tuple[float, pathlib.Path]]:
    imp = json.loads((case / "physics.json").read_text())["impact"]["t"] if (case / "physics.json").exists() else meta["time_of_event"]
    ts = sorted({round(min(meta["duration"] - 0.1, max(0.0, t)), 2) for t in (0.5, meta["duration"] * 0.45, imp)})
    cap = cv2.VideoCapture(str(case / "clip.mp4"))
    res = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if ok:
            p = out / f"k_{t:05.2f}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            res.append((t, p))
    cap.release()
    return res


def judge(frames, ocr_text: str, scene: dict) -> dict:
    """Muse Spark with web search proposes the location; a second call turns the answer into strict JSON."""
    from mercy.meta_client import data_url, meta_client, structured
    content = [{"type": "input_text", "text":
        "Geolocate this dashcam footage. Use only what is visible: road signs, route numbers, language, plate formats, "
        "driving side, road furniture, architecture, vegetation, terrain. Search the web to confirm named places. "
        f"OCR text read from the frames:\n{ocr_text or '(none)'}\nScene: {scene.get('summary', '')}\n"
        "Answer: the most likely place (as specific as the evidence allows: road + nearest junction + city + country), "
        "approximate coordinates, an honest error radius in metres, the evidence for it, and up to 2 alternatives."}]
    content += [{"type": "input_image", "image_url": data_url(p)} for _, p in frames]
    r = meta_client().with_options(timeout=180).responses.create(
        model=config.VISION_MODEL, input=[{"type": "message", "role": "user", "content": content}],
        tools=[{"type": "web_search", "search_context_size": "medium"}], reasoning={"effort": "medium"},
        max_output_tokens=6000)
    answer = "\n".join(b.text for it in r.output if it.type == "message" and getattr(it, "phase", None) != "commentary"
                       for b in it.content if getattr(b, "type", "") == "output_text") or r.output_text
    loc = structured(config.VISION_MODEL, [{"role": "user", "content":
        "Convert this geolocation analysis into JSON. geocode_query must be a short address string that "
        "OpenStreetMap Nominatim can resolve (e.g. 'Highway 400, Toronto, Ontario'). Use null when unknown.\n\n" + answer}],
        LOC_SCHEMA, "Location", effort="minimal")
    loc["analysis"] = answer
    return loc


def run(case: pathlib.Path, force: bool = False, rev: bool = False) -> dict:
    out_path = case / "location.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    meta = json.loads((case / "meta.json").read_text())
    scene = json.loads((case / "scene.json").read_text()) if (case / "scene.json").exists() else {}
    gdir = case / "geo"; gdir.mkdir(exist_ok=True)
    frames = keyframes(case, meta, gdir)
    with ThreadPoolExecutor(max_workers=3) as ex:
        ocr = list(ex.map(lambda tp: intake(tp[1], gdir / f"intake_{tp[0]:05.2f}", rev=rev and tp is frames[len(frames) // 2]), frames))
    lines = [(t, it["text"], it.get("conf", it.get("score", 1.0))) for (t, _), items in zip(frames, ocr) for it in items]
    ocr_text = "\n".join(f"t={t:.1f}s: {txt}" for t, txt, _ in lines)
    evidence, gps = [], []
    for t, txt, c in lines:
        for m in GPS_RE.finditer(txt):
            la, lo = float(m.group(1)), float(m.group(2))
            if abs(la) <= 90 and abs(lo) <= 180:
                gps.append((t, la, lo))
    gps = sorted({(round(t, 2), round(la, 5), round(lo, 5)) for t, la, lo in gps})    # OCR reads overlays twice
    loc: dict
    if gps:
        imp = json.loads((case / "physics.json").read_text())["impact"]["t"] if (case / "physics.json").exists() else meta["time_of_event"]
        t, la, lo = min(gps, key=lambda g: abs(g[0] - imp))
        heading = bearing((gps[0][1], gps[0][2]), (gps[-1][1], gps[-1][2])) if len(gps) >= 2 and haversine(gps[0][1:], gps[-1][1:]) > 15 else None
        loc = {"place_name": "", "lat": la, "lon": lo, "radius_m": 25.0, "confidence": "high",
               "camera_heading_deg": heading, "method": "GPS read from the dashcam overlay (geo-sleuth OCR)",
               "evidence": [f"GPS overlay at t={g[0]:.1f}s reads {g[1]:.5f}, {g[2]:.5f}" for g in gps],
               "alternatives": [], "gps_track": [[g[0], g[1], g[2]] for g in gps]}
    else:
        loc = judge(frames, ocr_text, scene)
        loc["method"] = "visual clues + OCR (geo-sleuth) judged with web search, geocoded with OpenStreetMap"
        hit = nominatim_search(loc["geocode_query"]) if loc.get("geocode_query") else None
        if hit and (loc.get("lat") is None or haversine((loc["lat"], loc["lon"]), hit[:2]) > 3000):
            loc["evidence"].append(f"OpenStreetMap geocoded '{loc['geocode_query']}' to {hit[0]:.5f}, {hit[1]:.5f}")
            loc["lat"], loc["lon"] = hit[0], hit[1]
        elif hit:
            loc["evidence"].append(f"OpenStreetMap confirms '{loc['geocode_query']}' within "
                                   f"{haversine((loc['lat'], loc['lon']), hit[:2]):.0f} m of the estimate")
    if loc.get("lat") is not None:
        snap = snap_to_road(loc["lat"], loc["lon"], gdir)
        if snap and snap["dist_m"] < max(60.0, loc["radius_m"]):
            loc["road"] = snap
            loc["evidence"].append(f"Nearest mapped road (OSM): {snap['road']} ({snap['dist_m']:.0f} m)")
        name, addr = nominatim_reverse(loc["lat"], loc["lon"])
        loc["address"] = name
        loc["place_name"] = loc.get("place_name") or ", ".join(x for x in [
            (loc.get("road") or {}).get("road") or addr.get("road"), addr.get("city") or addr.get("town") or addr.get("suburb"),
            addr.get("state"), addr.get("country")] if x)
    loc["ocr"] = [{"t": t, "text": txt} for t, txt, _ in lines]
    # dashcam overlays often print the camera car's own speed: ground truth for the reconstruction
    spd = sorted({(round(t, 2), int(m.group(1))) for t, txt, _ in lines
                  for m in re.finditer(r"(\d{1,3})\s*(?:KM/?H|KMH)", txt, re.I)})
    if spd:
        loc["overlay_speed_kmh"] = [list(s) for s in spd]
    loc["keyframes"] = [f"geo/{p.name}" for _, p in frames]
    out_path.write_text(json.dumps(loc, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("location: %s (%s, ±%.0f m) via %s", loc.get("place_name"), loc.get("confidence"), loc.get("radius_m", 0),
             loc.get("method"))
    return loc

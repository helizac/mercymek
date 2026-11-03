"""Open street-level imagery around the scene -> streetview.json.

Providers, in order: Mapillary (largest coverage; needs a free client token in MAPILLARY_TOKEN) and Panoramax
(fully open, federated, no key). The search widens from ~150 m to ~1.5 km until images are found; results are
sorted by distance, preferring pictures that face the crash point.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import urllib.parse
import urllib.request

from mercy.analysis.geolocate import UA, bearing, haversine
from mercy.config import log

RADII = (0.0015, 0.005, 0.014)          # degrees of latitude (~150 m, ~550 m, ~1.5 km)


def _get(url: str, headers: dict | None = None, timeout: float = 25):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _score(lat, lon, it):
    d = haversine((lat, lon), (it["lat"], it["lon"]))
    facing = 0.0
    if it.get("heading") is not None and d > 5:
        diff = abs((bearing((it["lat"], it["lon"]), (lat, lon)) - it["heading"] + 540) % 360 - 180)
        facing = 1 if diff < 45 else 0
    return d * (0.6 if facing else 1.0), d


def mapillary(lat: float, lon: float, r: float, token: str) -> list[dict]:
    k = 1 / math.cos(math.radians(lat))
    bbox = f"{lon - r * k},{lat - r},{lon + r * k},{lat + r}"
    q = urllib.parse.urlencode({"fields": "id,computed_geometry,compass_angle,captured_at,thumb_1024_url,is_pano",
                                "bbox": bbox, "limit": 60, "access_token": token})
    data = _get("https://graph.mapillary.com/images?" + q).get("data", [])
    out = []
    for d in data:
        g = (d.get("computed_geometry") or {}).get("coordinates")
        if not g:
            continue
        out.append({"provider": "Mapillary", "id": d["id"], "lat": g[1], "lon": g[0], "heading": d.get("compass_angle"),
                    "date": str(d.get("captured_at", ""))[:10], "url": d.get("thumb_1024_url"), "pano": d.get("is_pano"),
                    "page": f"https://www.mapillary.com/app/?pKey={d['id']}", "license": "CC BY-SA 4.0"})
    return out


def panoramax(lat: float, lon: float, r: float) -> list[dict]:
    k = 1 / math.cos(math.radians(lat))
    bbox = f"{lon - r * k},{lat - r},{lon + r * k},{lat + r}"
    feats = _get(f"https://api.panoramax.xyz/api/search?bbox={bbox}&limit=60").get("features", [])
    out = []
    for f in feats:
        lo, la = f["geometry"]["coordinates"][:2]
        p, a = f.get("properties", {}), f.get("assets", {})
        url = (a.get("sd") or a.get("hd") or a.get("thumb") or {}).get("href")
        if not url:
            continue
        out.append({"provider": "Panoramax", "id": f["id"], "lat": la, "lon": lo, "heading": p.get("view:azimuth"),
                    "date": str(p.get("datetime", ""))[:10], "url": url, "pano": p.get("pers:interior_orientation", {}).get("field_of_view") == 360,
                    "page": f"https://api.panoramax.xyz/#focus=pic&pic={f['id']}", "license": p.get("license", "open licence")})
    return out


def run(case: pathlib.Path, force: bool = False, keep: int = 8) -> dict:
    out = case / "streetview.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    locp = case / "location.json"
    loc = json.loads(locp.read_text(encoding="utf-8")) if locp.exists() else {}
    if loc.get("lat") is None:
        res = {"items": [], "note": "No location, so no street-level imagery."}
        out.write_text(json.dumps(res), encoding="utf-8"); return res
    lat, lon = loc["lat"], loc["lon"]
    token = os.environ.get("MAPILLARY_TOKEN", "")
    items, tried = [], []
    for r in RADII:
        for name, fn in ((("Mapillary", lambda: mapillary(lat, lon, r, token)),) if token else ()) + \
                        (("Panoramax", lambda: panoramax(lat, lon, r)),):
            try:
                got = fn()
                tried.append(f"{name} ±{int(r * 111000)} m: {len(got)}")
                items += got
            except Exception as e:
                tried.append(f"{name}: {e}")
        if len(items) >= 3:
            break
    for it in items:
        s, d = _score(lat, lon, it)
        it["distance_m"] = round(d)
        it["_s"] = s
    items = sorted({it["id"]: it for it in items}.values(), key=lambda it: it["_s"])[:keep]
    for it in items:
        it.pop("_s", None)
    res = {"items": items, "searched": tried, "center": [lat, lon]}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("street view: %d images (%s)", len(items), "; ".join(tried))
    return res

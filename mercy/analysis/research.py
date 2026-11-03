"""Muse Spark + web_search -> research.json (section 9.5). One question per request; cached at ingest."""
from __future__ import annotations

import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

from mercy import config
from mercy.config import log
from mercy.meta_client import meta_client


def research(question: str, key: str) -> dict:
    r = meta_client().responses.create(
        model=config.VISION_MODEL, input=question, reasoning={"effort": "low"}, max_output_tokens=3000,
        tools=[{"type": "web_search", "search_context_size": "medium",
                "user_location": {"type": "approximate", "country": "US"}}],
        include=["web_search_call.results"])
    cites, seen, answer, opened, considered = [], set(), [], [], []
    for item in r.output:
        if item.type == "message":
            if getattr(item, "phase", None) == "commentary":     # agent narration ("I'll search ..."), not the answer
                continue
            for block in item.content:
                if getattr(block, "type", "") != "output_text":
                    continue
                answer.append(block.text)
                for a in (block.annotations or []):
                    if getattr(a, "type", "") == "url_citation" and a.url not in seen:
                        seen.add(a.url)
                        cites.append({"title": a.title or a.url, "url": a.url,
                                      "quote_span": block.text[a.start_index:a.end_index][:300], "kind": "cited"})
        elif item.type == "web_search_call":
            act = getattr(item, "action", None)
            act = act if isinstance(act, dict) else (act.model_dump() if hasattr(act, "model_dump") else {})
            if act.get("type") == "open_page" and act.get("url"):
                url = act["url"].split("?")[0]
                opened.append({"title": url, "url": url, "kind": "opened"})
            for res in (getattr(item, "results", None) or []):
                g = res.get if isinstance(res, dict) else (lambda k, _r=res: getattr(_r, k, None))   # dicts or SDK objects
                if g("url"):
                    considered.append({"title": g("title") or g("url"), "url": g("url"), "kind": "searched"})
    # url_citation annotations first; then pages the model opened; then the top search results it read
    for c in opened + considered[:3]:
        if c["url"] not in seen and len(cites) < 5:
            seen.add(c["url"]); cites.append(c | {"quote_span": ""})
    return {"key": key, "question": question, "answer": "\n".join(answer).strip() or r.output_text.strip(),
            "citations": cites}


def questions(scene: dict, meta: dict) -> dict[str, str]:
    other = next((v for v in scene["vehicles"] if v["role"] != "camera_vehicle"), None)
    body = (other or {}).get("make_model_guess") or f"{(other or {}).get('color', '')} {(other or {}).get('body_type', 'car')}".strip()
    controls = ", ".join(scene.get("traffic_controls") or ["none visible"])
    return {
        "rule": (f"A dashcam recorded this crash on a {scene['road_type']} in the United States (state unknown). "
                 f"Traffic controls visible: {controls}. Summary: {scene['summary']} "
                 "Which right-of-way / traffic rules apply to each driver here? Use the Uniform Vehicle Code and typical "
                 "US state statutes. Answer in 4-6 short bullet points and cite sources."),
        "vehicle": (f"What are the typical overall length and width in metres of a {body}? "
                    "Give one line with numbers and cite a source."),
        "statistics": (f"What do NHTSA or IIHS data say about crashes of this type: {scene['summary'][:220]} "
                       "Give 2-3 short facts with numbers and cite sources."),
        "reaction": ("What driver perception-reaction time and dry/wet asphalt friction coefficients are commonly used in "
                     "US crash reconstruction (AASHTO, NHTSA, SAE)? Give the numbers in 2-3 bullet points with sources."),
    }


def run(case: pathlib.Path, force: bool = False) -> dict:
    out = case / "research.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    scene = json.loads((case / "scene.json").read_text())
    meta = json.loads((case / "meta.json").read_text())
    qs = questions(scene, meta)
    with ThreadPoolExecutor(max_workers=4) as ex:
        items = list(ex.map(lambda kv: research(kv[1], kv[0]), qs.items()))
    res = {"items": items}
    out.write_text(json.dumps(res, indent=1))
    log.info("research: %s", {i["key"]: len(i["citations"]) for i in items})
    return res

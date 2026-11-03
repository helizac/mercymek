"""End-to-end ingest for one case (section 12). Every step is idempotent: it skips when its output exists.

    python scripts/ingest_case.py --video data/raw/nexar/train/positive/00348.mp4 --case hero
    python scripts/ingest_case.py --video raw.mp4 --case cctv1 --t-event 23.4 --concepts car truck "white SUV"
    python scripts/ingest_case.py --case hero --force physics verdict     # redo selected steps

Order: trim -> frames -> upload -> scene (Spark) -> SAM (parallel) -> overlays -> VGGT -> scale + lift -> TSDF mesh
       -> physics -> 3D moments (AnySplat world + photo splats of the key instants) -> RAFT slow motion
       -> redaction + evidence -> research (web_search) -> verdict (cache) + Laya integrity -> diagram -> blueprint
       -> accident statement -> evidence seal
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mercy import config  # noqa: E402
from mercy.config import case_dir, log, timed  # noqa: E402

STEPS = ["video", "scene", "segment", "overlays", "recon", "lift", "mesh", "physics", "moments", "slowmo", "evidence", "plates", "location",
         "street", "research", "verdict", "diagram", "blueprint", "statement", "seal"]


def recon_python() -> str:
    """The reconstruction env (torch + vggt) if present, else this interpreter."""
    for p in (os.environ.get("RECON_PYTHON"), ROOT / ".venv-recon" / "Scripts" / "python.exe",
              ROOT / ".venv-recon" / "bin" / "python"):
        if p and pathlib.Path(p).exists():
            return str(p)
    return sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=pathlib.Path)
    ap.add_argument("--case", required=True)
    ap.add_argument("--t-event", type=float)
    ap.add_argument("--before", type=float, help="seconds retained before the event (default: 8)")
    ap.add_argument("--after", type=float, help="seconds retained after the event (default: 4)")
    ap.add_argument("--title")
    ap.add_argument("--source", default="user", help="provenance, e.g. 'Wikimedia Commons (public domain)'")
    ap.add_argument("--source-manifest", type=pathlib.Path,
                    help="required for new examples: open license, native HD, collision and unblurred readable plate review")
    ap.add_argument("--concepts", nargs="*")
    ap.add_argument("--recon", choices=["vggt", "mono"], default="vggt")
    ap.add_argument("--gpu-backend", choices=["kaggle", "local"], default="kaggle",
                    help="GPU model execution (default: private Kaggle jobs with cached weights)")
    ap.add_argument("--moments-backend", choices=["generated", "legacy"], default="generated")
    ap.add_argument("--max-frames", type=int, default=96)
    ap.add_argument("--prepare-only", action="store_true", help="stop after video, scene, masks and overlays; leave GPU work pending")
    ap.add_argument("--force", nargs="*", default=None, help="steps to redo (no value = all)")
    ap.add_argument("--skip", nargs="*", default=[], help="steps to skip (e.g. blueprint)")
    a = ap.parse_args()
    force = set(STEPS if a.force == [] else (a.force or []))
    case = case_dir(a.case)
    if a.video and not (case / "meta.json").exists():
        from mercy.ingest.source_quality import validate
        if not a.source_manifest:
            ap.error("New examples require --source-manifest (open license, 720p+, collision, clear unblurred plates)")
        provenance = validate(a.video, a.source_manifest)
        a.source = f"{provenance['source_url']} ({provenance['license']})"
    case.mkdir(parents=True, exist_ok=True)
    if a.source_manifest:
        import shutil
        shutil.copy2(a.source_manifest, case / "source_provenance.json")
    F = lambda s: s in force  # noqa: E731

    from mercy.analysis import enhance, geolocate, physics, research, streetview, verdict
    from mercy.generation import blueprint, diagram
    from mercy.analysis import seal, statement
    from mercy.ingest import evidence, lift, scene, segment, video

    with timed("video: trim + frames"):
        if a.video is None and not (case / "meta.json").exists():
            sys.exit("--video is required for a new case")
        meta = video.prepare(a.video, case, a.case, a.t_event, a.title, force=F("video"), source=a.source,
                             before_s=a.before, after_s=a.after) if a.video else \
            json.loads((case / "meta.json").read_text())
        if a.source_manifest:
            source_info = json.loads(a.source_manifest.read_text())
            attribution = {k: source_info.get(k) for k in ('author', 'title', 'source_url', 'license', 'license_url')}
            attribution['modifications'] = 'Excerpt trimmed; faces redacted. The 3D reconstruction is estimated.'
            if meta.get('source_attribution') != attribution:
                meta['source_attribution'] = attribution
                (case / 'meta.json').write_text(json.dumps(meta, indent=2))
    with timed("scene: Muse Spark video understanding"):
        sc = scene.understand_scene(case, meta, force=F("scene"))
    concepts = list(dict.fromkeys((a.concepts or segment.DEFAULT_CONCEPTS) + evidence.PRIVACY_CONCEPTS + ["sky"] +
                                  [v["label"] for v in sc["vehicles"] if v["role"] != "camera_vehicle"
                                   and len(v["label"].split()) <= 3]))
    (case / "concepts.json").write_text(json.dumps(concepts))
    if "segment" not in a.skip:
        with timed(f"segment: SAM 3.1 x {len(concepts)} concepts"):
            segment.segment_all(case, concepts, force=F("segment"))
    with timed("overlays"):
        segment.build_overlays(case, [c for c in concepts if c not in ("face", "sky")], force=F("overlays") or F("segment"))
    if a.prepare_only:
        log.info("case '%s': video and detections prepared; reconstruction is pending", a.case)
        return
    with timed(f"recon: {a.recon}"):
        if a.gpu_backend == "kaggle":
            if a.recon != "vggt":
                ap.error("The Kaggle motion worker uses VGGT; --recon mono requires --gpu-backend local")
            from mercy.ingest.kaggle_motion import run
            run(case, "recon", max_frames=min(a.max_frames, 32), force=F("recon"))
        else:
            cmd = [recon_python(), "-m", "mercy.ingest.reconstruct", "--case", a.case, "--cases-dir", str(config.CASES_DIR),
                   "--method", a.recon, "--max-frames", str(a.max_frames)] + (["--force"] if F("recon") else [])
            subprocess.run(cmd, check=True, cwd=ROOT)
    with timed("lift: scale + 3D tracks + cloud"):
        lift.lift(case, concepts, force=F("lift") or F("recon"))
    with timed("sky masks"):
        lift.sky_masks(case, force=F("mesh") or F("segment"))
    with timed("mesh: TSDF fusion -> scene.glb + 4D frames"):
        cmd = [recon_python(), "-m", "mercy.ingest.mesh", "--case", a.case, "--cases-dir", str(config.CASES_DIR)] + \
              (["--force"] if (F("mesh") or F("lift") or F("recon")) else [])
        subprocess.run(cmd, check=True, cwd=ROOT)
    with timed("physics"):
        physics.analyze(case, force=F("physics") or F("lift") or F("recon"))
    if a.moments_backend == "legacy":
        with timed("moments: legacy AnySplat photo layers"):
            cmd = [recon_python(), "-m", "mercy.ingest.moments", "--case", a.case, "--cases-dir", str(config.CASES_DIR)] + \
                  (["--force"] if (F("moments") or F("physics") or F("mesh") or F("lift") or F("recon")) else [])
            if subprocess.run(cmd, cwd=ROOT).returncode:
                log.warning("moments step failed; the timeline has no 3D moments")
    with timed("slowmo: RAFT optical-flow interpolation of the impact -> slowmo.mp4"):
        if a.gpu_backend == "kaggle":
            from mercy.ingest.kaggle_motion import run
            run(case, "slowmo", force=F("slowmo") or F("physics"))
        else:
            cmd = [recon_python(), "-m", "mercy.ingest.slowmo", "--case", a.case, "--cases-dir", str(config.CASES_DIR)] + \
                  (["--force"] if (F("slowmo") or F("physics")) else [])
            if subprocess.run(cmd, cwd=ROOT).returncode:
                log.warning("slow-motion step failed; replays fall back to quarter-speed playback")
    with timed("evidence: face redaction + stills"):
        evidence.redact(case, force=F("evidence"))
        evidence.stills(case, force=F("evidence") or F("physics"))
    if "plates" not in a.skip:
        with timed("plates: enhance + read"):
            enhance.plates_for_case(case, force=F("plates") or F("physics"))
    if a.moments_backend == "generated" and "moments" not in a.skip:
        with timed("moments: reviewed empty environment + Kaggle SHARP + complete vehicle models"):
            current = json.loads((case / "moments.json").read_text()) if (case / "moments.json").exists() else {}
            if F("moments") or F("plates") or F("physics") or F("recon") or F("lift") or not all(m.get("reconstruction") for m in current.get("moments", [])) or not current.get("moments"):
                subprocess.run([sys.executable, str(ROOT / "scripts" / "rebuild_4d.py"), "--case", a.case], cwd=ROOT, check=True)
    if "location" not in a.skip:
        with timed("location: geo-sleuth + OpenStreetMap"):
            try:
                geolocate.run(case, force=F("location"))
            except Exception as e:
                log.warning("geolocation failed: %s", e)
        with timed("street-level imagery (Mapillary / Panoramax)"):
            try:
                streetview.run(case, force=F("street") or F("location"))
            except Exception as e:
                log.warning("street-level imagery failed: %s", e)
    if "research" not in a.skip:
        with timed("research: Muse Spark + web_search"):
            research.run(case, force=F("research"))
    if "verdict" not in a.skip:
        with timed("verdict: Muse Wonder + Laya integrity (cached for the demo)"):
            verdict.run(case, force=F("verdict") or F("physics"))
    with timed("diagram"):
        diagram.draw(case, force=F("diagram") or F("physics"))
    if "blueprint" not in a.skip:
        with timed("blueprint: Muse Image"):
            try:
                blueprint.run(case, force=F("blueprint") or F("diagram") or F("physics"))
            except Exception as e:
                log.warning("blueprint failed (%s); the HUD will fall back to the diagram", e)
    if "statement" not in a.skip:
        with timed("statement: European Accident Statement, section 12"):
            try:
                statement.build(case, force=F("statement") or F("verdict"))
            except Exception as e:
                log.warning("accident statement failed: %s", e)
    with timed("seal: signed SHA-256 manifest of the footage and every artifact"):
        s = seal.seal(case)
        log.info("sealed %d files, root %s", len(s["files"]), s["root"][:16])
    log.info("case '%s' ready in %s", a.case, case)


if __name__ == "__main__":
    main()

"""Bundle the files the live demo serves into a zip small enough to share (GitHub release, Hugging Face, Drive).

    python scripts/pack_demo.py                       # hero only
    python scripts/pack_demo.py --cases hero snow_pileup cutin_rearend
    python scripts/pack_demo.py --unpack mercymek-demo-data.zip   # on the judge's machine, from the repo root

The repo itself ships no footage (a case is hundreds of MB); the pack holds exactly what `/cases/<id>/...` serves plus
the cached spoken phrases, so the demo runs offline with no API keys.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# what the server may hand out, minus the heavy debugging artefacts that the browser never requests
SKIP = {"environment/empty.attempt-1.png", "environment/source.png"}
SKIP_SUFFIX = (".legacy.json", ".anysplat.splat")


def case_files(d: pathlib.Path, files: set[str], dirs: set[str]):
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        if rel in SKIP or rel.endswith(SKIP_SUFFIX):
            continue
        # physics.json is read by the case list and the bundle even though it is not downloaded by the browser
        if rel in files or rel in {"physics.json", "meta.json"} or rel.split("/")[0] in dirs:
            yield p, rel


def pack(cases: list[str], out: pathlib.Path):
    from mercy.server.app import CASE_DIRS, CASE_FILES
    from mercy.analysis.seal import verify
    # A shareable case must retain its original signed manifest. Omitting a
    # server-side intermediate breaks verification on the recipient's machine.
    entries = []
    has_nexar = False
    for cid in cases:
        d = ROOT / "data" / "cases" / cid
        if not (d / "meta.json").exists():
            raise ValueError(f"unknown case {cid}")
        has_nexar |= "nexar" in (d / "meta.json").read_text(encoding="utf-8").lower()
        selected = {rel: p for p, rel in case_files(
            d, CASE_FILES | {"verdict.json", "claims.json", "moments.json"}, CASE_DIRS)}
        if (d / "seal.json").exists():
            result = verify(d)
            if not result["intact"]:
                raise ValueError(f"{cid}: evidence seal is not intact; refusing to package changed evidence")
            for item in json.loads((d / "seal.json").read_text(encoding="utf-8"))["files"]:
                rel = item["path"]
                p = (d / rel).resolve()
                if d.resolve() not in p.parents:
                    raise ValueError(f"unsafe sealed path: {rel}")
                selected[rel] = p
        entries.extend((p, f"data/cases/{cid}/{rel}") for rel, p in sorted(selected.items()))
    notices = [(ROOT/"third_party/NEXAR_LICENSE.txt", "data/licenses/NEXAR_LICENSE.txt"),
               (ROOT/"THIRD_PARTY_NOTICES.md", "data/licenses/THIRD_PARTY_NOTICES.md"),
               (ROOT/"docs/demo-credits.md", "data/licenses/DEMO_CREDITS.md")]
    if has_nexar and not notices[0][0].is_file():
        raise ValueError("Nexar redistribution requires third_party/NEXAR_LICENSE.txt")
    entries.extend((p, name) for p, name in notices if p.is_file())
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".part")
    n = size = 0
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, name in entries:
            z.write(p, name)
            n += 1
            size += p.stat().st_size
        tts = ROOT / "data" / "tts_cache"
        if tts.exists():
            for p in tts.iterdir():
                if p.is_file() and p.name != "usage.json":
                    z.write(p, f"data/tts_cache/{p.name}")
                    n += 1
    temporary.replace(out)
    print(f"{n} files, {size / 1e6:.0f} MB raw -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


def unpack(zip_path: pathlib.Path):
    with zipfile.ZipFile(zip_path) as z:
        # Only demo data belongs in this archive, never arbitrary project files.
        for name in z.namelist():
            rel = pathlib.PurePosixPath(name.replace("\\", "/"))
            if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 3
                    or rel.parts[0] != "data" or rel.parts[1] not in {"cases", "tts_cache", "licenses"}
                    or any(":" in part for part in rel.parts)):
                raise ValueError(f"unexpected archive path: {name}")
        z.extractall(ROOT)
    print(f"unpacked into {ROOT / 'data'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["hero"])
    ap.add_argument("--out", default="mercymek-demo-data.zip")
    ap.add_argument("--unpack", metavar="ZIP")
    a = ap.parse_args()
    unpack(pathlib.Path(a.unpack)) if a.unpack else pack(a.cases, pathlib.Path(a.out))

"""Download a subset of Nexar collision clips (positive cases) with their labels.

    python scripts/download_nexar.py --split train/positive --limit 100
    python scripts/download_nexar.py --ids 00417 00822          # specific clips

Writes data/raw/nexar/<split>/<id>.mp4 and data/raw/nexar/<split>/metadata.csv.
Nexar open data license: research use, no unethical use. See README.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download  # noqa: E402

from mercy.config import RAW_DIR, log  # noqa: E402

REPO = "nexar-ai/nexar_collision_prediction"


def fetch(path: str, out_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(hf_hub_download(REPO, path, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                                        local_dir=out_root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/positive")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--min-duration", type=float, default=38.0, help="skip clips shorter than this (s)")
    ap.add_argument("--daylight-first", action="store_true", default=True)
    args = ap.parse_args()

    out_root = RAW_DIR / "nexar"
    meta_path = fetch(f"{args.split}/metadata.csv", out_root)
    rows = list(csv.DictReader(open(meta_path, newline="", encoding="utf-8")))
    log.info("metadata: %d rows, columns=%s", len(rows), list(rows[0].keys()))

    if args.ids:
        wanted = [r for r in rows if r.get("id", r.get("file_name", "")).split(".")[0].zfill(5) in
                  {i.zfill(5) for i in args.ids}]
    else:
        def ok(r):
            try:
                return float(r.get("duration", 40) or 40) >= args.min_duration
            except ValueError:
                return True
        wanted = [r for r in rows if ok(r)]
        if args.daylight_first and "light_conditions" in rows[0]:
            wanted.sort(key=lambda r: r["light_conditions"] != "Normal")
        wanted = wanted[: args.limit]

    for i, r in enumerate(wanted):
        name = r.get("file_name") or f"{str(r['id']).zfill(5)}.mp4"
        name = name.split("/")[-1]
        try:
            p = fetch(f"{args.split}/{name}", out_root)
            log.info("[%d/%d] %s (%.1f MB)", i + 1, len(wanted), p.name, p.stat().st_size / 1e6)
        except Exception as e:  # keep going: one missing clip should not stop the batch
            log.warning("skip %s: %s", name, e)


if __name__ == "__main__":
    main()

"""One-command empty-scene -> Kaggle 3D -> tracked vehicle replay for any ingested cases.

python scripts/rebuild_4d.py --case cutin_rearend hero lanechange night_rain snow_pileup
Network and Muse calls run locally; all image-to-3D model inference runs on Kaggle CUDA.
Setup is a separate one-time job: python scripts/kaggle_gpu.py setup
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from mercy.analysis import seal
from mercy.generation.vehicles import profiles
from mercy.ingest.environment import assemble
from scripts.prepare_environments import prepare
from scripts.kaggle_gpu import client, SLUGS, STAGE


def refine_rgb_vehicles(names):
    """New cases use the same persisted Kaggle WildDet backend as existing cases."""
    subprocess.run([sys.executable, str(ROOT / 'scripts/kaggle_wilddet.py'), 'run', '--case', *names],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", required=True)
    parser.add_argument("--no-approach", action="store_true", help="skip the generated backwards road extension")
    args = parser.parse_args()
    cached = {json.loads(p.read_text())["case"]: p.parent for p in (STAGE / "output").rglob("result.json")}
    pending = []
    for name in args.case:
        case = config.case_dir(name)
        prepare(case)
        profiles(case)
        sha = hashlib.sha256((case / "environment" / "empty.png").read_bytes()).hexdigest()
        if name not in cached or json.loads((cached[name] / "result.json").read_text()).get("image_sha256") != sha:
            pending.append(name)
    if not pending:
        for name in args.case:
            info = assemble(config.case_dir(name), cached[name])
            seal.seal(config.case_dir(name))
            print(f"{name}: reused GPU result; {len(info['moments'])} rebuilt moments", flush=True)
        if not args.no_approach:
            from scripts.extend_approach import run
            run(args.case)
        refine_rgb_vehicles(args.case)
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_gpu.py"), "run", "--case", *pending],
                   cwd=ROOT, check=True)
    api = client()
    ref = f"{api.get_config_value('username')}/{SLUGS['run']}"
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        state = api.kernels_status(ref)
        print(f"Kaggle: {state}", flush=True)
        if "complete" in str(state.status).lower():
            break
        if any(word in str(state.status).lower() for word in ["error", "cancel", "fail"]):
            api.kernels_output(ref, path=str(STAGE / "failure"), file_pattern=r"\.log$")
            raise RuntimeError(f"Kaggle GPU job failed: {state}; see data/kaggle/failure")
        time.sleep(30)
    else:
        raise TimeoutError("Kaggle job still pending after one hour; check status before resubmitting")
    out = STAGE / "output"
    out.mkdir(exist_ok=True)
    api.kernels_output(ref, path=str(out))
    results = {json.loads(p.read_text())["case"]: p.parent for p in out.rglob("result.json")}
    for name in args.case:
        if name not in results:
            raise RuntimeError(f"GPU job omitted {name}")
        info = assemble(config.case_dir(name), results[name])
        seal.seal(config.case_dir(name))
        print(f"{name}: {len(info['moments'])} rebuilt 3D/4D moments", flush=True)
    if not args.no_approach:
        from scripts.extend_approach import run
        run(args.case)
    refine_rgb_vehicles(args.case)


if __name__ == "__main__":
    main()

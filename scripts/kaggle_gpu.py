"""Private Kaggle GPU execution with a reusable immutable setup output.

python scripts/kaggle_gpu.py setup
python scripts/kaggle_gpu.py status --job setup
python scripts/kaggle_gpu.py run --case hero cutin_rearend
python scripts/kaggle_gpu.py status
python scripts/kaggle_gpu.py fetch

KAGGLE_API_TOKEN comes from the ignored .env or process environment. It is never
included in notebooks, uploaded datasets, manifests or saved metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config

STAGE = ROOT / "data" / "kaggle"
SLUGS = {"setup": "mercy-environment-runtime-gpu-v1", "run": "mercy-empty-environments-v1"}


def client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def push(api, owner, job, sources=(), cache_version=None):
    dst = STAGE / job
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "worker.py").write_text(f"JOB = {job!r}\n" + (ROOT / "kaggle" / "environment_worker.py").read_text())
    metadata = {"id": f"{owner}/{SLUGS[job]}", "title": SLUGS[job], "code_file": "worker.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": job == "setup",
                "dataset_sources": list(sources), "competition_sources": [],
                "kernel_sources": [f"{owner}/{SLUGS['setup']}"] if job == "run" else [f"{owner}/mercy-environment-runtime-v1"]}
    if job == "run" and cache_version:
        metadata["kernel_sources"] = [f"{owner}/{SLUGS['setup']}/{cache_version}"]
    (dst / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    response = api.kernels_push(str(dst), acc="NvidiaTeslaT4")
    print(response)
    if response.error or response.invalid_dataset_sources or response.invalid_kernel_sources:
        raise RuntimeError("Kaggle rejected the job or one of its input dependencies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "run", "status", "fetch"])
    parser.add_argument("--job", choices=list(SLUGS), default="run")
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--cache-version", type=int, help="pin an already completed immutable setup version")
    args = parser.parse_args()
    api = client()
    owner = api.get_config_value("username")
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.action == "setup":
        push(api, owner, "setup")
    elif args.action == "run":
        if not args.case:
            parser.error("run needs --case")
        state = api.kernels_status(f"{owner}/{SLUGS['setup']}")
        if "complete" not in str(state.status).lower() and not args.cache_version:
            raise RuntimeError(f"Setup cache is not complete: {state}")
        dataset = STAGE / "inputs"
        dataset.mkdir(exist_ok=True)
        for existing in dataset.iterdir():
            if existing.is_dir():
                shutil.rmtree(existing)
        for case_id in args.case:
            src = config.case_dir(case_id) / "environment"
            info = json.loads((src / "input.json").read_text())
            review = json.loads((src / info["review"]).read_text())
            from mercy.generation.cleanplate import digest
            if not review.get("accepted") or review["image_sha256"] != digest(src / info["image"]):
                raise RuntimeError(f"{case_id}: image has no valid successful Muse review")
            dst = dataset / case_id
            dst.mkdir()
            for name in ("input.json", info["image"], info["review"]):
                shutil.copy2(src / name, dst / name)
        ref = f"{owner}/mercy-reviewed-environments"
        (dataset / "dataset-metadata.json").write_text(json.dumps({"id": ref, "title": "MERCY reviewed environments",
            "licenses": [{"name": "other"}]}, indent=2))
        exists = any(d.ref == ref for d in api.dataset_list(mine=True, search="mercy-reviewed-environments"))
        fingerprint = hashlib.sha256("".join(
            f"{p.relative_to(dataset)}:{hashlib.sha256(p.read_bytes()).hexdigest()}"
            for p in sorted(dataset.rglob("*")) if p.is_file() and p.name != "dataset-metadata.json").encode()).hexdigest()
        uploaded = STAGE / "uploaded.json"
        previous = json.loads(uploaded.read_text()) if uploaded.exists() else {}
        if not exists or previous.get("fingerprint") != fingerprint:
            if exists:
                api.dataset_create_version(str(dataset), version_notes="Reviewed empty environments", dir_mode="zip", quiet=True)
            else:
                api.dataset_create_new(str(dataset), public=False, dir_mode="zip", quiet=True)
            uploaded.write_text(json.dumps({"fingerprint": fingerprint}))
        for _ in range(60):
            state = api.dataset_status(ref)
            if "ready" in str(state).lower():
                break
            print(f"Waiting for private dataset: {state}", flush=True)
            time.sleep(2)
        else:
            raise RuntimeError("Kaggle input dataset did not become ready")
        push(api, owner, "run", [ref], args.cache_version)
    elif args.action == "status":
        print(api.kernels_status(f"{owner}/{SLUGS[args.job]}"))
    else:
        dst = STAGE / "output"
        dst.mkdir(exist_ok=True)
        api.kernels_output(f"{owner}/{SLUGS[args.job]}", path=str(dst))
        print(f"Downloaded outputs: {dst}")


if __name__ == "__main__":
    main()

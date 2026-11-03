"""Kaggle-only GPU worker. Configuration is prepended by scripts/kaggle_gpu.py.

The setup kernel's output is an immutable dependency of inference kernels: weights,
source and extra Python packages are downloaded once, then loaded from /kaggle/input.
No API keys are embedded in this file or uploaded inputs.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def setup():
    cache = WORK / "mercy-runtime"
    cache.mkdir(exist_ok=True)
    repo = cache / "ml-sharp"
    bootstrap = next(INPUT.rglob("mercy-runtime/manifest.json"), None)
    if bootstrap:
        shutil.copytree(bootstrap.parent / "ml-sharp", repo)
        shutil.copy2(bootstrap.parent / "sharp.pt", cache / "sharp.pt")
    else:
        run("git", "clone", "--depth", "1", "https://github.com/apple-aiml-research/ml-sharp.git", repo)
    # Torch/CUDA/numpy/scipy are supplied by the Kaggle image, never reinstall them.
    run(sys.executable, "-m", "pip", "install", "--no-deps", "--target", cache / "site",
        "timm==1.0.22", "plyfile==1.1.3", "pillow-heif==1.1.1", "gsplat==1.5.3")
    checkpoint = cache / "sharp.pt"
    if not checkpoint.exists():
        urllib.request.urlretrieve("https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt", checkpoint)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Setup must run in the same CUDA image as inference")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache / "extensions")
    os.environ["MAX_JOBS"] = "2"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    sys.path.insert(0, str(cache / "site"))
    from gsplat import rasterization
    rasterization(means=torch.tensor([[0., 0., 3.]], device="cuda"),
        quats=torch.tensor([[1., 0., 0., 0.]], device="cuda"),
        scales=torch.full((1, 3), .1, device="cuda"), opacities=torch.ones(1, device="cuda"),
        colors=torch.ones((1, 3), device="cuda"), viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]]], device="cuda"), width=32, height=32)
    manifest = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("MERCY runtime saved", manifest, flush=True)


def infer():
    cache = next(INPUT.rglob("mercy-runtime/manifest.json")).parent
    sys.path[:0] = [str(cache / "site"), str(cache / "ml-sharp" / "src")]
    import numpy as np
    import torch
    from PIL import Image
    from sharp.models import PredictorParams, create_predictor
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import save_ply
    if not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA GPU is required; refusing CPU fallback")
    cached = json.loads((cache / "manifest.json").read_text())
    if cached["python"].split()[0] != sys.version.split()[0] or cached["torch"] != torch.__version__:
        raise RuntimeError("Kaggle runtime changed. Rebuild the cache once before inference.")
    compiled = next((cache / "extensions").rglob("gsplat_cuda.so"), None)
    if compiled:
        spec = importlib.util.spec_from_file_location("gsplat_cuda", compiled)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules["gsplat.csrc"] = native  # Load the persisted CUDA binary; never invoke JIT on every job.
    else:
        print("This cache predates the optional renderer; generating the 3D environment without preview renders.", flush=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(WORK / "extensions")
    os.environ["MAX_JOBS"] = "2"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    print("GPU:", torch.cuda.get_device_name(0), "runtime:", cached, flush=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(torch.load(cache / "sharp.pt", map_location="cpu", weights_only=True))
    predictor.eval().cuda()
    cases = list(INPUT.rglob("input.json"))
    if not cases:
        unpacked = WORK / "inputs"
        for archive in INPUT.rglob("*.zip"):
            with zipfile.ZipFile(archive) as bundle:
                if "input.json" not in bundle.namelist():
                    continue
                target = unpacked / archive.stem
                for member in bundle.infolist():
                    dest = (target / member.filename).resolve()
                    if not dest.is_relative_to(target.resolve()):
                        raise ValueError("Unsafe archive path")
                bundle.extractall(target)
        cases = list(unpacked.rglob("input.json"))
    results = []
    for info_path in sorted(cases):
        info = json.loads(info_path.read_text())
        if "anchor_k" not in info:
            continue
        out = WORK / "environments" / info["case"]
        out.mkdir(parents=True, exist_ok=True)
        image = np.asarray(Image.open(info_path.parent / info["image"]).convert("RGB"))
        start = time.time()
        with torch.inference_mode():
            gaussians = predict_image(predictor, image, info["focal_px"], torch.device("cuda"))
        save_ply(gaussians, info["focal_px"], image.shape[:2], out / "environment.ply")
        # These are genuine renders of one 3D scene, at the same height, not independently imagined views.
        from gsplat import rasterization
        from sharp.utils.color_space import linearRGB2sRGB
        width, height = 1280, round(image.shape[0] * 1280 / image.shape[1])
        focal = info["focal_px"] * width / image.shape[1]
        K = torch.tensor([[[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]]], device="cuda")
        for name, x in ([("left", -.6), ("anchor", 0.), ("right", .6)] if compiled else []):
            view = torch.eye(4, device="cuda")[None]
            view[0, 0, 3] = -x
            with torch.inference_mode():
                rgb, alpha, _ = rasterization(means=gaussians.mean_vectors[0], quats=gaussians.quaternions[0],
                    scales=gaussians.singular_values[0], opacities=gaussians.opacities[0], colors=gaussians.colors[0],
                    viewmats=view, Ks=K, width=width, height=height, packed=True)
            preview = linearRGB2sRGB(rgb[0]).clamp(0, 1).cpu().numpy()
            Image.fromarray((preview * 255).astype(np.uint8)).save(out / f"view-{name}.png")
        result = dict(info, seconds=round(time.time() - start, 2), backend="SHARP", gpu=torch.cuda.get_device_name(0),
                      image_sha256=hashlib.sha256((info_path.parent / info["image"]).read_bytes()).hexdigest(),
                      coordinate_system="OpenCV camera: x right, y down, z forward", runtime=cached)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        del gaussians
        torch.cuda.empty_cache()
        print("DONE", info["case"], result["seconds"], flush=True)
    if not results:
        raise RuntimeError("No reviewed environment inputs were attached")
    (WORK / "results.json").write_text(json.dumps(results, indent=2))


if JOB == "setup":
    setup()
else:
    infer()

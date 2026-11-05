"""Package the current source and verify the demo in a clean, API-key-free copy.

    python scripts/prepare_delivery.py package
    python scripts/prepare_delivery.py verify

Source is collected from git's tracked and nonignored files, including uncommitted
work. Neither .git, credentials nor local environments enter the source archive.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "deliverables"


def package():
    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/check_public_repo.py"), "--history"], check=True, cwd=ROOT)
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    excluded = {"data", "deliverables", ".git", ".venv", ".venv-recon", "node_modules", "__pycache__"}
    archive = OUT / "mercymek-source.zip"
    count = 0
    with zipfile.ZipFile(archive.with_suffix(".zip.part"), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(set(filter(None, names))):
            p = ROOT / name
            parts = Path(name).parts
            if any(part in excluded for part in parts) or p.name == ".env" or not p.is_file():
                continue
            if p.suffix.lower() in {".pem", ".key"}:
                raise ValueError(f"Private-key-like file must not enter release: {name}")
            z.write(p, name)
            count += 1
    archive.with_suffix(".zip.part").replace(archive)
    print(f"Source: {count} files, {archive.stat().st_size/1e6:.1f} MB", flush=True)
    write_checksums()


def write_checksums():
    names = ["mercymek-source.zip", "mercymek-demo-data.zip", "mercymek-extra-cases.zip", "mercymek-3-minute-demo.mp4", "mercymek-3-minute-demo-v2.mp4", "mercymek-3-minute-demo-v3.mp4", "mercymek-3-minute-demo-v4.mp4", "mercymek-3-minute-demo-v5.mp4", "mercymek-3-minute-demo-v6.mp4", "mercymek-demo.en.srt", "mercymek-narration.mp3", "mercymek-thumbnail.png"]
    lines = []
    for name in names:
        p = OUT / name
        if p.exists():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda:f.read(4 << 20), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def verify():
    import httpx
    from mercy.analysis.seal import verify as verify_seal
    clean = OUT / "production" / "clean-install"
    if clean.exists():
        shutil.rmtree(clean)
    clean.mkdir(parents=True)
    for name in ["mercymek-source.zip", "mercymek-demo-data.zip", "mercymek-extra-cases.zip"]:
        print(f"Extracting {name} into clean install", flush=True)
        with zipfile.ZipFile(OUT / name) as z:
            z.extractall(clean)
    assert not (clean / ".env").exists()
    assert not (clean / "data" / "keys").exists()
    seals = {}
    for case in sorted((clean / "data" / "cases").iterdir()):
        if not (case / "meta.json").exists():
            continue
        result = verify_seal(case)
        assert result["intact"] and result["signature_valid"], (case.name, result)
        seals[case.name] = {"intact": True, "files": result["files"], "root": result["root"]}
        print(f"{case.name}: original seal intact ({result['files']} files)", flush=True)
    env = dict(os.environ)
    for name in ("MODEL_API_KEY", "ELEVENLABS_API_KEY", "HF_TOKEN", "KAGGLE_API_TOKEN", "MAPILLARY_TOKEN",
                 "CASES_DIR", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(TTS_CHAR_BUDGET="0", MERCY_VISER="0", PYTHONUNBUFFERED="1")
    log_path = OUT / "production" / "clean-server.log"
    log = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "scripts/run_demo.py", "--case", "hero", "--port", "8002", "--no-viser"], cwd=clean, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url="http://localhost:8002", timeout=30) as client:
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError(log_path.read_text(encoding="utf-8")[-5000:])
                try:
                    status = client.get("/api/status").json()
                    if status["laya"]:
                        break
                except httpx.RequestError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Clean install did not start within 120 seconds")
                time.sleep(.5)
            assert status["meta_key"] is False
            inventory = client.get("/api/cases").json()
            assert len(inventory) == 7
            assets = []
            for item in inventory:
                cid = item["id"]
                bundle = client.get(f"/api/case/{cid}").json()
                response = client.get(bundle["urls"]["video"], headers={"Range":"bytes=0-1023"})
                assert response.status_code == 206 and len(response.content) == 1024
                assert bundle["moments"]
                for moment in bundle["moments"]:
                    response = client.get(moment["url"], headers={"Range":"bytes=0-63"})
                    assert response.status_code == 206, (cid, moment["id"])
                assets.append({"case":cid,"video_range_request":206,"moments":len(bundle["moments"])})
            interactions = asyncio.run(verify_commands(client))
        from mercy import __version__
        report = {"application_version": __version__, "source_archive_sha256": hashlib.sha256((OUT / "mercymek-source.zip").read_bytes()).hexdigest(), "validated_clean_install": True, "api_keys_present":False,"private_signing_key_present":False,"cases":seals,"assets":assets,"commands":interactions}
        (OUT / "release-validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("Clean installation: all seven cases, video ranges, 3D assets and API-free commands PASS",flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        log.close()
    write_checksums()


async def verify_commands(client):
    import websockets
    results = []
    async with websockets.connect("ws://localhost:8002/ws", max_size=16 << 20) as ws:
        tests = [("Show the impact in 3D", "state", lambda d:d.get("view")=="scene"),
                 ("What if the camera car had been going 30 km/h?", "whatif", lambda d:d.get("verdict")=="avoided"),
                 ("Who is at fault?", "verdict", lambda d:d.get("status")=="ready"),
                 ("Fill in the accident statement", "statement", lambda d:bool(d.get("A"))),
                 ("Verify the evidence", "seal", lambda d:d.get("intact") is True),
                 ("Show the evidence", "state", lambda d:d.get("view")=="evidence")]
        for text, kind, predicate in tests:
            await ws.send(json.dumps({"type":"cmd","text":text}))
            deadline = time.monotonic()+45
            found = speech = False
            while time.monotonic()<deadline:
                message = json.loads(await asyncio.wait_for(ws.recv(),timeout=45))
                data = message.get("data",{})
                if message["type"]=="error":
                    raise AssertionError((text,data))
                if message["type"]==kind and predicate(data):
                    found = True
                if message["type"]=="say":
                    url = data.get("url")
                    if url:
                        response = client.get(url)
                        assert response.status_code == 200 and len(response.content)>100
                        speech = True
                    if found:
                        break
            assert found, text
            # At least this deterministic line is prewarmed during recording;
            # the no-key server must actually return cached MP3 bytes.
            if text=="Show the evidence":
                assert speech, "Cached ElevenLabs speech unavailable without an API key"
            results.append({"command":text,"passed":True,"cached_speech":speech})
        await ws.send(json.dumps({"type":"load_case","id":"london_rearend"}))
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(),timeout=30))
            if message["type"]=="case" and message["data"]["id"]=="london_rearend":
                break
        # The named target and compound paperwork command must work after clean
        # extraction too, without falling back to a paid live segmentation call.
        for text, expected in [
            ("Zoom in on the silver BMW", {"zoom"}),
            ("Fill in the accident statement and verify the evidence", {"statement", "seal"}),
        ]:
            await ws.send(json.dumps({"type": "cmd", "text": text}))
            seen = set()
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
                data = message.get("data", {})
                if message["type"] == "error":
                    raise AssertionError((text, data))
                if message["type"] == "state" and data.get("zoom") == "car:3":
                    seen.add("zoom")
                if message["type"] == "statement" and data.get("A"):
                    seen.add("statement")
                if message["type"] == "seal" and data.get("intact") is True:
                    seen.add("seal")
                if message["type"] == "say" and expected <= seen:
                    break
            assert expected <= seen, (text, seen)
            results.append({"command": text, "passed": True, "results": sorted(seen)})
        await ws.send(json.dumps({"type":"ui","patch":{"t":7.4,"playing":False}}))
        await ws.send(json.dumps({"type":"cmd","text":"Zoom in on the licence plate"}))
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(),timeout=45))
            data = message.get("data",{})
            if message["type"]=="enhance" and data.get("status")=="failed":
                raise AssertionError(data)
            if message["type"]=="enhance" and data.get("status")=="ready":
                assert data["cached"] is True and data["frames_used"]==9
                for url in data["images"].values():
                    response = client.get(url)
                    assert response.status_code==200 and len(response.content)>100
                results.append({"command":"London plate enhancement","passed":True,"cached":True,"frames_used":9})
                break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["package","verify","checksums"])
    action = parser.parse_args().action
    {"package":package,"verify":verify,"checksums":write_checksums}[action]()

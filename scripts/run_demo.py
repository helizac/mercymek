"""Start the cockpit (FastAPI + HUD on :APP_PORT) and the viser 4D scene (:VISER_PORT) with one command.

    python scripts/run_demo.py --case hero
    python scripts/run_demo.py --case hero --no-viser --mic server
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from mercy import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="hero")
    ap.add_argument("--port", type=int, default=config.APP_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-viser", action="store_true")
    ap.add_argument("--mic", choices=["browser", "server"], default=config.MIC_SOURCE)
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    os.environ["MERCY_CASE"] = a.case
    os.environ["MERCY_VISER"] = "0" if a.no_viser else "1"
    os.environ["MIC_SOURCE"] = a.mic
    config.MIC_SOURCE = a.mic
    import uvicorn
    url = f"http://localhost:{a.port}/"
    print(f"\n  MERCY cockpit  {url}\n  viser 4D scene http://localhost:{config.VISER_PORT}/\n")
    if a.open:
        webbrowser.open(url)
    uvicorn.run("mercy.server.app:app", host=a.host, port=a.port, log_level="warning", ws_max_size=16 * 1024 * 1024)


if __name__ == "__main__":
    main()

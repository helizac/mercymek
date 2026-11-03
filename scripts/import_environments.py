"""Convert downloaded Kaggle environments and update all moments atomically per case."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from mercy.analysis import seal
from mercy.ingest.environment import assemble

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
for p in sorted(args.output.rglob("result.json")):
    import json
    name = json.loads(p.read_text())["case"]
    info = assemble(config.case_dir(name), p.parent)
    seal.seal(config.case_dir(name))
    print(name, len(info["moments"]), "3D moments", info["road"], flush=True)

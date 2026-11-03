"""Review road-relative body headings in full source videos, using the saved Meta cache."""
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mercy.generation.heading_review import review

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(review,sys.argv[1:] or ['cutin_rearend','hero','lanechange','night_rain','snow_pileup']))

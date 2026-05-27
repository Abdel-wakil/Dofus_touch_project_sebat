"""
harvest_map.py
--------------
Harvest the current map once (used by the UI "Harvest map" button).

Usage:
    python harvest_map.py --pos 16,-33
"""

import sys
import os
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from farm import farm_current_map, _load_db_and_spots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=str, required=True,
                        help="Map position as 'x,y', e.g. '16,-33'")
    args = parser.parse_args()
    x, y = args.pos.split(",")
    pos  = (int(x), int(y))

    db, spots_map, color_range = _load_db_and_spots()

    if pos not in db:
        print(f"[Harvest] {pos} is not in the resource database.")
        return

    spots = spots_map.get(pos)
    mode  = "color check" if color_range else "blink detection"
    print(f"[Harvest] {pos} — {len(spots) if spots else 0} known spot(s) ({mode})")
    print("Switch to the game. Starting in 3s...")
    for i in range(3, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  GO!   ")

    count = farm_current_map(pos, spots=spots, color_range=color_range)
    print(f"[Harvest] Done — {count} resource(s) harvested.")


if __name__ == "__main__":
    main()

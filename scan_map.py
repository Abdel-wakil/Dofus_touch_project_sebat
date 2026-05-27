"""
scan_map.py
-----------
Scan the current map for resources and save spots to the active resource JSON.
Used by the UI "Scan current map" button.

Usage:
    python scan_map.py --pos -11,5
"""

import sys
import os
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from farm import capture_frames, blink_detect, _center, DEFAULT_BLINK
from scout import _save_debug, _save, _load
from config.loader import get_resource_path, get_active_resource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=str, required=True,
                        help="Map position as 'x,y', e.g. '-11,5'")
    args = parser.parse_args()
    x, y = args.pos.split(",")
    pos = (int(x), int(y))

    resource  = get_active_resource()
    data      = _load()
    blink_cfg = {**DEFAULT_BLINK, **data.get("blink", {})}
    db        = {(m["x"], m["y"]) for m in data["maps"]}

    if pos not in db:
        print(f"[Scan] {pos} is not in {resource}.json — nothing to save.")
        return

    expected = next(
        (m.get("count") for m in data["maps"] if m["x"] == pos[0] and m["y"] == pos[1]),
        None
    )
    print(f"[Scan] Scanning {pos} ({resource}) — hold still...")

    frames = capture_frames(blink_cfg)
    zones  = blink_detect(frames, blink_cfg)
    spots  = [[cx, cy] for cx, cy in (_center(z) for z in zones)]

    exp_str = f"/{expected}" if expected is not None else ""
    print(f"[Scan] Found {len(spots)}{exp_str} spot(s): {spots}")

    _save_debug(frames, zones, spots, pos)

    for m in data["maps"]:
        if m["x"] == pos[0] and m["y"] == pos[1]:
            m["spots"] = spots
            break
    _save(data)
    print(f"[Scan] Saved to resources/{resource}.json")


if __name__ == "__main__":
    main()

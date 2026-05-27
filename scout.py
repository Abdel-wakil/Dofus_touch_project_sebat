"""
scout.py
--------
One-time scout run: visits every map in kalyptus.json, uses hold-click
blink detection to find resource screen positions, and saves them as
"spots": [[cx, cy], ...] in kalyptus.json.

After scout completes, farm.py clicks saved positions directly on each
visit — no blink detection needed, making each map visit much faster.

Progress is saved after every map, so you can Ctrl-C and resume later
(already-scanned maps are skipped on the next run).

Usage:
    python scout.py
"""

import sys
import os
import json
import time
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import vision
from planner import _DELTAS
from farm import capture_frames, blink_detect, _center, navigate
from config.loader import get_resource_path

_DET_DIR = ROOT / "screenshots" / "scout" / "detection"
_DET_DIR.mkdir(parents=True, exist_ok=True)


def _save_debug(frames, zones, spots, pos):
    tag = f"{pos[0]}_{pos[1]}"
    out = frames[len(frames) // 2].copy()
    for x1, y1, x2, y2 in zones:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for cx, cy in spots:
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(out, f"({cx},{cy})", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(_DET_DIR / f"{tag}.png"), out)
    print(f"[Scout] Saved screenshots/scout/detection/{tag}.png ({len(spots)} spots)")


def _load():
    with open(get_resource_path(), encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    """Write resource JSON with one compact line per map entry."""
    path  = get_resource_path()
    maps  = data["maps"]
    lines = [
        "{",
        f'  "resource": {json.dumps(data["resource"])},',
        f'  "respawn_minutes": {data["respawn_minutes"]},',
        '  "maps": [',
    ]
    for i, m in enumerate(maps):
        comma = "," if i < len(maps) - 1 else ""
        lines.append(f"    {json.dumps(m, ensure_ascii=False)}{comma}")
    lines += ["  ]", "}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _choose_next(pos, db, visited, prev_pos):
    """Prefer unvisited neighbours; backtrack through visited ones if needed."""
    cx, cy = pos
    all_moves = [
        (d, (cx + dx, cy + dy))
        for d, (dx, dy) in _DELTAS.items()
        if (cx + dx, cy + dy) in db
    ]
    if not all_moves:
        return None, None
    unvisited = [(d, p) for d, p in all_moves if p not in visited]
    if unvisited:
        return random.choice(unvisited)
    forward = [(d, p) for d, p in all_moves if p != prev_pos]
    return random.choice(forward if forward else all_moves)


def _parse_start_pos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-pos", type=str, default=None,
                        help="Starting map coords as 'x,y', e.g. '-11,5'")
    args, _ = parser.parse_known_args()
    if args.start_pos:
        x, y = args.start_pos.split(",")
        return int(x), int(y)
    return None


def main():
    data = _load()
    db   = {(m["x"], m["y"]) for m in data["maps"]}

    already = sum(1 for m in data["maps"] if "spots" in m)
    print(f"=== Scout Run === ({len(db)} maps total, {already} already scanned)")
    if already == len(db):
        print("All maps already scanned. Use Reset spots to re-scout.")
        return

    pos = _parse_start_pos()
    if pos:
        print(f"Starting at {pos} (user-provided)")
    else:
        print("No start pos given — waiting for OCR...")

    print("Switch to the game. Starting in 5s...\n")
    for i in range(5, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  GO!\n")

    # Block until we have a position
    while pos is None:
        pos = vision.read_current_position()
        if pos:
            print(f"[Pos] {pos} (OCR initial)")
        else:
            print("[Pos] OCR failed — retrying in 2s...")
            time.sleep(2)

    visited       = {(m["x"], m["y"]) for m in data["maps"] if "spots" in m}
    prev_pos      = None
    ocr_candidate = None  # OCR value that disagreed with tracker; checked next move

    try:
        while True:
            ocr = vision.read_current_position()
            if ocr is None:
                print(f"[Pos] {pos} (tracking | OCR failed)")
            elif ocr == pos:
                print(f"[Pos] {pos} OK")
                ocr_candidate = None
            else:
                print(f"[Pos] {pos} (tracking) | OCR says {ocr} — watching")
                ocr_candidate = ocr

            if pos in db and pos not in visited:
                visited.add(pos)
                print(f"[Scout] {pos} — scanning ({len(visited)}/{len(db)})...")

                frames = capture_frames()
                zones  = blink_detect(frames)
                spots  = [[cx, cy] for cx, cy in (_center(z) for z in zones)]
                print(f"[Scout] Found {len(spots)} spot(s): {spots}")

                _save_debug(frames, zones, spots, pos)

                for m in data["maps"]:
                    if m["x"] == pos[0] and m["y"] == pos[1]:
                        m["spots"] = spots
                        break
                _save(data)

            if len(visited) >= len(db):
                print(f"\n[Scout] All {len(db)} maps scanned. kalyptus.json updated.")
                break

            direction, _ = _choose_next(pos, db, visited, prev_pos)
            if direction is None:
                print("[Scout] No moves available — done.")
                break

            prev_pos    = pos
            ok, nav_ocr = navigate(direction, current_pos=pos)
            if ok:
                dx, dy    = _DELTAS[direction]
                delta_pos = (pos[0] + dx, pos[1] + dy)
                if nav_ocr == delta_pos:
                    pos = nav_ocr
                elif (ocr_candidate is not None and nav_ocr is not None
                      and nav_ocr == (ocr_candidate[0] + dx, ocr_candidate[1] + dy)):
                    print(f"[Pos] Candidate {ocr_candidate} confirmed -> correcting to {nav_ocr}")
                    pos = nav_ocr
                else:
                    pos = delta_pos
                ocr_candidate = None

    except KeyboardInterrupt:
        print(f"\n[Scout] Stopped. ({len(visited)}/{len(db)} maps scanned)")
        print("Progress saved — resume any time.")


if __name__ == "__main__":
    main()

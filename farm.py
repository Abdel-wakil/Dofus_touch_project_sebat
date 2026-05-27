"""
farm.py
-------
Each iteration:
  1. Pre-compute snake (boustrophedon) route through all resource maps
  2. Navigate to each map in order, one step at a time
  3. Hold-click blink-detect to find available spots
  4. Click available spots and wait for harvest
  5. Repeat from start of route when all maps visited

Run scout.py first to populate spot coordinates in boumu.json (or relevant file).

Usage:
    python farm.py
    python farm.py --start-pos=16,-33
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
import mss
import numpy as np
import pyautogui

from config.loader import get_screen_regions, get_timing, get_resource_path
import input as bot_input
import vision
from planner import snake_route, step_toward, _DELTAS


# ── Per-map navigation hints ──────────────────────────────────────────────────
# Some maps have narrow or partially-blocked edges.
# Each entry: (map_x, map_y, direction) → (x_frac0, y_frac0, x_frac1, y_frac1)
# The fractions clip the full edge region to the passable sub-area (0.0–1.0).
#   x/y_frac0 = start of passable zone, x/y_frac1 = end of passable zone.
# Examples:
#   bottom-half of left edge  → (0.0, 0.5, 1.0, 1.0)
#   left-half of bottom edge  → (0.0, 0.0, 0.5, 1.0)
NAV_HINTS: dict[tuple, tuple] = {
    (18, -35, "left"):   (0.0, 0.7, 1.0, 1.0),   # click bottom 30% of left edge
    (18, -35, "bottom"): (0.0, 0.0, 0.5, 1.0),   # click left half of bottom edge
}


def _hint_region(region, pos, direction):
    """Clip region to the passable sub-area defined in NAV_HINTS, if any."""
    if pos is None:
        return region
    hint = NAV_HINTS.get((pos[0], pos[1], direction))
    if hint is None:
        return region
    x1, y1, x2, y2 = region
    xf0, yf0, xf1, yf1 = hint
    w, h = x2 - x1, y2 - y1
    return (
        round(x1 + xf0 * w), round(y1 + yf0 * h),
        round(x1 + xf1 * w), round(y1 + yf1 * h),
    )


def _segmented_click(region, attempt, max_retries):
    """
    Click in a different segment of the edge region on each retry.

    Attempt 0 → fully random (covers whole region).
    Attempts 1+ → divide the region's long axis into (max_retries-1)
    equal strips and cycle through them, so each retry targets a
    different passage and we don't keep hitting the same obstacle.
    """
    x1, y1, x2, y2 = region
    if attempt == 0:
        bot_input.click_random_in_region(region)
        return
    w, h    = x2 - x1, y2 - y1
    n_segs  = max(max_retries - 1, 1)
    seg_idx = (attempt - 1) % n_segs
    if w >= h:                          # wide strip (top/bottom) — split along x
        seg_w = w / n_segs
        sx1   = x1 + seg_idx * seg_w
        sx2   = sx1 + seg_w
        x     = random.randint(round(sx1), round(sx2))
        y     = random.randint(y1, y2)
    else:                               # tall strip (left/right) — split along y
        seg_h = h / n_segs
        sy1   = y1 + seg_idx * seg_h
        sy2   = sy1 + seg_h
        x     = random.randint(x1, x2)
        y     = random.randint(round(sy1), round(sy2))
    bot_input.click(x, y)


def navigate(direction, current_pos=None, max_retries=3):
    """
    Click the edge, wait for the map transition, then validate the new position.

    Returns (True, ocr_pos) on success, (False, None) if all nav attempts fail.
    ocr_pos may differ from expected if OCR never agreed — caller handles fallback.
    """
    regions  = get_screen_regions()
    timing   = get_timing()
    dx, dy   = _DELTAS[direction]
    expected = (current_pos[0] + dx, current_pos[1] + dy) if current_pos else None

    for attempt in range(max_retries):
        if attempt == 0:
            print(f"[Nav] Moving {direction} (expected: {expected})...")
        else:
            print(f"[Nav] Nav retry {attempt}/{max_retries - 1}...")

        _segmented_click(_hint_region(regions[direction], current_pos, direction), attempt, max_retries)
        vision.wait_for_map_change()
        time.sleep(timing["post_map_change_delay"])

        best_ocr = None
        for ocr_try in range(3):
            reading = vision.read_current_position()

            if reading is None:
                print(f"[Nav] OCR failed (try {ocr_try + 1}/3)")
                if ocr_try < 2:
                    time.sleep(0.3)
                continue

            if current_pos is not None and reading == current_pos:
                print(f"[Nav] OCR still at {current_pos} — click missed, retrying nav...")
                best_ocr = None
                break

            best_ocr = reading
            if reading == expected:
                print(f"[Nav] {direction} confirmed: OCR={reading}")
                return True, reading

            print(f"[Nav] OCR={reading}, expected={expected} — OCR retry {ocr_try + 1}/2...")
            if ocr_try < 2:
                time.sleep(0.3)

        if best_ocr is not None:
            print(f"[Nav] OCR={best_ocr} != expected={expected}, returning best reading")
            return True, best_ocr

    print(f"[Nav] Could not confirm {direction} after {max_retries} attempts.")
    return False, None


# ── Per-spot blink detection ──────────────────────────────────────────────────

_regions_cache = get_screen_regions()

SPOT_WIN_X     = 45    # horizontal pixel radius around each spot center
SPOT_WIN_Y_TOP = 140   # upward extension from spot center
SPOT_WIN_Y_BOT = 70    # downward extension from spot center
SPOT_FRAMES    = 5     # screenshots to take (3 consecutive pairs)
SPOT_INTERVAL  = 0.3  # seconds between screenshots
BLINK_DIFF     = 10    # per-pixel brightness change threshold
MIN_BLINK_PX   = 2000   # blink-event sum per box → plant is available
                       # fire plant ~500-2000+, stump ~0-20


def check_spots_available(spots, return_mask=False):
    """
    Per-spot blink check: hold left click at screen centre to trigger blinking,
    then count pixels that change between consecutive grayscale screenshots.
    Fire-animated plant → many blinking pixels.  Static stump → near zero.
    """
    fz = _regions_cache["farm_zone"]
    fx1, fy1, fx2, fy2 = fz
    hold_x = (fx1 + fx2) // 2
    hold_y = (fy1 + fy2) // 2

    frames = []
    img_bgr = None
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        pyautogui.moveTo(hold_x, hold_y)
        time.sleep(0.05)
        pyautogui.mouseDown(button="left")
        time.sleep(0.15)
        for i in range(SPOT_FRAMES):
            t0  = time.perf_counter()
            raw = sct.grab(monitor)
            frames.append(cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR))
            elapsed = time.perf_counter() - t0
            if SPOT_INTERVAL > elapsed:
                time.sleep(SPOT_INTERVAL - elapsed)
        pyautogui.mouseUp(button="left")
        time.sleep(0.05)

    img_bgr = frames[-1]
    h, w = img_bgr.shape[:2]
    blink_sum = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(frames) - 1):
        # Max change across all three colour channels — catches hue/saturation
        # shifts (background trees cycling colour) that grayscale would miss.
        diff = np.abs(frames[i].astype(np.int16) - frames[i + 1].astype(np.int16))
        blink_sum += (diff.max(axis=2) > BLINK_DIFF).astype(np.uint8)

    available = []
    for cx, cy in spots:
        x1 = max(0, cx - SPOT_WIN_X); x2 = min(w, cx + SPOT_WIN_X)
        y1 = max(0, cy - SPOT_WIN_Y_TOP); y2 = min(h, cy + SPOT_WIN_Y_BOT)
        count = int(np.sum(blink_sum[y1:y2, x1:x2]))
        if count >= MIN_BLINK_PX:
            available.append((cx, cy))
            print(f"[Farm] ({cx},{cy}) blink={count} -> available")
        else:
            print(f"[Farm] ({cx},{cy}) blink={count} -> stump")

    if return_mask:
        return available, blink_sum, img_bgr
    return available


# ── Harvest ───────────────────────────────────────────────────────────────────

def farm_current_map(pos=None, spots=None):
    """Check which spots are available and click them. Returns number harvested."""
    timing = get_timing()
    if not spots:
        print(f"[Farm] {pos}: no spots defined, skipping")
        return 0
    time.sleep(0.1)
    available = check_spots_available(spots)
    if not available:
        print(f"[Farm] {pos}: all {len(spots)} spot(s) are stumps, skipping")
        return 0
    print(f"[Farm] {pos}: {len(available)}/{len(spots)} available — clicking")
    for cx, cy in available:
        bot_input.click(cx, cy)
        time.sleep(0.1)
    print(f"[Farm] Waiting {timing['harvest_wait_seconds']}s...")
    time.sleep(timing["harvest_wait_seconds"])
    return len(available)


def _load_db_and_spots():
    with open(get_resource_path(), encoding="utf-8") as f:
        data = json.load(f)
    db = {(m["x"], m["y"]) for m in data["maps"]}
    spots_map = {
        (m["x"], m["y"]): m["spots"]
        for m in data["maps"]
        if m.get("spots")
    }
    print(f"[DB] {len(db)} maps, {len(spots_map)} pre-scouted")
    return db, spots_map


# ── Main loop ─────────────────────────────────────────────────────────────────

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
    db, spots_map = _load_db_and_spots()
    route_fwd = snake_route(db)
    route_rev = list(reversed(route_fwd))
    routes    = [route_fwd, route_rev]
    print(f"=== Farm Bot === {len(route_fwd)} maps in snake route")

    pos = _parse_start_pos()
    if pos:
        print(f"Starting at {pos}")
    else:
        print("No start pos — waiting for OCR...")

    print("Switch to the game. Starting in 5s...\n")
    for i in range(5, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  GO!\n")

    while pos is None:
        pos = vision.read_current_position()
        if pos:
            print(f"[Pos] {pos} (OCR initial)")
        else:
            print("[Pos] OCR failed — retrying in 2s...")
            time.sleep(2)

    # Resume mid-route if we're already at a known position
    sweep         = 1
    current_route = routes[0]
    idx           = next((i for i, p in enumerate(current_route) if p == pos), 0)
    print(f"[Loop] Route index {idx}/{len(current_route)}, pos={pos}\n")

    try:
        while True:
            target = current_route[idx]

            # Navigate to target one step at a time
            while pos != target:
                direction = step_toward(pos, target, db)
                if direction is None:
                    print(f"[Nav] No path from {pos} toward {target}")
                    break
                ok, nav_ocr = navigate(direction, current_pos=pos)
                if ok and nav_ocr:
                    pos = nav_ocr
                else:
                    ocr = vision.read_current_position()
                    if ocr:
                        pos = ocr
                    if pos != target:
                        print(f"[Nav] Skipping unreachable {target}")
                        break

            # Harvest at target
            if pos == target:
                ocr = vision.read_current_position()
                if ocr and ocr != pos:
                    print(f"[Pos] Tracker={pos} | OCR={ocr} — trusting OCR")
                    pos = ocr
                farm_current_map(pos, spots=spots_map.get(pos))

            idx += 1
            if idx >= len(current_route):
                idx = 0
                sweep += 1
                current_route = routes[(sweep - 1) % 2]
                direction_lbl = "S→N" if (sweep % 2 == 1) else "N→S"
                print(f"\n[Loop] Sweep {sweep - 1} done — sweep {sweep} ({direction_lbl})...\n")

    except KeyboardInterrupt:
        print("\n[Bot] Stopped.")


if __name__ == "__main__":
    main()

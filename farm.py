"""
farm.py
-------
Each iteration:
  1. OCR current position
  2. If map has pre-scouted spots: click them directly (fast path)
     Otherwise: hold-click blink-detect to find resources (scout path)
  3. Wait for harvest
  4. Use planner to pick next adjacent map (never backtracks unless stuck)
  5. Navigate there, wait for transition
  6. Repeat

Run scout.py first to populate spot coordinates in kalyptus.json.

Usage:
    python farm.py
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

from config.loader import get_screen_regions, get_timing, get_active_profile, get_resource_path
import input as bot_input
import vision
from planner import choose_next, _DELTAS


def navigate(direction, current_pos=None, max_retries=3):
    """
    Click the edge, wait for the map transition, then validate the new position.

    OCR flow after each navigation attempt:
      1. Read OCR. If it matches expected (current + delta) → confirmed, return it.
      2. If OCR still shows current_pos → click missed, retry the whole navigation.
      3. If OCR changed but doesn't match expected → retry OCR up to 2 more times
         (re-read only, no extra click) before returning the best reading anyway.

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

        bot_input.click_random_in_region(regions[direction])
        vision.wait_for_map_change()
        time.sleep(timing["post_map_change_delay"])

        best_ocr = None
        for ocr_try in range(3):  # 1 initial read + 2 OCR-only retries
            reading = vision.read_current_position()

            if reading is None:
                print(f"[Nav] OCR failed (try {ocr_try + 1}/3)")
                if ocr_try < 2:
                    time.sleep(0.3)
                continue

            if current_pos is not None and reading == current_pos:
                print(f"[Nav] OCR still at {current_pos} — click missed, retrying nav...")
                best_ocr = None
                break   # re-click, don't retry OCR

            best_ocr = reading  # OCR shows new map
            if reading == expected:
                print(f"[Nav] {direction} confirmed: OCR={reading}")
                return True, reading

            print(f"[Nav] OCR={reading}, expected={expected} — OCR retry {ocr_try + 1}/2...")
            if ocr_try < 2:
                time.sleep(0.3)

        if best_ocr is not None:
            # OCR changed but never matched expected — return what we got
            print(f"[Nav] OCR={best_ocr} != expected={expected}, returning best reading")
            return True, best_ocr

    print(f"[Nav] Could not confirm {direction} after {max_retries} attempts.")
    return False, None


# ── Blink detect ──────────────────────────────────────────────────────────────

_profile  = get_active_profile()
_W, _H    = _profile["resolution"]
_X_LIM    = int(0.75 * _W)
_Y_LIM    = int(0.93 * _H)

_regions_cache = get_screen_regions()

N_FRAMES        = 10
FRAME_INTERVAL  = 0.35   # 10 frames × 0.35 s = 3.5 s total
DIFF_THRESHOLD  = 20     # per-channel change to count as "blinked"
MIN_BLINK_COUNT = 4      # pixel must blink in at least N of the 9 pairs
MIN_BLOB_AREA   = 300
MAX_BLOB_AREA   = 20000
PADDING         = 18


def capture_frames(n=N_FRAMES, interval=FRAME_INTERVAL):
    """Hold left click at a random central position — only harvestable objects blink."""
    fz = _regions_cache["farm_zone"]
    x1, y1, x2, y2 = fz
    # Pick randomly within the inner 40% of the game area to avoid edges/UI
    mx1 = x1 + (x2 - x1) // 3
    mx2 = x2 - (x2 - x1) // 3
    my1 = y1 + (y2 - y1) // 3
    my2 = y2 - (y2 - y1) // 3
    hold_x = random.randint(mx1, mx2)
    hold_y = random.randint(my1, my2)

    frames = []
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        pyautogui.moveTo(hold_x, hold_y)
        time.sleep(0.10)
        pyautogui.mouseDown(button="left")
        time.sleep(0.15)
        for _ in range(n):
            t0  = time.perf_counter()
            raw = sct.grab(monitor)
            frames.append(cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR))
            wait = interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        pyautogui.mouseUp(button="left")
        time.sleep(0.10)
    return frames


def blink_detect(frames, diff_threshold=DIFF_THRESHOLD, min_blink_count=MIN_BLINK_COUNT):
    """
    Count how many consecutive frame pairs show a colour change at each pixel.
    Blinking trees change repeatedly at the same location (high count).
    Moving monsters hit any pixel in only 1-2 pairs (low count).
    Static ground/stones never change (count = 0).
    """
    h, w = frames[0].shape[:2]
    change_count = np.zeros((h, w), dtype=np.float32)
    for i in range(len(frames) - 1):
        diff = np.abs(
            frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)
        ).max(axis=2)
        change_count += (diff > diff_threshold)

    mask = (change_count >= min_blink_count).astype(np.uint8) * 255
    mask[:, _X_LIM:] = 0
    mask[_Y_LIM:, :] = 0

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        zones.append((
            max(0,      x - PADDING),
            max(0,      y - PADDING),
            min(_X_LIM, x + bw + PADDING),
            min(_Y_LIM, y + bh + PADDING),
        ))
    return _merge_boxes(zones)


def _merge_boxes(zones):
    merged  = list(zones)
    changed = True
    while changed:
        changed = False
        result, used = [], [False] * len(merged)
        for i, (ax1, ay1, ax2, ay2) in enumerate(merged):
            if used[i]:
                continue
            for j, (bx1, by1, bx2, by2) in enumerate(merged):
                if i == j or used[j]:
                    continue
                if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j]  = True
                    changed  = True
            result.append((ax1, ay1, ax2, ay2))
            used[i] = True
        merged = result
    return merged


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


# Trees animate continuously; stumps are completely static.
# Blink detection is the only reliable signal — no color check needed.
#
# Tuning guide:
#   SPOT_WINDOW          — smaller = less grass/background noise
#   SPOT_BLINK_DURATION  — longer = more chances for slow-blinking trees to register
#   SPOT_BLINK_MIN_PAIRS — higher = stricter (fewer stump false-positives)
#     Trees blink ~1 Hz, so in 2.5s you get ~10+ changed pairs out of 14.
#     Grass/static noise typically produces 0-2 changed pairs.
#     Setting min to 4 gives a wide gap between tree (10+) and stump (0-2).
SPOT_WINDOW          = 20   # tight focus on tree center, reduces grass noise
SPOT_BLINK_DURATION  = 3.0  # seconds — longer window catches slow/subtle blinks
SPOT_BLINK_FRAMES    = 18   # 17 consecutive frame pairs @ ~0.167 s each
SPOT_BLINK_MIN_PAIRS = 4    # out of 17 — trees easily hit 6+, stumps hit 0-2


def check_spots_available(spots):
    """
    Observe each saved spot for SPOT_BLINK_DURATION seconds.
    Trees animate continuously -> many changed frame-pairs.
    Stumps are static        -> 0-2 changed pairs (just background noise).
    No colour check — both are brown; animation is the only reliable signal.
    """
    frames   = []
    interval = SPOT_BLINK_DURATION / SPOT_BLINK_FRAMES
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        for _ in range(SPOT_BLINK_FRAMES):
            t0  = time.perf_counter()
            raw = sct.grab(monitor)
            frames.append(cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR))
            wait = interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    h, w = frames[0].shape[:2]
    total_pairs = len(frames) - 1
    available   = []

    for cx, cy in spots:
        x1 = max(0, cx - SPOT_WINDOW);  x2 = min(w, cx + SPOT_WINDOW)
        y1 = max(0, cy - SPOT_WINDOW);  y2 = min(h, cy + SPOT_WINDOW)

        blink_pairs = sum(
            1 for i in range(total_pairs)
            if np.abs(
                frames[i  ][y1:y2, x1:x2].astype(np.float32) -
                frames[i+1][y1:y2, x1:x2].astype(np.float32)
            ).max() > DIFF_THRESHOLD
        )

        if blink_pairs >= SPOT_BLINK_MIN_PAIRS:
            available.append((cx, cy))
            print(f"[Farm] ({cx},{cy}) blinked {blink_pairs}/{total_pairs} -> tree, click")
        else:
            print(f"[Farm] ({cx},{cy}) blinked {blink_pairs}/{total_pairs} -> stump, skip")

    return available


# ── Harvest ───────────────────────────────────────────────────────────────────

_RAW_DIR  = ROOT / "screenshots" / "raw"
_DET_DIR  = ROOT / "screenshots" / "detection"
_RAW_DIR.mkdir(parents=True, exist_ok=True)
_DET_DIR.mkdir(parents=True, exist_ok=True)

_save_idx = 0

def _save_debug(frames, zones, pos):
    global _save_idx
    _save_idx += 1
    tag = f"{_save_idx:04d}_{'_'.join(str(c) for c in pos) if pos else 'unknown'}"

    # Save middle raw frame
    ref = frames[len(frames) // 2]
    cv2.imwrite(str(_RAW_DIR / f"{tag}.png"), ref)

    # Save annotated detection frame
    out = ref.copy()
    for x1, y1, x2, y2 in zones:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "harvestable", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(str(_DET_DIR / f"{tag}.png"), out)
    print(f"[Debug] saved {tag}.png")


def farm_current_map(pos=None, spots=None):
    timing = get_timing()

    if spots:
        time.sleep(0.1)  # let map elements finish rendering
        print(f"[Farm] Watching {len(spots)} spot(s) ({SPOT_BLINK_DURATION}s, {SPOT_BLINK_FRAMES} frames)...")
        available = check_spots_available([(cx, cy) for cx, cy in spots])
        if not available:
            print("[Farm] No blinking spots found — all stumps, skipping harvest.")
            return 0
        print(f"[Farm] {len(available)}/{len(spots)} spot(s) available")
        for cx, cy in available:
            print(f"[Farm] Clicking ({cx}, {cy})")
            bot_input.click(cx, cy)
            time.sleep(0.3)
        print(f"[Farm] Waiting {timing['harvest_wait_seconds']}s...")
        time.sleep(timing["harvest_wait_seconds"])
        return len(available)

    print("[Farm] Blink-detecting resources...")
    frames = capture_frames()
    zones  = blink_detect(frames)
    print(f"[Farm] {len(zones)} resource(s) detected")

    _save_debug(frames, zones, pos)

    for zone in zones:
        cx, cy = _center(zone)
        print(f"[Farm] Clicking ({cx}, {cy})")
        bot_input.click(cx, cy)
        time.sleep(0.3)

    if zones:
        print(f"[Farm] Waiting {timing['harvest_wait_seconds']}s...")
        time.sleep(timing["harvest_wait_seconds"])

    return len(zones)


def _load_db_and_spots():
    with open(get_resource_path(), encoding="utf-8") as f:
        data = json.load(f)
    db = {(m["x"], m["y"]) for m in data["maps"]}
    spots_map = {
        (m["x"], m["y"]): m["spots"]
        for m in data["maps"]
        if m.get("spots")
    }
    scouted = len(spots_map)
    print(f"[DB] {len(db)} maps loaded, {scouted} pre-scouted")
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

    pos = _parse_start_pos()
    if pos:
        print(f"=== Farm Bot === ({len(db)} maps) — starting at {pos}")
    else:
        print(f"=== Farm Bot === ({len(db)} maps) — no start pos, waiting for OCR...")

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

    prev_pos      = None
    ocr_candidate = None  # OCR value that disagreed with tracker; checked next move

    try:
        while True:
            # Validate position — store mismatches as candidates for next-move check
            ocr = vision.read_current_position()
            if ocr is None:
                print(f"[Pos] {pos} (tracking | OCR failed)")
            elif ocr == pos:
                print(f"[Pos] {pos} OK")
                ocr_candidate = None
            else:
                print(f"[Pos] {pos} (tracking) | OCR says {ocr} — watching")
                ocr_candidate = ocr

            farm_current_map(pos, spots=spots_map.get(pos))

            direction, _ = choose_next(pos, db, prev_pos)
            if direction is None:
                break

            prev_pos    = pos
            ok, nav_ocr = navigate(direction, current_pos=pos)
            if ok:
                dx, dy    = _DELTAS[direction]
                delta_pos = (pos[0] + dx, pos[1] + dy)
                if nav_ocr == delta_pos:
                    pos = nav_ocr   # OCR matches expected — confirmed
                elif (ocr_candidate is not None and nav_ocr is not None
                      and nav_ocr == (ocr_candidate[0] + dx, ocr_candidate[1] + dy)):
                    print(f"[Pos] Candidate {ocr_candidate} confirmed -> correcting to {nav_ocr}")
                    pos = nav_ocr
                else:
                    pos = delta_pos
                ocr_candidate = None

    except KeyboardInterrupt:
        print("\n[Bot] Stopped.")


if __name__ == "__main__":
    main()

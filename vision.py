"""
bot/vision.py
-------------
Screen capture, coordinate OCR, black-screen detection,
and HSV-based pink flower detection for Kalyptus (and other resources).
"""

import re
import time
from pathlib import Path
from typing import Tuple, Optional, List

import mss
import numpy as np
import cv2

from config.loader import get_screen_regions, get_detection_config, get_timing

_regions   = get_screen_regions()
_detection = get_detection_config()
_timing    = get_timing()


# ─────────────────────────────────────────────
# Screen capture
# ─────────────────────────────────────────────

def capture_region(region: Tuple[int, int, int, int]) -> np.ndarray:
    """Capture a screen region (x1,y1,x2,y2) → BGR array."""
    x1, y1, x2, y2 = region
    with mss.mss() as sct:
        monitor = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
        shot = sct.grab(monitor)
        img = np.array(shot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def capture_full_screen() -> np.ndarray:
    """Capture the entire primary monitor → BGR array."""
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        img = np.array(shot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def save_screenshot(img: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)


# ─────────────────────────────────────────────
# Coordinate reading (OCR)
# ─────────────────────────────────────────────

_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789-,."


def _has_leading_minus(inverted: np.ndarray) -> bool:
    """
    Pixel-level check: is there a minus sign before the first digit?

    A minus sign is a thin horizontal bar → dark pixels in only ~5-10% of rows.
    A digit (e.g. '1', '8') is tall → dark pixels in 50-80% of rows.
    We look at x=[18:60] (just after the left padding, where the first character lives)
    and count what fraction of rows have dark content.
    """
    h, w = inverted.shape
    x1, x2 = 18, min(62, w // 5)
    y1, y2 = int(h * 0.15), int(h * 0.85)
    region = inverted[y1:y2, x1:x2]
    if region.size == 0:
        return False
    strip_w      = x2 - x1
    row_dark     = np.sum(region < 128, axis=1)
    active_rows  = int(np.sum(row_dark / strip_w > 0.15))
    active_ratio = active_rows / (y2 - y1)
    # Minus: thin bar → active_ratio < 0.22
    # Digit: tall glyph → active_ratio > 0.50
    return bool(active_ratio < 0.22 and np.max(row_dark) > 0)


def _resolve_sign(x: int, y: int) -> Tuple[int, int]:
    """
    Correct x's sign using the active resource's map list.

    If every map in the resource has negative X → x must be negative (Kalyptus).
    If every map has positive X              → x must be positive (Bambu Sombre).
    Mixed-sign resources                     → fall back to map-list lookup.
    """
    import json
    from config.loader import get_resource_path
    try:
        with open(get_resource_path(), encoding="utf-8") as f:
            data = json.load(f)
        xs      = [m["x"] for m in data["maps"]]
        map_set = {(m["x"], m["y"]) for m in data["maps"]}

        if all(v < 0 for v in xs):      # e.g. Kalyptus — X always negative
            return (-abs(x), y)
        if all(v > 0 for v in xs):      # e.g. Bambu Sombre — X always positive
            return (abs(x), y)

        # Mixed-sign resource: check map list
        if (x, y) in map_set:
            return (x, y)
        if (-x, y) in map_set:
            return (-x, y)
    except Exception:
        pass
    return (x, y)               # can't determine — return as-is


def read_current_position() -> Optional[Tuple[int, int]]:
    """
    OCR the HUD coordinate display.
    Tries thresholds 180 → 150 → 120 → OTSU until one parses successfully.
    Returns (x, y) or None on failure.
    """
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    img  = capture_region(_regions["coord"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (gray.shape[1] * 4, gray.shape[0] * 4),
                      interpolation=cv2.INTER_CUBIC)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    minus_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # horizontal — thickens minus sign

    for thresh in [180, 150, 120, 0]:   # 0 = OTSU
        if thresh == 0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        binary   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        binary   = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  open_k)
        inverted = cv2.bitwise_not(binary)
        inverted = cv2.dilate(inverted, minus_k, iterations=1)
        inverted = cv2.copyMakeBorder(inverted, 10, 10, 20, 10,
                                      cv2.BORDER_CONSTANT, value=255)

        raw    = pytesseract.image_to_string(inverted, config=_OCR_CONFIG).strip()
        result = _parse_ocr(raw)
        if result:
            x, y  = result
            result = _resolve_sign(x, y)
            return result

    print(f"[Vision] OCR failed (all thresholds). Last raw: '{raw}'")
    return None


def _parse_ocr(raw: str) -> Optional[tuple]:
    """Parse a raw OCR string into (x, y) coordinates."""
    clean = re.sub(r"\s", "", raw)

    # Primary: digit pair with separator
    matches = re.findall(r"(-?\d{1,3})[,.](-?\d{1,3})", clean)
    if matches:
        return int(matches[-1][0]), int(matches[-1][1])

    # Fallback: separator dropped — split merged run e.g. "-913" → (-9, 13)
    for token in re.findall(r"-?\d{2,6}", clean):
        result = _split_coords(token)
        if result:
            return result

    return None


def _split_coords(s):
    """Split a merged coordinate string like '-913' into (-9, 13)."""
    if s.startswith("-"):
        # Try 1 or 2 digits after the minus sign as x
        for x_len in range(2, min(4, len(s))):
            try:
                x = int(s[:x_len])
                y = int(s[x_len:])
                if -99 <= x <= 99 and -99 <= y <= 99 and s[x_len:]:
                    return x, y
            except ValueError:
                pass
    else:
        # Try 1 or 2 digits as x
        for x_len in range(1, min(3, len(s))):
            try:
                x = int(s[:x_len])
                y = int(s[x_len:])
                if -99 <= x <= 99 and -99 <= y <= 99 and s[x_len:]:
                    return x, y
            except ValueError:
                pass
    return None


# ─────────────────────────────────────────────
# Black-screen detection (map change / monster)
# ─────────────────────────────────────────────

def is_screen_black(threshold: int = None) -> bool:
    if threshold is None:
        threshold = _detection["black_screen_threshold"]
    img = capture_full_screen()
    h, w, _ = img.shape
    center = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    return float(np.mean(center)) < threshold


def wait_for_map_change(
    timeout: float = None,
    poll: float = None,
) -> bool:
    """
    Wait for a black-screen transition to start then end.
    Returns True if transition detected within timeout.
    """
    if timeout is None:
        timeout = _timing["map_change_timeout_seconds"]
    if poll is None:
        poll = _timing["map_change_poll_interval"]

    elapsed = 0.0
    went_black = False
    while elapsed < timeout:
        time.sleep(poll)
        elapsed += poll
        if is_screen_black():
            went_black = True
            break

    if not went_black:
        return False

    while True:
        time.sleep(poll)
        if not is_screen_black():
            return True


# ─────────────────────────────────────────────
# Pink flower HSV detection (resource locator)
# ─────────────────────────────────────────────

def detect_resource_zones(
    img: np.ndarray,
    hsv_lower: List[int],
    hsv_upper: List[int],
) -> List[Tuple[int, int, int, int]]:
    """
    Detect harvestable resource zones in a BGR image using HSV masking.

    Looks for the distinctive pink flower colour that only appears on
    harvestable (not yet cut) trees.

    Returns list of (x1, y1, x2, y2) bounding boxes in image coordinates.
    """
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_area = _detection["min_blob_area"]
    max_area = _detection["max_blob_area"]
    pad      = _detection["box_padding"]
    zones    = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        zones.append((
            max(0, x - pad),
            max(0, y - pad),
            x + w + pad,
            y + h + pad,
        ))

    # Merge overlapping boxes
    zones = _merge_overlapping(zones)
    return zones


def _merge_overlapping(
    boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that overlap into a single bounding box."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    for bx1, by1, bx2, by2 in boxes[1:]:
        mx1, my1, mx2, my2 = merged[-1]
        if bx1 <= mx2 and by1 <= my2:
            merged[-1] = (min(mx1, bx1), min(my1, by1),
                          max(mx2, bx2), max(my2, by2))
        else:
            merged.append((bx1, by1, bx2, by2))
    return merged


def annotate_detections(
    img: np.ndarray,
    zones: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw red rectangles around each detected zone. Returns annotated copy."""
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(zones):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"#{i+1}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def zone_center(zone: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = zone
    return (x1 + x2) // 2, (y1 + y2) // 2


def is_in_farm_zone(x: int, y: int) -> bool:
    x1, y1, x2, y2 = _regions["farm_zone"]
    return x1 <= x <= x2 and y1 <= y <= y2

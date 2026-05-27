"""
tools/test_ocr.py
-----------------
Test OCR. Saves to tools/ocr_pics/:
  - full.png              : full screenshot with red box showing the crop region
  - crop.png              : the raw crop
  - gray.png              : after grayscale + 4x upscale
  - thresh_180.png        : binary threshold at 180, inverted
  - thresh_150.png        : binary threshold at 150, inverted
  - thresh_120.png        : binary threshold at 120, inverted
  - thresh_otsu.png       : OTSU threshold, inverted
  - final.png             : the image that actually produced a result (or last tried)

Usage:
    python tools/test_ocr.py
"""

import sys
import os
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

COUNTDOWN = 5
OUT_DIR   = ROOT / "tools" / "ocr_pics"
OUT_DIR.mkdir(exist_ok=True)

print(f"Switch to the game. Capturing in {COUNTDOWN}s...")
for i in range(COUNTDOWN, 0, -1):
    print(f"  {i}...", end="\r", flush=True)
    time.sleep(1)
print("  Capturing!   ")

import cv2
import mss
import numpy as np
import pytesseract
from config.loader import get_active_profile
from vision import _parse_ocr, _OCR_CONFIG, _has_leading_minus

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Screenshot ────────────────────────────────────────────────────────────────
with mss.mss() as sct:
    raw  = sct.grab(sct.monitors[1])
    full = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

H, W = full.shape[:2]
profile = get_active_profile()
x1r, y1r, x2r, y2r = profile["coord_region"]
px1, py1 = int(x1r * W), int(y1r * H)
px2, py2 = int(x2r * W), int(y2r * H)

# Save full screenshot with red box
annotated = full.copy()
cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)
cv2.imwrite(str(OUT_DIR / "full.png"), annotated)

# Save raw crop
crop = full[py1:py2, px1:px2]
cv2.imwrite(str(OUT_DIR / "crop.png"), crop)

print(f"Full screenshot -> tools/ocr_pics/full.png  (red box = crop region)")
print(f"Crop            -> tools/ocr_pics/crop.png")
print(f"Region pixels   : ({px1},{py1}) -> ({px2},{py2})  of {W}x{H}")
print()

# ── Preprocessing pipeline ────────────────────────────────────────────────────
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (gray.shape[1] * 4, gray.shape[0] * 4),
                  interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(OUT_DIR / "gray.png"), gray)

close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
minus_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # horizontal — thickens minus sign

final_img   = None
final_label = None

for thresh in [180, 150, 120, 0]:
    label = f"thresh_{thresh}" if thresh > 0 else "thresh_otsu"

    if thresh == 0:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    binary   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    binary   = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  open_k)
    inverted = cv2.bitwise_not(binary)
    inverted = cv2.dilate(inverted, minus_k, iterations=1)  # thicken minus sign horizontally
    inverted = cv2.copyMakeBorder(inverted, 10, 10, 20, 10,
                                  cv2.BORDER_CONSTANT, value=255)  # pad so edge chars aren't clipped

    cv2.imwrite(str(OUT_DIR / f"{label}.png"), inverted)

    raw    = pytesseract.image_to_string(inverted, config=_OCR_CONFIG).strip()
    result = _parse_ocr(raw)
    minus  = _has_leading_minus(inverted)

    # Debug: print region details for first threshold only
    if label == "thresh_180":
        h, w = inverted.shape
        x1, x2 = 18, min(62, w // 5)
        y1, y2 = int(h * 0.15), int(h * 0.85)
        region = inverted[y1:y2, x1:x2]
        strip_w  = x2 - x1
        row_dark = np.sum(region < 128, axis=1)
        active_rows = int(np.sum(row_dark / strip_w > 0.15))
        active_ratio = active_rows / (y2 - y1)
        print(f"  [debug] img={w}x{h}  check=x[{x1}:{x2}] y[{y1}:{y2}]  active_rows={active_rows}/{y2-y1}  active_ratio={active_ratio:.3f}  max_row_dark={int(np.max(row_dark))}")

    if result:
        x, y = result
        if x > 0 and minus:
            result = (-x, y)

    status = f"-> '{raw}' => {result}  [minus_pixel={minus}]" if result else f"-> '{raw}' => no match  [minus_pixel={minus}]"
    print(f"  {label:18s}  {status}")

    if final_img is None:
        final_img   = inverted
        final_label = label
    if result:
        final_img   = inverted
        final_label = label
        print(f"  ** used this threshold **")
        break

if final_img is not None:
    cv2.imwrite(str(OUT_DIR / "final.png"), final_img)

print()
print("Saved images:")
for name in ["full.png", "crop.png", "gray.png",
             "thresh_180.png", "thresh_150.png", "thresh_120.png",
             "thresh_otsu.png", "final.png"]:
    path = OUT_DIR / name
    print(f"  {'[ok]' if path.exists() else '[--]'}  tools/ocr_pics/{name}")

#!/usr/bin/env python3
"""
tools/test_detection.py
-----------------------
Offline tester for resource spot availability detection.

Sample frames in tools/blink_detect/frames/:
  grown*.png     -- crops around an AVAILABLE (harvestable) spot
  harvested*.png -- crops around the SAME spot after harvest (stump)

The tool runs every detection method against those samples and reports
accuracy, then sweeps parameters to find the best profile for each resource.

Usage
-----
  python tools/test_detection.py
      Test all resources. Outputs annotated images + report to tools/results/

  python tools/test_detection.py --resource boumu
      Focus on one resource.

  python tools/test_detection.py --sweep
      Full colour-threshold sweep (try many min_px values).

  python tools/test_detection.py --hue-scan
      Print a per-hue-bin table to understand what colours distinguish the states.

  python tools/test_detection.py --auto-profile
      Auto-search the best HSV range and print the JSON snippet to paste.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

TOOLS_DIR   = Path(__file__).resolve().parent
ROOT        = TOOLS_DIR.parent
FRAMES_DIR  = TOOLS_DIR / "blink_detect" / "frames"
RESULTS_DIR = TOOLS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_frames(glob_pat):
    paths = sorted(FRAMES_DIR.glob(glob_pat))
    if not paths:
        sys.exit(f"[Error] No frames: {FRAMES_DIR / glob_pat}")
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [Warn] cannot read {p.name}")
            continue
        out.append((p.name, img))
    return out


def _hsv_count(frame_bgr, lo, hi):
    """Count pixels inside the HSV range. Returns (count, mask)."""
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return int(np.count_nonzero(mask)), mask


def _annotate(frame_bgr, mask, label, count, tint=(0, 220, 80)):
    """Tint matched pixels and add a status label."""
    out = frame_bgr.copy()
    overlay = np.zeros_like(out)
    overlay[mask > 0] = tint
    out = cv2.addWeighted(out, 0.55, overlay, 0.45, 0)
    text_color = tint if count >= 8 else (60, 60, 255)
    cv2.putText(out, f"{label}  [{count} px]",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)
    return out


def _side_by_side(imgs, labels):
    """Stack images horizontally, padding shorter ones to equal height."""
    max_h = max(i.shape[0] for i in imgs)
    padded = []
    for img, lbl in zip(imgs, labels):
        h, w = img.shape[:2]
        pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
        tile = np.vstack([img, pad]) if pad.size else img.copy()
        cv2.putText(tile, lbl, (4, max_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        padded.append(tile)
    return np.hstack(padded)


# ─── hue histogram ────────────────────────────────────────────────────────────

def _hue_histogram(frames_bgr):
    """Return 36-bin hue histogram (each bin = 5deg of OpenCV's 0-180 range)."""
    hist = np.zeros(36, dtype=np.float64)
    for frame in frames_bgr:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bins = np.minimum(hsv[:, :, 0].flatten() // 5, 35).astype(np.int32)
        hist += np.bincount(bins, minlength=36)
    return hist


# ─── auto HSV search ──────────────────────────────────────────────────────────

def _auto_hsv(grown_frames, harv_frames, s_min=60, v_min=60, fp_penalty=4):
    """
    Sweep hue bands [h_lo, h_lo+w] and find the range that maximises
    grown_pixels - fp_penalty * harvested_pixels.
    Returns (h_lo, h_hi, score, grown_totals, harv_totals).
    """
    grown_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for _, f in grown_frames]
    harv_hsv  = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for _, f in harv_frames]

    best_score = float("-inf")
    best       = (0, 30)
    best_g     = best_h = 0

    for h_lo in range(0, 175, 5):
        for h_w in range(10, 65, 5):
            h_hi = min(h_lo + h_w, 180)
            lo   = np.array([h_lo, s_min, v_min], np.uint8)
            hi   = np.array([h_hi, 255,   255  ], np.uint8)
            g_px = sum(np.count_nonzero(cv2.inRange(h, lo, hi)) for h in grown_hsv)
            h_px = sum(np.count_nonzero(cv2.inRange(h, lo, hi)) for h in harv_hsv)
            sc   = g_px - fp_penalty * h_px
            if sc > best_score:
                best_score, best, best_g, best_h = sc, (h_lo, h_hi), g_px, h_px

    return best[0], best[1], best_score, best_g, best_h


# ─── blink / animation variance ───────────────────────────────────────────────

def _blink_variance(frames_bgr):
    """
    Mean pixel std-dev across a set of frames (all resized to common size).
    High = lots of animation / blinking.  Low = static stump.
    Returns 0.0 if fewer than 2 frames.
    """
    if len(frames_bgr) < 2:
        return 0.0
    # Resize all to the smallest common size to allow stacking
    min_h = min(f.shape[0] for f in frames_bgr)
    min_w = min(f.shape[1] for f in frames_bgr)
    resized = [cv2.resize(f, (min_w, min_h)) for f in frames_bgr]
    stack = np.stack([r.astype(np.float32) for r in resized], axis=0)
    return float(np.std(stack, axis=0).mean())


def _frame_diff(a, b):
    """Mean absolute per-pixel difference between two same-shape frames."""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


# ─── report builder ───────────────────────────────────────────────────────────

class _Report:
    def __init__(self):
        self._lines = []

    def h1(self, t):
        b = "=" * (len(t) + 4)
        self._lines += ["", b, f"  {t}", b]

    def h2(self, t):
        self._lines += ["", f"  --- {t} ---"]

    def ln(self, text=""):
        self._lines.append(text)

    def save(self, path):
        txt = "\n".join(self._lines)
        path.write_text(txt, encoding="utf-8")
        print(f"[Report] -> {path}")

    def echo(self):
        print("\n".join(self._lines))


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resource",     default=None,
                    help="Resource stem to focus on (e.g. boumu, kalyptus)")
    ap.add_argument("--sweep",        action="store_true",
                    help="Sweep min_px threshold values")
    ap.add_argument("--hue-scan",     action="store_true",
                    help="Print per-hue-bin counts for grown vs harvested")
    ap.add_argument("--auto-profile", action="store_true",
                    help="Auto-search best HSV range and print JSON snippet")
    ap.add_argument("--s-min",  type=int, default=60,
                    help="Min saturation for auto HSV search (default 60)")
    ap.add_argument("--v-min",  type=int, default=60,
                    help="Min value/brightness for auto HSV search (default 60)")
    ap.add_argument("--fp-penalty", type=int, default=4,
                    help="False-positive penalty weight in auto search (default 4)")
    args = ap.parse_args()

    rep = _Report()
    rep.h1("Detection Test Report")

    # ── load frames ────────────────────────────────────────────────────────────
    grown_frames = _load_frames("grown*.png")
    harv_frames  = _load_frames("harvested*.png")
    grown_bgr    = [f for _, f in grown_frames]
    harv_bgr     = [f for _, f in harv_frames]

    rep.ln(f"Grown frames    : {[n for n,_ in grown_frames]}")
    rep.ln(f"Harvested frames: {[n for n,_ in harv_frames]}")
    rep.ln(f"Frame sizes (grown)    : {[f.shape[1::-1] for f in grown_bgr]}")
    rep.ln(f"Frame sizes (harvested): {[f.shape[1::-1] for f in harv_bgr]}")

    # ── blink / animation analysis ─────────────────────────────────────────────
    rep.h2("Animation variance (higher = more movement between frames)")
    g_var = _blink_variance(grown_bgr)
    h_var = _blink_variance(harv_bgr)
    rep.ln(f"  Grown     frames : {g_var:.3f}")
    rep.ln(f"  Harvested frames : {h_var:.3f}")
    rep.ln(f"  Ratio grown/harv : {g_var / h_var:.2f}x" if h_var > 0 else "  (no harvested variance)")

    # Check same-size pairs within grown set
    same_pairs = [
        (n1, n2, f1, f2)
        for i, (n1, f1) in enumerate(grown_frames)
        for j, (n2, f2) in enumerate(grown_frames)
        if i < j and f1.shape == f2.shape
    ]
    if same_pairs:
        rep.h2("Frame-pair diffs (grown, same resolution)")
        for n1, n2, f1, f2 in same_pairs:
            rep.ln(f"  {n1} vs {n2}: mean diff = {_frame_diff(f1, f2):.2f}")
    else:
        rep.ln("  (no same-resolution grown pairs -- provide sequential frames to test blink)")

    # Blink-detect visualisation: change heatmap on same-size grown pairs
    if same_pairs:
        for n1, n2, f1, f2 in same_pairs[:1]:
            diff_img = cv2.absdiff(f1, f2)
            diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
            heat = cv2.applyColorMap(
                cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_HOT
            )
            cv2.imwrite(str(RESULTS_DIR / f"blink_heatmap_{n1}_vs_{n2}.png"), heat)

    # ── hue scan ──────────────────────────────────────────────────────────────
    if args.hue_scan:
        rep.h1("Hue Distribution (grown vs harvested)")
        gh = _hue_histogram(grown_bgr)
        hh = _hue_histogram(harv_bgr)
        rep.ln(f"  {'bin':>4}  {'H range':>10}  {'grown':>10}  {'harvested':>10}  {'diff':>10}")
        rep.ln(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for i in range(36):
            h_lo, h_hi = i * 5, i * 5 + 5
            g, h = int(gh[i]), int(hh[i])
            mark = "  <-- distinctive" if (g - h) > max(gh) * 0.05 and g > 20 else ""
            rep.ln(f"  {i:>4}  H={h_lo:3d}-{h_hi:3d}  {g:>10d}  {h:>10d}  {g-h:>+10d}{mark}")

        # Visualise hue histograms as a bar chart image
        bar_h, bar_w = 200, 36 * 14
        bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        max_v = max(gh.max(), hh.max(), 1)
        for i in range(36):
            x = i * 14
            hue_color = np.array([[[i * 5, 200, 200]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hue_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            gh_bar = int(gh[i] / max_v * (bar_h - 10))
            hh_bar = int(hh[i] / max_v * (bar_h - 10))
            cv2.rectangle(bar, (x, bar_h - gh_bar), (x + 6, bar_h), bgr_color, -1)
            cv2.rectangle(bar, (x + 7, bar_h - hh_bar), (x + 13, bar_h), (60, 60, 200), -1)
        cv2.putText(bar, "Color=grown  Blue=harvested", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.imwrite(str(RESULTS_DIR / "hue_histogram.png"), bar)
        rep.ln(f"\n  Histogram saved -> {RESULTS_DIR / 'hue_histogram.png'}")

    # ── auto HSV profile search ────────────────────────────────────────────────
    rep.h1("Auto HSV Range Search")
    h_lo, h_hi, score, g_total, h_total = _auto_hsv(
        grown_frames, harv_frames, args.s_min, args.v_min, args.fp_penalty
    )
    auto_lo = [h_lo, args.s_min, args.v_min]
    auto_hi = [h_hi, 255, 255]
    rep.ln(f"  Best range  : H={h_lo}-{h_hi},  S>={args.s_min},  V>={args.v_min}")
    rep.ln(f"  Score       : {score}  (grown_px={g_total}, harv_px={h_total})")
    rep.ln(f"  To use this range, add to your resource JSON:")
    rep.ln(f'    "spot_color": {{')
    rep.ln(f'      "hsv_lower": [{h_lo}, {args.s_min}, {args.v_min}],')
    rep.ln(f'      "hsv_upper": [{h_hi}, 255, 255]')
    rep.ln(f'    }}')

    if args.auto_profile:
        print("\n--- Suggested spot_color profile ---")
        print(json.dumps({"spot_color": {
            "hsv_lower": auto_lo, "hsv_upper": auto_hi
        }}, indent=2))

    # ── load resources ────────────────────────────────────────────────────────
    all_resources = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "resources").glob("*.json"))
    }
    if args.resource:
        if args.resource not in all_resources:
            sys.exit(f"[Error] '{args.resource}' not found. "
                     f"Available: {', '.join(all_resources)}")
        test_resources = {args.resource: all_resources[args.resource]}
    else:
        test_resources = all_resources

    # ── per-resource colour detection tests ───────────────────────────────────
    for res_stem, data in test_resources.items():
        res_name = data.get("resource", res_stem)
        rep.h1(f"Resource: {res_name}  [{res_stem}.json]")

        color = data.get("spot_color")
        if not color:
            rep.ln("  No spot_color defined -- colour detection not applicable.")
            rep.ln("  This resource uses blink detection (no offline frames available).")
            continue

        lo = color["hsv_lower"]
        hi = color["hsv_upper"]
        rep.ln(f"  Current profile: hsv_lower={lo}  hsv_upper={hi}")

        MIN_PX = 8  # default threshold

        # ── test current profile ────────────────────────────────────────────
        rep.h2("Colour detection -- current profile")
        rep.ln(f"  {'frame':30s}  {'state':>10}  {'px':>6}  {'verdict':>12}")
        rep.ln(f"  {'-'*30}  {'-'*10}  {'-'*6}  {'-'*12}")

        results = []
        tiles   = []

        for name, frame in grown_frames:
            count, mask = _hsv_count(frame, lo, hi)
            available   = count >= MIN_PX
            results.append(("grown", available))
            verdict = "AVAILABLE" if available else "MISSED"
            rep.ln(f"  {name:30s}  {'grown':>10}  {count:>6}  {verdict:>12}")
            ann = _annotate(frame, mask, f"grown | {verdict}", count)
            cv2.imwrite(str(RESULTS_DIR / f"{res_stem}_cur_grown_{name}"), ann)
            tiles.append((ann, name))

        for name, frame in harv_frames:
            count, mask = _hsv_count(frame, lo, hi)
            available   = count >= MIN_PX
            results.append(("harvested", not available))  # correct = NOT available
            verdict = "stump OK" if not available else "FALSE POS!"
            rep.ln(f"  {name:30s}  {'harvested':>10}  {count:>6}  {verdict:>12}")
            tint = (60, 60, 255) if available else (80, 80, 80)
            ann = _annotate(frame, mask, f"harvested | {verdict}", count, tint=tint)
            cv2.imwrite(str(RESULTS_DIR / f"{res_stem}_cur_harvested_{name}"), ann)
            tiles.append((ann, name))

        tp = sum(1 for s, c in results if s == "grown"     and c)
        tn = sum(1 for s, c in results if s == "harvested" and c)
        fn = sum(1 for s, c in results if s == "grown"     and not c)
        fp = sum(1 for s, c in results if s == "harvested" and not c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rep.ln(f"\n  TP={tp}  FN={fn}  TN={tn}  FP={fp}  |  "
               f"Precision={prec:.2f}  Recall={rec:.2f}")

        # composite side-by-side
        if tiles:
            composite = _side_by_side([t for t, _ in tiles], [l for _, l in tiles])
            cv2.imwrite(str(RESULTS_DIR / f"{res_stem}_cur_composite.png"), composite)

        # ── test auto-found profile ─────────────────────────────────────────
        rep.h2(f"Colour detection -- auto-found H={h_lo}-{h_hi} S>={args.s_min} V>={args.v_min}")
        tiles_auto = []
        results_auto = []
        for name, frame in grown_frames:
            count, mask = _hsv_count(frame, auto_lo, auto_hi)
            available   = count >= MIN_PX
            results_auto.append(("grown", available))
            verdict = "AVAILABLE" if available else "MISSED"
            rep.ln(f"  grown     {name}: {count:>4}px  -> {verdict}")
            ann = _annotate(frame, mask, f"auto | {verdict}", count, tint=(0, 180, 255))
            cv2.imwrite(str(RESULTS_DIR / f"{res_stem}_auto_grown_{name}"), ann)
            tiles_auto.append((ann, name))
        for name, frame in harv_frames:
            count, mask = _hsv_count(frame, auto_lo, auto_hi)
            available   = count >= MIN_PX
            results_auto.append(("harvested", not available))
            verdict = "stump OK" if not available else "FALSE POS!"
            rep.ln(f"  harvested {name}: {count:>4}px  -> {verdict}")
            ann = _annotate(frame, mask, f"auto | {verdict}", count,
                            tint=(60, 60, 255) if available else (80, 80, 80))
            cv2.imwrite(str(RESULTS_DIR / f"{res_stem}_auto_harvested_{name}"), ann)
            tiles_auto.append((ann, name))

        if tiles_auto:
            composite_auto = _side_by_side([t for t, _ in tiles_auto], [l for _, l in tiles_auto])
            cv2.imwrite(str(RESULTS_DIR / f"{res_stem}_auto_composite.png"), composite_auto)

        tp2 = sum(1 for s, c in results_auto if s == "grown"     and c)
        tn2 = sum(1 for s, c in results_auto if s == "harvested" and c)
        fn2 = sum(1 for s, c in results_auto if s == "grown"     and not c)
        fp2 = sum(1 for s, c in results_auto if s == "harvested" and not c)
        prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0
        rec2  = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
        rep.ln(f"  TP={tp2}  FN={fn2}  TN={tn2}  FP={fp2}  |  "
               f"Precision={prec2:.2f}  Recall={rec2:.2f}")

        # ── parameter sweep ─────────────────────────────────────────────────
        if args.sweep:
            rep.h2(f"Min-pixel sweep -- current profile")
            rep.ln(f"  {'min_px':>6}  {'grown_pass':>12}  {'harv_FP':>10}  notes")
            best_min = None
            for min_px in [2, 4, 6, 8, 10, 12, 16, 20, 30, 50]:
                g_pass = sum(
                    1 for _, f in grown_frames
                    if _hsv_count(f, lo, hi)[0] >= min_px
                )
                h_fp = sum(
                    1 for _, f in harv_frames
                    if _hsv_count(f, lo, hi)[0] >= min_px
                )
                note = ""
                if g_pass == len(grown_frames) and h_fp == 0:
                    note = "  *** perfect ***"
                    if best_min is None:
                        best_min = min_px
                elif g_pass == len(grown_frames):
                    note = f"  all grown pass, {h_fp} FP"
                elif h_fp == 0:
                    note = f"  no FP, but misses {len(grown_frames)-g_pass} grown"
                rep.ln(f"  {min_px:>6}  {g_pass:>5}/{len(grown_frames):>5}     "
                       f"{h_fp:>5}/{len(harv_frames):>5}  {note}")
            if best_min is not None:
                rep.ln(f"\n  -> Optimal min_px for current profile: {best_min}")
            else:
                rep.ln("\n  -> No single min_px value achieves perfect separation.")

            rep.h2(f"Min-pixel sweep -- auto-found profile H={h_lo}-{h_hi}")
            rep.ln(f"  {'min_px':>6}  {'grown_pass':>12}  {'harv_FP':>10}  notes")
            best_min_auto = None
            for min_px in [2, 4, 6, 8, 10, 12, 16, 20, 30, 50]:
                g_pass = sum(
                    1 for _, f in grown_frames
                    if _hsv_count(f, auto_lo, auto_hi)[0] >= min_px
                )
                h_fp = sum(
                    1 for _, f in harv_frames
                    if _hsv_count(f, auto_lo, auto_hi)[0] >= min_px
                )
                note = ""
                if g_pass == len(grown_frames) and h_fp == 0:
                    note = "  *** perfect ***"
                    if best_min_auto is None:
                        best_min_auto = min_px
                elif g_pass == len(grown_frames):
                    note = f"  all grown pass, {h_fp} FP"
                elif h_fp == 0:
                    note = f"  no FP, but misses {len(grown_frames)-g_pass} grown"
                rep.ln(f"  {min_px:>6}  {g_pass:>5}/{len(grown_frames):>5}     "
                       f"{h_fp:>5}/{len(harv_frames):>5}  {note}")
            if best_min_auto is not None:
                rep.ln(f"\n  -> Optimal min_px for auto profile: {best_min_auto}")

    # ── save ──────────────────────────────────────────────────────────────────
    rep.h1("Summary")
    rep.ln(f"  Annotated images saved to: {RESULTS_DIR}/")
    rep.ln(f"  Prefixes: <resource>_cur_*  (current profile)")
    rep.ln(f"            <resource>_auto_* (auto-found profile)")
    rep.ln(f"            <resource>_*_composite.png (side-by-side)")
    if args.hue_scan:
        rep.ln(f"  hue_histogram.png -- per-hue-bin bar chart")

    rep.save(RESULTS_DIR / "report.txt")
    rep.echo()
    print(f"\n[Done] All results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

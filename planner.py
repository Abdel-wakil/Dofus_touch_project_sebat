"""
planner.py
----------
Answers one question: given where I am, where should I go next?

Scoring system (weighted random — not greedy):
  +20  destination has not yet been visited this run
  +dist_reduction * 3  moving toward the centroid of all unvisited maps
  ×0.05  going back to prev_pos (soft penalty — allowed as last resort)

Direction → coordinate delta:
    right  = x + 1
    left   = x - 1
    bottom = y + 1
    top    = y - 1
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_DELTAS = {
    "right":  ( 1,  0),
    "left":   (-1,  0),
    "bottom": ( 0,  1),
    "top":    ( 0, -1),
}


def load_db(path=None):
    """Load resource JSON and return a set of (x, y) tuples."""
    p = Path(path) if path else ROOT / "kalyptus.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {(m["x"], m["y"]) for m in data["maps"]}


def adjacent_in_db(pos, db):
    """Return all (direction, next_pos) where next_pos is in the database."""
    cx, cy = pos
    return [
        (d, (cx + dx, cy + dy))
        for d, (dx, dy) in _DELTAS.items()
        if (cx + dx, cy + dy) in db
    ]


def choose_next(pos, db, prev_pos=None, visited=None):
    """
    Pick the next direction using weighted random scoring.

    visited  — set of (x, y) already visited this run (optional).
                Unvisited destinations score much higher, steering the
                route toward unexplored territory.

    Returns (direction, next_pos), or (None, None) if no moves available.
    """
    all_moves = adjacent_in_db(pos, db)
    if not all_moves:
        print(f"[Plan] {pos}: no adjacent maps — stuck")
        return None, None

    visited   = visited or set()
    cx, cy    = pos
    unvisited = db - visited

    # Centroid of unvisited maps (for directional bias)
    if unvisited:
        ucx      = sum(x for x, y in unvisited) / len(unvisited)
        ucy      = sum(y for x, y in unvisited) / len(unvisited)
        dist_now = math.sqrt((cx - ucx) ** 2 + (cy - ucy) ** 2)
    else:
        ucx = ucy = dist_now = None

    scores = []
    for _d, nxt in all_moves:
        score = 1.0

        # Large bonus for going somewhere new
        if nxt not in visited:
            score += 20.0

        # Reward moving toward the cluster of unvisited maps
        if ucx is not None:
            dist_next = math.sqrt((nxt[0] - ucx) ** 2 + (nxt[1] - ucy) ** 2)
            score += max(0.0, dist_now - dist_next) * 3.0

        # Heavy backtrack penalty — still possible as a last resort
        if nxt == prev_pos:
            score *= 0.05

        scores.append(max(score, 0.01))

    # Weighted random selection
    total  = sum(scores)
    r      = random.uniform(0, total)
    cumsum = 0.0
    chosen_dir, chosen_pos = all_moves[-1]   # fallback
    for (d, nxt), score in zip(all_moves, scores):
        cumsum += score
        if r <= cumsum:
            chosen_dir, chosen_pos = d, nxt
            break

    _log(pos, all_moves, scores, chosen_dir, chosen_pos, visited)
    return chosen_dir, chosen_pos


def _log(pos, all_moves, scores, chosen_dir, chosen_pos, visited):
    total = sum(scores)
    parts = []
    for (d, nxt), s in zip(all_moves, scores):
        pct   = round(100 * s / total)
        mark  = "*" if d == chosen_dir else " "
        tag   = "" if nxt in visited else " [new]"
        parts.append(f"{mark}{d}({pct}%){tag}")
    print(f"[Plan] {pos} -> {chosen_dir} {chosen_pos}  |  " + "  ".join(parts))

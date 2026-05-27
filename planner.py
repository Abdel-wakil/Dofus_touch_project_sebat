"""
planner.py
----------
Answers one question: given where I am, where should I go next?

Rules:
  1. Only move to maps that exist in kalyptus.json
  2. A valid move is ±1 on exactly one axis (right/left/bottom/top)
  3. Never go back to the map we just came from (prev_pos)
  4. If all neighbours are prev_pos (dead end), allow going back
  5. If multiple options remain, pick randomly

Direction → coordinate delta:
    right  = x + 1
    left   = x - 1
    bottom = y + 1
    top    = y - 1
"""

import json
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
    """Load kalyptus.json and return a set of (x, y) tuples."""
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


def choose_next(pos, db, prev_pos=None):
    """
    Pick the next direction to move.
    Returns (direction, next_pos), or (None, None) if no moves available.
    """
    all_moves = adjacent_in_db(pos, db)

    if not all_moves:
        print(f"[Plan] {pos}: no adjacent Kalyptus maps — stuck")
        return None, None

    # Exclude the map we just came from
    forward = [(d, p) for d, p in all_moves if p != prev_pos]
    moves   = forward if forward else all_moves   # fall back if dead end

    direction, next_pos = random.choice(moves)

    _log(pos, all_moves, forward, direction, next_pos)
    return direction, next_pos


def _log(pos, all_moves, forward, chosen_dir, chosen_pos):
    all_dirs = ", ".join(d for d, _ in all_moves)
    fwd_dirs = ", ".join(d for d, _ in forward) if forward else "none (backtrack allowed)"
    print(f"[Plan] {pos}")
    print(f"       accessible : {all_dirs}")
    print(f"       forward    : {fwd_dirs}")
    print(f"       chosen     : {chosen_dir} -> {chosen_pos}")

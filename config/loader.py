"""
config/loader.py
----------------
Single source of truth for all runtime settings.
Resolves ratio-based screen regions to absolute pixels
based on the active resolution profile.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any

_CONFIG_PATH   = Path(__file__).parent.parent / "settings.json"
_RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def load_settings() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_active_profile() -> Dict[str, Any]:
    s = load_settings()
    return s["profiles"][s["active_profile"]]


def switch_profile(name: str) -> None:
    """Switch between '2k' and '1080p'. Persists to settings.json."""
    s = load_settings()
    if name not in s["profiles"]:
        raise ValueError(f"Unknown profile '{name}'")
    s["active_profile"] = name
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def _resolve(ratio: list, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = ratio
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def get_screen_regions() -> Dict[str, Tuple[int, int, int, int]]:
    p = get_active_profile()
    w, h = p["resolution"]
    regions = {}
    for direction, ratio in p["map_change_regions"].items():
        regions[direction] = _resolve(ratio, w, h)
    regions["minimap"]   = _resolve(p["minimap_region"], w, h)
    regions["coord"]     = _resolve(p["coord_region"], w, h)
    regions["farm_zone"] = _resolve(p["farm_zone"], w, h)
    return regions


def get_resolution() -> Tuple[int, int]:
    return tuple(get_active_profile()["resolution"])


def get_timing() -> Dict[str, float]:
    return load_settings()["timing"]


def get_detection_config() -> Dict[str, Any]:
    return load_settings()["detection"]


def get_inventory_config() -> Dict[str, Any]:
    return load_settings()["inventory"]


def get_scout_config() -> Dict[str, str]:
    return load_settings()["scout"]


# ── Resource management ────────────────────────────────────────────────────────

def list_resources() -> list:
    """Return sorted list of resource stems from the resources/ folder."""
    return sorted(p.stem for p in _RESOURCES_DIR.glob("*.json"))


def get_active_resource() -> str:
    """Return the active resource stem (e.g. 'kalyptus')."""
    return load_settings().get("active_resource", "kalyptus")


def get_resource_path(name: str = None) -> Path:
    """Return the Path to a resource JSON file. Uses active resource if name is None."""
    stem = name if name is not None else get_active_resource()
    return _RESOURCES_DIR / f"{stem}.json"


def set_active_resource(name: str) -> None:
    """Persist the active resource selection to settings.json."""
    s = load_settings()
    s["active_resource"] = name
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

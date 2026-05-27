"""
bot/input.py
------------
Human-like mouse and keyboard simulation.
All timing values pulled from config/settings.json.
"""

import time
import random
from typing import Tuple

import pyautogui
from humancursor import SystemCursor
import keyboard as kb

from config.loader import get_timing

_t = get_timing()


def move(x: int, y: int) -> None:
    dur = random.uniform(_t["mouse_move_duration_min"], _t["mouse_move_duration_max"])
    SystemCursor().move_to([x, y], duration=dur)
    time.sleep(random.uniform(_t["mouse_pre_click_delay_min"],
                              _t["mouse_pre_click_delay_max"]))


def click(x: int, y: int) -> None:
    move_dur   = random.uniform(_t["mouse_move_duration_min"], _t["mouse_move_duration_max"])
    click_dur  = random.uniform(_t["click_duration_min"], _t["click_duration_max"])
    time.sleep(random.uniform(_t["mouse_pre_click_delay_min"],
                              _t["mouse_pre_click_delay_max"]))
    SystemCursor().move_to([x, y], duration=move_dur)
    pyautogui.mouseDown()
    time.sleep(click_dur)
    pyautogui.mouseUp()


def click_random_in_region(region: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = region
    x, y = random.randint(x1, x2), random.randint(y1, y2)
    click(x, y)
    return x, y


def press_key(key: str) -> None:
    kb.press(key)
    time.sleep(random.uniform(0.05, 0.15))
    kb.release(key)


def wait_for_key(key: str) -> None:
    kb.wait(key)


def add_hotkey(hotkey: str, callback) -> None:
    kb.add_hotkey(hotkey, callback)

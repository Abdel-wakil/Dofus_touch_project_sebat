"""
ui.py
-----
Control panel for the Dofus Touch bot.

Usage:
    python ui.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import json
import sys
import re
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

ROOT          = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "settings.json"
RESOURCES_DIR = ROOT / "resources"

# Preview coordinate mapping — must match the crop in _update_map_preview
_CROP        = (280, 130, 1960, 1280)           # (x1, y1, x2, y2) full-screen pixels
_CROP_W      = _CROP[2] - _CROP[0]              # 1680
_CROP_H      = _CROP[3] - _CROP[1]              # 1150
_scale       = min(640 / _CROP_W, 360 / _CROP_H)
_THUMB_W     = round(_CROP_W * _scale)           # 526
_THUMB_H     = round(_CROP_H * _scale)           # 360

# Use venv python if available, otherwise fall back to current interpreter
_venv_python = ROOT / "venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_settings():
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_profile(name):
    s = _load_settings()
    s["active_profile"] = name
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def _list_resources():
    """Scan resources/ and return {display_name: stem} — auto-discovers new files."""
    result = {}
    for p in sorted(RESOURCES_DIR.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                name = json.load(f).get("resource", p.stem.title())
        except Exception:
            name = p.stem.title()
        result[name] = p.stem
    return result


def _resource_path(display_name):
    stem = _list_resources().get(display_name, display_name.lower())
    return RESOURCES_DIR / f"{stem}.json"


def _scout_progress(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        total   = len(data["maps"])
        scouted = sum(1 for m in data["maps"] if m.get("spots") is not None)
        return scouted, total
    except Exception:
        return 0, 0


# ── Main UI ────────────────────────────────────────────────────────────────────

class BotUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dofus Touch Bot")
        self.root.resizable(False, False)

        self._process: subprocess.Popen | None = None

        # State variables
        s = _load_settings()
        self._profile   = tk.StringVar(value=s["active_profile"])
        _res = _list_resources()
        _active_stem    = s.get("active_resource", "kalyptus")
        _active_display = next((n for n, stem in _res.items() if stem == _active_stem),
                               next(iter(_res), "Kalyptus"))
        self._resource  = tk.StringVar(value=_active_display)
        self._mode      = tk.StringVar(value="scout")
        self._start_x   = tk.StringVar(value="")
        self._start_y   = tk.StringVar(value="")

        self._status_text  = tk.StringVar(value="Idle")
        self._position     = tk.StringVar(value="—")
        self._progress_txt = tk.StringVar(value=self._progress_label())
        self._timing_var   = tk.DoubleVar(value=s["timing"]["harvest_wait_seconds"])

        self._current_map_xy     = None   # (x, y) of map currently shown in preview
        self._draw_start         = None   # canvas coords where drag started
        self._draw_rect_id       = None   # canvas rectangle item id
        self._selected_spot      = None   # (screen_x, screen_y) of selected spot
        self._spot_count_var     = tk.StringVar(value="")
        self._availability_result: dict = {}  # {(sx, sy): bool} from last Check spots run
        self._stop_event: threading.Event | None = None

        self._start_x.trace_add("write", self._on_start_pos_change)
        self._start_y.trace_add("write", self._on_start_pos_change)

        self._build()
        self._refresh_progress()

    # ── Build layout ───────────────────────────────────────────────────────────

    def _build(self):
        P = 10

        # Header
        hdr = tk.Frame(self.root, bg="#1a1a2e", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Dofus Touch Bot", bg="#1a1a2e", fg="#e0e0e0",
                 font=("Segoe UI", 13, "bold")).pack()

        # ── Settings ──────────────────────────────────────────────────────────
        sf = ttk.LabelFrame(self.root, text="Settings", padding=P)
        sf.pack(fill="x", padx=P, pady=(P, 4))
        sf.columnconfigure(1, weight=1)

        # Resolution
        tk.Label(sf, text="Resolution").grid(row=0, column=0, sticky="w", padx=(0, 8))
        rf = tk.Frame(sf)
        rf.grid(row=0, column=1, sticky="w")
        for label, val in [("1080p", "1080p"), ("2k / 1440p", "2k")]:
            ttk.Radiobutton(rf, text=label, variable=self._profile,
                            value=val, command=self._on_profile_change
                            ).pack(side="left", padx=6)

        # Resource
        tk.Label(sf, text="Resource").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        _cb = ttk.Combobox(sf, textvariable=self._resource,
                           values=list(_list_resources()), state="readonly", width=14)
        _cb.grid(row=1, column=1, sticky="w")
        _cb.bind("<<ComboboxSelected>>", self._on_resource_change)

        # Mode
        tk.Label(sf, text="Mode").grid(row=2, column=0, sticky="w", padx=(0, 8))
        mf = tk.Frame(sf)
        mf.grid(row=2, column=1, sticky="w")
        for label, val in [("Scout", "scout"), ("Harvest", "harvest")]:
            ttk.Radiobutton(mf, text=label, variable=self._mode,
                            value=val).pack(side="left", padx=6)

        # Start position
        tk.Label(sf, text="Start position").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=5)
        sp_frame = tk.Frame(sf)
        sp_frame.grid(row=3, column=1, sticky="w")
        tk.Label(sp_frame, text="X").pack(side="left")
        ttk.Entry(sp_frame, textvariable=self._start_x, width=5).pack(side="left", padx=(2, 8))
        tk.Label(sp_frame, text="Y").pack(side="left")
        ttk.Entry(sp_frame, textvariable=self._start_y, width=5).pack(side="left", padx=2)
        tk.Button(sp_frame, text="⌖ Use OCR", width=9,
                  bg="#1565c0", fg="white", relief="flat",
                  font=("Segoe UI", 8, "bold"), cursor="hand2",
                  command=self._use_ocr_as_start
                  ).pack(side="left", padx=(20, 4))
        tk.Button(sp_frame, text="✕ Clear", width=7,
                  bg="#c0392b", fg="white", relief="flat",
                  font=("Segoe UI", 8, "bold"), cursor="hand2",
                  command=lambda: (self._start_x.set(""), self._start_y.set(""))
                  ).pack(side="left", padx=(4, 4))
        tk.Label(sp_frame, text="(leave blank to use OCR)", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 0))

        # Harvest wait
        tk.Label(sf, text="Harvest wait (s)").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=5)
        hw_frame = tk.Frame(sf)
        hw_frame.grid(row=4, column=1, sticky="w")
        self._timing_label = tk.Label(hw_frame, text=f"{self._timing_var.get():.1f}s", width=4)
        self._timing_label.pack(side="right")
        ttk.Scale(hw_frame, from_=2, to=20, orient="horizontal", length=140,
                  variable=self._timing_var, command=self._on_timing_change
                  ).pack(side="left")

        # ── Status ────────────────────────────────────────────────────────────
        stf = ttk.LabelFrame(self.root, text="Status", padding=P)
        stf.pack(fill="x", padx=P, pady=4)
        stf.columnconfigure(1, weight=1)

        tk.Label(stf, text="State").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._status_label = tk.Label(stf, textvariable=self._status_text,
                                      fg="#888888", font=("Segoe UI", 9, "bold"))
        self._status_label.grid(row=0, column=1, sticky="w")

        tk.Label(stf, text="Position").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        tk.Label(stf, textvariable=self._position).grid(row=1, column=1, sticky="w")

        tk.Label(stf, text="Scout progress").grid(row=2, column=0, sticky="w", padx=(0, 8))
        tk.Label(stf, textvariable=self._progress_txt).grid(row=2, column=1, sticky="w")

        self._progress_bar = ttk.Progressbar(stf, length=200, mode="determinate")
        self._progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # ── Controls ──────────────────────────────────────────────────────────
        bf = tk.Frame(self.root, pady=8)
        bf.pack()

        row1 = tk.Frame(bf)
        row1.pack()
        row2 = tk.Frame(bf)
        row2.pack(pady=(4, 0))

        self._start_btn = ttk.Button(row1, text="▶  START", command=self._start, width=14)
        self._start_btn.pack(side="left", padx=6)

        self._stop_btn = ttk.Button(row1, text="■  STOP", command=self._stop,
                                    width=14, state="disabled")
        self._stop_btn.pack(side="left", padx=6)

        ttk.Button(row1, text="⟳  Reset spots", command=self._reset_spots, width=14
                   ).pack(side="left", padx=6)

        ttk.Button(row1, text="⊕  Scan map", command=self._scan_current_map, width=14
                   ).pack(side="left", padx=6)

        ttk.Button(row2, text="✓  Check spots", command=self._check_current_spots, width=14
                   ).pack(side="left", padx=6)

        ttk.Button(row2, text="⛏  Harvest map", command=self._harvest_current_map, width=14
                   ).pack(side="left", padx=6)

        ttk.Button(row2, text="⌖  Read OCR", command=self._read_ocr, width=14
                   ).pack(side="left", padx=6)

        # ── Map preview ───────────────────────────────────────────────────────
        if _PIL_AVAILABLE:
            pf = ttk.LabelFrame(self.root,
                                text="Map preview  —  drag to add  |  click spot + Delete to remove",
                                padding=(P, 4))
            pf.pack(fill="x", padx=P, pady=(0, 4))
            self._canvas = tk.Canvas(pf, width=_THUMB_W, height=_THUMB_H,
                                     bg="#1e1e1e", cursor="crosshair", highlightthickness=0)
            self._canvas.pack()
            self._canvas.create_text(_THUMB_W // 2, _THUMB_H // 2,
                                     text="No map loaded", fill="#888888", tags="placeholder")
            self._canvas.bind("<ButtonPress-1>",   self._on_preview_press)
            self._canvas.bind("<B1-Motion>",       self._on_preview_drag)
            self._canvas.bind("<ButtonRelease-1>", self._on_preview_release)
            self._canvas.bind("<Delete>",          self._on_delete_spot)
            self._canvas.bind("<BackSpace>",       self._on_delete_spot)
            self._preview_image = None
            tk.Label(pf, textvariable=self._spot_count_var,
                     fg="#aaaaaa", font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))
        else:
            self._canvas = None

        # ── Log ───────────────────────────────────────────────────────────────
        lf = ttk.LabelFrame(self.root, text="Log", padding=(P, 4))
        lf.pack(fill="both", expand=True, padx=P, pady=(0, P))

        ttk.Button(lf, text="Clear", command=self._clear_log, width=6
                   ).pack(anchor="ne", pady=(0, 2))

        self._log = scrolledtext.ScrolledText(
            lf, height=16, width=64, state="disabled",
            font=("Consolas", 8), bg="#1e1e1e", fg="#d4d4d4",
        )
        self._log.pack(fill="both", expand=True)

        # colour tags for log
        self._log.tag_config("pos",   foreground="#4fc3f7")
        self._log.tag_config("scout", foreground="#81c784")
        self._log.tag_config("nav",   foreground="#ffb74d")
        self._log.tag_config("err",   foreground="#e57373")
        self._log.tag_config("farm",  foreground="#ce93d8")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_start_pos_change(self, *_):
        try:
            x = int(self._start_x.get().strip())
            y = int(self._start_y.get().strip())
        except ValueError:
            return
        self._update_map_preview(x, y)

    def _on_profile_change(self):
        _save_profile(self._profile.get())
        self._log_line(f"[Config] Profile → {self._profile.get()}")

    def _on_timing_change(self, _=None):
        val = round(self._timing_var.get(), 1)
        self._timing_label.config(text=f"{val:.1f}s")
        # Persist to settings.json
        s = _load_settings()
        s["timing"]["harvest_wait_seconds"] = val
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)

    def _start(self):
        mode = self._mode.get()
        if mode == "harvest":
            self._run_harvest_loop()
            return
        script = "scout.py"
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_status("Running", "#4caf50")

        cmd = [PYTHON, "-u", str(ROOT / script)]
        sx, sy = self._start_x.get().strip(), self._start_y.get().strip()
        if sx and sy:
            cmd += [f"--start-pos={sx},{sy}"]
            self._log_line(f"[UI] Starting {script} at ({sx}, {sy})")
        else:
            self._log_line(f"[UI] Starting {script} (OCR will detect start position)")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _stop(self):
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log_line("[UI] Stopped by user.")
        self._set_status("Idle", "#888888")
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._refresh_progress()

    def _scan_current_map(self):
        sx, sy = self._start_x.get().strip(), self._start_y.get().strip()
        if not sx or not sy:
            self._log_line("[UI] Enter X and Y position before scanning.")
            return
        self._log_line(f"[UI] Scanning map ({sx}, {sy})...")
        self._last_scan_pos = (int(sx), int(sy))
        cmd = [PYTHON, "-u", str(ROOT / "scan_map.py"), f"--pos={sx},{sy}"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        threading.Thread(target=self._read_scan_output, args=(proc,), daemon=True).start()

    def _harvest_current_map(self):
        # Fast path: Check spots was already run — use those results directly.
        if self._availability_result:
            avail_spots = [(sx, sy) for (sx, sy), ok in self._availability_result.items() if ok]
            mx, my = self._current_map_xy or (None, None)
            if not avail_spots:
                self._log_line(f"[Harvest] ({mx},{my}) — 0 available spots, nothing to click.", "farm")
                return
            self._log_line(
                f"[Harvest] ({mx},{my}) — {len(avail_spots)} spot(s) ready, clicking in 3s...",
                "farm"
            )
            def _do_fast():
                import time as _t
                _t.sleep(3.0)
                try:
                    import sys as _sys, os as _os
                    _sys.path.insert(0, str(ROOT)); _os.chdir(ROOT)
                    import input as _bot
                    from config.loader import get_timing
                    timing = get_timing()
                    for sx, sy in avail_spots:
                        _bot.click(sx, sy); _t.sleep(0.1)
                    _t.sleep(timing["harvest_wait_seconds"])
                    self.root.after(0, self._log_line,
                                    f"[Harvest] Done — {len(avail_spots)} resource(s) clicked.", "farm")
                except Exception as e:
                    self.root.after(0, self._log_line, f"[Harvest] Error: {e}", "err")
            threading.Thread(target=_do_fast, daemon=True).start()
            return

        # Full pipeline: OCR -> check spots -> harvest, same as clicking each button.
        self._log_line("[Harvest] Starting in 3s — switch to game!", "farm")

        def _do_full():
            import time as _t
            _t.sleep(3.0)

            # Step 1: Use OCR as start (same as button).
            ocr_done = threading.Event()
            self.root.after(0, self._use_ocr_as_start, ocr_done.set)
            if not ocr_done.wait(timeout=10) or self._current_map_xy is None:
                self.root.after(0, self._log_line, "[Harvest] OCR failed.", "err")
                return

            # Step 2: Check spots (same as button, no pre-delay — already in game).
            check_done = threading.Event()
            self.root.after(0, self._check_current_spots, check_done.set, 0)
            if not check_done.wait(timeout=30):
                self.root.after(0, self._log_line, "[Harvest] Check spots timed out.", "err")
                return

            # Step 3: Harvest available spots.
            avail_spots = [(sx, sy) for (sx, sy), ok in self._availability_result.items() if ok]
            mx, my = self._current_map_xy
            if not avail_spots:
                self.root.after(0, self._log_line,
                                f"[Harvest] ({mx},{my}) — all stumps, nothing to harvest.", "farm")
                return

            self.root.after(0, self._log_line,
                            f"[Harvest] Clicking {len(avail_spots)} spot(s)...", "farm")
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, str(ROOT)); _os.chdir(ROOT)
                import input as _bot
                from config.loader import get_timing
                timing = get_timing()
                for sx, sy in avail_spots:
                    _bot.click(sx, sy); _t.sleep(0.1)
                _t.sleep(timing["harvest_wait_seconds"])
                self.root.after(0, self._log_line,
                                f"[Harvest] Done — {len(avail_spots)} resource(s) clicked.", "farm")
            except Exception as e:
                self.root.after(0, self._log_line, f"[Harvest] Error: {e}", "err")

        threading.Thread(target=_do_full, daemon=True).start()

    # ── Harvest loop (START button, harvest mode) ──────────────────────────────

    def _run_harvest_loop(self):
        """
        Boustrophedon (snake) harvest loop.
        Pre-computes the full zone traversal order once, then for each map:
          navigate step-by-step → check spots → click available → next map.
        Sweeps repeat indefinitely until STOP is pressed.
        """
        self._stop_event = threading.Event()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_status("Harvesting", "#ce93d8")

        sx_str = self._start_x.get().strip()
        sy_str = self._start_y.get().strip()
        manual_start = None
        if sx_str and sy_str:
            try:
                manual_start = (int(sx_str), int(sy_str))
            except ValueError:
                pass

        self._log_line("[Loop] Harvest loop starting in 5s — switch to game!", "farm")

        def _loop():
            import time as _t
            import sys as _sys, os as _os
            _sys.path.insert(0, str(ROOT)); _os.chdir(ROOT)
            from farm import _load_db_and_spots, navigate
            from planner import snake_route, step_toward

            _t.sleep(5.0)
            if self._stop_event.is_set():
                self.root.after(0, self._end_harvest_loop)
                return

            db, _ = _load_db_and_spots()
            route_fwd = snake_route(db)           # south → north
            route_rev = list(reversed(route_fwd)) # north → south
            routes    = [route_fwd, route_rev]
            self.root.after(0, self._log_line,
                            f"[Loop] Snake route: {len(route_fwd)} maps.", "farm")

            # Determine starting position
            if manual_start:
                pos = manual_start
                done = threading.Event()
                def _set_start(p=pos):
                    self._start_x.set(str(p[0]))
                    self._start_y.set(str(p[1]))
                    self._update_map_preview(p[0], p[1])
                    self._log_line(f"[Loop] Starting at {p}", "pos")
                    done.set()
                self.root.after(0, _set_start)
                done.wait(timeout=5)
            else:
                pos = None
                while not self._stop_event.is_set():
                    ocr_done = threading.Event()
                    self.root.after(0, self._use_ocr_as_start, ocr_done.set)
                    ocr_done.wait(timeout=10)
                    if self._current_map_xy:
                        pos = self._current_map_xy
                        break
                    self.root.after(0, self._log_line,
                                    "[Loop] OCR failed, retrying in 2s...", "err")
                    _t.sleep(2)

            if self._stop_event.is_set() or pos is None:
                self.root.after(0, self._end_harvest_loop)
                return

            # Resume from current position in the route if possible
            sweep         = 1
            current_route = routes[0]
            idx           = next((i for i, p in enumerate(current_route) if p == pos), 0)

            while not self._stop_event.is_set():
                target = current_route[idx]

                # Navigate to target one step at a time
                while pos != target and not self._stop_event.is_set():
                    direction = step_toward(pos, target, db)
                    if direction is None:
                        self.root.after(0, self._log_line,
                                        f"[Loop] No path from {pos} to {target}", "err")
                        break
                    ok, nav_ocr = navigate(direction, current_pos=pos)
                    if ok and nav_ocr:
                        pos = nav_ocr
                        nav_done = threading.Event()
                        def _on_nav(p=pos):
                            self._start_x.set(str(p[0]))
                            self._start_y.set(str(p[1]))
                            self._update_map_preview(p[0], p[1])
                            self._log_line(f"[Loop] → {p}", "pos")
                            nav_done.set()
                        self.root.after(0, _on_nav)
                        nav_done.wait(timeout=5)
                    else:
                        # Nav failed — OCR re-sync
                        self.root.after(0, self._log_line,
                                        "[Loop] Nav failed — re-reading position...", "err")
                        ocr_done2 = threading.Event()
                        self.root.after(0, self._use_ocr_as_start, ocr_done2.set)
                        ocr_done2.wait(timeout=10)
                        if self._current_map_xy:
                            pos = self._current_map_xy
                        else:
                            self.root.after(0, self._log_line,
                                            "[Loop] OCR re-sync failed — stopping.", "err")
                            self._stop_event.set()
                        break

                if self._stop_event.is_set():
                    break

                if pos != target:
                    self.root.after(0, self._log_line,
                                    f"[Loop] Skipped unreachable {target}", "nav")
                else:
                    # Check spots (no pre-delay — already in game)
                    check_done = threading.Event()
                    self.root.after(0, self._check_current_spots, check_done.set, 0)
                    check_done.wait(timeout=30)

                    if self._stop_event.is_set():
                        break

                    avail = [(sx, sy) for (sx, sy), ok in
                             self._availability_result.items() if ok]
                    if avail:
                        self.root.after(0, self._log_line,
                                        f"[Loop] {pos} — clicking {len(avail)} spot(s)...",
                                        "farm")
                        try:
                            import input as _bot
                            from config.loader import get_timing
                            timing = get_timing()
                            for sx, sy in avail:
                                _bot.click(sx, sy); _t.sleep(0.1)
                            _t.sleep(timing["harvest_wait_seconds"])
                            self.root.after(0, self._log_line,
                                            f"[Loop] {pos} — {len(avail)} harvested.",
                                            "farm")
                        except Exception as e:
                            self.root.after(0, self._log_line,
                                            f"[Loop] Harvest error: {e}", "err")
                    else:
                        self.root.after(0, self._log_line,
                                        f"[Loop] {pos} — all stumps.", "farm")

                idx += 1
                if idx >= len(current_route):
                    idx = 0
                    sweep += 1
                    current_route = routes[(sweep - 1) % 2]
                    direction_lbl = "↑ S→N" if (sweep % 2 == 1) else "↓ N→S"
                    self.root.after(0, self._log_line,
                                    f"[Loop] Sweep {sweep - 1} done — sweep {sweep} {direction_lbl}.",
                                    "farm")

            self.root.after(0, self._end_harvest_loop)

        threading.Thread(target=_loop, daemon=True).start()

    def _end_harvest_loop(self):
        self._set_status("Idle", "#888888")
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._log_line("[Loop] Harvest loop ended.", "farm")

    def _use_ocr_as_start(self, _callback=None):
        self._log_line("[OCR] Reading position for start...")
        def _do():
            try:
                import sys, os
                sys.path.insert(0, str(ROOT))
                os.chdir(ROOT)
                from vision import read_current_position
                pos = read_current_position()
                if pos:
                    def _apply():
                        self._start_x.set(str(pos[0]))
                        self._start_y.set(str(pos[1]))
                        self._log_line(f"[OCR] Start position set to ({pos[0]}, {pos[1]})", "pos")
                        self._update_map_preview(pos[0], pos[1])
                        if _callback:
                            _callback()
                    self.root.after(0, _apply)
                else:
                    def _fail():
                        self._log_line("[OCR] Failed — no match", "err")
                        if _callback:
                            _callback()
                    self.root.after(0, _fail)
            except Exception as e:
                def _err():
                    self._log_line(f"[OCR] Error: {e}", "err")
                    if _callback:
                        _callback()
                self.root.after(0, _err)
        threading.Thread(target=_do, daemon=True).start()

    def _read_ocr(self):
        self._log_line("[OCR] Reading position...")
        def _do():
            try:
                import sys, os
                sys.path.insert(0, str(ROOT))
                os.chdir(ROOT)
                from vision import read_current_position
                pos = read_current_position()
                if pos:
                    msg = f"[OCR] Result: ({pos[0]}, {pos[1]})"
                    self.root.after(0, self._log_line, msg, "pos")
                else:
                    self.root.after(0, self._log_line, "[OCR] Failed — no match", "err")
            except Exception as e:
                self.root.after(0, self._log_line, f"[OCR] Error: {e}", "err")
        threading.Thread(target=_do, daemon=True).start()

    def _check_current_spots(self, _callback=None, pre_delay=1.5):
        if self._current_map_xy is None:
            self._log_line("[Check] No map loaded -- enter X/Y first.", "err")
            if _callback:
                _callback()
            return
        mx, my = self._current_map_xy
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._log_line(f"[Check] Could not load resource: {e}", "err")
            if _callback:
                _callback()
            return
        spots = next(
            (m.get("spots") or [] for m in data["maps"] if m["x"] == mx and m["y"] == my),
            None
        )
        if not spots:
            self._log_line(f"[Check] No spots defined for ({mx}, {my}).", "err")
            if _callback:
                _callback()
            return

        resource_stem = _list_resources().get(self._resource.get(), self._resource.get().lower())
        self._log_line(
            f"[Check] Switch to game! Holding mouse + capturing {len(spots)} spot(s) on ({mx},{my})...",
            "farm"
        )

        def _do():
            import time as _time
            _time.sleep(pre_delay)
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, str(ROOT))
                _os.chdir(ROOT)
                from farm import (check_spots_available,
                                  SPOT_WIN_X, SPOT_WIN_Y_TOP, SPOT_WIN_Y_BOT, SPOT_FRAMES)
                import cv2 as _cv2
                import numpy as _np

                available, blink_sum, img = check_spots_available(spots, return_mask=True)
                available_set = {tuple(p) for p in available}
                results = {(sx, sy): (sx, sy) in available_set for sx, sy in spots}
                n_avail = len(available)

                # Heatmap overlay: bright = lots of blinking pixels
                n_pairs = SPOT_FRAMES - 1
                norm = _np.clip(
                    (blink_sum.astype(_np.float32) / n_pairs * 255), 0, 255
                ).astype(_np.uint8)
                heat = _cv2.applyColorMap(norm, _cv2.COLORMAP_HOT)
                ann  = _cv2.addWeighted(img, 0.65, heat, 0.35, 0)

                lines = []
                ch, cw = blink_sum.shape
                for sx, sy in spots:
                    x1 = max(0, sx - SPOT_WIN_X); x2 = min(cw, sx + SPOT_WIN_X)
                    y1 = max(0, sy - SPOT_WIN_Y_TOP); y2 = min(ch, sy + SPOT_WIN_Y_BOT)
                    count = int(_np.sum(blink_sum[y1:y2, x1:x2]))
                    avail = results[(sx, sy)]
                    tag_str = "available" if avail else "stump"
                    lines.append((f"[Check]   ({sx},{sy})  blink={count}  -> {tag_str}", "farm"))
                    col = (0, 220, 0) if avail else (0, 120, 255)
                    # Measurement rectangle
                    _cv2.rectangle(ann, (x1, y1), (x2, y2), col, 1)
                    # Center dot + label
                    _cv2.circle(ann, (sx, sy), 5, col, -1)
                    _cv2.putText(ann, "OK" if avail else "stump",
                                 (sx + 28, sy + 5),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, _cv2.LINE_AA)

                det_dir = ROOT / "screenshots" / resource_stem / "detection"
                det_dir.mkdir(parents=True, exist_ok=True)
                _cv2.imwrite(str(det_dir / f"{mx}_{my}.png"), ann)

                def _update():
                    self._availability_result = results
                    for text, tag in lines:
                        self._log_line(text, tag)
                    self._log_line(
                        f"[Check] {n_avail}/{len(spots)} available -- preview updated"
                    )
                    if self._current_map_xy == (mx, my):
                        self._update_map_preview(mx, my, keep_availability=True)
                    if _callback:
                        _callback()
                self.root.after(0, _update)
            except Exception as e:
                self.root.after(0, self._log_line, f"[Check] Error: {e}", "err")
                if _callback:
                    self.root.after(0, _callback)

        threading.Thread(target=_do, daemon=True).start()

    def _read_scan_output(self, proc):
        for line in proc.stdout:
            self.root.after(0, self._handle_line, line.rstrip())
        self.root.after(0, self._refresh_progress)
        pos = getattr(self, "_last_scan_pos", None)
        if pos:
            self.root.after(0, self._update_map_preview, pos[0], pos[1])

    def _on_resource_change(self, *_):
        stem = _list_resources().get(self._resource.get(), self._resource.get().lower())
        s = _load_settings()
        s["active_resource"] = stem
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        self._log_line(f"[Config] Resource -> {self._resource.get()}")
        self._refresh_progress()

    def _reset_spots(self):
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for m in data["maps"]:
                m.pop("spots", None)
            maps = data["maps"]
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
            self._log_line(f"[UI] Spots cleared from {path.name}")
            self._refresh_progress()
        except Exception as e:
            self._log_line(f"[UI] Error resetting spots: {e}")

    # ── Output reader ──────────────────────────────────────────────────────────

    def _read_output(self):
        for raw_line in self._process.stdout:
            line = raw_line.rstrip()
            self.root.after(0, self._handle_line, line)
        self.root.after(0, self._on_process_end)

    def _handle_line(self, line):
        # Determine colour tag
        tag = None
        if "[Pos]" in line:    tag = "pos"
        elif "[Scout]" in line: tag = "scout"
        elif "[Nav]" in line:   tag = "nav"
        elif "[Farm]" in line:  tag = "farm"
        elif "Error" in line or "Traceback" in line or "FAIL" in line: tag = "err"
        self._log_line(line, tag)

        # Update position
        m = re.search(r"\[Pos\]\s*\((-?\d+),\s*(-?\d+)\)", line)
        if m:
            self._position.set(f"({m.group(1)}, {m.group(2)})")
            self._update_map_preview(int(m.group(1)), int(m.group(2)))

        # Auto-load preview after scout saves a detection screenshot
        m = re.search(r"\[Scout\] Saved .+/(-?\d+)_(-?\d+)\.png", line)
        if m:
            self._update_map_preview(int(m.group(1)), int(m.group(2)))
            self._refresh_progress()

    def _on_process_end(self):
        self._set_status("Finished", "#2196f3")
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._refresh_progress()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _progress_label(self):
        scouted, total = _scout_progress(_resource_path(self._resource.get()))
        return f"{scouted} / {total} maps"

    def _refresh_progress(self):
        scouted, total = _scout_progress(_resource_path(self._resource.get()))
        self._progress_txt.set(f"{scouted} / {total} maps")
        self._progress_bar["maximum"] = max(total, 1)
        self._progress_bar["value"]   = scouted

    def _set_status(self, text, colour):
        self._status_text.set(text)
        self._status_label.config(fg=colour)

    def _log_line(self, text, tag=None):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.config(state="normal")
        self._log.insert("end", f"[{ts}] {text}\n", tag or "")
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _update_map_preview(self, x, y, keep_availability=False):
        self._current_map_xy = (x, y)
        if not keep_availability:
            self._availability_result = {}   # reset when map changes
        if not _PIL_AVAILABLE or self._canvas is None:
            return
        stem = _list_resources().get(self._resource.get(), self._resource.get().lower())
        path = ROOT / "screenshots" / stem / "detection" / f"{x}_{y}.png"
        self._canvas.delete("all")
        self._selected_spot  = None
        self._draw_rect_id   = None
        if not path.exists():
            self._canvas.create_text(_THUMB_W // 2, _THUMB_H // 2,
                                     text=f"No scan for ({x}, {y})", fill="#888888")
            self._preview_image = None
            self._draw_existing_spots(x, y)   # still update the spot count label
            return
        try:
            img = Image.open(path)
            img = img.crop(_CROP)
            img.thumbnail((_THUMB_W, _THUMB_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview_image = photo
            self._canvas.create_image(0, 0, anchor="nw", image=photo, tags="bg")
            # Draw any already-saved spots as green dots
            self._draw_existing_spots(x, y)
        except Exception as e:
            self._canvas.create_text(_THUMB_W // 2, _THUMB_H // 2,
                                     text=f"Preview error: {e}", fill="#e57373")
            self._preview_image = None


    # ── Manual spot drawing / selection / deletion ────────────────────────────

    _SPOT_R      = 7    # dot radius in canvas pixels
    _SELECT_DIST = 14   # click within this many canvas pixels to select a spot

    def _on_preview_press(self, event):
        self._canvas.focus_set()   # so Delete key reaches the canvas
        # Check if clicking near an existing spot (select it)
        hit = self._spot_at(event.x, event.y)
        if hit is not None:
            self._selected_spot = hit
            self._draw_existing_spots(*self._current_map_xy)
            self._draw_start = None  # don't start a drag
            return
        # Otherwise deselect and start a drag-to-add
        self._selected_spot = None
        self._draw_start = (event.x, event.y)
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

    def _on_preview_drag(self, event):
        if self._draw_start is None:
            return
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
        x0, y0 = self._draw_start
        self._draw_rect_id = self._canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#00ff00", width=2
        )

    def _on_preview_release(self, event):
        if self._draw_start is None:
            return
        x0, y0 = self._draw_start
        x1, y1 = event.x, event.y
        self._draw_start = None
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return  # too small — likely accidental click, not a drag
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        screen_x = round(_CROP[0] + cx * (_CROP_W / _THUMB_W))
        screen_y = round(_CROP[1] + cy * (_CROP_H / _THUMB_H))
        self._save_manual_spot(screen_x, screen_y)

    def _on_delete_spot(self, _=None):
        if self._selected_spot is None or self._current_map_xy is None:
            return
        mx, my = self._current_map_xy
        sx, sy = self._selected_spot
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for m in data["maps"]:
                if m["x"] == mx and m["y"] == my:
                    spots = m.get("spots") or []
                    try:
                        spots.remove([sx, sy])
                    except ValueError:
                        return
                    m["spots"] = spots
                    break
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._selected_spot = None
            self._log_line(f"[UI] Spot ({sx}, {sy}) removed from map ({mx}, {my})", "scout")
            self._refresh_progress()
            self._update_map_preview(mx, my)
        except Exception as e:
            self._log_line(f"[UI] Error deleting spot: {e}", "err")

    def _spot_at(self, cx, cy):
        """Return (screen_x, screen_y) of the spot closest to canvas point (cx,cy), or None."""
        if self._current_map_xy is None:
            return None
        mx, my = self._current_map_xy
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for m in data["maps"]:
                if m["x"] == mx and m["y"] == my:
                    best, best_d = None, self._SELECT_DIST
                    for sx, sy in m.get("spots") or []:
                        dcx = (sx - _CROP[0]) * (_THUMB_W / _CROP_W)
                        dcy = (sy - _CROP[1]) * (_THUMB_H / _CROP_H)
                        d   = ((cx - dcx) ** 2 + (cy - dcy) ** 2) ** 0.5
                        if d < best_d:
                            best, best_d = (sx, sy), d
                    return best
        except Exception:
            pass
        return None

    def _save_manual_spot(self, screen_x, screen_y):
        if self._current_map_xy is None:
            self._log_line("[UI] No map position known — navigate to a map first.", "err")
            return
        # Clear the drag rectangle immediately
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None
        mx, my = self._current_map_xy
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for m in data["maps"]:
                if m["x"] == mx and m["y"] == my:
                    m.setdefault("spots", []).append([screen_x, screen_y])
                    break
            else:
                self._log_line(f"[UI] Map ({mx}, {my}) not in {path.name}", "err")
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._log_line(
                f"[UI] Spot added at ({screen_x}, {screen_y}) on map ({mx}, {my})", "scout"
            )
            self._refresh_progress()
            self._update_map_preview(mx, my)
        except Exception as e:
            self._log_line(f"[UI] Error saving spot: {e}", "err")

    def _draw_existing_spots(self, x, y):
        self._canvas.delete("spot")   # remove only spot items, keep the background image
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for m in data["maps"]:
                if m["x"] == x and m["y"] == y:
                    spots    = m.get("spots") or []
                    expected = m.get("count")
                    checked  = bool(self._availability_result)

                    if checked and spots:
                        n_avail = sum(
                            1 for s in spots
                            if self._availability_result.get((s[0], s[1])) is True
                        )
                        count_str = f"{n_avail} available / {len(spots)} spots"
                    elif expected is not None:
                        count_str = f"{len(spots)} / {expected} spots"
                    else:
                        count_str = f"{len(spots)} spot(s)"
                    self._spot_count_var.set(count_str)

                    for sx, sy in spots:
                        dcx = (sx - _CROP[0]) * (_THUMB_W / _CROP_W)
                        dcy = (sy - _CROP[1]) * (_THUMB_H / _CROP_H)
                        r        = self._SPOT_R
                        selected = (self._selected_spot == (sx, sy))
                        avail    = self._availability_result.get((sx, sy))

                        if not checked:
                            fill, outline = "#00ff00", "#007700"
                            lbl = None
                        elif avail is True:
                            fill, outline = "#00ff44", "#00aa44"   # green = available
                            lbl = "OK"
                        elif avail is False:
                            fill, outline = "#ff8800", "#884400"   # orange = stump
                            lbl = "stump"
                        else:
                            fill, outline = "#ffff00", "#888800"   # yellow = not in check
                            lbl = None

                        if selected:
                            outline = "#ffffff"

                        self._canvas.create_oval(
                            dcx - r, dcy - r, dcx + r, dcy + r,
                            fill=fill, outline=outline,
                            width=2 if selected else 1,
                            tags="spot"
                        )
                        if lbl:
                            lbl_color = "#00ff44" if avail else "#ff8800"
                            self._canvas.create_text(
                                dcx + r + 4, dcy, text=lbl,
                                fill=lbl_color, font=("Consolas", 7),
                                anchor="w", tags="spot"
                            )
                    break
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-topmost", True)
    BotUI(root)
    root.mainloop()

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

        self._current_map_xy  = None   # (x, y) of map currently shown in preview
        self._draw_start      = None   # canvas coords where drag started
        self._draw_rect_id    = None   # canvas rectangle item id

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
        tk.Button(sp_frame, text="✕ Clear", width=7,
                  bg="#c0392b", fg="white", relief="flat",
                  font=("Segoe UI", 8, "bold"), cursor="hand2",
                  command=lambda: (self._start_x.set(""), self._start_y.set(""))
                  ).pack(side="left", padx=(20, 4))
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

        self._start_btn = ttk.Button(bf, text="▶  START", command=self._start, width=14)
        self._start_btn.pack(side="left", padx=6)

        self._stop_btn = ttk.Button(bf, text="■  STOP", command=self._stop,
                                    width=14, state="disabled")
        self._stop_btn.pack(side="left", padx=6)

        ttk.Button(bf, text="⟳  Reset spots", command=self._reset_spots, width=14
                   ).pack(side="left", padx=6)

        ttk.Button(bf, text="⊕  Scan map", command=self._scan_current_map, width=14
                   ).pack(side="left", padx=6)

        ttk.Button(bf, text="⌖  Read OCR", command=self._read_ocr, width=14
                   ).pack(side="left", padx=6)

        # ── Map preview ───────────────────────────────────────────────────────
        if _PIL_AVAILABLE:
            pf = ttk.LabelFrame(self.root, text="Map preview  —  drag to add a spot", padding=(P, 4))
            pf.pack(fill="x", padx=P, pady=(0, 4))
            self._canvas = tk.Canvas(pf, width=_THUMB_W, height=_THUMB_H,
                                     bg="#1e1e1e", cursor="crosshair", highlightthickness=0)
            self._canvas.pack()
            self._canvas.create_text(_THUMB_W // 2, _THUMB_H // 2,
                                     text="No map loaded", fill="#888888", tags="placeholder")
            self._canvas.bind("<ButtonPress-1>",   self._on_preview_press)
            self._canvas.bind("<B1-Motion>",       self._on_preview_drag)
            self._canvas.bind("<ButtonRelease-1>", self._on_preview_release)
            self._preview_image = None
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
        mode   = self._mode.get()
        script = "scout.py" if mode == "scout" else "farm.py"
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

    def _read_scan_output(self, proc):
        for line in proc.stdout:
            self.root.after(0, self._handle_line, line.rstrip())
        self.root.after(0, self._refresh_progress)

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

        # Refresh scout progress after each map save
        if "[Scout]" in line and "spot" in line.lower():
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

    def _update_map_preview(self, x, y):
        self._current_map_xy = (x, y)
        if not _PIL_AVAILABLE or self._canvas is None:
            return
        stem = _list_resources().get(self._resource.get(), self._resource.get().lower())
        path = ROOT / "screenshots" / stem / "detection" / f"{x}_{y}.png"
        self._canvas.delete("all")
        if not path.exists():
            self._canvas.create_text(_THUMB_W // 2, _THUMB_H // 2,
                                     text=f"No scan for ({x}, {y})", fill="#888888")
            self._preview_image = None
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


    # ── Manual spot drawing ────────────────────────────────────────────────────

    def _on_preview_press(self, event):
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
            return  # too small — likely accidental click
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        screen_x = round(_CROP[0] + cx * (_CROP_W / _THUMB_W))
        screen_y = round(_CROP[1] + cy * (_CROP_H / _THUMB_H))
        self._save_manual_spot(screen_x, screen_y)

    def _save_manual_spot(self, screen_x, screen_y):
        if self._current_map_xy is None:
            self._log_line("[UI] No map position known — navigate to a map first.", "err")
            return
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
            # Redraw dots so the new spot shows immediately
            self._draw_existing_spots(mx, my)
        except Exception as e:
            self._log_line(f"[UI] Error saving spot: {e}", "err")

    def _draw_existing_spots(self, x, y):
        path = _resource_path(self._resource.get())
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for m in data["maps"]:
                if m["x"] == x and m["y"] == y:
                    for sx, sy in m.get("spots") or []:
                        cx = (sx - _CROP[0]) * (_THUMB_W / _CROP_W)
                        cy = (sy - _CROP[1]) * (_THUMB_H / _CROP_H)
                        r = 5
                        self._canvas.create_oval(
                            cx - r, cy - r, cx + r, cy + r,
                            fill="#00ff00", outline="#007700"
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

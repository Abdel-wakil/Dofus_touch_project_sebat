# Dofus Touch Lumberjack Bot

Automated resource farming bot for Dofus Touch. Navigates a defined set of maps,
detects harvestable trees by blink animation, clicks them, and loops.

---

## Requirements

### Virtual environment (recommended)

```
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### Python packages

```
pip install -r requirements.txt
```

### Tesseract OCR (required for map coordinate reading)

Download and install the Windows binary:
https://github.com/UB-Mannheim/tesseract/wiki

During install, note the path (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
After installing, either:
- Add Tesseract to your system PATH, **or**
- Set the path in `vision.py` if it is not found automatically

---

## Project structure

```
Dofus_touch_project_sebat/
├── ui.py                  # Launch this — the main interface
├── farm.py                # Farming loop (navigate + harvest)
├── scout.py               # Scout loop (walk maps, record spot positions)
├── scan_map.py            # CLI tool: scan the current map for spots
├── vision.py              # Screen capture, OCR, blink detection
├── input.py               # Human-like mouse movement and clicks
├── planner.py             # Route planning and direction logic
│
├── config/
│   ├── settings.json      # All tunable settings (resolution, timing, regions)
│   └── loader.py          # Reads settings, resolves regions to pixels
│
├── resources/
│   ├── kalyptus.json      # Kalyptus map list + harvested spot positions
│   ├── boumu.json         # Bambu Sombre map list
│   ├── olioli.json        # Oliviolet map list
│   └── bumbu.json         # Bumbu map list
│
└── screenshots/           # Auto-saved debug images from scout/detection
```

---

## How to run

```
python ui.py
```

The UI lets you:
- Choose a **resource** (Kalyptus, Bambu Sombre, etc.) from the dropdown
- Choose a **resolution profile** (2K or 1080p)
- Set a **start position** (the map coordinate where the bot begins)
- **Start / Stop** the farming loop
- **Scout** — walk every map in the resource's list and save spot positions
- **Scan current map** — re-scan just the current map for spots
- **Read OCR** — test that coordinate reading is working

---

## First-time setup for a resource

Each resource JSON (`resources/*.json`) contains a list of maps the bot is
allowed to visit. Before farming, spots must be scouted on each map.

### Step 1 — Verify the map list

Open the relevant JSON (e.g. `resources/kalyptus.json`). Each entry looks like:

```json
{"x": -11, "y": 5, "count": 2}
```

`count` is the expected number of trees on that map.
`spots` is added automatically after scanning.

### Step 2 — Scout spots

1. Open Dofus Touch on your PC (BlueStacks or native)
2. Navigate your character to any map in the resource list
3. In the UI, select the resource and set the start position to your current map
4. Click **Scout**

The bot will walk every map, detect blinking tree animations, and save pixel
coordinates into the JSON under `"spots"`.

### Step 3 — Farm

Once spots are saved, click **Start**. The bot navigates the map list,
clicks every live tree (detected by blink animation), and loops.

---

## Resolution setup

The default profile is **2K (2560x1440)**. If you play at 1080p, change it in
`settings.json`:

```json
"active_profile": "1080p"
```

Or switch it from the UI dropdown. All click regions are stored as ratios so
they scale automatically to the correct pixels for each profile.

---

## Coordinate reading (OCR)

The bot reads the in-game coordinate display (top-left of screen) using Tesseract
to validate map navigation. Use the **Read OCR** button in the UI to confirm
it is reading correctly before starting a run.

If OCR reads wrong values, check:
- Tesseract is installed and on PATH
- The `coord_region` ratio in `settings.json` covers the coordinate text
- The game's UI scale matches the active resolution profile

---

## Adding a new resource

1. Create `resources/<name>.json`:

```json
{
  "resource": "Display Name",
  "respawn_minutes": 60,
  "maps": [
    {"x": 10, "y": -5, "count": 3},
    {"x": 11, "y": -5, "count": 2}
  ]
}
```

2. The UI will pick it up automatically from the dropdown (no code changes needed)
3. Scout the maps to populate `"spots"` on each entry

---

## Emergency stop

Press `*` (numpad multiply) at any time to stop the bot immediately.

---

## Timing tuning

All delays are in `settings.json` under `"timing"`:

| Key | What it controls |
|-----|-----------------|
| `harvest_wait_seconds` | How long to wait after clicking a tree |
| `post_map_change_delay` | Settle time after entering a new map |
| `mouse_move_duration_min/max` | Mouse travel speed range (seconds) |
| `map_change_timeout_seconds` | How long to wait for a map transition |

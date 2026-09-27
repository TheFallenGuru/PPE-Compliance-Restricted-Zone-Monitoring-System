# PPE Safety Monitoring Project

**Last updated:** 2026-09-06 (Audit pass — documentation corrected to match code)
**Agent:** Antigravity (Claude Sonnet 4.6)

---

## 1. Project Overview

A local Python computer-vision application that monitors PPE (Personal Protective Equipment) compliance on worksites using a YOLOv8 SH17 model and DeepSORT tracking. The software-only CV/detection phase is complete. Hardware integration, email alerts, and a proper web frontend are planned for future phases.

**Core principle:** Detect individual persons → spatially associate PPE items to each person → classify and track compliance per worker → log events and save evidence.

**This document covers only the current software/CV implementation. Hardware (Arduino, sensors, buzzer) is a future phase. Do not treat planned features as complete.**

---

## 2. Current Architecture

```
Video / Webcam Input
        │
        ▼
YOLOv8m SH17 Model (17-class inference, conf ≥ 0.4)
        │
        ├── person boxes → DeepSort (max_age=70, n_init=1) → stable track_id per worker
        │
        └── PPE item boxes → assign_ppe_to_persons()
                                (winner-takes-all by overlap ratio ≥ 0.3)
        │
        ▼
smooth_compliance(track_id, detected_ppe, required_ppe)
        Appends frame to per-worker deque(maxlen=5)
        Modal vote for status (SAFE / PARTIAL PPE / UNSAFE)
        Majority vote per item for stable missing set (only after window fills)
        │
        ▼
draw_worker_compliance_panel()   ← drawn on frame BEFORE snapshot
        │
        ▼
log_violation()   ← event-state machine; no cooldown; state-change only
        │
        ├── violation_log.csv  (dynamic columns per active checklist)
        └── snapshots/         (frame with compliance card drawn)

Intrusion Detection (runs BEFORE YOLO/DeepSORT, in parallel):
        grayframe2 vs grayframe1 (first frame)
        → absdiff → threshold → polygon mask → count pixels
        → if pixels > 100: trigger; 5s cooldown; log to intrusion_log.txt; save snapshot
```

---

## 3. Current Features

### 3.1 Input Sources

| Source | Support | Details |
|--------|---------|---------|
| Video file | ✅ Working | Any path via `--source` CLI or GUI file browser (`.mp4 .avi .mov .mkv`) |
| Webcam | ✅ Working | Integer index (`--source 0`); GUI auto-detects available cameras via `probe_cameras()` |
| Headless mode | ✅ Working | `--headless` suppresses all `cv2.imshow` windows; prints `[Headless] Processed frame N...` every 10 frames |
| Interactive GUI mode | ✅ Working | Default when `--headless` not passed; renders BGR + grayscale side-by-side at 0.8× scale; ESC/Q to stop |
| RTSP / network streams | ❌ Not implemented | No RTSP URL handling; not tested |

**Per-session output isolation:** Each video run writes to `output/<source_name>/`. Each webcam run writes to `output/webcam_<YYYYMMDD_HHMMSS>/`.

### 3.2 GUI (app.py — Industrial Control Room HMI)

`python app.py` opens the professional industrial control room HMI dashboard. All detection still runs via `main.run_detection()`.

**HMI Layout & Features:**
- **Header:** Operational status badge (`IDLE` / `MONITORING ACTIVE`), real-time FPS counter, processed frame counter, active session indicator.
- **Left Control Panel:**
  - **Source selection:** Native Windows file picker supporting arbitrary video files from any directory, plus webcam selection with camera index probing.
  - **Required PPE Checklist:** All 10 supported SH17 PPE classes with Select All, Clear All, and Reset Default.
  - **Restricted Zone (Polygon ROI):** Interactive `PolygonDrawingDialog` on the first frame with click-to-add vertices, live guideline, translucent fill, undo, reset, and coordinate resolution mapping back to original video dimensions.
  - **Execution controls:** Start monitoring, Stop monitoring, and annotated video recording toggle.
- **Right Live Monitoring & Telemetry Area:**
  - **Embedded Video Canvas:** Real-time video frame display with isolated fail-safe handling (never crashes detection loop).
  - **Alarm & Count Strip:** Live intrusion status (`● ZONE SECURE` vs `▲ ALARM: INTRUSION DETECTED`) and compliance breakdown badges (`Total`, `Safe`, `Partial`, `Unsafe`).
  - **Telemetry Tabs:** Active Workers real-time table (`Track ID`, `Compliance Status`, `Missing PPE`, `Detected PPE`, `Bounding Box`), Violation Events live log, and System Output console.
| Stop Monitoring | Sets `threading.Event` → detection loop exits cleanly |

**On startup:** Checks all required packages are importable, validates `models/yolo8m.pt` against the SH17 17-class schema.

**Limitation:** The detection window (OpenCV) is separate from the Tkinter GUI window. The Stop button works via `threading.Event`; the window does not block the GUI thread.

### 3.3 YOLO Model

| Item | Detail |
|------|--------|
| File | `models/yolo8m.pt` |
| Architecture | YOLOv8m |
| Total classes | 17 (SH17 dataset) |
| Validation | On every startup — mismatched schema causes `ValueError` + exit |
| Detection threshold | `conf ≥ 0.4` (hard-coded in detection loop, line 967) |
| Device | CPU only (`torch.device("cpu")` — no CUDA on this machine) |

**17 SH17 classes:**

| ID | Name | Category |
|----|------|----------|
| 0 | `person` | Subject |
| 1 | `ear` | Anatomy |
| 2 | `ear-mufs` | PPE |
| 3 | `face` | Anatomy |
| 4 | `face-guard` | PPE |
| 5 | `face-mask` | PPE |
| 6 | `foot` | Anatomy |
| 7 | `tool` | Non-PPE |
| 8 | `glasses` | PPE |
| 9 | `gloves` | PPE |
| 10 | `helmet` | PPE |
| 11 | `hands` | Anatomy |
| 12 | `head` | Anatomy |
| 13 | `medical-suit` | PPE |
| 14 | `shoes` | PPE |
| 15 | `safety-suit` | PPE |
| 16 | `safety-vest` | PPE |

Anatomy classes (`ear, face, foot, hands, head`) and `tool` are detected but ignored by compliance logic.

### 3.4 PPE Classes

**10 selectable PPE classes** (all can be required):
`ear-mufs`, `face-guard`, `face-mask`, `glasses`, `gloves`, `helmet`, `medical-suit`, `shoes`, `safety-suit`, `safety-vest`

**Default required PPE:** `helmet, safety-vest, gloves, shoes, face-mask`

Any subset of the 10 classes can be required via GUI checkboxes or `--required-ppe "helmet,safety-vest,shoes"` CLI argument. The compliance logic, CSV columns, and temporal smoothing all dynamically use whatever is currently required.

### 3.5 Person-PPE Association

**Function:** `assign_ppe_to_persons()` (`main.py:230`)

**Method:**
1. For every detected PPE item, compute `compute_overlap_ratio(person_box, ppe_box)` — fraction of the PPE box area that overlaps the person bounding box (0.0–1.0).
2. Assign the PPE item to the person with the highest overlap ratio, only if that ratio exceeds `overlap_threshold=0.3`.
3. Winner-takes-all: each PPE item goes to exactly one person; no PPE is shared.

**Limitation:** Association is purely spatial/geometric per frame. If PPE boxes are equidistant between two workers (heavily overlapping workers), assignment may be incorrect. Occlusion of person bounding box reduces the overlap ratio and can cause misassignment.

### 3.6 DeepSORT Tracking

**Library:** `deep_sort_realtime` v1.3.2

**Parameters:**
- `max_age=70` — track kept alive for up to 70 frames (~2.3–2.8s) without a matched detection
- `n_init=1` — track confirmed after 1 detection (instant tracking, no warm-up)

**Behavior:**
- Only `person` detections are fed to DeepSORT (PPE items are excluded)
- Confirmed tracks produce a stable `track_id` (integer) used as the worker key
- If a track is lost, the track_id eventually changes on re-detection → creates a new worker entry

**Stale track cleanup (`TRACK_GRACE_FRAMES = 80`):**
- Each frame, tracks absent from `current_active_ids` increment `grace_counter`
- After 80 frames of absence, `worker_state[track_id]` is deleted (history forgotten)
- This means if the same physical person re-enters after 80+ frames, they appear as a new worker

**IOU matching to compliance results:**
- After DeepSORT produces track boxes, the main loop matches each confirmed person track to the closest YOLO-detected person bounding box using IOU > 0.3
- If no YOLO person box matches (IOU ≤ 0.3), the track receives no compliance update that frame

### 3.7 PPE Compliance

**Function:** `classify_compliance(detected_ppe, required_ppe)` (`main.py:246`)

| Status | Condition |
|--------|-----------|
| `SAFE` | All required PPE detected |
| `PARTIAL PPE` | Some but not all required PPE detected |
| `UNSAFE` | No required PPE detected at all |

**Temporal smoothing:** `smooth_compliance(track_id, detected_ppe, required_ppe)` (`main.py:262`)

- Maintains a `deque(maxlen=5)` of detected PPE sets per worker (`SMOOTHING_WINDOW = 5`)
- **Status vote:** Modal (most frequent) status across the window. Tie-break: `UNSAFE > PARTIAL PPE > SAFE` (always err towards danger)
- **Stable missing set:** Per-item majority vote — an item is considered "stably missing" only if absent in the strict majority of frames in the window (`count ≤ window_size / 2`)
- **Warm-up guard:** During the first `SMOOTHING_WINDOW` frames of a new track, `smoothed_missing` returns `[]` even if the status is non-SAFE. This prevents premature violation logging while the window is filling. Logging only begins once the window is full.

### 3.8 PPE Violation Logging

**Function:** `log_violation()` (`main.py:386`)

**Event-based state machine — no time-based cooldown:**

| Condition | Action |
|-----------|--------|
| `status == 'SAFE'` or `missing_items == []` | Never logs; silently resets state to SAFE |
| First non-SAFE event for a new/reset track | Logs `INITIAL VIOLATION` |
| Violation returns after worker was SAFE | Logs `PPE STATE CHANGE` |
| Missing PPE set changes (e.g. `{helmet}` → `{helmet, shoes}`) | Logs `PPE STATE CHANGE` |
| Same missing PPE set continues (unchanged) | **Suppressed — no duplicate row** |

**CSV file:** `output/<session>/violation_log.csv`

**CSV columns (dynamic):** `Date, Time, Person, Status, [Required PPE columns...], Missing PPE, Event, Snapshot`
- Required PPE columns are dynamically generated per active checklist (alphabetical order)
- Each column is `YES` / `NO` for whether that PPE item was detected

**Snapshots:** Saved to `output/<session>/snapshots/event_NNN_person_M.jpg`
- The compliance panel (bounding box + status card) is drawn onto the frame **before** saving
- Filename: `event_{counter:03d}_person_{track_id}.jpg`
- The relative path `snapshots/event_NNN_person_M.jpg` is stored in the CSV

**Known limitations:**
- If DeepSORT loses and re-assigns a track ID (person briefly leaves frame > 80 frames), the new track ID is treated as a fresh worker — a new INITIAL VIOLATION may be logged even if the person's PPE state is unchanged
- During the 5-frame warm-up window, the live display shows status without listing specific missing items
- Track-ID reassignment after short occlusions (< 70 frames) does not cause duplicate events because the track_id is stable
- YOLO detection flicker (PPE appearing/disappearing frame-to-frame) is suppressed by the majority-vote smoothing

### 3.9 Restricted Polygon

**Function:** `select_roi_interactive()` (`main.py:602`), `validate_polygon()` (`main.py:121`)

**How it works:**
- User clicks vertices on the first video frame in an OpenCV window
- Minimum 3 non-collinear points required; unlimited maximum points
- Keyboard shortcuts: S/Enter = save, U/Backspace = undo, R = reset, ESC/Q = cancel
- Live canvas shows: numbered point markers, edge lines, translucent purple fill (≥3 points), preview line to cursor

**Storage:** `restricted_polygon = [(x1,y1), (x2,y2), ...]` — plain Python list of int tuples

**Display during detection:** Translucent filled overlay (green = clear, red = intrusion), solid boundary contour, status text pinned to top-most vertex of the polygon

**CLI input:** `--roi "x1,y1;x2,y2;x3,y3..."` (semicolon-separated) or legacy `x1,y1,x2,y2` (auto-converted to 4-point polygon)

**Rectangle ROI:** No standalone rectangle ROI system exists. The legacy 4-value format (`x1,y1,x2,y2`) is automatically converted to a 4-vertex polygon by `validate_polygon()`. The system is fully polygon-only.

### 3.10 Intrusion Detection

**Two-Tier Intrusion Architecture (Raw Detection + Standalone Filter):**

1. **Raw Motion Detector (`main.py` -> `intrusion_log.txt`):**
   - **Method:** Pixel/motion difference via `cv2.absdiff`
   - **Reference frame:** Compared against `grayframe1` (first frame of the video session)
   - **Polygon masking:** `cv2.fillPoly` mask -> `cv2.bitwise_and` counts active pixels strictly inside polygon
   - **Trigger threshold:** >100 motion pixels inside polygon (intensity threshold 130/255)
   - **Cooldown for logging:** 5 seconds (`intrusion_cooldown_seconds = 5`)
   - **Log file:** `output/<session>/intrusion_log.txt` (plain text)
   - **Snapshots:** Saved to `output/<session>/intrusion_snapshots/intrusion_<YYYYMMDD_HHMMSS>.jpg`
   - **Person awareness:** **None** — raw detector does not know which person entered; runs before YOLO/DeepSORT.

2. **Real-Time Log Watcher & Event Filter (`scripts/intrusion_log_watcher.py` -> `intrusion_log.csv` + `intrusion_gifs/`):**
   - **Architecture:** Standalone consumer script completely decoupled from `main.py`
   - **Tailing mechanism:** Monitors `intrusion_log.txt` in real time (`tail -f` style), processing newly appended lines
   - **10-Second Grouping Rule:** Consecutive raw detections with gaps $\le 10$ seconds (`INTRUSION_EVENT_GAP_SECONDS = 10`) belong to the same continuous intrusion event. Gaps $> 10$ seconds start a new event.
   - **Animated GIF Generation:** For each grouped intrusion event, all evidence snapshots from `intrusion_snapshots/` are sorted chronologically and compiled into an animated GIF (`intrusion_gifs/intrusion_event_XXX.gif`). Ongoing events have their GIF updated in real time as new snapshots arrive.
   - **Preservation of JPGs:** All original JPG snapshots in `intrusion_snapshots/` are 100% preserved untouched.
   - **In-Place CSV Updates:** While an event is ongoing, its CSV row is updated in real time (`Last Detection`, `Duration`, `Detection Count`, `GIF`).
   - **Duration definition:** Duration in `intrusion_log.csv` is **strictly the time between the first and last raw intrusion detection** belonging to that grouped event. It does **not** represent verified person tracking or physical dwell time.
   - **No entry/exit tracking:** Does not claim person-specific identity, entry/exit boundaries, or DeepSORT-based tracking.

### 3.11 Output Files

**Session directory structure:**
```
output/
  <session_name>/          ← one per video run (named after source file)
    violation_log.csv      ← PPE violation events (event-based, dynamic columns)
    intrusion_log.txt      ← Raw motion intrusion detections (plain text, 5s cooldown)
    intrusion_log.csv      ← Filtered intrusion events (grouped by 10s gap, real-time watcher)
    intrusion_gifs/        ← Animated evidence GIFs (one per grouped intrusion event)
      intrusion_event_001.gif
      intrusion_event_002.gif
      ...
    snapshots/             ← PPE violation evidence frames
      event_001_person_1.jpg
      event_002_person_3.jpg
      ...
    annotated/
      <source_name>_annotated.mp4   ← full annotated video
    intrusion_snapshots/   ← raw motion intrusion frames (preserved untouched)
      intrusion_<YYYYMMDD_HHMMSS>.jpg
      ...

  webcam_<YYYYMMDD_HHMMSS>/   ← one per webcam session
    (same structure as above; no annotated video saved for webcam)

  annotated/               ← legacy mirror copy of annotated video (for backward compatibility)
    <source_name>_annotated.mp4
```

**`violation_log.csv` — example row (policy: helmet, safety-vest, shoes):**
```
Date,Time,Person,Status,Helmet,Safety-Vest,Shoes,Missing PPE,Event,Snapshot
2026-09-03,15:11:59,Person 3,PARTIAL PPE,NO,YES,YES,"helmet",PPE STATE CHANGE,snapshots/event_001_person_3.jpg
```

**`intrusion_log.csv` — example row:**
```
Event ID,Start Time,Last Detection,Duration (seconds),Detection Count,Polygon,GIF
1,2026-09-02 23:16:56,2026-09-02 23:17:14,18,3,"[(100, 100), (400, 100), (400, 400), (100, 400)]",intrusion_gifs/intrusion_event_001.gif
```

**`intrusion_log.txt` — example entry:**
```
Intrusion detected at Wed Sep  3 23:44:23 2026 | Polygon=[(100, 100), (400, 100), (400, 400), (100, 400)]
```

---

## 4. Testing Status

**Test suite location:** `tests/`
**Test count:** 10 test modules, 66 test methods total
**Last confirmed result:** 66/66 PASS (verified 2026-09-07)

| Test Module | What It Tests | Count |
|-------------|--------------|-------|
| `test_model_validation.py` | Model file exists, 17 SH17 classes confirmed, mismatch detection | 3 |
| `test_compliance_classifier.py` | `classify_compliance()` for all statuses; custom checklists; extra PPE ignored | 4 |
| `test_spatial_association.py` | `compute_overlap_ratio()`, `assign_ppe_to_persons()` winner-takes-all | 3 |
| `test_temporal_smoothing.py` | Single-frame glitch suppression, tie-break severity, custom checklist window | 3 |
| `test_violation_logging.py` | Initial violation, unchanged-state suppression, state change, SAFE→re-violation, SAFE never logs | 5 |
| `test_stable_missing_pipeline.py` | Full `smooth_compliance → log_violation` pipeline; 7 scenarios including flicker, state changes, SAFE transitions, CSV integrity | 7 |
| `test_polygon_roi.py` | Polygon validation (point count, degenerate, clamping), CLI string parsing, strict pixel masking | 5 |
| `test_app_gui.py` | Dependency check, model validation, GUI init/navigation, PPE checklist controls, headless execution with/without ROI | 5 |
| `test_cli_headless.py` | Full headless run via subprocess, missing file error handling | 2 |
| `test_intrusion_log_watcher.py` | Parsing, 10s grouping, exact/greater gaps, real-time live appending, auto-discovery, GIF generation, chronological ordering, edge cases | 23 |
| `__init__.py` | (empty, required for test discovery) | — |

**What is NOT tested:**
- Intrusion detection behavior end-to-end (no `test_person_intrusion.py`)
- Webcam input (requires physical camera)
- Multi-person scenarios with overlapping bounding boxes
- YOLO detection at various confidence thresholds
- DeepSORT re-identification accuracy
- Track-loss and track-ID reassignment behavior
- GUI detection start/stop via the Tkinter buttons (only internal logic tested)

---

## 5. Known Limitations

### Detection & Model
1. **CPU-only inference** — no CUDA; inference speed is limited by CPU performance (~2–8 FPS on typical hardware with this model)
2. **Single-camera only** — one video source per session
3. **Small/distant persons** — YOLO detects poorly at distances where persons appear < ~40 pixels tall
4. **Overlapping persons** — PPE assignment may be incorrect when two workers' bounding boxes heavily overlap
5. **Occlusion of PPE** — any PPE hidden by hands, other workers, or equipment is not detected; the worker's compliance may be incorrectly classified as worse than actual
6. **Helmet at extreme angles** — top-down camera angles reduce helmet detection confidence
7. **Shoes/feet out of frame** — if the camera doesn't capture the lower body, shoe detection fails entirely
8. **Confidence threshold is fixed** — 0.4 is hard-coded; no per-class adjustment

### Tracking
9. **Track-ID reassignment** — if a worker leaves the frame for >70 frames and returns, they get a new track_id, resetting PPE history and potentially triggering a new INITIAL VIOLATION
10. **5-frame warm-up delay** — the first SMOOTHING_WINDOW frames of any worker appearance produce no violation log even if compliance is bad

### Intrusion Detection
11. **Motion-based only** — cannot identify which person entered the restricted zone
12. **Accumulated reference drift** — reference frame is the first video frame and never updates; slow lighting changes or background drift can cause continuous false positives
13. **No entry/exit tracking** — cannot distinguish first entry from continued presence; logs every 5 seconds during sustained intrusion
14. **No person-level intrusion ID** — snapshots capture the full frame with no indication of who caused the intrusion
15. **Camera movement = false positive** — any camera shake inside the polygon area triggers the alarm

### Logging
16. **Track-ID loss creates new events** — if a person's DeepSORT track is lost and re-assigned, any resumed violation appears as a new event even if PPE state is unchanged
17. **Raw intrusion log is unstructured plain text** — the separate `intrusion_log_watcher.py` provides a structured CSV representation, but raw detection is still motion-pixel based.

### Environment
18. **No GPU support configured** — torch loaded on CPU; CUDA/MPS not detected or used
19. **Windows path assumptions** — output paths use `os.path.join`; should work cross-platform but only tested on Windows
20. **Light/environment sensitivity** — PPE detection degrades in poor lighting, glare, or heavily shadowed scenes

---

## 6. Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Video file input | ✅ Working | CLI and GUI, any common video format |
| Webcam input | ✅ Working | Index-based, GUI auto-detects |
| Headless execution | ✅ Working | `--headless --no-roi` fully verified |
| GUI launcher (app.py) | ✅ Working | Tkinter, webcam + video screens |
| YOLOv8 PPE detection | ✅ Working | 17-class SH17 model, conf ≥ 0.4 |
| Person detection | ✅ Working | SH17 class 0, feeds DeepSORT |
| SH17 model validation | ✅ Working | Schema check on every startup |
| PPE-to-person spatial association | ✅ Working | `compute_overlap_ratio()` + winner-takes-all |
| DeepSORT person tracking | ✅ Working | Stable track_ids, max_age=70, n_init=1 |
| Configurable PPE checklist | ✅ Working | 10 classes selectable via GUI or CLI |
| Compliance classification | ✅ Working | SAFE / PARTIAL PPE / UNSAFE per worker |
| 5-frame temporal smoothing | ✅ Working | Modal vote + majority-vote missing set |
| Warm-up guard (no premature logging) | ✅ Working | First 5 frames suppressed |
| Per-worker compliance card overlay | ✅ Working | ID, status, [REQ], [MISSING], [OTHER] |
| Event-based violation logging | ✅ Working | State-change only; no cooldown |
| Duplicate violation suppression | ✅ Working | Same missing set → no repeat row |
| Dynamic CSV columns | ✅ Working | Columns match active checklist |
| Violation snapshots | ✅ Working | Compliance card drawn before save |
| Annotated video output | ✅ Working | Saved to session folder |
| Session folder isolation | ✅ Working | Per-video and per-webcam dirs |
| Polygon ROI — drawing | ✅ Working | Interactive multi-point canvas |
| Polygon ROI — validation | ✅ Working | ≥3 points, non-degenerate, clamped |
| Polygon ROI — CLI input | ✅ Working | Semicolon-separated or legacy rect |
| Polygon ROI — overlay | ✅ Working | Translucent fill, green/red status |
| Raw intrusion detection | ⚠️ Limited | Motion-pixel based; no person ID; 5s cooldown; repeated logging to TXT |
| Intrusion log watcher & GIF generator | ✅ Working | Standalone `scripts/intrusion_log_watcher.py`, 10s gap grouping, animated GIF generation in `intrusion_gifs/` |
| Person-specific intrusion tracking | ❌ Not implemented | Was planned/built then reverted |
| Entry/exit event tracking | ❌ Not implemented | Reverted with person intrusion system |
| Email alerts | ❌ Not implemented | Planned Phase 2 |
| Web frontend / dashboard | ❌ Not implemented | Planned Phase 3 |
| Arduino / hardware integration | ❌ Not implemented | Planned Phase 4 |
| Audio alerts (buzzer/speaker) | ❌ Not implemented | Planned Phase 4 |

---

## 7. Planned Next Phases

> These are future roadmap items. None of these are currently implemented.

### Phase 1 — Stabilize Current CV/Software System
- Improve intrusion detection: person-specific entry/exit tracking using existing YOLO + DeepSORT bounding boxes
- Replace motion-pixel intrusion with foot-position polygon containment (`cv2.pointPolygonTest`)
- Add entry/exit event log as proper CSV
- Add dwell suppression (no re-logging while inside)
- Add track-loss grace period for intrusion state

### Phase 2 — Email Alert System
- Email notifications triggered by software events only:
  - PPE violation (new INITIAL VIOLATION or PPE STATE CHANGE)
  - Restricted zone intrusion (ENTRY event)
- No gas/temperature/humidity alerts in this phase
- Configurable recipient list
- Rate limiting to prevent email floods

### Phase 3 — Proper Frontend
- Web dashboard (local or hosted)
- Live session monitoring and log visualization
- Recipient management for alerts
- PPE policy configuration UI
- Historical log browser

### Phase 4 — Hardware Integration
- Arduino Uno
- Gas sensor
- Temperature sensor
- Humidity sensor
- Buzzer / audio output (DFPlayer or equivalent)
- Hardware events fed into the alert system

### Phase 5 — Final Integration
- Combine CV pipeline + email alerts + frontend + hardware
- Hardware-generated safety events trigger alerts
- Audio/buzzer warnings on violation/intrusion
- End-to-end system testing

---

## 8. Important Development Notes

### Model
- **Active model:** `models/yolo8m.pt` — SH17 17-class. **Do not replace with COCO weights.**
- The model schema is validated on every startup via `validate_sh17_model()`. Any mismatch raises `ValueError` and exits.
- Class 0 = `person` (non-PPE subject). Classes 2,4,5,8,9,10,13,14,15,16 = PPE.

### Installed Packages (verified 2026-08-25)
| Package | Version | Status |
|---------|---------|--------|
| Python | 3.12.10 | ✅ |
| opencv-python | 4.10.0.84 | ✅ |
| numpy | 2.3.5 | ✅ |
| pandas | 2.3.3 | ✅ |
| shapely | 2.1.2 | ✅ |
| ultralytics | 8.4.30 | ✅ |
| torch | 2.11.0+cpu | ✅ CPU-only |
| torchvision | 0.26.0 | ✅ |
| deep_sort_realtime | 1.3.2 | ✅ |
| matplotlib | 3.10.8 | ✅ |

> `deep_sort_realtime` produces a harmless `pkg_resources` deprecation warning on startup. This does not affect functionality.

### Run Commands
```bash
cd <project-directory>

# 1. GUI Launcher (recommended on Windows: .\run_gui.bat or python app.py)
python app.py

# 2. Headless video — no ROI
python main.py --source my_video.mp4 --headless --no-roi --required-ppe "helmet,safety-vest,shoes"

# 3. Headless video — with polygon ROI
python main.py --source my_video.mp4 --headless --roi "100,100;400,100;400,400;100,400" --required-ppe "helmet,safety-vest"

# 4. Interactive video (prompts for polygon ROI interactively)
python main.py --source my_video.mp4 --required-ppe "helmet,safety-vest,shoes"

# 5. Webcam (interactive)
python main.py --source 0 --required-ppe "helmet,safety-vest"

# 6. Run real-time intrusion log watcher (runs alongside main.py or standalone)
python scripts/intrusion_log_watcher.py --input output/my_video/intrusion_log.txt

# 7. Run full test suite (66 tests across 10 modules)
python -m unittest discover -s tests -p "test_*.py" -v

# 8. Validate model schema only
python scripts/validate_model.py
```

### Default PPE Policy
```python
DEFAULT_REQUIRED_PPE = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}
```
This is the default when no `--required-ppe` argument is given. The GUI pre-selects these 5 items.

### Key Constants
```python
SMOOTHING_WINDOW            = 5      # frames per worker history window
TRACK_GRACE_FRAMES          = 80     # frames before deleting lost track state
DEEPSORT_MAX_AGE            = 70     # frames DeepSORT keeps an unmatched track alive
DEEPSORT_N_INIT             = 1      # detections required to confirm a track
INTRUSION_COOLDOWN          = 5      # seconds between raw intrusion log entries (main.py)
INTRUSION_EVENT_GAP_SECONDS = 10     # seconds for grouping continuous intrusion events (watcher)
GIF_FRAME_DURATION_MS       = 800    # milliseconds per frame for generated intrusion GIFs
DETECTION_CONF_THRESH       = 0.4    # YOLO confidence threshold (hard-coded in loop)
```

### Updates Log
- **2026-09-08 (Industrial HMI Upgrade):** Upgraded `app.py` into a dark control-room industrial HMI dashboard. Added native Windows file browser for arbitrary video files from any drive, 10-class PPE policy panel, interactive multi-point polygon editor with exact resolution mapping, fail-safe live monitoring canvas, real-time active worker telemetry table, live intrusion alarm banner, and violation event viewer. Added optional `frame_callback` to `main.run_detection()`. Added `tests/test_hmi_integration.py` (total 71 tests across 11 modules, all PASS).
- **2026-09-07 (GIF generation):** Added animated GIF generation to `scripts/intrusion_log_watcher.py`. Each grouped intrusion event compiles evidence snapshots into `output/<session>/intrusion_gifs/intrusion_event_XXX.gif` (chronologically ordered, updated live). `intrusion_log.csv` references the GIF path. All original JPG snapshots are preserved untouched. 23 tests in `tests/test_intrusion_log_watcher.py` (total 66 tests across 10 modules, all PASS).
- **2026-09-07:** Implemented standalone `scripts/intrusion_log_watcher.py` to monitor `output/<session>/intrusion_log.txt` in real time and generate grouped `intrusion_log.csv`.
- **2026-09-06:** Audit pass — removed outdated references to reverted person-based intrusion tracker and documented actual motion-pixel detector. Fixed majority-vote stable missing PPE smoothing.

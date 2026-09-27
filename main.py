#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import time
import os
import sys
import argparse
import torch
import math
import shapely.geometry
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# --- Core Verified SH17 Model Schema ---
VERIFIED_MODEL_CLASSES = {
    0: 'person',    1: 'ear',          2: 'ear-mufs',   3: 'face',
    4: 'face-guard',5: 'face-mask',    6: 'foot',       7: 'tool',
    8: 'glasses',   9: 'gloves',      10: 'helmet',    11: 'hands',
   12: 'head',     13: 'medical-suit',14: 'shoes',     15: 'safety-suit',
   16: 'safety-vest'
}

# SH17 detectable PPE classes (what the model can detect)
SH17_PPE_CLASSES = {
    'ear-mufs', 'face-guard', 'face-mask', 'glasses', 'gloves',
    'helmet', 'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
}

# Display titles for CSV headers and reporting
PPE_COLUMN_TITLES = {
    'helmet': 'Helmet',
    'safety-vest': 'Safety-Vest',
    'gloves': 'Gloves',
    'shoes': 'Shoes',
    'face-mask': 'Face-Mask',
    'glasses': 'Glasses',
    'face-guard': 'Face-Guard',
    'ear-mufs': 'Ear-Muffs',
    'safety-suit': 'Safety-Suit',
    'medical-suit': 'Medical-Suit',
}

# Active compliance policy (what is REQUIRED for a worker to be SAFE)
DEFAULT_REQUIRED_PPE = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}
REQUIRED_PPE = DEFAULT_REQUIRED_PPE

# System thresholds and parameters
SMOOTHING_WINDOW = 5
_STATUS_SEVERITY = {'UNSAFE': 0, 'PARTIAL PPE': 1, 'SAFE': 2}
TRACK_GRACE_FRAMES = 80

# State dictionaries
worker_state = {}
violation_cooldown = {}
motion_pixels = 0


def parse_required_ppe(ppe_input):
    """Validate and resolve a required PPE checklist into a set of SH17 PPE class names.

    Args:
        ppe_input: None, set, list, tuple, or comma/semicolon-separated string.

    Returns:
        set of valid SH17 PPE class names (falls back to DEFAULT_REQUIRED_PPE if None or invalid).
    """
    if ppe_input is None:
        return set(DEFAULT_REQUIRED_PPE)

    if isinstance(ppe_input, str):
        delimiters = [',', ';', ' ']
        raw_items = [ppe_input]
        for d in delimiters:
            new_items = []
            for item in raw_items:
                new_items.extend(item.split(d))
            raw_items = new_items
        items = {it.strip().lower() for it in raw_items if it.strip()}
    elif isinstance(ppe_input, (set, list, tuple)):
        items = {str(it).strip().lower() for it in ppe_input if str(it).strip()}
    else:
        return set(DEFAULT_REQUIRED_PPE)

    valid_items = {it for it in items if it in SH17_PPE_CLASSES}
    if not valid_items:
        print(f"WARNING: No valid SH17 PPE classes found in input ({ppe_input}). Using default checklist.")
        return set(DEFAULT_REQUIRED_PPE)

    return valid_items


def validate_sh17_model(model_path="models/yolo8m.pt"):
    """Load and strictly validate the SH17 17-class model weights."""
    if not os.path.exists(model_path):
        print(f"FATAL: Model weights not found at: {model_path}")
        raise FileNotFoundError(f"Model file missing: {model_path}")

    model = YOLO(model_path)
    class_names = model.names

    mismatches = [
        (k, VERIFIED_MODEL_CLASSES[k], class_names.get(k))
        for k in VERIFIED_MODEL_CLASSES
        if class_names.get(k) != VERIFIED_MODEL_CLASSES[k]
    ]
    if mismatches:
        print("FATAL: Model class validation failed. This is not the expected SH17 model.")
        for k, expected, got in mismatches:
            print(f"  class {k}: expected '{expected}', got '{got}'")
        raise ValueError("Model class schema mismatch. Expected 17 SH17 classes.")

    print(f"Model validation PASSED — {len(class_names)} SH17 classes confirmed.")
    return model, class_names


def validate_polygon(points, frame_width=640, frame_height=480):
    """Validate and clamp polygon coordinates for restricted zone monitoring.

    Args:
        points: List or tuple of (x, y) coordinate pairs, or legacy 4-val box.
        frame_width: Image frame width for bounds checking.
        frame_height: Image frame height for bounds checking.

    Returns:
        list of (int, int) tuples if valid polygon with >=3 points and non-zero area, else None.
    """
    if not points:
        return None

    # Handle string formatted input
    if isinstance(points, str):
        return parse_polygon_string(points, frame_width, frame_height)

    # Convert legacy 4-tuple rectangle (x1, y1, x2, y2) to 4-point polygon
    if isinstance(points, (tuple, list)) and len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
        rx1, ry1, rx2, ry2 = points
        rx1, rx2 = sorted([int(rx1), int(rx2)])
        ry1, ry2 = sorted([int(ry1), int(ry2)])
        points = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]

    cleaned_pts = []
    try:
        for pt in points:
            if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                return None
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            x = max(0, min(frame_width, x))
            y = max(0, min(frame_height, y))
            cleaned_pts.append((x, y))
    except (ValueError, TypeError):
        return None

    if len(cleaned_pts) < 3:
        return None

    # Verify polygon has non-zero area (not degenerate/collinear)
    np_pts = np.array(cleaned_pts, dtype=np.int32)
    area = cv2.contourArea(np_pts)
    if area <= 0:
        return None

    return cleaned_pts


def parse_polygon_string(poly_str, frame_width=640, frame_height=480):
    """Parse CLI string into a verified list of polygon (x, y) tuples."""
    if not poly_str or not isinstance(poly_str, str):
        return None

    poly_str = poly_str.strip()

    # Semicolon-separated pairs: "100,100;300,80;500,200;450,400"
    if ';' in poly_str:
        pts = []
        for p in poly_str.split(';'):
            p = p.strip()
            if not p:
                continue
            parts = [int(v.strip()) for v in p.split(',') if v.strip()]
            if len(parts) == 2:
                pts.append((parts[0], parts[1]))
        return validate_polygon(pts, frame_width, frame_height)

    # Space-separated pairs: "100,100 300,80 500,200 450,400"
    elif ' ' in poly_str:
        pts = []
        for p in poly_str.split():
            p = p.strip()
            if not p:
                continue
            parts = [int(v.strip()) for v in p.split(',') if v.strip()]
            if len(parts) == 2:
                pts.append((parts[0], parts[1]))
        return validate_polygon(pts, frame_width, frame_height)

    # Comma-separated format
    elif ',' in poly_str:
        parts = [int(v.strip()) for v in poly_str.split(',') if v.strip()]
        # Legacy 4-number rectangle: x1,y1,x2,y2 -> convert to 4-point polygon
        if len(parts) == 4:
            rx1, ry1, rx2, ry2 = parts
            return validate_polygon([(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)], frame_width, frame_height)
        elif len(parts) >= 6 and len(parts) % 2 == 0:
            pts = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
            return validate_polygon(pts, frame_width, frame_height)

    return None


def compute_overlap_ratio(person_box, ppe_box):
    """Return fraction of ppe_box area that lies inside person_box (0.0 to 1.0)."""
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = ppe_box
    ix1 = max(px1, ox1)
    iy1 = max(py1, oy1)
    ix2 = min(px2, ox2)
    iy2 = min(py2, oy2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    ppe_area = max(1, (ox2 - ox1) * (oy2 - oy1))
    return inter_area / ppe_area


def assign_ppe_to_persons(persons, ppe_detections, overlap_threshold=0.3):
    """Assign each detected PPE item to exactly one person (winner-takes-all)."""
    person_ppe = {i: set() for i in range(len(persons))}
    for ppe_box, class_name, conf in ppe_detections:
        best_ratio = overlap_threshold
        best_person = None
        for i, person_box in enumerate(persons):
            ratio = compute_overlap_ratio(person_box, ppe_box)
            if ratio > best_ratio:
                best_ratio = ratio
                best_person = i
        if best_person is not None:
            person_ppe[best_person].add(class_name)
    return person_ppe


def classify_compliance(detected_ppe, required_ppe=None):
    """Classify a single worker's compliance status against required PPE."""
    if required_ppe is None:
        required_ppe = DEFAULT_REQUIRED_PPE
    else:
        required_ppe = set(required_ppe)

    missing = required_ppe - detected_ppe
    if not missing:
        return 'SAFE', (0, 200, 0), []
    elif len(missing) < len(required_ppe):
        return 'PARTIAL PPE', (0, 200, 200), sorted(missing)
    else:
        return 'UNSAFE', (0, 0, 255), sorted(missing)


def smooth_compliance(track_id, current_detected_ppe, required_ppe=None):
    """Update per-worker history and return temporally smoothed compliance status evaluated against required_ppe."""
    if required_ppe is None:
        required_ppe = DEFAULT_REQUIRED_PPE
    else:
        required_ppe = set(required_ppe)

    if track_id not in worker_state:
        worker_state[track_id] = {
            'ppe_history': deque(maxlen=SMOOTHING_WINDOW),
            'grace_counter': 0,
            'last_logged_missing': None,    # frozenset of missing required items, or None initially
            'last_confirmed_status': None,  # 'SAFE', 'PARTIAL PPE', 'UNSAFE', or None
        }

    ws = worker_state[track_id]
    ws['ppe_history'].append(current_detected_ppe)
    ws['grace_counter'] = 0

    status_counts = {}
    for ppe_set in ws['ppe_history']:
        s, _, _ = classify_compliance(ppe_set, required_ppe=required_ppe)
        status_counts[s] = status_counts.get(s, 0) + 1

    smoothed_status = min(
        status_counts,
        key=lambda s: (-status_counts[s], _STATUS_SEVERITY[s])
    )
    smoothed_color = {'SAFE': (0, 200, 0), 'PARTIAL PPE': (0, 200, 200), 'UNSAFE': (0, 0, 255)}[smoothed_status]
    # Build the stable missing-PPE set using per-item majority vote across the smoothing window.
    # This prevents single-frame YOLO detection flicker from changing the missing set and
    # triggering spurious PPE STATE CHANGE events in log_violation().
    #
    # IMPORTANT: only compute once the window is full (>= SMOOTHING_WINDOW frames).
    # During warm-up the partial window can misclassify items, log an initial state, then
    # correct itself once full — generating a spurious extra state-change event.
    # Returning [] during warm-up suppresses logging via log_violation's existing
    # `not missing_items` guard, until a stable PPE state has been established.
    if smoothed_status != 'SAFE':
        window_size = len(ws['ppe_history'])
        if window_size >= SMOOTHING_WINDOW:
            per_item_count = {}
            for frame_ppe in ws['ppe_history']:
                for item in required_ppe:
                    if item in frame_ppe:
                        per_item_count[item] = per_item_count.get(item, 0) + 1
            stably_detected = {item for item in required_ppe
                                if per_item_count.get(item, 0) > window_size / 2}
            smoothed_missing = sorted(required_ppe - stably_detected)
        else:
            # Window still warming up: suppress logging; display shows status without item breakdown
            smoothed_missing = []
    else:
        smoothed_missing = []

    return smoothed_status, smoothed_color, smoothed_missing


def draw_worker_compliance_panel(frame, tx1, ty1, tx2, ty2, track_id, status, color, required_detected, required_missing, other_detected=None):
    """Render a compact, high-contrast compliance status card and bounding box for a tracked worker."""
    # 1. Main bounding box
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 2)

    # 2. Prepare text lines: (text, text_color_bgr, is_bold)
    lines = []
    lines.append((f"ID:{track_id} {status}", color, True))

    if required_detected:
        req_det_str = ", ".join(sorted(required_detected))
        lines.append((f"[REQ] {req_det_str}", (0, 240, 0), False))

    if required_missing:
        req_miss_str = ", ".join(sorted(required_missing))
        lines.append((f"[MISSING] {req_miss_str}", (0, 0, 255), False))
    elif not required_missing and status == 'SAFE':
        lines.append(("All Required PPE Detected", (0, 240, 0), False))

    if other_detected:
        other_str = ", ".join(sorted(other_detected))
        lines.append((f"[OTHER] {other_str}", (190, 190, 190), False))

    # 3. Calculate panel dimensions
    line_h = 16
    panel_h = len(lines) * line_h + 6
    max_text_w = max(cv2.getTextSize(l[0], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] for l in lines)
    panel_w = max(130, max_text_w + 12)

    # 4. Position above bounding box if space, otherwise below, or inside box if tight
    frame_h, frame_w = frame.shape[:2]
    px1 = tx1
    py2 = ty1 - 4
    py1 = py2 - panel_h

    # If too close to top edge, try below bounding box
    if py1 < 5:
        py1 = ty2 + 4
        py2 = py1 + panel_h

    # If extending past bottom edge, place inside bounding box near the top
    if py2 > frame_h - 5:
        py1 = max(5, ty1 + 5)
        py2 = min(frame_h - 5, py1 + panel_h)

    # Clamp horizontal position
    if px1 + panel_w > frame_w:
        px1 = max(0, frame_w - panel_w - 4)
    if px1 < 0:
        px1 = 0

    # 5. Draw semi-transparent dark card background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px1 + panel_w, py2), (15, 23, 42), -1)
    cv2.rectangle(overlay, (px1, py1), (px1 + panel_w, py2), color, 1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 6. Render text lines
    curr_y = py1 + 14
    for text, text_color, is_bold in lines:
        thickness = 2 if is_bold else 1
        font_scale = 0.45 if is_bold else 0.38
        cv2.putText(frame, text, (px1 + 6, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        curr_y += line_h


def log_violation(
    track_id,
    status,
    missing_items,
    track_box,
    frame=None,
    save_dir=None,
    log_file=None,
    active_required_ppe=None,
    detected_ppe=None,
    event_counter=0,
    current_time=None
):
    """Log an event-based PPE violation when a worker's confirmed state changes into a violation state.

    Args:
        track_id: DeepSORT track ID (int or str).
        status: Smoothed status ('SAFE', 'PARTIAL PPE', 'UNSAFE').
        missing_items: List or set of currently missing required PPE class names.
        track_box: Worker bounding box [x1, y1, x2, y2].
        frame: Video frame (with worker compliance panel drawn).
        save_dir: Directory where event snapshots are saved.
        log_file: Path to session violation_log.csv.
        active_required_ppe: Set of active required PPE classes.
        detected_ppe: Set of PPE classes detected on this worker.
        event_counter: Current session event count.
        current_time: Optional float timestamp.

    Returns:
        tuple: (did_log: bool, record: dict or None, new_event_counter: int)
    """
    if current_time is None:
        current_time = time.time()

    if active_required_ppe is None:
        active_required_ppe = DEFAULT_REQUIRED_PPE
    else:
        active_required_ppe = set(active_required_ppe)

    if detected_ppe is None:
        detected_ppe = set()
    else:
        detected_ppe = set(detected_ppe)

    if track_id not in worker_state:
        worker_state[track_id] = {
            'ppe_history': deque(maxlen=SMOOTHING_WINDOW),
            'grace_counter': 0,
            'last_logged_missing': None,
            'last_confirmed_status': None,
        }

    ws = worker_state[track_id]

    # 1. If worker is SAFE
    if status == 'SAFE' or not missing_items:
        if ws.get('last_confirmed_status') != 'SAFE':
            # Worker transitioned from violation to SAFE -> reset state without writing CSV row
            ws['last_confirmed_status'] = 'SAFE'
            ws['last_logged_missing'] = frozenset()
        return False, None, event_counter

    # 2. Worker is in Violation (PARTIAL PPE or UNSAFE with missing items)
    current_missing = frozenset(missing_items)
    prev_missing = ws.get('last_logged_missing')
    prev_status = ws.get('last_confirmed_status')

    # Determine if this is a state change that must be logged
    if prev_status == 'SAFE' or prev_missing is None:
        # Initial violation for this worker, or resumed violation after being SAFE
        event_type = 'INITIAL VIOLATION' if prev_missing is None else 'PPE STATE CHANGE'
        should_log = True
    elif current_missing != prev_missing:
        # Transition to a different missing PPE combination
        event_type = 'PPE STATE CHANGE'
        should_log = True
    else:
        # Unchanged violation state -> suppress logging
        should_log = False

    if not should_log:
        return False, None, event_counter

    # Update confirmed state
    ws['last_confirmed_status'] = status
    ws['last_logged_missing'] = current_missing

    # Increment event counter
    new_event_counter = event_counter + 1

    # Date and Time
    date_str = time.strftime('%Y-%m-%d', time.localtime(current_time))
    time_str = time.strftime('%H:%M:%S', time.localtime(current_time))
    person_str = f"Person {track_id}"

    # Required PPE columns (YES / NO)
    sorted_req = sorted(list(active_required_ppe))
    ppe_values = ["YES" if p in detected_ppe else "NO" for p in sorted_req]

    # Missing PPE string
    missing_str = "; ".join(sorted(missing_items))

    # Snapshot filename and paths
    snapshot_filename = f"event_{new_event_counter:03d}_person_{track_id}.jpg"
    snapshot_path = os.path.join(save_dir, snapshot_filename) if save_dir else snapshot_filename
    snapshot_rel_path = f"snapshots/{snapshot_filename}"

    # Save evidence frame (frame has worker compliance card & bounding box drawn!)
    if frame is not None and save_dir:
        cv2.imwrite(snapshot_path, frame)

    # Construct CSV row: Date,Time,Person,Status,[Dynamic PPE Cols],Missing PPE,Event,Snapshot
    row_cells = [date_str, time_str, person_str, status] + ppe_values + [f'"{missing_str}"', event_type, snapshot_rel_path]
    log_line = ",".join(row_cells) + "\n"

    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_line)

    record = {
        'date': date_str,
        'time': time_str,
        'track_id': track_id,
        'person': person_str,
        'status': status,
        'missing_ppe': list(missing_items),
        'event': event_type,
        'snapshot_path': snapshot_path,
        'snapshot_rel': snapshot_rel_path,
        'bbox': track_box
    }
    print(f"PPE event logged: {person_str} [{event_type}] ({status}: {missing_str}) -> {snapshot_filename}")
    return True, record, new_event_counter


def intrusion(cframe, rframe, polygon, thresh=130, motion_pixel_count=100):
    """Detect motion intrusion strictly inside a polygon restricted zone.

    Args:
        cframe: Current grayscale frame.
        rframe: Reference grayscale frame.
        polygon: List of (x, y) tuples defining the polygon.
        thresh: Binary motion threshold (default: 130).
        motion_pixel_count: Minimum active pixels inside polygon to trigger alarm (default: 100).

    Returns:
        bool: True if motion inside polygon exceeds motion_pixel_count, else False.
    """
    global motion_pixels
    if not polygon or len(polygon) < 3:
        motion_pixels = 0
        return False

    # 1. Create binary polygon mask
    poly_mask = np.zeros_like(cframe, dtype=np.uint8)
    np_poly = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(poly_mask, [np_poly], 255)

    # 2. Compute absolute difference between frames
    diff = cv2.absdiff(cframe, rframe)

    # 3. Binary threshold
    _, motion_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # 4. Strictly isolate motion inside the polygon (outside pixels are zeroed)
    motion_inside_polygon = cv2.bitwise_and(motion_mask, motion_mask, mask=poly_mask)

    # 5. Count non-zero motion pixels inside polygon
    motion_pixels = cv2.countNonZero(motion_inside_polygon)

    return motion_pixels > motion_pixel_count


def fps_counts(frame, persons, ppe_results, prev_time, frame_height=480):
    """Compute FPS and overlay count statistics."""
    new_time = time.time()
    fps = 1 / (new_time - prev_time) if prev_time else 0
    prev_time = new_time
    
    total_persons = len(persons)
    total_violations = sum(1 for res in ppe_results if res[1] != "SAFE")

    cv2.putText(frame, f"FPS:{fps:.1f}", (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 150), 2)
    cv2.putText(frame, f"Persons:{total_persons}", (10, frame_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 150), 2)
    cv2.putText(frame, f"PPE Violations:{total_violations}", (10, frame_height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 150), 2)
    return frame, prev_time


def iou(boxA, boxB):
    """Compute Intersection over Union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = boxA 
    bx1, by1, bx2, by2 = boxB 

    xi1 = max(ax1, bx1)
    yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2)
    yi2 = min(ay2, by2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    boxA_area = (ax2 - ax1) * (ay2 - ay1)
    boxB_area = (bx2 - bx1) * (by2 - by1)
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def is_inside(person_box, object_box):
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = object_box
    ix1 = max(px1, ox1)
    iy1 = max(py1, oy1)
    ix2 = min(px2, ox2)
    iy2 = min(py2, oy2)
    return ix1 < ix2 and iy1 < iy2


def select_roi_interactive(source=None, firstframe=None):
    """Open an interactive OpenCV window to draw a multi-point restricted polygon zone.

    Interaction:
      - Left Mouse Click: Add polygon vertex
      - Mouse Move: Real-time preview line to cursor
      - Press 'S' or Enter: Confirm and save polygon (minimum 3 points required)
      - Press 'U' or Backspace: Undo last point
      - Press 'R': Reset all points
      - Press ESC or 'Q': Cancel (returns None)

    Args:
        source: Video file path (str) or camera index (int) if firstframe not provided.
        firstframe: BGR numpy image frame if already available.

    Returns:
        list of (x, y) tuples if confirmed, or None if cancelled.
    """
    cap = None
    if firstframe is None:
        if source is None:
            return None
        video_src = int(source) if (isinstance(source, int) or (isinstance(source, str) and str(source).isdigit())) else source
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            print(f"Error opening source for ROI selection: {source}")
            return None
        ret, firstframe = cap.read()
        if not ret or firstframe is None:
            print(f"Error reading frame for ROI selection: {source}")
            cap.release()
            return None

    frame_h, frame_w = firstframe.shape[:2]
    points = []
    current_mouse = None
    confirmed_polygon = None

    def render_canvas():
        canvas = firstframe.copy()

        # Instruction Header Banner
        cv2.rectangle(canvas, (0, 0), (frame_w, 38), (15, 23, 42), -1)
        pt_count = len(points)
        status_text = f"Points: {pt_count}" if pt_count < 3 else f"Points: {pt_count} (Ready to Save)"
        cv2.putText(
            canvas,
            f"Click to add Polygon points | 'S'/Enter: SAVE | 'U': UNDO | 'R': RESET | ESC: CANCEL  [{status_text}]",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        # Translucent fill if >= 3 points
        if len(points) >= 3:
            overlay = canvas.copy()
            np_pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [np_pts], (139, 92, 246))  # purple translucent fill
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            cv2.polylines(canvas, [np_pts], isClosed=True, color=(139, 92, 246), thickness=2)

        # Connect consecutive lines
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], (0, 255, 255), 2)

        # Preview line to current mouse position
        if points and current_mouse is not None:
            cv2.line(canvas, points[-1], current_mouse, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw point markers
        for idx, (px, py) in enumerate(points):
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1)
            cv2.putText(canvas, str(idx + 1), (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return canvas

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_mouse, points
        if event == cv2.EVENT_MOUSEMOVE:
            current_mouse = (x, y)
            cv2.imshow(window_name, render_canvas())
        elif event == cv2.EVENT_LBUTTONDOWN:
            clamped_x = max(0, min(frame_w, x))
            clamped_y = max(0, min(frame_h, y))
            points.append((clamped_x, clamped_y))
            cv2.imshow(window_name, render_canvas())

    window_name = "Define Restricted Polygon Zone (Click to Add Points)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, render_canvas())

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('s'), ord('S'), 13):  # 's' or Enter
            if len(points) >= 3:
                val_poly = validate_polygon(points, frame_w, frame_h)
                if val_poly:
                    confirmed_polygon = val_poly
                    print(f"Restricted Polygon confirmed ({len(confirmed_polygon)} points): {confirmed_polygon}")
                    break
                else:
                    print("Invalid polygon (zero area or degenerate). Add more points or reset.")
            else:
                print("Cannot save: Polygon requires at least 3 points.")
        elif key in (ord('u'), ord('U'), 8):  # 'u' or Backspace (Undo)
            if points:
                points.pop()
                cv2.imshow(window_name, render_canvas())
        elif key in (ord('r'), ord('R')):  # Reset
            points = []
            cv2.imshow(window_name, render_canvas())
        elif key in (27, ord('q'), ord('Q')):  # ESC or 'q' (Cancel)
            print("Polygon ROI selection cancelled by user.")
            confirmed_polygon = None
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()

    return confirmed_polygon


def run_detection(
    source=None,
    headless=False,
    no_roi=False,
    roi=None,
    output=None,
    no_save_video=False,
    stop_event=None,
    model_path=None,
    required_ppe=None,
    session_dir=None,
    frame_callback=None,
    show_window=None,
    max_frames=None
):
    """Execute the full end-to-end PPE Compliance and Polygon Restricted Zone detection pipeline.

    Args:
        source:        Video file path (str) or camera index (int)
        headless:      If True, skips GUI window rendering (no cv2.imshow)
        no_roi:        If True, disables restricted zone ROI selection
        roi:           Optional polygon points [(x1,y1), (x2,y2)...] or string
        output:        Custom output video path
        no_save_video: If True, skips saving annotated video
        stop_event:    threading.Event or callable to interrupt execution
        model_path:    Custom weights path (defaults to models/yolo8m.pt)
        required_ppe:  Set, list, or comma-separated string of required PPE classes
        session_dir:   Optional custom session directory
        frame_callback: Callable receiving per-frame telemetry dict
        show_window:   If True, renders cv2.imshow window. If False, suppresses window. Defaults to (not headless).

    Returns:
        dict: Summary statistics of the detection session
    """
    global motion_pixels
    base_dir = os.getcwd()
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "yolo8m.pt")

    if show_window is None:
        show_window = not headless

    # 1. Load and validate model & resolve required PPE
    model, class_names = validate_sh17_model(model_path)
    active_required_ppe = parse_required_ppe(required_ppe)
    print(f"Active compliance policy (REQUIRED_PPE): {sorted(list(active_required_ppe))}")

    # 2. Resolve video source
    if source is None:
        print("FATAL: No video source specified. Provide --source <path> or a webcam index.")
        sys.exit(1)
    is_webcam = False
    if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
        video_source = int(source)
        is_webcam = True
        source_name = f"webcam_{video_source}"
    else:
        video_source = os.path.abspath(source) if not os.path.isabs(source) else source
        if not os.path.exists(video_source):
            rel_path = os.path.join(base_dir, source)
            if os.path.exists(rel_path):
                video_source = rel_path
            else:
                print(f"FATAL: Source video file not found: {source}")
                sys.exit(1)
        source_name = os.path.splitext(os.path.basename(video_source))[0]

    # 3. Setup isolated session directory
    if session_dir is not None:
        session_path = os.path.abspath(session_dir)
        session_name = os.path.basename(session_path)
    else:
        if is_webcam:
            ts = time.strftime('%Y%m%d_%H%M%S')
            session_name = f"webcam_{ts}"
        else:
            session_name = source_name
        session_path = os.path.join(base_dir, "output", session_name)

    snapshots_dir = os.path.join(session_path, "snapshots")
    annotated_dir = os.path.join(session_path, "annotated")
    intrusion_snapshots_dir = os.path.join(session_path, "intrusion_snapshots")
    violation_log_file = os.path.join(session_path, "violation_log.csv")
    intrusion_log_file = os.path.join(session_path, "intrusion_log.txt")

    os.makedirs(session_path, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(intrusion_snapshots_dir, exist_ok=True)

    # Initialize session CSV with dynamic required PPE headers
    sorted_req = sorted(list(active_required_ppe))
    req_headers = [PPE_COLUMN_TITLES.get(p, p.title()) for p in sorted_req]
    csv_headers = ["Date", "Time", "Person", "Status"] + req_headers + ["Missing PPE", "Event", "Snapshot"]
    with open(violation_log_file, 'w') as _f:
        _f.write(",".join(csv_headers) + "\n")

    print(f"Session Output Directory: {session_path}")
    print(f"Session CSV Log: {violation_log_file}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"FATAL: Error opening video source: {video_source}")
        sys.exit(1)

    ret, firstframe = cap.read()
    if not ret or firstframe is None:
        print(f"FATAL: Error reading first frame from video source: {video_source}")
        cap.release()
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(raw_fps) if raw_fps > 0 else 25
    print(f"Video Source: {video_source} | Resolution: {frame_width} x {frame_height} | FPS: {fps}")

    # 4. Setup VideoWriter
    out_writer = None
    out_video_path = None
    if not no_save_video and not is_webcam:
        if output:
            out_video_path = output
            os.makedirs(os.path.dirname(os.path.abspath(out_video_path)), exist_ok=True)
        else:
            out_video_path = os.path.join(annotated_dir, f"{source_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))
        if out_writer.isOpened():
            print(f"Annotated video output: {out_video_path}")
        else:
            print(f"WARNING: VideoWriter failed for {out_video_path}")
            out_writer = None

    # 5. Handle Restricted Polygon
    restricted_polygon = None
    if roi:
        restricted_polygon = validate_polygon(roi, frame_width, frame_height)
        if restricted_polygon:
            print(f"Using configured Polygon ROI ({len(restricted_polygon)} points): {restricted_polygon}")
        else:
            print(f"WARNING: Invalid polygon input ({roi}). Running without restricted zone.")
    elif no_roi or headless or not show_window:
        restricted_polygon = None
        if headless:
            print("Headless mode: ROI selection skipped.")
        elif not show_window:
            print("Window display disabled: Interactive ROI selection skipped.")
    else:
        # Interactive Polygon Selector prompt if not headless and no explicit ROI flag passed
        selected = select_roi_interactive(firstframe=firstframe)
        if selected is not None:
            restricted_polygon = selected
        else:
            restricted_polygon = None

    if not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 6. Initialize tracker and session state
    tracker = DeepSort(max_age=70, n_init=1)
    grayframe1 = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    last_save_time = 0
    intrusion_cooldown_seconds = 5
    prev_time = None
    frame_count = 0
    total_violations_logged = 0
    total_intrusions_logged = 0
    session_event_counter = 0

    # Clear in-memory worker state for a fresh session
    worker_state.clear()

    print(f"Starting detection processing (Headless={headless}, Source={video_source}, Polygon_ROI={bool(restricted_polygon)}, Required_PPE={sorted(list(active_required_ppe))})...")

    try:
        while True:
            if stop_event is not None:
                if (callable(stop_event) and stop_event()) or (hasattr(stop_event, 'is_set') and stop_event.is_set()):
                    print("Detection interrupted by stop signal.")
                    break

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"End of video stream. Processed {frame_count} frames.")
                break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                print(f"Processing complete. Reached max-frames limit ({max_frames}). Processed {frame_count} frames.")
                break
            results = model(frame, stream=True, verbose=False)
            grayframe2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Polygon Intrusion Detection & Visualization
            is_intrusive = False
            if restricted_polygon:
                np_poly = np.array(restricted_polygon, dtype=np.int32)
                is_intrusive = intrusion(grayframe2, grayframe1, restricted_polygon, thresh=130, motion_pixel_count=100)
                if is_intrusive:
                    color = (0, 0, 255)  # Red
                    status = 'INTRUSION DETECTED'
                    current_time = time.time()
                    if current_time - last_save_time > intrusion_cooldown_seconds:
                        timestamp = time.strftime('%Y%m%d_%H%M%S')
                        image_path = os.path.join(intrusion_snapshots_dir, f"intrusion_{timestamp}.jpg")
                        cv2.imwrite(image_path, frame)
                        with open(intrusion_log_file, 'a') as log:
                            log.write(f'Intrusion detected at {time.ctime()} | Polygon={restricted_polygon}\n')
                        print(f"Restricted polygon intrusion detected and saved at {timestamp}")
                        last_save_time = current_time
                        total_intrusions_logged += 1
                else:
                    color = (0, 255, 0)  # Green
                    status = 'RESTRICTED ZONE (CLEAR)'

                # Translucent filled polygon overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [np_poly], color)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

                # Boundary contour
                cv2.polylines(frame, [np_poly], isClosed=True, color=color, thickness=2)

                # Anchor status text at top-most point of polygon
                top_pt = min(restricted_polygon, key=lambda pt: pt[1])
                cv2.putText(frame, status, (top_pt[0], max(20, top_pt[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Grayscale visualization
                gray_visual = cv2.cvtColor(grayframe2, cv2.COLOR_GRAY2BGR)
                cv2.polylines(gray_visual, [np_poly], isClosed=True, color=color, thickness=2)
            else:
                gray_visual = cv2.cvtColor(grayframe2, cv2.COLOR_GRAY2BGR)

            # Collect detections
            persons = []
            ppe_detections = []
            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                coords = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for i in range(len(coords)):
                    x1, y1, x2, y2 = [int(v) for v in coords[i]]
                    conf_val = float(conf[i])
                    if conf_val < 0.4:
                        continue
                    conf_dis = math.ceil(conf_val * 100) / 100
                    cls = int(classes[i])
                    label = class_names[cls]
                    w, h = x2 - x1, y2 - y1

                    if label == 'person':
                        detections.append(([x1, y1, w, h], conf_val, cls))
                        persons.append([x1, y1, x2, y2])
                        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                        cv2.putText(frame, f"{label}:{conf_dis}", (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    elif label in SH17_PPE_CLASSES:
                        ppe_detections.append(([x1, y1, x2, y2], label, conf_val))
                        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                        cv2.putText(frame, f"{label}:{conf_dis}", (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Spatial assignment & compliance classification evaluated against active_required_ppe
            person_ppe_map = assign_ppe_to_persons(persons, ppe_detections)
            ppe_results = []
            for idx, person in enumerate(persons):
                detected_ppe = person_ppe_map.get(idx, set())
                status, color, missing_items = classify_compliance(detected_ppe, required_ppe=active_required_ppe)
                ppe_results.append((person, status, color, missing_items, detected_ppe))

            # DeepSORT tracking
            tracks = tracker.update_tracks(detections, frame=frame)
            active_workers_telemetry = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                track_class = track.det_class
                label = class_names[int(track_class)] if track_class is not None else "unknown"
                tx1, ty1, tx2, ty2 = [int(v) for v in track.to_ltrb()]

                if label == 'person':
                    track_box = [tx1, ty1, tx2, ty2]
                    best_iou = 0
                    best_match = None
                    for (pbox, status, color, missing_items, detected_ppe) in ppe_results:
                        iou_val = iou(pbox, track_box)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_match = (status, color, missing_items, detected_ppe)

                    if best_iou > 0.3:
                        _, _, _, detected_ppe = best_match
                        current_time = time.time()
                        status, color, missing_items = smooth_compliance(track_id, detected_ppe, required_ppe=active_required_ppe)

                        required_detected = active_required_ppe.intersection(detected_ppe)
                        required_missing = set(missing_items)
                        other_detected = detected_ppe - active_required_ppe

                        # 1. DRAW worker compliance card & bounding box onto frame FIRST
                        draw_worker_compliance_panel(
                            frame, tx1, ty1, tx2, ty2,
                            track_id=track_id,
                            status=status,
                            color=color,
                            required_detected=required_detected,
                            required_missing=required_missing,
                            other_detected=other_detected
                        )

                        # 2. LOG event-based violation (frame now contains the compliance card and bounding box!)
                        did_log, rec, session_event_counter = log_violation(
                            track_id=track_id,
                            status=status,
                            missing_items=missing_items,
                            track_box=track_box,
                            frame=frame,
                            save_dir=snapshots_dir,
                            log_file=violation_log_file,
                            active_required_ppe=active_required_ppe,
                            detected_ppe=detected_ppe,
                            event_counter=session_event_counter,
                            current_time=current_time
                        )
                        if did_log:
                            total_violations_logged += 1

                        active_workers_telemetry.append({
                            'track_id': track_id,
                            'bbox': track_box,
                            'status': status,
                            'color': color,
                            'detected_ppe': sorted(list(detected_ppe)),
                            'missing_ppe': sorted(list(missing_items)),
                            'required_detected': sorted(list(required_detected)),
                            'other_detected': sorted(list(other_detected)),
                        })

            # Stale track cleanup
            current_active_ids = {track.track_id for track in tracks if track.is_confirmed() and track.time_since_update == 0}
            stale = []
            for tid in worker_state:
                if tid not in current_active_ids:
                    worker_state[tid]['grace_counter'] += 1
                    if worker_state[tid]['grace_counter'] > TRACK_GRACE_FRAMES:
                        stale.append(tid)
            for tid in stale:
                del worker_state[tid]

            # Display FPS overlay
            frame, prev_time = fps_counts(frame, persons, ppe_results, prev_time, frame_height)

            # Optional GUI / Telemetry Callback
            if frame_callback is not None:
                try:
                    active_safe = sum(1 for w in active_workers_telemetry if w['status'] == 'SAFE')
                    active_partial = sum(1 for w in active_workers_telemetry if w['status'] == 'PARTIAL PPE')
                    active_unsafe = sum(1 for w in active_workers_telemetry if w['status'] == 'UNSAFE')

                    intrusion_info = {
                        'is_intrusion': bool(is_intrusive) if restricted_polygon else False,
                        'status': 'INTRUSION DETECTED' if (restricted_polygon and is_intrusive) else ('RESTRICTED ZONE (CLEAR)' if restricted_polygon else 'NO RESTRICTED ZONE'),
                        'motion_pixels': int(motion_pixels) if restricted_polygon else 0,
                        'polygon': restricted_polygon
                    }

                    frame_callback({
                        'monitoring_status': 'RUNNING',
                        'source': video_source,
                        'session_dir': session_path,
                        'restricted_polygon': restricted_polygon,
                        'frame_count': frame_count,
                        'fps': float(fps),
                        'current_timestamp': time.time(),
                        'active_workers': active_workers_telemetry,
                        'worker_counts': {
                            'total': len(active_workers_telemetry),
                            'safe': active_safe,
                            'partial': active_partial,
                            'unsafe': active_unsafe
                        },
                        'intrusion_state': intrusion_info,
                        'frame': frame
                    })
                except Exception:
                    pass

            # Write annotated frame
            if out_writer is not None:
                out_writer.write(frame)

            # GUI / Window Display
            if show_window:
                scale = 0.8
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                bgr_resized = cv2.resize(frame, (new_width, new_height))
                gray_resized = cv2.resize(gray_visual, (new_width, new_height))
                stacked = np.hstack((bgr_resized, gray_resized))
                cv2.imshow("BGR and GRAYSCALE", stacked)
                
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    print("User terminated detection playback.")
                    break
                # Check if window was closed via 'X' button
                if cv2.getWindowProperty("BGR and GRAYSCALE", cv2.WND_PROP_VISIBLE) < 1:
                    print("Detection window closed by user.")
                    break
            else:
                if frame_count % 10 == 0 or frame_count == 1:
                    if headless:
                        print(f"[Headless] Processed frame {frame_count}...")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
            print(f"Annotated video saved successfully to: {out_video_path}")
            # Ensure backward compatibility by also providing video in legacy output/annotated/
            legacy_dir = os.path.join(base_dir, 'output', 'annotated')
            os.makedirs(legacy_dir, exist_ok=True)
            legacy_video_path = os.path.join(legacy_dir, f"{source_name}_annotated.mp4")
            if os.path.abspath(out_video_path) != os.path.abspath(legacy_video_path):
                import shutil
                try:
                    shutil.copyfile(out_video_path, legacy_video_path)
                except Exception:
                    pass
        if show_window:
            cv2.destroyAllWindows()
        print(f"Processing complete. Total frames processed: {frame_count}")

        if frame_callback is not None:
            try:
                frame_callback({
                    'monitoring_status': 'STOPPED' if (stop_event and ((callable(stop_event) and stop_event()) or (hasattr(stop_event, 'is_set') and stop_event.is_set()))) else 'COMPLETED',
                    'source': video_source,
                    'session_dir': session_path,
                    'restricted_polygon': restricted_polygon,
                    'frame_count': frame_count,
                    'fps': 0.0,
                    'current_timestamp': time.time(),
                    'active_workers': [],
                    'worker_counts': {'total': 0, 'safe': 0, 'partial': 0, 'unsafe': 0},
                    'intrusion_state': {
                        'is_intrusion': False,
                        'status': 'CLEAR',
                        'motion_pixels': 0,
                        'polygon': restricted_polygon
                    },
                    'frame': None
                })
            except Exception:
                pass

    return {
        'status': 'completed',
        'frames_processed': frame_count,
        'violations_logged': total_violations_logged,
        'intrusions_logged': total_intrusions_logged,
        'output_video': out_video_path,
        'session_dir': session_path,
        'session_name': session_name,
        'csv_path': violation_log_file
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Compliance & Polygon Restricted Zone Monitoring System (SH17 YOLOv8 + DeepSORT)")
    parser.add_argument("--source", default=None, help="Video file path or webcam index (required)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI windows")
    parser.add_argument("--no-roi", action="store_true", help="Disable restricted zone ROI selection")
    parser.add_argument("--roi", "--polygon", type=str, default=None, help="Polygon points as 'x1,y1;x2,y2;x3,y3...' (e.g. '100,100;300,80;500,200;450,400')")
    parser.add_argument("--required-ppe", type=str, default=None, help="Comma-separated list of required PPE classes (e.g. 'helmet,safety-vest,gloves')")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated video output")
    parser.add_argument("--no-save-video", action="store_true", help="Disable saving annotated video file")
    parser.add_argument("--session-dir", type=str, default=None, help="Custom output session directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this many frames (default: process entire video)")
    args, _ = parser.parse_known_args()

    run_detection(
        source=args.source,
        headless=args.headless,
        no_roi=args.no_roi,
        roi=args.roi,
        output=args.output,
        no_save_video=args.no_save_video,
        required_ppe=args.required_ppe,
        session_dir=args.session_dir,
        max_frames=args.max_frames,
    )

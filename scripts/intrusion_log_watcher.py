#!/usr/bin/env python
"""
intrusion_log_watcher.py — Real-time Intrusion Log Watcher and Event Filter with GIF Generation.

Monitors output/<session>/intrusion_log.txt in real time (tailing newly appended lines),
groups raw detections occurring within a configurable gap (default: 10s) into continuous
intrusion events, compiles evidence snapshots into chronological animated GIFs in
output/<session>/intrusion_gifs/, and updates output/<session>/intrusion_log.csv in place.

Supports two operational modes:
1. Direct File Mode:
   python scripts/intrusion_log_watcher.py --input output/<session>/intrusion_log.txt
2. Auto-Discovery Mode:
   python scripts/intrusion_log_watcher.py --watch-dir output
   (Monitors output/ for the active session and automatically attaches to its log)

This script is completely decoupled from main.py and the detection pipeline.
All original JPG snapshots in intrusion_snapshots/ are preserved untouched.
"""

import os
import sys
import time
import re
import csv
import argparse
from datetime import datetime, timedelta
from PIL import Image

INTRUSION_EVENT_GAP_SECONDS = 10
GIF_FRAME_DURATION_MS = 800

CSV_HEADERS = [
    "Event ID",
    "Start Time",
    "Last Detection",
    "Duration (seconds)",
    "Detection Count",
    "Polygon",
    "GIF"
]


def parse_intrusion_line(line):
    """Parse a single raw line from intrusion_log.txt.

    Format expected:
        Intrusion detected at <ctime> | Polygon=<polygon_str>
    Example:
        Intrusion detected at Wed Sep  3 23:44:23 2026 | Polygon=[(100, 100), (400, 100), (400, 400), (100, 400)]

    Returns:
        (datetime_obj, polygon_str) or (None, None) if malformed.
    """
    if not line or not isinstance(line, str):
        return None, None

    line = line.strip()
    if not line:
        return None, None

    pattern = r"^Intrusion detected at\s+(.+?)\s*\|\s*Polygon=(.+)$"
    match = re.match(pattern, line, re.IGNORECASE)
    if not match:
        return None, None

    raw_time_str = match.group(1).strip()
    polygon_str = match.group(2).strip()

    # Normalize whitespace (handles single-digit day padding in ctime, e.g. "Sep  3" -> "Sep 3")
    norm_time_str = re.sub(r"\s+", " ", raw_time_str)

    try:
        dt = datetime.strptime(norm_time_str, "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None, None

    return dt, polygon_str


def find_snapshot(dt, session_dir):
    """Find corresponding snapshot in session_dir/intrusion_snapshots.

    main.py writes snapshot as: intrusion_YYYYMMDD_HHMMSS.jpg
    Because time.strftime and time.ctime may occur across a 1-second clock tick,
    we check dt, dt - 1s, and dt + 1s.

    Returns:
        Relative path "intrusion_snapshots/intrusion_YYYYMMDD_HHMMSS.jpg" or None.
    """
    if not session_dir:
        return None

    snapshots_dir = os.path.join(session_dir, "intrusion_snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    candidates = [
        dt,
        dt - timedelta(seconds=1),
        dt + timedelta(seconds=1)
    ]

    for cand in candidates:
        fname = f"intrusion_{cand.strftime('%Y%m%d_%H%M%S')}.jpg"
        full_path = os.path.join(snapshots_dir, fname)
        if os.path.isfile(full_path):
            return f"intrusion_snapshots/{fname}"

    return None


class IntrusionEvent:
    """Represents a grouped continuous intrusion event."""

    def __init__(self, event_id, start_time, polygon, snapshot=None):
        self.event_id = event_id
        self.start_time = start_time
        self.last_detection = start_time
        self.duration = 0
        self.detection_count = 1
        self.polygon = polygon
        self.snapshots = [snapshot] if snapshot else []
        self.gif_path = ""

    def update(self, dt, snapshot=None):
        """Update existing event with a new raw detection."""
        self.last_detection = dt
        self.duration = max(0, int((self.last_detection - self.start_time).total_seconds()))
        self.detection_count += 1
        if snapshot and snapshot not in self.snapshots:
            self.snapshots.append(snapshot)

    def create_or_update_gif(self, session_dir, frame_duration_ms=GIF_FRAME_DURATION_MS):
        """Create or update animated GIF in session_dir/intrusion_gifs/ from chronological snapshots.

        All original JPG files in intrusion_snapshots/ are preserved untouched.
        Returns relative path to the GIF, or empty string if no snapshots are available.
        """
        if not session_dir or not self.snapshots:
            self.gif_path = ""
            return ""

        # Sort snapshots chronologically (intrusion_YYYYMMDD_HHMMSS.jpg sorts naturally)
        sorted_snaps = sorted(set(self.snapshots))
        valid_full_paths = []
        for s in sorted_snaps:
            full_p = os.path.join(session_dir, s)
            if os.path.isfile(full_p):
                valid_full_paths.append(full_p)

        if not valid_full_paths:
            self.gif_path = ""
            return ""

        gifs_dir = os.path.join(session_dir, "intrusion_gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        gif_filename = f"intrusion_event_{self.event_id:03d}.gif"
        gif_full_path = os.path.join(gifs_dir, gif_filename)
        self.gif_path = f"intrusion_gifs/{gif_filename}"

        images = []
        for path in valid_full_paths:
            try:
                with Image.open(path) as img:
                    # Convert to RGB and copy into memory so original file handle is released
                    images.append(img.convert("RGB").copy())
            except Exception as e:
                print(f"WARNING: Could not open snapshot {path} for GIF: {e}")

        if not images:
            self.gif_path = ""
            return ""

        try:
            first_frame = images[0]
            first_frame.save(
                gif_full_path,
                save_all=True,
                append_images=images[1:],
                duration=frame_duration_ms,
                loop=0
            )
        except Exception as e:
            print(f"ERROR: Failed to save GIF {gif_full_path}: {e}")
        finally:
            for img in images:
                img.close()

        return self.gif_path

    def to_csv_row(self):
        """Return row matching CSV_HEADERS."""
        return [
            str(self.event_id),
            self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.last_detection.strftime("%Y-%m-%d %H:%M:%S"),
            str(self.duration),
            str(self.detection_count),
            self.polygon,
            self.gif_path
        ]


class IntrusionAggregator:
    """Aggregates raw intrusion detections into grouped events using a gap threshold."""

    def __init__(self, gap_seconds=INTRUSION_EVENT_GAP_SECONDS, gif_duration_ms=GIF_FRAME_DURATION_MS):
        self.gap_seconds = gap_seconds
        self.gif_duration_ms = gif_duration_ms
        self.events = []

    def process_detection(self, dt, polygon, session_dir=None):
        """Process a single parsed detection and create/update its GIF.

        Returns:
            (event, is_new: bool)
        """
        snapshot = find_snapshot(dt, session_dir) if session_dir else None

        if not self.events:
            new_event = IntrusionEvent(event_id=1, start_time=dt, polygon=polygon, snapshot=snapshot)
            if session_dir:
                new_event.create_or_update_gif(session_dir, self.gif_duration_ms)
            self.events.append(new_event)
            return new_event, True

        last_event = self.events[-1]
        gap = (dt - last_event.last_detection).total_seconds()

        # Grouping rule: gap <= gap_seconds belongs to same event
        if 0 <= gap <= self.gap_seconds:
            last_event.update(dt, snapshot)
            if session_dir:
                last_event.create_or_update_gif(session_dir, self.gif_duration_ms)
            return last_event, False
        else:
            new_id = len(self.events) + 1
            new_event = IntrusionEvent(event_id=new_id, start_time=dt, polygon=polygon, snapshot=snapshot)
            if session_dir:
                new_event.create_or_update_gif(session_dir, self.gif_duration_ms)
            self.events.append(new_event)
            return new_event, True

    def write_csv(self, csv_path):
        """Write all current events to CSV atomically."""
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for event in self.events:
                writer.writerow(event.to_csv_row())
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def find_active_session_log(watch_dir):
    """Find the most recently active session directory in watch_dir and return (session_name, intrusion_log_path)."""
    if not os.path.isdir(watch_dir):
        return None, None

    candidates = []
    try:
        entries = list(os.scandir(watch_dir))
    except OSError:
        return None, None

    for entry in entries:
        if entry.is_dir() and entry.name != "annotated":
            v_csv = os.path.join(entry.path, "violation_log.csv")
            i_txt = os.path.join(entry.path, "intrusion_log.txt")
            times = [entry.stat().st_mtime]
            if os.path.isfile(v_csv):
                times.append(os.path.getmtime(v_csv))
            if os.path.isfile(i_txt):
                times.append(os.path.getmtime(i_txt))
            candidates.append((max(times), entry.name, entry.path))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, session_name, session_path = candidates[0]
    return session_name, os.path.join(session_path, "intrusion_log.txt")


def watch_log_file(
    input_txt,
    output_csv=None,
    gap_seconds=INTRUSION_EVENT_GAP_SECONDS,
    poll_interval=0.5,
    stop_event=None,
    once=False,
    no_wait=False,
    switch_checker=None,
    gif_duration_ms=GIF_FRAME_DURATION_MS
):
    """Continuously monitor input_txt and update output_csv in real time.

    Args:
        input_txt: Path to raw intrusion_log.txt.
        output_csv: Path to output intrusion_log.csv (defaults to same folder as input_txt).
        gap_seconds: Max seconds between detections to be considered same event.
        poll_interval: Sleep duration when at EOF.
        stop_event: Optional threading.Event to signal termination.
        once: If True, processes until EOF and returns immediately (batch mode).
        no_wait: If True and input_txt does not exist, returns immediately.
        switch_checker: Optional callable returning True if a newer active session should be switched to.
        gif_duration_ms: Frame duration for generated GIFs in milliseconds.

    Returns:
        IntrusionAggregator instance with final events.
    """
    input_txt = os.path.abspath(input_txt)
    session_dir = os.path.dirname(input_txt)

    if output_csv is None:
        output_csv = os.path.join(session_dir, "intrusion_log.csv")
    else:
        output_csv = os.path.abspath(output_csv)

    print(f"Watching: {input_txt}")
    print(f"Writing:  {output_csv}")
    print(f"GIF dir:  {os.path.join(session_dir, 'intrusion_gifs')}")
    print(f"Grouping gap: {gap_seconds}s (INTRUSION_EVENT_GAP_SECONDS)")

    # Wait for file if it doesn't exist
    if not os.path.exists(input_txt):
        if no_wait:
            print(f"File not found: {input_txt}")
            return IntrusionAggregator(gap_seconds=gap_seconds, gif_duration_ms=gif_duration_ms)

        print(f"Waiting for {input_txt} to be created... (Press Ctrl+C to exit)")
        while not os.path.exists(input_txt):
            if stop_event and stop_event.is_set():
                return IntrusionAggregator(gap_seconds=gap_seconds, gif_duration_ms=gif_duration_ms)
            if switch_checker and switch_checker():
                return IntrusionAggregator(gap_seconds=gap_seconds, gif_duration_ms=gif_duration_ms)
            time.sleep(poll_interval)

    aggregator = IntrusionAggregator(gap_seconds=gap_seconds, gif_duration_ms=gif_duration_ms)

    # Initial CSV write with header if not exists
    if not os.path.exists(output_csv):
        aggregator.write_csv(output_csv)

    try:
        with open(input_txt, "r", encoding="utf-8", errors="replace") as f:
            while True:
                if stop_event and stop_event.is_set():
                    break

                if switch_checker and switch_checker():
                    print("\nNewer active session detected. Switching...")
                    break

                curr_pos = f.tell()
                line = f.readline()

                if line:
                    # Check if line was completely written (ends with newline)
                    if not line.endswith("\n"):
                        # Writer may be in the middle of writing this line; back up and wait
                        f.seek(curr_pos)
                        time.sleep(poll_interval)
                        continue

                    dt, polygon = parse_intrusion_line(line)
                    if dt is None:
                        # Malformed or unrecognized line
                        clean_line = line.strip()
                        if clean_line:
                            print(f"WARNING: Skipping malformed line: {clean_line}")
                        continue

                    event, is_new = aggregator.process_detection(dt, polygon, session_dir)
                    gif_info = f" -> {event.gif_path}" if event.gif_path else ""
                    if is_new:
                        print(f"Event {event.event_id} started ({dt.strftime('%Y-%m-%d %H:%M:%S')}){gif_info}")
                    else:
                        print(f"Event {event.event_id} updated (Count: {event.detection_count}, Duration: {event.duration}s){gif_info}")

                    aggregator.write_csv(output_csv)

                else:
                    # EOF reached
                    if once:
                        break

                    # Check if file was truncated by an external process
                    try:
                        if os.path.getsize(input_txt) < curr_pos:
                            print("Log file was truncated; rewinding to start.")
                            f.seek(0)
                    except OSError:
                        pass

                    time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nWatcher stopped by user (Ctrl+C).")

    print(f"Finished watching session. Total grouped events: {len(aggregator.events)}")
    return aggregator


def watch_sessions_directory(
    watch_dir="output",
    output_csv=None,
    gap_seconds=INTRUSION_EVENT_GAP_SECONDS,
    poll_interval=0.5,
    stop_event=None,
    once=False,
    gif_duration_ms=GIF_FRAME_DURATION_MS
):
    """Monitor a directory containing multiple sessions (e.g. output/), auto-detecting the active session."""
    watch_dir = os.path.abspath(watch_dir)
    print("=======================================================")
    print("       PPE REAL-TIME INTRUSION LOG WATCHER")
    print(f"  Auto-detecting active session in: {watch_dir}")
    print(f"  Grouping gap: {gap_seconds}s (INTRUSION_EVENT_GAP_SECONDS)")
    print("=======================================================")

    current_session = None

    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            session_name, log_path = find_active_session_log(watch_dir)
            if session_name is None:
                time.sleep(poll_interval)
                continue

            current_session = session_name
            session_dir = os.path.dirname(log_path)
            target_csv = output_csv if output_csv else os.path.join(session_dir, "intrusion_log.csv")
            print(f"\n[Active Session Detected] {session_name}")

            def check_session_switch():
                newest, _ = find_active_session_log(watch_dir)
                return newest is not None and newest != current_session

            aggregator = watch_log_file(
                input_txt=log_path,
                output_csv=target_csv,
                gap_seconds=gap_seconds,
                poll_interval=poll_interval,
                stop_event=stop_event,
                once=once,
                switch_checker=check_session_switch,
                gif_duration_ms=gif_duration_ms
            )

            if once or (stop_event and stop_event.is_set()):
                return aggregator

    except KeyboardInterrupt:
        print("\nWatcher stopped by user (Ctrl+C).")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Intrusion Log Watcher & Event Filter with GIF Generation (tail -f intrusion_log.txt -> intrusion_log.csv + intrusion_gifs/)"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to specific intrusion_log.txt (e.g. output/<session_name>/intrusion_log.txt)"
    )
    parser.add_argument(
        "--watch-dir", "-w",
        default=None,
        help="Directory containing sessions to auto-detect and monitor (default: output if --input is omitted)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output intrusion_log.csv (default: intrusion_log.csv in same folder as input)"
    )
    parser.add_argument(
        "--gap", "-g",
        type=int,
        default=INTRUSION_EVENT_GAP_SECONDS,
        help=f"Continuous intrusion grouping gap in seconds (default: {INTRUSION_EVENT_GAP_SECONDS})"
    )
    parser.add_argument(
        "--poll-interval", "-p",
        type=float,
        default=0.5,
        help="Polling interval in seconds when waiting for new lines (default: 0.5)"
    )
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=GIF_FRAME_DURATION_MS,
        help=f"Frame display duration for generated GIFs in milliseconds (default: {GIF_FRAME_DURATION_MS})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process current file to EOF and exit immediately (batch mode)"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait if input file does not exist (exit immediately)"
    )
    args = parser.parse_args()

    if args.input:
        watch_log_file(
            input_txt=args.input,
            output_csv=args.output,
            gap_seconds=args.gap,
            poll_interval=args.poll_interval,
            once=args.once,
            no_wait=args.no_wait,
            gif_duration_ms=args.gif_duration
        )
    else:
        watch_dir = args.watch_dir if args.watch_dir else "output"
        watch_sessions_directory(
            watch_dir=watch_dir,
            output_csv=args.output,
            gap_seconds=args.gap,
            poll_interval=args.poll_interval,
            once=args.once,
            gif_duration_ms=args.gif_duration
        )


if __name__ == "__main__":
    main()

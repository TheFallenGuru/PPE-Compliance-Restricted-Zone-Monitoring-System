import unittest
import os
import sys
import tempfile
import time
import threading
import csv
from datetime import datetime, timedelta

# Add project root and scripts directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import intrusion_log_watcher as watcher


class TestIntrusionLogWatcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.txt_path = os.path.join(self.temp_dir.name, "intrusion_log.txt")
        self.csv_path = os.path.join(self.temp_dir.name, "intrusion_log.csv")
        self.snapshots_dir = os.path.join(self.temp_dir.name, "intrusion_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Parse a valid intrusion line
    def test_1_parse_valid_intrusion_line(self):
        line = "Intrusion detected at Wed Sep  3 23:44:23 2026 | Polygon=[(100, 100), (400, 100), (400, 400), (100, 400)]\n"
        dt, poly = watcher.parse_intrusion_line(line)
        self.assertIsNotNone(dt)
        self.assertEqual(poly, "[(100, 100), (400, 100), (400, 400), (100, 400)]")
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 9)
        self.assertEqual(dt.day, 3)
        self.assertEqual(dt.hour, 23)
        self.assertEqual(dt.minute, 44)
        self.assertEqual(dt.second, 23)

    # 2. Parse timestamp correctly (including single-digit day and two-digit day)
    def test_2_parse_timestamp_formats(self):
        # Single digit day with double space from ctime
        line1 = "Intrusion detected at Mon Sep  7 11:02:18 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]"
        dt1, _ = watcher.parse_intrusion_line(line1)
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1, datetime(2026, 9, 7, 11, 2, 18))

        # Two digit day
        line2 = "Intrusion detected at Fri Sep 25 15:30:00 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]"
        dt2, _ = watcher.parse_intrusion_line(line2)
        self.assertIsNotNone(dt2)
        self.assertEqual(dt2, datetime(2026, 9, 25, 15, 30, 0))

    # 3. One detection -> one CSV event
    def test_3_one_detection_one_event(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        dt = datetime(2026, 9, 7, 12, 0, 0)
        event, is_new = agg.process_detection(dt, "[(10, 10), (20, 20), (10, 20)]")
        self.assertTrue(is_new)
        self.assertEqual(len(agg.events), 1)
        self.assertEqual(event.event_id, 1)
        self.assertEqual(event.detection_count, 1)
        self.assertEqual(event.duration, 0)

    # 4. Close detections (<= 10s) -> same event
    def test_4_close_detections_same_event(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        t2 = datetime(2026, 9, 7, 12, 0, 9)

        _, is_new0 = agg.process_detection(t0, poly)
        _, is_new1 = agg.process_detection(t1, poly)
        ev, is_new2 = agg.process_detection(t2, poly)

        self.assertTrue(is_new0)
        self.assertFalse(is_new1)
        self.assertFalse(is_new2)
        self.assertEqual(len(agg.events), 1)
        self.assertEqual(ev.detection_count, 3)
        self.assertEqual(ev.duration, 9)

    # 5. Gap > 10 seconds -> new event
    def test_5_gap_greater_than_10s_new_event(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        t2 = datetime(2026, 9, 7, 12, 0, 20)  # Gap = 15s > 10s

        agg.process_detection(t0, poly)
        agg.process_detection(t1, poly)
        ev2, is_new2 = agg.process_detection(t2, poly)

        self.assertTrue(is_new2)
        self.assertEqual(len(agg.events), 2)
        self.assertEqual(ev2.event_id, 2)
        self.assertEqual(ev2.detection_count, 1)
        self.assertEqual(ev2.duration, 0)

    # 6. Exactly 10 seconds -> same event
    def test_6_exact_10s_gap_same_event(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 10)  # Gap = exactly 10s

        agg.process_detection(t0, poly)
        ev, is_new = agg.process_detection(t1, poly)

        self.assertFalse(is_new)
        self.assertEqual(len(agg.events), 1)
        self.assertEqual(ev.detection_count, 2)
        self.assertEqual(ev.duration, 10)

    # 7. Duration calculation
    def test_7_duration_calculation(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        t2 = datetime(2026, 9, 7, 12, 0, 12)
        t3 = datetime(2026, 9, 7, 12, 0, 18)

        for t in [t0, t1, t2, t3]:
            agg.process_detection(t, poly)

        self.assertEqual(len(agg.events), 1)
        # Total duration from 12:00:00 to 12:00:18 is 18 seconds
        self.assertEqual(agg.events[0].duration, 18)

    # 8. Detection count
    def test_8_detection_count(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        for i in range(5):
            agg.process_detection(datetime(2026, 9, 7, 12, 0, i * 2), poly)
        self.assertEqual(agg.events[0].detection_count, 5)

    # 9. CSV updates when an existing event receives another detection
    def test_9_csv_updates_existing_row(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(100, 100), (400, 100), (400, 400), (100, 400)]"
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        agg.process_detection(t0, poly)
        agg.write_csv(self.csv_path)

        # Check CSV content at step 1
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 2)  # Header + 1 data row
        self.assertEqual(reader[1][0], "1")
        self.assertEqual(reader[1][3], "0")  # Duration
        self.assertEqual(reader[1][4], "1")  # Count

        # Add second detection belonging to same event
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        agg.process_detection(t1, poly)
        agg.write_csv(self.csv_path)

        # Check CSV content at step 2
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 2)  # Still exactly 1 data row, updated in place!
        self.assertEqual(reader[1][0], "1")
        self.assertEqual(reader[1][3], "5")  # Duration updated to 5
        self.assertEqual(reader[1][4], "2")  # Count updated to 2

    # 10. Multiple events
    def test_10_multiple_events(self):
        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        # Event 1: 12:00:00, 12:00:05 (duration 5s, count 2)
        agg.process_detection(datetime(2026, 9, 7, 12, 0, 0), poly)
        agg.process_detection(datetime(2026, 9, 7, 12, 0, 5), poly)

        # Event 2: 12:00:30 (gap = 25s, duration 0s, count 1)
        agg.process_detection(datetime(2026, 9, 7, 12, 0, 30), poly)

        # Event 3: 12:01:00, 12:01:06, 12:01:10 (gap = 30s, duration 10s, count 3)
        agg.process_detection(datetime(2026, 9, 7, 12, 1, 0), poly)
        agg.process_detection(datetime(2026, 9, 7, 12, 1, 6), poly)
        agg.process_detection(datetime(2026, 9, 7, 12, 1, 10), poly)

        agg.write_csv(self.csv_path)

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 4)  # Header + 3 events
        # Event 1
        self.assertEqual(reader[1][0], "1")
        self.assertEqual(reader[1][3], "5")
        self.assertEqual(reader[1][4], "2")
        # Event 2
        self.assertEqual(reader[2][0], "2")
        self.assertEqual(reader[2][3], "0")
        self.assertEqual(reader[2][4], "1")
        # Event 3
        self.assertEqual(reader[3][0], "3")
        self.assertEqual(reader[3][3], "10")
        self.assertEqual(reader[3][4], "3")

    # 11. Malformed line doesn't crash
    def test_11_malformed_line_skipped(self):
        dt, poly = watcher.parse_intrusion_line("GARBAGE LINE NOT AN INTRUSION\n")
        self.assertIsNone(dt)
        self.assertIsNone(poly)

        dt2, poly2 = watcher.parse_intrusion_line("Intrusion detected at INVALID_DATE | Polygon=[]\n")
        self.assertIsNone(dt2)
        self.assertIsNone(poly2)

        # Write malformed line then valid line to file and run watcher --once
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("Random junk\n")
            f.write("Intrusion detected at INVALID_TIME | Polygon=[(1,1)]\n")
            f.write("Intrusion detected at Mon Sep  7 12:00:00 2026 | Polygon=[(1, 1), (2, 2), (1, 2)]\n")

        agg = watcher.watch_log_file(self.txt_path, self.csv_path, once=True)
        self.assertEqual(len(agg.events), 1)
        self.assertEqual(agg.events[0].detection_count, 1)

    # 12. Empty log file
    def test_12_empty_log_file(self):
        with open(self.txt_path, "w", encoding="utf-8") as f:
            pass  # empty

        agg = watcher.watch_log_file(self.txt_path, self.csv_path, once=True)
        self.assertEqual(len(agg.events), 0)
        self.assertTrue(os.path.exists(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 1)  # Only header

    # 13. Existing log processed correctly
    def test_13_existing_log_batch_mode(self):
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("Intrusion detected at Mon Sep  7 12:00:00 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]\n")
            f.write("Intrusion detected at Mon Sep  7 12:00:05 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]\n")
            f.write("Intrusion detected at Mon Sep  7 12:00:25 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]\n")

        agg = watcher.watch_log_file(self.txt_path, self.csv_path, gap_seconds=10, once=True)
        self.assertEqual(len(agg.events), 2)
        self.assertEqual(agg.events[0].duration, 5)
        self.assertEqual(agg.events[1].duration, 0)

    # 14. Real-time behavior: append lines in a thread while watcher is running
    def test_14_realtime_appending(self):
        # Create initially empty log file
        with open(self.txt_path, "w", encoding="utf-8") as f:
            pass

        stop_event = threading.Event()

        def run_watcher():
            watcher.watch_log_file(
                self.txt_path,
                self.csv_path,
                gap_seconds=10,
                poll_interval=0.1,
                stop_event=stop_event
            )

        watcher_thread = threading.Thread(target=run_watcher, daemon=True)
        watcher_thread.start()

        # Let watcher start and create initial CSV
        time.sleep(0.3)

        # 1. Append first detection
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write("Intrusion detected at Mon Sep  7 12:00:00 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]\n")
            f.flush()

        time.sleep(0.3)

        # Verify Event 1 exists in CSV
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[1][3], "0")
        self.assertEqual(rows[1][4], "1")

        # 2. Append second detection 5 seconds later (within 10s gap)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write("Intrusion detected at Mon Sep  7 12:00:05 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]\n")
            f.flush()

        time.sleep(0.3)

        # Verify Event 1 updated in place (still 2 rows total)
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[1][3], "5")
        self.assertEqual(rows[1][4], "2")

        # 3. Append third detection 30 seconds later (gap > 10s -> Event 2)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write("Intrusion detected at Mon Sep  7 12:00:35 2026 | Polygon=[(0, 0), (10, 10), (0, 10)]\n")
            f.flush()

        time.sleep(0.3)

        # Verify Event 2 added (now 3 rows total)
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2][0], "2")
        self.assertEqual(rows[2][3], "0")
        self.assertEqual(rows[2][4], "1")

        # Stop watcher
        stop_event.set()
        watcher_thread.join(timeout=2.0)

    # 15. Snapshot association (test file exists in intrusion_snapshots vs test file missing)
    def test_15_snapshot_association(self):
        # Create a real dummy snapshot in the session's intrusion_snapshots dir
        dt = datetime(2026, 9, 7, 12, 0, 0)
        snap_name = f"intrusion_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        snap_full = os.path.join(self.snapshots_dir, snap_name)
        with open(snap_full, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0")  # dummy JPEG header

        # Match exact timestamp
        found = watcher.find_snapshot(dt, self.temp_dir.name)
        self.assertEqual(found, f"intrusion_snapshots/{snap_name}")

        # Match timestamp 1 second off (boundary tolerance)
        dt_off = dt + timedelta(seconds=1)
        found_off = watcher.find_snapshot(dt_off, self.temp_dir.name)
        self.assertEqual(found_off, f"intrusion_snapshots/{snap_name}")

        # Timestamp with no matching file -> None
        dt_none = datetime(2026, 1, 1, 0, 0, 0)
        found_none = watcher.find_snapshot(dt_none, self.temp_dir.name)
        self.assertIsNone(found_none)

    # 16. Auto-discovery of active session directory
    def test_16_find_active_session_log(self):
        output_dir = os.path.join(self.temp_dir.name, "test_output")
        os.makedirs(output_dir, exist_ok=True)

        # Empty directory -> None, None
        name, path = watcher.find_active_session_log(output_dir)
        self.assertIsNone(name)
        self.assertIsNone(path)

        # Create session 1
        s1 = os.path.join(output_dir, "session_one")
        os.makedirs(s1, exist_ok=True)
        time.sleep(0.05)

        # Create session 2 with a violation_log.csv touched more recently
        s2 = os.path.join(output_dir, "session_two")
        os.makedirs(s2, exist_ok=True)
        time.sleep(0.05)
        with open(os.path.join(s2, "violation_log.csv"), "w") as f:
            f.write("Date,Time\n")

        name, path = watcher.find_active_session_log(output_dir)
        self.assertEqual(name, "session_two")
        self.assertEqual(path, os.path.join(s2, "intrusion_log.txt"))

    # 17. watch_sessions_directory batch mode
    def test_17_watch_sessions_directory_batch(self):
        output_dir = os.path.join(self.temp_dir.name, "test_output_batch")
        sess_dir = os.path.join(output_dir, "my_session")
        os.makedirs(sess_dir, exist_ok=True)
        log_file = os.path.join(sess_dir, "intrusion_log.txt")

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Intrusion detected at Mon Sep  7 12:00:00 2026 | Polygon=[(1, 1), (2, 2), (1, 2)]\n")

        agg = watcher.watch_sessions_directory(watch_dir=output_dir, once=True)
        self.assertIsNotNone(agg)
        self.assertEqual(len(agg.events), 1)
        csv_file = os.path.join(sess_dir, "intrusion_log.csv")
        self.assertTrue(os.path.exists(csv_file))

    # 18. GIF creation from multiple snapshots
    def test_18_gif_creation_from_multiple_snapshots(self):
        from PIL import Image

        # Create two real JPEG snapshots
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        s0_name = f"intrusion_{t0.strftime('%Y%m%d_%H%M%S')}.jpg"
        s1_name = f"intrusion_{t1.strftime('%Y%m%d_%H%M%S')}.jpg"

        Image.new("RGB", (64, 64), color="red").save(os.path.join(self.snapshots_dir, s0_name))
        Image.new("RGB", (64, 64), color="blue").save(os.path.join(self.snapshots_dir, s1_name))

        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        agg.process_detection(t0, poly, session_dir=self.temp_dir.name)
        agg.process_detection(t1, poly, session_dir=self.temp_dir.name)

        event = agg.events[0]
        self.assertEqual(event.gif_path, "intrusion_gifs/intrusion_event_001.gif")

        gif_full = os.path.join(self.temp_dir.name, "intrusion_gifs", "intrusion_event_001.gif")
        self.assertTrue(os.path.isfile(gif_full))

        with Image.open(gif_full) as gif:
            self.assertEqual(getattr(gif, "n_frames", 1), 2)

        # Verify CSV references the GIF
        agg.write_csv(self.csv_path)
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[1][6], "intrusion_gifs/intrusion_event_001.gif")

    # 19. Chronological snapshot ordering in GIF
    def test_19_chronological_snapshot_ordering(self):
        from PIL import Image

        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 4)
        t2 = datetime(2026, 9, 7, 12, 0, 8)

        # Save with distinct pixel colors: Frame 0 = pure red, Frame 1 = pure green, Frame 2 = pure blue
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(
            os.path.join(self.snapshots_dir, f"intrusion_{t0.strftime('%Y%m%d_%H%M%S')}.jpg")
        )
        Image.new("RGB", (32, 32), color=(0, 255, 0)).save(
            os.path.join(self.snapshots_dir, f"intrusion_{t1.strftime('%Y%m%d_%H%M%S')}.jpg")
        )
        Image.new("RGB", (32, 32), color=(0, 0, 255)).save(
            os.path.join(self.snapshots_dir, f"intrusion_{t2.strftime('%Y%m%d_%H%M%S')}.jpg")
        )

        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        for t in [t0, t1, t2]:
            agg.process_detection(t, poly, session_dir=self.temp_dir.name)

        gif_full = os.path.join(self.temp_dir.name, "intrusion_gifs", "intrusion_event_001.gif")
        with Image.open(gif_full) as gif:
            self.assertEqual(gif.n_frames, 3)
            # Frame 0: Red (allowing minor JPEG compression tolerance)
            gif.seek(0)
            p0 = gif.convert("RGB").getpixel((10, 10))
            self.assertGreater(p0[0], 200)
            self.assertLess(p0[1], 50)
            self.assertLess(p0[2], 50)
            # Frame 1: Green
            gif.seek(1)
            p1 = gif.convert("RGB").getpixel((10, 10))
            self.assertLess(p1[0], 50)
            self.assertGreater(p1[1], 200)
            self.assertLess(p1[2], 50)
            # Frame 2: Blue
            gif.seek(2)
            p2 = gif.convert("RGB").getpixel((10, 10))
            self.assertLess(p2[0], 50)
            self.assertLess(p2[1], 50)
            self.assertGreater(p2[2], 200)

    # 20. Separate GIFs for separate events
    def test_20_separate_gifs_for_separate_events(self):
        from PIL import Image

        # Event 1 snapshots (12:00:00, 12:00:05) — distinct colors so frames remain distinct in GIF
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        Image.new("RGB", (32, 32), color="red").save(
            os.path.join(self.snapshots_dir, f"intrusion_{t0.strftime('%Y%m%d_%H%M%S')}.jpg")
        )
        Image.new("RGB", (32, 32), color="yellow").save(
            os.path.join(self.snapshots_dir, f"intrusion_{t1.strftime('%Y%m%d_%H%M%S')}.jpg")
        )

        # Event 2 snapshot (12:00:30, gap = 25s > 10s)
        t2 = datetime(2026, 9, 7, 12, 0, 30)
        Image.new("RGB", (32, 32), color="blue").save(
            os.path.join(self.snapshots_dir, f"intrusion_{t2.strftime('%Y%m%d_%H%M%S')}.jpg")
        )

        agg = watcher.IntrusionAggregator(gap_seconds=10)
        poly = "[(0, 0), (10, 10), (0, 10)]"
        agg.process_detection(t0, poly, session_dir=self.temp_dir.name)
        agg.process_detection(t1, poly, session_dir=self.temp_dir.name)
        agg.process_detection(t2, poly, session_dir=self.temp_dir.name)

        self.assertEqual(len(agg.events), 2)
        gif1 = os.path.join(self.temp_dir.name, "intrusion_gifs", "intrusion_event_001.gif")
        gif2 = os.path.join(self.temp_dir.name, "intrusion_gifs", "intrusion_event_002.gif")

        self.assertTrue(os.path.isfile(gif1))
        self.assertTrue(os.path.isfile(gif2))

        with Image.open(gif1) as g1:
            self.assertEqual(g1.n_frames, 2)
        with Image.open(gif2) as g2:
            self.assertEqual(g2.n_frames, 1)

    # 21. Updating an ongoing event's GIF
    def test_21_updating_ongoing_event_gif(self):
        from PIL import Image

        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        t2 = datetime(2026, 9, 7, 12, 0, 9)
        poly = "[(0, 0), (10, 10), (0, 10)]"

        agg = watcher.IntrusionAggregator(gap_seconds=10)
        gif_path = os.path.join(self.temp_dir.name, "intrusion_gifs", "intrusion_event_001.gif")

        # Step 1: First snapshot
        Image.new("RGB", (32, 32), color="red").save(
            os.path.join(self.snapshots_dir, f"intrusion_{t0.strftime('%Y%m%d_%H%M%S')}.jpg")
        )
        agg.process_detection(t0, poly, session_dir=self.temp_dir.name)
        with Image.open(gif_path) as g:
            self.assertEqual(g.n_frames, 1)

        # Step 2: Second snapshot added to same event -> GIF updated to 2 frames
        Image.new("RGB", (32, 32), color="green").save(
            os.path.join(self.snapshots_dir, f"intrusion_{t1.strftime('%Y%m%d_%H%M%S')}.jpg")
        )
        agg.process_detection(t1, poly, session_dir=self.temp_dir.name)
        with Image.open(gif_path) as g:
            self.assertEqual(g.n_frames, 2)

        # Step 3: Third snapshot added to same event -> GIF updated to 3 frames
        Image.new("RGB", (32, 32), color="blue").save(
            os.path.join(self.snapshots_dir, f"intrusion_{t2.strftime('%Y%m%d_%H%M%S')}.jpg")
        )
        agg.process_detection(t2, poly, session_dir=self.temp_dir.name)
        with Image.open(gif_path) as g:
            self.assertEqual(g.n_frames, 3)

    # 22. Missing / unavailable snapshot handling
    def test_22_missing_snapshot_handling(self):
        # Detections where NO snapshot files exist on disk
        t0 = datetime(2026, 9, 7, 12, 0, 0)
        t1 = datetime(2026, 9, 7, 12, 0, 5)
        poly = "[(0, 0), (10, 10), (0, 10)]"

        agg = watcher.IntrusionAggregator(gap_seconds=10)
        agg.process_detection(t0, poly, session_dir=self.temp_dir.name)
        agg.process_detection(t1, poly, session_dir=self.temp_dir.name)

        event = agg.events[0]
        self.assertEqual(event.gif_path, "")
        self.assertEqual(event.detection_count, 2)
        self.assertEqual(event.duration, 5)

        agg.write_csv(self.csv_path)
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[1][6], "")  # GIF column is empty string

    # 23. Original JPG snapshots are preserved untouched
    def test_23_original_jpgs_preserved(self):
        from PIL import Image

        t0 = datetime(2026, 9, 7, 12, 0, 0)
        s_name = f"intrusion_{t0.strftime('%Y%m%d_%H%M%S')}.jpg"
        s_full = os.path.join(self.snapshots_dir, s_name)
        Image.new("RGB", (64, 64), color="yellow").save(s_full)

        orig_size = os.path.getsize(s_full)
        with open(s_full, "rb") as f:
            orig_bytes = f.read()

        agg = watcher.IntrusionAggregator(gap_seconds=10)
        agg.process_detection(t0, "[(0, 0), (1, 1)]", session_dir=self.temp_dir.name)

        # Verify original JPG is still there with identical size and content
        self.assertTrue(os.path.isfile(s_full))
        self.assertEqual(os.path.getsize(s_full), orig_size)
        with open(s_full, "rb") as f:
            self.assertEqual(f.read(), orig_bytes)


if __name__ == "__main__":
    unittest.main()



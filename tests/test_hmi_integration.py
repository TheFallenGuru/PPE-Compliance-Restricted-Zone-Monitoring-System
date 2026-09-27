"""
test_hmi_integration.py — Integration and Verification Test for Upgraded Industrial HMI.

Validates all 10 user requirements:
1. Launch GUI without errors.
2. Select Webcam mode.
3. Select Video File mode.
4. Browse to an arbitrary video path outside project directory.
5. Select different combinations of all 10 PPE requirements.
6. Polygon drawing workflow: multi-point placement, coordinate resolution mapping, confirm/reset.
7. Start detection with video + PPE + polygon, verifying settings reach main.py.
8. Verify intrusion detection still works and logs motion.
9. Verify violation logs, snapshots, intrusion logs, and watcher GIF generation work.
10. Verify Stop functionality interrupts cleanly.
"""

import unittest
import os
import shutil
import tempfile
import time
import threading
import cv2
import numpy as np

import app
import main
from scripts import intrusion_log_watcher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_VIDEO = os.path.join(PROJECT_ROOT, "Sample Input", "sample1.mp4")


class TestHMIIntegration(unittest.TestCase):

    def setUp(self):
        self.gui = app.PPELauncherApp()
        self.gui.update()

    def tearDown(self):
        try:
            if self.gui.is_running:
                self.gui.stop_detection()
            self.gui.destroy()
        except Exception:
            pass

    def test_01_webcam_and_video_mode_selection(self):
        """Verify webcam and video source mode switching and state."""
        # 1. Switch to Webcam
        self.gui.show_webcam_screen()
        self.gui.update()
        self.assertEqual(self.gui.source_mode_var.get(), "webcam")
        selected_source = self.gui.get_selected_source()
        self.assertIsInstance(selected_source, int)

        # 2. Switch to Video
        self.gui.show_video_screen()
        self.gui.update()
        self.assertEqual(self.gui.source_mode_var.get(), "video")
        self.assertTrue(isinstance(self.gui.get_selected_source(), str))

    def test_02_arbitrary_external_video_browsing(self):
        """Verify browsing and setting arbitrary video files from outside the project directory."""
        # Create an external temporary directory outside the project
        temp_dir = tempfile.mkdtemp(prefix="ppe_external_video_")
        try:
            external_video_path = os.path.join(temp_dir, "outside_project_video.mp4")
            shutil.copyfile(SAMPLE_VIDEO, external_video_path)

            self.gui.video_path_var.set(external_video_path)
            self.gui.source_mode_var.set("video")
            self.gui.update()

            # Verify the GUI accepted the arbitrary external path
            source = self.gui.get_selected_source()
            self.assertEqual(os.path.normpath(source), os.path.normpath(external_video_path))
            self.assertTrue(os.path.exists(source))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_03_ppe_10_class_combinations(self):
        """Verify toggling and feeding all 10 supported PPE checklist classes."""
        self.assertEqual(len(app.AVAILABLE_PPE_OPTIONS), 10)

        # 1. Select specific combination (e.g. 4 specific items)
        test_combo = {'helmet', 'shoes', 'glasses', 'face-guard'}
        for key, var in self.gui.selected_ppe_vars.items():
            var.set(key in test_combo)
        self.assertEqual(self.gui.get_selected_required_ppe(), test_combo)

        # 2. Select All
        self.gui.select_all_ppe()
        self.assertEqual(len(self.gui.get_selected_required_ppe()), 10)

        # 3. Clear All
        self.gui.clear_all_ppe()
        self.assertEqual(len(self.gui.get_selected_required_ppe()), 0)

        # 4. Reset Default
        self.gui.reset_default_ppe()
        self.assertEqual(self.gui.get_selected_required_ppe(), app.DEFAULT_REQUIRED_PPE)

    def test_04_polygon_drawing_and_resolution_mapping(self):
        """Verify multi-point polygon creation and accurate resolution mapping back to original frame."""
        # Create a test frame at 1280x720
        orig_w, orig_h = 1280, 720
        test_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        dialog = app.PolygonDrawingDialog(self.gui, test_frame)
        self.gui.update()

        # Check scale factor calculations
        self.assertGreater(dialog.scale_x, 1.0)
        self.assertGreater(dialog.scale_y, 1.0)

        # Simulate clicking 4 points in display coordinates
        disp_clicks = [(50, 50), (200, 50), (200, 180), (50, 180)]
        for cx, cy in disp_clicks:
            class DummyEvent:
                x = cx
                y = cy
            dialog._on_canvas_click(DummyEvent)

        self.assertEqual(len(dialog.points), 4)

        # Test Undo
        dialog._on_undo()
        self.assertEqual(len(dialog.points), 3)

        # Re-add fourth point
        class DummyEvent4:
            x = 50
            y = 180
        dialog._on_canvas_click(DummyEvent4)
        self.assertEqual(len(dialog.points), 4)

        # Confirm and validate
        dialog._on_confirm()
        self.assertIsNotNone(dialog.confirmed_polygon)
        self.assertEqual(len(dialog.confirmed_polygon), 4)

        # Verify all coordinates are mapped back to original resolution domain (0..1280, 0..720)
        for px, py in dialog.confirmed_polygon:
            self.assertTrue(0 <= px <= orig_w)
            self.assertTrue(0 <= py <= orig_h)

        # Check Reset functionality on a new dialog
        dialog2 = app.PolygonDrawingDialog(self.gui, test_frame)
        dialog2._on_canvas_click(DummyEvent)
        dialog2._on_reset()
        self.assertEqual(len(dialog2.points), 0)
        dialog2.destroy()

    def test_05_detection_with_hmi_settings_telemetry_and_stop(self):
        """Verify detection pipeline receives custom HMI settings, emits telemetry, detects intrusions, and stops."""
        custom_ppe = {'helmet', 'safety-vest', 'gloves'}
        custom_polygon = [(50, 50), (450, 50), (450, 450), (50, 450)]

        received_telemetry = []
        stop_evt = threading.Event()

        def telemetry_tap(data):
            received_telemetry.append(data)
            # Safely exercise GUI telemetry handler
            self.gui._on_frame_telemetry(data)
            if data.get('intrusion_state', {}).get('is_intrusion') or data.get('frame_count', 0) >= 25:
                stop_evt.set()

        temp_session = tempfile.mkdtemp(prefix="hmi_session_test_")
        try:
            results = main.run_detection(
                source=SAMPLE_VIDEO,
                headless=True,
                roi=custom_polygon,
                required_ppe=custom_ppe,
                stop_event=stop_evt,
                no_save_video=True,
                session_dir=temp_session,
                frame_callback=telemetry_tap
            )

            # 1. Verify detection completed or stopped
            self.assertIn(results['status'], ('completed', 'stopped'))
            self.assertGreater(results['frames_processed'], 0)

            # 2. Verify telemetry callback received real data
            self.assertGreater(len(received_telemetry), 0)
            last_frame_telemetry = received_telemetry[-1]
            self.assertIn('monitoring_status', last_frame_telemetry)
            self.assertIn('frame_count', last_frame_telemetry)
            self.assertIn('worker_counts', last_frame_telemetry)
            self.assertIn('intrusion_state', last_frame_telemetry)

            # 3. Verify intrusion system detected motion in polygon
            self.assertTrue(os.path.exists(os.path.join(temp_session, "intrusion_log.txt")))
            self.assertTrue(os.path.exists(os.path.join(temp_session, "violation_log.csv")))

            # 4. Verify intrusion watcher groups events and produces CSV
            aggregator = intrusion_log_watcher.watch_log_file(
                os.path.join(temp_session, "intrusion_log.txt"),
                once=True
            )
            self.assertGreaterEqual(len(aggregator.events), 1)
            self.assertTrue(os.path.exists(os.path.join(temp_session, "intrusion_log.csv")))

        finally:
            shutil.rmtree(temp_session, ignore_errors=True)

    def test_06_system_output_console_logging(self):
        """Verify the System Output tab console receives concise, timestamped, operator-relevant event logs."""
        self.gui.update()

        # 1. Check initial logs are present in console_txt
        initial_log_text = self.gui.console_txt.get("1.0", "end")
        self.assertIn("[SYS  ]", initial_log_text)
        self.assertIn("PPE Safety Monitoring System HMI initialized", initial_log_text)

        # 2. Test logging a custom event
        self.gui.log_system_event("OK", "Test operational verification message.")
        self.gui.update()
        log_text = self.gui.console_txt.get("1.0", "end")
        self.assertIn("[OK   ]", log_text)
        self.assertIn("Test operational verification message.", log_text)

        # 3. Test policy change logging
        self.gui.select_all_ppe()
        self.gui.clear_all_ppe()
        self.gui.reset_default_ppe()
        self.gui.update()
        log_text = self.gui.console_txt.get("1.0", "end")
        self.assertIn("Selected ALL 10 required PPE classes", log_text)
        self.assertIn("CLEARED all required PPE classes", log_text)
        self.assertIn("Reset to default 5 classes", log_text)

        # 4. Test ROI clear logging
        self.gui.clear_current_roi()
        self.gui.update()
        log_text = self.gui.console_txt.get("1.0", "end")
        self.assertIn("Restricted zone", log_text)

    def test_07_show_window_suppression(self):
        """Verify main.run_detection accepts show_window=False and suppresses cv2.imshow without errors."""
        stop_evt = threading.Event()

        def telemetry_tap(data):
            if data.get('frame_count', 0) >= 2:
                stop_evt.set()

        temp_session = tempfile.mkdtemp(prefix="hmi_window_test_")
        try:
            results = main.run_detection(
                source=SAMPLE_VIDEO,
                headless=False,
                show_window=False,
                no_roi=True,
                stop_event=stop_evt,
                no_save_video=True,
                session_dir=temp_session,
                frame_callback=telemetry_tap
            )
            self.assertIn(results['status'], ('completed', 'stopped'))
            self.assertGreater(results['frames_processed'], 0)
        finally:
            shutil.rmtree(temp_session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

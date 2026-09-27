import unittest
import os
import threading
import numpy as np
import app
import main

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_VIDEO = os.path.join(PROJECT_ROOT, "Sample Input", "sample1.mp4")

class TestAppGUI(unittest.TestCase):
    def test_dependency_check(self):
        missing = app.check_dependencies()
        self.assertEqual(missing, [], f"All dependencies should be installed, but missing: {missing}")

    def test_model_weights_check(self):
        ok, msg = app.check_model_weights()
        self.assertTrue(ok, f"SH17 model should validate: {msg}")

    def test_gui_initialization_and_teardown(self):
        gui_app = app.PPELauncherApp()
        gui_app.update()
        
        # Test navigation to webcam screen
        gui_app.show_webcam_screen()
        gui_app.update()
        self.assertIsNotNone(gui_app.cam_choice)
        self.assertIsNone(gui_app.webcam_roi)

        # Set ROI manually and clear
        gui_app.webcam_roi = [(50, 50), (200, 50), (200, 200), (50, 200)]
        gui_app.clear_webcam_roi()
        self.assertIsNone(gui_app.webcam_roi)

        # Test navigation to video screen
        gui_app.show_video_screen()
        gui_app.update()
        self.assertIsNotNone(gui_app.video_path_var)
        self.assertIsNone(gui_app.video_roi)

        # Set ROI manually and clear
        gui_app.video_roi = [(100, 100), (300, 80), (500, 200), (450, 400), (150, 350)]
        gui_app.clear_video_roi()
        self.assertIsNone(gui_app.video_roi)

        # Test navigation back to main screen
        gui_app.show_main_screen()
        gui_app.update()

        gui_app.destroy()

    def test_gui_ppe_checklist_controls(self):
        gui_app = app.PPELauncherApp()
        gui_app.update()

        # 1. Verify default selections
        default_selected = gui_app.get_selected_required_ppe()
        self.assertEqual(default_selected, app.DEFAULT_REQUIRED_PPE)

        # 2. Select all
        gui_app.select_all_ppe()
        self.assertEqual(len(gui_app.get_selected_required_ppe()), len(app.AVAILABLE_PPE_OPTIONS))

        # 3. Clear all
        gui_app.clear_all_ppe()
        self.assertEqual(len(gui_app.get_selected_required_ppe()), 0)

        # 4. Reset default
        gui_app.reset_default_ppe()
        self.assertEqual(gui_app.get_selected_required_ppe(), app.DEFAULT_REQUIRED_PPE)

        gui_app.destroy()

    def test_roi_headless_execution(self):
        """Run detection with an explicit polygon ROI in headless mode; stop after 5 frames."""
        stop_evt = threading.Event()

        def _stop_after_frames(data):
            if data.get('frame_count', 0) >= 5:
                stop_evt.set()

        results = main.run_detection(
            source=SAMPLE_VIDEO,
            headless=True,
            roi=[(100, 100), (400, 100), (400, 400), (100, 400)],
            required_ppe={'helmet', 'safety-vest'},
            no_save_video=True,
            stop_event=stop_evt,
            frame_callback=_stop_after_frames,
        )
        self.assertIn(results['status'], ('completed', 'stopped'))
        self.assertGreater(results['frames_processed'], 0)

    def test_no_roi_headless_execution(self):
        """Run detection with no ROI in headless mode; stop after 5 frames."""
        stop_evt = threading.Event()

        def _stop_after_frames(data):
            if data.get('frame_count', 0) >= 5:
                stop_evt.set()

        results = main.run_detection(
            source=SAMPLE_VIDEO,
            headless=True,
            no_roi=True,
            required_ppe={'gloves', 'shoes', 'glasses'},
            no_save_video=True,
            stop_event=stop_evt,
            frame_callback=_stop_after_frames,
        )
        self.assertIn(results['status'], ('completed', 'stopped'))
        self.assertGreater(results['frames_processed'], 0)

if __name__ == "__main__":
    unittest.main()


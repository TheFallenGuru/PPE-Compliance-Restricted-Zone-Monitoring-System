import unittest
import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_VIDEO = os.path.join(PROJECT_ROOT, "Sample Input", "sample1.mp4")

class TestCLIHeadless(unittest.TestCase):
    def test_headless_execution(self):
        """Verify the CLI runs headless detection and exits cleanly."""
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "main.py"),
            "--source", SAMPLE_VIDEO,
            "--headless", "--no-roi",
            "--no-save-video",
            "--max-frames", "5",
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Headless execution failed: {result.stderr}")
        self.assertIn("Model validation PASSED", result.stdout)
        self.assertIn("Processing complete", result.stdout)

    def test_missing_file_error_handling(self):
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "main.py"), "--source", "nonexistent_test_video.mp4", "--headless"]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, "Must exit with code 1 on missing source")
        self.assertIn("FATAL: Source video file not found", result.stdout)

if __name__ == "__main__":
    unittest.main()

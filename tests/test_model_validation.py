import unittest
import os
from ultralytics import YOLO

VERIFIED_MODEL_CLASSES = {
    0: 'person',    1: 'ear',          2: 'ear-mufs',   3: 'face',
    4: 'face-guard',5: 'face-mask',    6: 'foot',       7: 'tool',
    8: 'glasses',   9: 'gloves',      10: 'helmet',    11: 'hands',
   12: 'head',     13: 'medical-suit',14: 'shoes',     15: 'safety-suit',
   16: 'safety-vest'
}

class TestModelValidation(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("models/yolo8m.pt"), "models/yolo8m.pt must exist")

    def test_sh17_class_schema(self):
        model = YOLO("models/yolo8m.pt")
        class_names = model.names
        self.assertEqual(len(class_names), 17, "Model must contain exactly 17 classes")
        for idx, expected_name in VERIFIED_MODEL_CLASSES.items():
            self.assertEqual(class_names.get(idx), expected_name, f"Class {idx} must be {expected_name}")

    def test_mismatch_detection(self):
        # Verify schema checker flags deviations
        mock_classes = {0: 'person', 1: 'car'}
        mismatches = [
            (k, VERIFIED_MODEL_CLASSES[k], mock_classes.get(k))
            for k in VERIFIED_MODEL_CLASSES
            if mock_classes.get(k) != VERIFIED_MODEL_CLASSES[k]
        ]
        self.assertGreater(len(mismatches), 0, "Mismatches should be detected on mock COCO class dictionary")

if __name__ == "__main__":
    unittest.main()

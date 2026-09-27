import unittest
import numpy as np
import cv2
import main

class TestPolygonROI(unittest.TestCase):
    def test_polygon_validation_point_counts(self):
        # 2 points -> must reject
        self.assertIsNone(main.validate_polygon([(10, 10), (100, 100)]))
        # 0 or empty -> must reject
        self.assertIsNone(main.validate_polygon([]))
        self.assertIsNone(main.validate_polygon(None))
        # 3 valid non-collinear points -> must accept
        triangle = main.validate_polygon([(10, 10), (100, 10), (50, 100)])
        self.assertIsNotNone(triangle)
        self.assertEqual(len(triangle), 3)
        # 5 valid points -> must accept
        pentagon = main.validate_polygon([(100, 100), (300, 80), (500, 200), (450, 400), (150, 350)])
        self.assertIsNotNone(pentagon)
        self.assertEqual(len(pentagon), 5)

    def test_polygon_validation_degenerate_zero_area(self):
        # Collinear points (zero area) -> must reject
        collinear = [(10, 10), (20, 20), (30, 30)]
        self.assertIsNone(main.validate_polygon(collinear))

    def test_polygon_validation_clamping(self):
        # Coordinates exceeding bounds -> must clamp cleanly
        out_of_bounds = [(-50, -10), (800, 50), (400, 700)]
        clamped = main.validate_polygon(out_of_bounds, frame_width=640, frame_height=480)
        self.assertIsNotNone(clamped)
        for x, y in clamped:
            self.assertTrue(0 <= x <= 640)
            self.assertTrue(0 <= y <= 480)

    def test_cli_polygon_string_parsing(self):
        # Semicolon-separated format
        parsed1 = main.parse_polygon_string("100,100;300,80;500,200;450,400;150,350")
        self.assertIsNotNone(parsed1)
        self.assertEqual(len(parsed1), 5)
        self.assertEqual(parsed1[0], (100, 100))
        self.assertEqual(parsed1[2], (500, 200))

        # Legacy 4-number rectangle format
        parsed2 = main.parse_polygon_string("100,100,400,400")
        self.assertIsNotNone(parsed2)
        self.assertEqual(len(parsed2), 4)
        self.assertEqual(parsed2, [(100, 100), (400, 100), (400, 400), (100, 400)])

    def test_polygon_masking_strict_isolation(self):
        # Frame of 480x640 (black background)
        rframe = np.zeros((480, 640), dtype=np.uint8)
        cframe = np.zeros((480, 640), dtype=np.uint8)

        # Polygon occupying [100..300, 100..300]
        polygon = [(100, 100), (300, 100), (300, 300), (100, 300)]

        # 1. Motion COMPLETELY OUTSIDE the polygon (at [400..450, 400..450])
        cframe[400:450, 400:450] = 255
        is_intrusive = main.intrusion(cframe, rframe, polygon, thresh=130, motion_pixel_count=10)
        self.assertFalse(is_intrusive, "Motion outside polygon must NOT trigger intrusion")
        self.assertEqual(main.motion_pixels, 0, "Motion pixels inside polygon must be 0")

        # 2. Motion COMPLETELY INSIDE the polygon (at [150..200, 150..200] -> 2500 pixels)
        cframe_in = np.zeros((480, 640), dtype=np.uint8)
        cframe_in[150:200, 150:200] = 255
        is_intrusive_in = main.intrusion(cframe_in, rframe, polygon, thresh=130, motion_pixel_count=10)
        self.assertTrue(is_intrusive_in, "Motion inside polygon MUST trigger intrusion")
        self.assertEqual(main.motion_pixels, 2500)

        # 3. Motion PARTLY INSIDE, PARTLY OUTSIDE
        # Box from [80..120, 80..120] -> total 40x40 = 1600 pixels, but only [100..120, 100..120] = 20x20 = 400 pixels inside
        cframe_partial = np.zeros((480, 640), dtype=np.uint8)
        cframe_partial[80:120, 80:120] = 255
        main.intrusion(cframe_partial, rframe, polygon, thresh=130, motion_pixel_count=10)
        self.assertEqual(main.motion_pixels, 400, "Only the 400 pixels inside polygon should be counted")

if __name__ == "__main__":
    unittest.main()

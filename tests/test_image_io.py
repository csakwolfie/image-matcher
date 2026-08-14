import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_matcher.image_io import get_native_longer_side, load_image_smart


class ImageIOTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "test.png"
        # alacsony kontrasztú szintetikus kép (szűk értéktartomány)
        img = np.full((300, 500), 128, dtype=np.uint8)
        img[100:200, 150:350] = 135
        cv2.imwrite(str(self.path), img)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_downscales_when_larger_than_max_size(self):
        gray, scale = load_image_smart(self.path, max_size=200, min_size=50, use_clahe=False)
        self.assertIsNotNone(gray)
        self.assertLessEqual(max(gray.shape), 200)
        self.assertLess(scale, 1.0)

    def test_upscales_when_smaller_than_min_size(self):
        gray, scale = load_image_smart(self.path, max_size=2000, min_size=600, use_clahe=False)
        self.assertIsNotNone(gray)
        self.assertGreaterEqual(max(gray.shape), 600)
        self.assertGreater(scale, 1.0)

    def test_clahe_increases_local_contrast(self):
        gray_plain, _ = load_image_smart(self.path, max_size=500, min_size=50, use_clahe=False)
        gray_clahe, _ = load_image_smart(self.path, max_size=500, min_size=50, use_clahe=True)
        self.assertGreater(float(gray_clahe.std()), float(gray_plain.std()))

    def test_missing_file_returns_none(self):
        gray, scale = load_image_smart(Path(self.tmpdir.name) / "nope.png", max_size=500)
        self.assertIsNone(gray)
        self.assertEqual(scale, 1.0)

    def test_get_native_longer_side_matches_actual_dimensions(self):
        # a setUp-beli teszt kép natívan 500x300 (szélesség x magasság)
        self.assertEqual(get_native_longer_side(self.path), 500)

    def test_get_native_longer_side_missing_file_returns_none(self):
        self.assertIsNone(get_native_longer_side(Path(self.tmpdir.name) / "nope.png"))


if __name__ == "__main__":
    unittest.main()

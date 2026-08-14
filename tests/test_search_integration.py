"""
Végponttól-végpontig teszt szintetikus képekkel: egy referencia (a valódi
forrás egy kivágata) megtalálása több jelölt közül, és a NEM TALÁLHATÓ eset
(csak "idegen" jelöltek), a config.yaml gyári (szigorú) küszöbeivel.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_matcher.cache_disk import DescriptorCache
from image_matcher.config import load_default_config
from image_matcher.detectors import ThreadLocalDetectors, create_detector
from image_matcher.search import find_best_match_for_reference


def _make_textured_image(seed: int, size: int = 1200) -> np.ndarray:
    """Kellően "dús" szintetikus textúra, hogy SIFT sok stabil keypointot találjon."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 255, (size, size), dtype=np.uint8)
    img = cv2.GaussianBlur(noise, (0, 0), sigmaX=3)
    for _ in range(60):
        x, y = rng.randint(60, size - 60, size=2)
        r = int(rng.randint(10, 45))
        color = int(rng.randint(0, 255))
        cv2.circle(img, (int(x), int(y)), r, color, -1)
    return img


class EndToEndMatchTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.cfg = load_default_config()

        true_source = _make_textured_image(seed=1)
        decoy1 = _make_textured_image(seed=2)
        decoy2 = _make_textured_image(seed=3)

        self.true_source_path = self.root / "true_source.png"
        self.decoy1_path = self.root / "decoy1.png"
        self.decoy2_path = self.root / "decoy2.png"
        cv2.imwrite(str(self.true_source_path), true_source)
        cv2.imwrite(str(self.decoy1_path), decoy1)
        cv2.imwrite(str(self.decoy2_path), decoy2)

        # referencia = a valódi forrás egy körbevágott kivágata
        crop = true_source[250:950, 250:950]
        self.reference_path = self.root / "reference.png"
        cv2.imwrite(str(self.reference_path), crop)

        self.detectors = {}
        for name in self.cfg.detector_priority:
            det = create_detector(name, self.cfg)
            if det is not None:
                self.detectors[name] = det

        self.cache = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg), persist_dir=None)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_finds_true_source_among_decoys(self):
        best_src, info, near_miss, all_results = find_best_match_for_reference(
            self.reference_path,
            [self.true_source_path, self.decoy1_path, self.decoy2_path],
            self.cache, self.detectors, self.cfg, max_workers=2,
        )
        self.assertEqual(best_src, self.true_source_path)
        self.assertTrue(info["success"])
        self.assertGreaterEqual(info["good_matches"], self.cfg.min_good_matches)
        self.assertGreaterEqual(info["inliers"], self.cfg.min_inliers)

    def test_reports_not_found_among_pure_decoys(self):
        best_src, info, near_miss, all_results = find_best_match_for_reference(
            self.reference_path,
            [self.decoy1_path, self.decoy2_path],
            self.cache, self.detectors, self.cfg, max_workers=2,
        )
        self.assertIsNone(best_src)
        self.assertFalse(info["success"])


if __name__ == "__main__":
    unittest.main()

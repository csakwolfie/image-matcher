"""
Fix 3: az 1. köri ("stage1") diagnosztika bővítése – a match_pair mostantól a
forrás saját keypoint-számát is visszaadja, és stage1_rank_candidates a "senki
nem érte el a technikai minimumot" esetben egy top-5 nyers rangsort ad a
generikus szöveg helyett.
"""

import tempfile
import unittest
from pathlib import Path

import cv2

from image_matcher.cache_disk import DescriptorCache
from image_matcher.config import load_default_config
from image_matcher.detectors import ThreadLocalDetectors, create_detector
from image_matcher.matching import match_pair
from image_matcher.search import stage1_rank_candidates
from tests.test_search_integration import _make_textured_image


class MatchPairSrcKeypointsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.cfg = load_default_config()

        src_img = _make_textured_image(seed=10)
        self.src_path = self.root / "src.png"
        cv2.imwrite(str(self.src_path), src_img)

        self.thread_detectors = ThreadLocalDetectors(self.cfg)
        self.cache = DescriptorCache(self.cfg, self.thread_detectors, persist_dir=None)
        self.detector = create_detector("SIFT", self.cfg)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_src_keypoints_matches_actual_descriptor_count(self):
        kp2, des2, _ = self.cache.get_descriptors(self.src_path, "SIFT")

        # csak a src_keypoints mezőt ellenőrizzük, nem a tényleges egyezést -
        # a referenciának elég néhány (>=4) keypointtal rendelkeznie
        ref_img = _make_textured_image(seed=11)
        kp1, des1 = self.detector.detectAndCompute(ref_img, None)
        self.assertGreaterEqual(len(kp1), 4)

        result = match_pair(kp1, des1, self.src_path, "SIFT", self.cache, self.cfg.ratio_test_sift, self.cfg)
        self.assertEqual(result["src_keypoints"], len(kp2))
        self.assertGreater(result["src_keypoints"], 0)

    def test_src_keypoints_zero_when_reference_has_no_features(self):
        result = match_pair([], None, self.src_path, "SIFT", self.cache, self.cfg.ratio_test_sift, self.cfg)
        self.assertEqual(result["src_keypoints"], 0)
        self.assertEqual(result["reject_reason"], "ref_no_features")


class Stage1EmptyLeaderboardTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        # elérhetetlenül magas technikai minimum -> senki sem érheti el,
        # így garantáltan az üres-scored ágat futtatjuk
        self.cfg = load_default_config().replace(stage1_min_good_matches=100000)

        ref_img = _make_textured_image(seed=20)
        self.ref_cache_path = self.root / "reference_cache.jpg"
        cv2.imwrite(str(self.ref_cache_path), ref_img)

        self.source_cache_files = []
        for i in range(3):
            src_img = _make_textured_image(seed=30 + i)
            p = self.root / f"src{i}.jpg"
            cv2.imwrite(str(p), src_img)
            self.source_cache_files.append(p)

        self.thread_detectors = ThreadLocalDetectors(self.cfg)
        self.cache = DescriptorCache(self.cfg, self.thread_detectors, persist_dir=None)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_scored_reports_top5_leaderboard(self):
        candidates, diag = stage1_rank_candidates(
            self.ref_cache_path, self.source_cache_files, "SIFT", self.cache,
            top_k=8, cfg=self.cfg, thread_detectors=self.thread_detectors, max_workers=2,
        )
        self.assertEqual(candidates, [])
        self.assertIn("src_kp=", diag)
        self.assertTrue(any(p.name in diag for p in self.source_cache_files))


if __name__ == "__main__":
    unittest.main()

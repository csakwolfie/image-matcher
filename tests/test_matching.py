import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_matcher.cache_disk import DescriptorCache
from image_matcher.config import load_default_config
from image_matcher.detectors import ThreadLocalDetectors, create_detector
from image_matcher.image_io import load_image_smart
from image_matcher.matching import classify_decision, compute_score, is_homography_plausible, match_pair
from tests.test_search_integration import _make_textured_image


class HomographyPlausibilityTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_default_config()

    def test_identity_is_plausible(self):
        H = np.eye(3)
        ok, reason = is_homography_plausible(H, self.cfg)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_mild_rotation_scale_is_plausible(self):
        theta = np.deg2rad(15)
        s = 1.2
        H = np.array([
            [s * np.cos(theta), -s * np.sin(theta), 5],
            [s * np.sin(theta), s * np.cos(theta), -3],
            [0, 0, 1],
        ])
        ok, reason = is_homography_plausible(H, self.cfg)
        self.assertTrue(ok)

    def test_mirror_is_rejected_as_shear(self):
        H = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        ok, reason = is_homography_plausible(H, self.cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "shear")

    def test_extreme_anisotropic_shear_is_rejected(self):
        H = np.array([[20.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        ok, reason = is_homography_plausible(H, self.cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "shear")

    def test_extreme_scale_is_rejected(self):
        H = np.array([[500.0, 0, 0], [0, 500.0, 0], [0, 0, 1.0]])
        ok, reason = is_homography_plausible(H, self.cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "scale")

    def test_none_homography_is_rejected(self):
        ok, reason = is_homography_plausible(None, self.cfg)
        self.assertFalse(ok)


class ScoreAndDecisionTest(unittest.TestCase):
    def test_compute_score_rewards_higher_inlier_ratio(self):
        low = compute_score(good_matches=50, inliers=10, inlier_ratio=0.2)
        high = compute_score(good_matches=50, inliers=10, inlier_ratio=0.9)
        self.assertGreater(high, low)

    def test_compute_score_bonus_for_many_inliers(self):
        few = compute_score(good_matches=50, inliers=5, inlier_ratio=0.9)
        many = compute_score(good_matches=50, inliers=30, inlier_ratio=0.9)
        self.assertGreater(many, few)

    def test_classify_decision_accept_strong_geometry(self):
        result = classify_decision(
            geo_reason="", inliers=300, min_inliers=220,
            inlier_ratio=0.95, min_inlier_ratio=0.93,
            score=1.0, score_threshold=0.97, success=True,
            decision_strong_ratio=0.85,
        )
        self.assertEqual(result, "ACCEPT_STRONG_GEOMETRY")

    def test_classify_decision_accept_inlier(self):
        result = classify_decision(
            geo_reason="", inliers=225, min_inliers=220,
            inlier_ratio=0.94, min_inlier_ratio=0.93,
            score=0.98, score_threshold=0.97, success=True,
            decision_strong_ratio=0.99,
        )
        self.assertEqual(result, "ACCEPT_INLIER")

    def test_classify_decision_reject_scale(self):
        result = classify_decision(
            geo_reason="implausible_homography_scale", inliers=0, min_inliers=220,
            inlier_ratio=0.0, min_inlier_ratio=0.93,
            score=0.0, score_threshold=0.97, success=False,
            decision_strong_ratio=0.85,
        )
        self.assertEqual(result, "REJECT_SCALE")

    def test_classify_decision_reject_homography(self):
        result = classify_decision(
            geo_reason="implausible_homography_shear", inliers=0, min_inliers=220,
            inlier_ratio=0.0, min_inlier_ratio=0.93,
            score=0.0, score_threshold=0.97, success=False,
            decision_strong_ratio=0.85,
        )
        self.assertEqual(result, "REJECT_HOMOGRAPHY")

    def test_classify_decision_reject_no_inliers(self):
        result = classify_decision(
            geo_reason="too_few_good_matches", inliers=0, min_inliers=220,
            inlier_ratio=0.0, min_inlier_ratio=0.93,
            score=0.0, score_threshold=0.97, success=False,
            decision_strong_ratio=0.85,
        )
        self.assertEqual(result, "REJECT_NO_INLIERS")

    def test_classify_decision_reject_inlier_ratio(self):
        result = classify_decision(
            geo_reason="", inliers=100, min_inliers=220,
            inlier_ratio=0.5, min_inlier_ratio=0.93,
            score=0.5, score_threshold=0.97, success=False,
            decision_strong_ratio=0.85,
        )
        self.assertEqual(result, "REJECT_INLIER_RATIO")

    def test_classify_decision_accept_ratio_compensated(self):
        result = classify_decision(
            geo_reason="", inliers=190, min_inliers=220,
            inlier_ratio=0.945, min_inlier_ratio=0.93,
            score=1.05, score_threshold=0.97, success=True,
            decision_strong_ratio=0.85, accept_branch="B",
        )
        self.assertEqual(result, "ACCEPT_RATIO_COMPENSATED")

    def test_classify_decision_accept_inlier_compensated(self):
        result = classify_decision(
            geo_reason="", inliers=340, min_inliers=220,
            inlier_ratio=0.90, min_inlier_ratio=0.93,
            score=1.15, score_threshold=0.97, success=True,
            decision_strong_ratio=0.85, accept_branch="C",
        )
        self.assertEqual(result, "ACCEPT_INLIER_COMPENSATED")

    def test_classify_decision_no_branch_falls_back_to_legacy_categories(self):
        # accept_branch alapértéke None -> visszafelé kompatibilis a régi hívókkal
        result = classify_decision(
            geo_reason="", inliers=300, min_inliers=220,
            inlier_ratio=0.95, min_inlier_ratio=0.93,
            score=1.0, score_threshold=0.97, success=True,
            decision_strong_ratio=0.85,
        )
        self.assertEqual(result, "ACCEPT_STRONG_GEOMETRY")


class MatchPairBranchLogicTest(unittest.TestCase):
    """
    match_pair szintű integrációs tesztek a Fix 1 trade-off ágakra – valódi
    (nem mockolt) SIFT/RANSAC eredményt használva egy tökéletes kivágat-forrás
    párra (nagyon magas good_matches/inliers/inlier_ratio), és a cfg küszöbeit
    úgy állítva, hogy determinisztikusan csak egy adott ág (B vagy C) tudjon
    tüzelni.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.cfg = load_default_config()

        source_img = _make_textured_image(seed=42, size=1200)
        self.source_path = root / "source.png"
        cv2.imwrite(str(self.source_path), source_img)

        crop = source_img[250:950, 250:950]
        self.reference_path = root / "reference.png"
        cv2.imwrite(str(self.reference_path), crop)

        detector = create_detector("SIFT", self.cfg)
        # ugyanaz a betöltési út (load_image_smart, CLAHE-vel), mint amit a
        # valós keresési kód (search.py) használ a referenciára
        ref_gray, _ = load_image_smart(
            self.reference_path, max_size=self.cfg.max_process_size,
            min_size=self.cfg.min_process_size, use_clahe=self.cfg.use_clahe,
            clahe_clip_limit=self.cfg.clahe_clip_limit, clahe_tile_size=self.cfg.clahe_tile_size,
        )
        self.kp1, self.des1 = detector.detectAndCompute(ref_gray, None)

        self.cache = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg), persist_dir=None)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_accepts_via_branch_b_when_only_ratio_clears(self):
        # A ág letiltva (elérhetetlen min_inliers), B ág könnyen elérhető
        cfg = self.cfg.replace(
            min_inliers=10**9,
            ratio_compensation_bar=0.5, relaxed_min_inliers=100,
        )
        result = match_pair(self.kp1, self.des1, self.source_path, "SIFT",
                             self.cache, cfg.ratio_test_sift, cfg)
        self.assertTrue(result["success"])
        self.assertEqual(result["decision_reason"], "ACCEPT_RATIO_COMPENSATED")

    def test_accepts_via_branch_c_when_only_inliers_clear(self):
        # A és B ág is letiltva (elérhetetlen ratio-küszöbök), C ág könnyen elérhető
        cfg = self.cfg.replace(
            min_inlier_ratio=10.0, ratio_compensation_bar=10.0,
            inlier_compensation_bar=100, relaxed_min_inlier_ratio=0.5,
        )
        result = match_pair(self.kp1, self.des1, self.source_path, "SIFT",
                             self.cache, cfg.ratio_test_sift, cfg)
        self.assertTrue(result["success"])
        self.assertEqual(result["decision_reason"], "ACCEPT_INLIER_COMPENSATED")

    def test_rejects_when_no_branch_clears(self):
        cfg = self.cfg.replace(
            min_inliers=10**9, min_inlier_ratio=10.0,
            ratio_compensation_bar=10.0, relaxed_min_inliers=10**9,
            inlier_compensation_bar=10**9, relaxed_min_inlier_ratio=10.0,
        )
        result = match_pair(self.kp1, self.des1, self.source_path, "SIFT",
                             self.cache, cfg.ratio_test_sift, cfg)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()

"""
Fix 2: downscale-retry kis natív felbontású referenciákhoz.

Megjegyzés a lefedettségről: a "normál 2. kör elbukik, de a downscale-retry
megtalálja" jelenség valós fotókon (JPEG-tömörítés, nem-sík felület, eltérő
megvilágítás) figyelhető meg – ezt szintetikus, tisztán generált textúrával
(éles kör-alakzatok) nem sikerült megbízhatóan reprodukálni (a szintetikus
körök túl robusztus SIFT-feature-öket adnak, még extrém kicsinyítés után is).
Ezért itt a MECHANIKAI helyességet teszteljük determinisztikusan: a downscale
célméret-képletet, a jelölt-sorrend helyes követését, és hogy a mechanizmus
nem tör el egy egyébként is működő esetet, illetve helyesen utasítja el a
csak idegen jelölteket tartalmazó esetet.
"""

import tempfile
import unittest
from pathlib import Path

import cv2

from image_matcher.config import load_default_config
from image_matcher.detectors import ThreadLocalDetectors, create_detector
from image_matcher.search import rank_candidates_by_score, retry_with_downscale
from tests.test_search_integration import _make_textured_image


class RankCandidatesByScoreTest(unittest.TestCase):
    def test_orders_by_max_score_across_detectors(self):
        p1, p2, p3 = Path("a.jpg"), Path("b.jpg"), Path("c.jpg")
        all_results = [
            {"candidate": p1, "score": 0.5},
            {"candidate": p2, "score": 0.9},
            {"candidate": p1, "score": 0.7},  # p1 legjobb score-ja 0.7, nem 0.5
            {"candidate": p3, "score": 0.1},
        ]
        ranked = rank_candidates_by_score(all_results, [p1, p2, p3])
        self.assertEqual(ranked, [p2, p1, p3])

    def test_missing_candidates_appended_in_fallback_order(self):
        p1, p2 = Path("a.jpg"), Path("b.jpg")
        never_tried = Path("never_tried.jpg")
        all_results = [{"candidate": p1, "score": 0.5}]
        ranked = rank_candidates_by_score(all_results, [never_tried, p1, p2])
        # p1-nek van eredménye -> elöl, a maradék a fallback_order sorrendjében
        self.assertEqual(ranked, [p1, never_tried, p2])


class DownscaleTargetFormulaTest(unittest.TestCase):
    """A célméret-képlet: candidate_processing_size / (downscale_retry_factor * upscale_ratio),
    [min_process_size, max_process_size]-re szorítva."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        # downscale_retry_factor=1.0 -> tiszta, kerekítés nélküli célérték
        self.cfg = load_default_config().replace(downscale_retry_factor=1.0)

        ref_img = _make_textured_image(seed=500, size=1200)
        self.ref_path = root / "reference.png"
        cv2.imwrite(str(self.ref_path), cv2.resize(ref_img, (200, 200)))  # natívan 200px

        # candidate_processing_size = min(1200, max_process_size) = 1200
        candidate_img = _make_textured_image(seed=501, size=1200)
        self.candidate_path = root / "candidate.png"
        cv2.imwrite(str(self.candidate_path), candidate_img)

        self.detectors = {}
        for name in self.cfg.detector_priority:
            det = create_detector(name, self.cfg)
            if det is not None:
                self.detectors[name] = det
        self.thread_detectors = ThreadLocalDetectors(self.cfg)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_target_size_matches_expected_formula(self):
        # upscale_ratio = min_process_size(400) / ref_native(200) = 2.0
        # osztó = downscale_retry_factor(1.0) * 2.0 = 2.0
        # célméret = candidate_processing_size(1200) / 2.0 = 600
        _, _, retry_results = retry_with_downscale(
            self.ref_path, ref_native_longer_side=200,
            ranked_candidates=[self.candidate_path],
            detectors=self.detectors, thread_detectors=self.thread_detectors, cfg=self.cfg,
        )
        self.assertTrue(retry_results)
        self.assertEqual(retry_results[0]["downscale_target_size"], 600)

    def test_target_size_clamped_to_min_process_size(self):
        # nagyon nagy downscale_retry_factor -> a nyers célérték a
        # min_process_size alá esne, ezért fel kell szorítani rá
        cfg = self.cfg.replace(downscale_retry_factor=100.0)
        _, _, retry_results = retry_with_downscale(
            self.ref_path, ref_native_longer_side=200,
            ranked_candidates=[self.candidate_path],
            detectors=self.detectors, thread_detectors=self.thread_detectors, cfg=cfg,
        )
        self.assertTrue(retry_results)
        self.assertEqual(retry_results[0]["downscale_target_size"], cfg.min_process_size)

    def test_target_size_clamped_to_max_process_size(self):
        # nagyon kicsi downscale_retry_factor -> a nyers célérték a
        # max_process_size fölé menne, ezért le kell fogni rá
        cfg = self.cfg.replace(downscale_retry_factor=0.001)
        _, _, retry_results = retry_with_downscale(
            self.ref_path, ref_native_longer_side=200,
            ranked_candidates=[self.candidate_path],
            detectors=self.detectors, thread_detectors=self.thread_detectors, cfg=cfg,
        )
        self.assertTrue(retry_results)
        self.assertEqual(retry_results[0]["downscale_target_size"], cfg.max_process_size)


class RetryWithDownscaleMechanicsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.cfg = load_default_config()

        true_source = _make_textured_image(seed=600, size=1200)
        decoy = _make_textured_image(seed=601, size=1200)
        self.true_source_path = root / "true_source.png"
        self.decoy_path = root / "decoy.png"
        cv2.imwrite(str(self.true_source_path), true_source)
        cv2.imwrite(str(self.decoy_path), decoy)

        crop = true_source[250:950, 250:950]
        self.reference_path = root / "reference.png"
        cv2.imwrite(str(self.reference_path), crop)

        self.detectors = {}
        for name in self.cfg.detector_priority:
            det = create_detector(name, self.cfg)
            if det is not None:
                self.detectors[name] = det
        self.thread_detectors = ThreadLocalDetectors(self.cfg)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_finds_true_source_when_it_is_the_only_ranked_candidate(self):
        best_src, info, retry_results = retry_with_downscale(
            self.reference_path, ref_native_longer_side=200,
            ranked_candidates=[self.true_source_path, self.decoy_path],
            detectors=self.detectors, thread_detectors=self.thread_detectors, cfg=self.cfg,
        )
        self.assertEqual(best_src, self.true_source_path)
        self.assertTrue(info["success"])
        self.assertIn("via_downscale_retry", info["reject_reason"])
        # a decision_reason a match_pair trade-off ágának jelzését őrzi meg,
        # NEM egy flat "ACCEPT_DOWNSCALE_RETRY" string
        self.assertTrue(info["decision_reason"].startswith("ACCEPT_"))

    def test_returns_none_for_pure_decoys(self):
        best_src, info, retry_results = retry_with_downscale(
            self.reference_path, ref_native_longer_side=200,
            ranked_candidates=[self.decoy_path],
            detectors=self.detectors, thread_detectors=self.thread_detectors, cfg=self.cfg,
        )
        self.assertIsNone(best_src)
        self.assertFalse(info["success"])

    def test_stops_at_first_successful_candidate(self):
        # a valódi forrás áll elöl a rangsorban -> a decoy-t meg sem kellene
        # próbálni utána (nincs könnyen megfigyelhető side effect erre, de
        # legalább azt ellenőrizzük, hogy a helyes jelöltet találja meg
        # akkor is, ha nem az egyetlen jelölt)
        best_src, info, retry_results = retry_with_downscale(
            self.reference_path, ref_native_longer_side=200,
            ranked_candidates=[self.true_source_path, self.decoy_path],
            detectors=self.detectors, thread_detectors=self.thread_detectors, cfg=self.cfg,
        )
        tried_candidates = {r["candidate"] for r in retry_results}
        self.assertIn(self.true_source_path, tried_candidates)
        self.assertNotIn(self.decoy_path, tried_candidates)


if __name__ == "__main__":
    unittest.main()

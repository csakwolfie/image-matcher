import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_matcher.cache_disk import DescriptorCache, processing_fingerprint
from image_matcher.config import load_default_config
from image_matcher.detectors import ThreadLocalDetectors


class DescriptorCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.img_path = Path(self.tmpdir.name) / "src.png"
        img = np.random.RandomState(0).randint(0, 255, (400, 400)).astype(np.uint8)
        cv2.imwrite(str(self.img_path), img)
        self.cfg = load_default_config()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_fingerprint_changes_when_detector_params_change(self):
        fp1 = processing_fingerprint(self.cfg)
        cfg2 = self.cfg.replace(sift_nfeatures=1234)
        fp2 = processing_fingerprint(cfg2)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_when_processing_size_changes(self):
        fp1 = processing_fingerprint(self.cfg)
        cfg2 = self.cfg.replace(max_process_size=999)
        fp2 = processing_fingerprint(cfg2)
        self.assertNotEqual(fp1, fp2)

    def test_persistent_cache_hit_across_runs(self):
        persist_dir = Path(self.tmpdir.name) / "descriptors"

        cache1 = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg), persist_dir=persist_dir)
        kps1, des1, _ = cache1.get_descriptors(self.img_path, "ORB")
        self.assertEqual(cache1.persist_misses, 1)
        self.assertEqual(cache1.persist_hits, 0)
        self.assertGreater(len(kps1), 0)

        # Új DescriptorCache példány = egy "másik futtatás" szimulációja
        cache2 = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg), persist_dir=persist_dir)
        kps2, des2, _ = cache2.get_descriptors(self.img_path, "ORB")
        self.assertEqual(cache2.persist_hits, 1)
        self.assertEqual(len(kps1), len(kps2))

    def test_config_change_invalidates_persistent_cache(self):
        persist_dir = Path(self.tmpdir.name) / "descriptors_inval"
        cache1 = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg), persist_dir=persist_dir)
        cache1.get_descriptors(self.img_path, "ORB")

        cfg2 = self.cfg.replace(orb_nfeatures=self.cfg.orb_nfeatures + 500)
        cache2 = DescriptorCache(cfg2, ThreadLocalDetectors(cfg2), persist_dir=persist_dir)
        cache2.get_descriptors(self.img_path, "ORB")
        # más fingerprint -> más fájl -> nem talál a régi cache-ben -> miss, nem hit
        self.assertEqual(cache2.persist_hits, 0)
        self.assertEqual(cache2.persist_misses, 1)

    def test_rebuild_flag_bypasses_disk_cache(self):
        persist_dir = Path(self.tmpdir.name) / "descriptors_rebuild"
        cache1 = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg), persist_dir=persist_dir)
        cache1.get_descriptors(self.img_path, "ORB")

        cache2 = DescriptorCache(self.cfg, ThreadLocalDetectors(self.cfg),
                                  persist_dir=persist_dir, rebuild=True)
        cache2.get_descriptors(self.img_path, "ORB")
        self.assertEqual(cache2.persist_hits, 0)


if __name__ == "__main__":
    unittest.main()

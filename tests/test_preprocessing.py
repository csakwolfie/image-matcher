import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_matcher.config import load_default_config
from image_matcher.preprocessing import list_images, preprocess_directory, to_cache_path


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name) / "src"
        self.root.mkdir()
        for i in range(3):
            img = np.random.RandomState(i).randint(0, 255, (300, 300)).astype(np.uint8)
            cv2.imwrite(str(self.root / f"img{i}.png"), img)
        self.cache_root = Path(self.tmpdir.name) / "cache"
        self.cfg = load_default_config()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_preprocesses_all_images_first_run(self):
        mapping = preprocess_directory(self.root, self.cache_root, 200, "TEST", self.cfg, 2, 90)
        self.assertEqual(len(mapping), 3)
        for cache_path in mapping:
            self.assertTrue(cache_path.exists())

    def test_incremental_run_only_processes_new_files(self):
        preprocess_directory(self.root, self.cache_root, 200, "TEST", self.cfg, 2, 90)
        img = np.random.RandomState(99).randint(0, 255, (300, 300)).astype(np.uint8)
        cv2.imwrite(str(self.root / "img_new.png"), img)
        mapping2 = preprocess_directory(self.root, self.cache_root, 200, "TEST", self.cfg, 2, 90)
        self.assertEqual(len(mapping2), 4)

    def test_force_rebuild_reprocesses_existing_files(self):
        preprocess_directory(self.root, self.cache_root, 200, "TEST2", self.cfg, 2, 90)
        example = to_cache_path(self.root, sorted(list_images(self.root))[0], self.cache_root)
        mtime_before = example.stat().st_mtime
        time.sleep(1.1)  # a mtime-felbontás miatt kell az érdemi különbséghez
        preprocess_directory(self.root, self.cache_root, 200, "TEST2", self.cfg, 2, 90, force=True)
        mtime_after = example.stat().st_mtime
        self.assertGreater(mtime_after, mtime_before)


if __name__ == "__main__":
    unittest.main()

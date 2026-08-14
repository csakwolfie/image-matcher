import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_matcher.gui.live_panel_render import (
    BORDER_FOUND, BORDER_IDLE, BORDER_NEAR_MISS, BORDER_NOT_FOUND, CARD_SIZE,
    build_card, status_color,
)


class StatusColorTest(unittest.TestCase):
    def test_known_statuses_map_to_distinct_colors(self):
        self.assertEqual(status_color("found"), BORDER_FOUND)
        self.assertEqual(status_color("exists"), BORDER_FOUND)
        self.assertEqual(status_color("not_found"), BORDER_NOT_FOUND)
        self.assertEqual(status_color("near_miss"), BORDER_NEAR_MISS)
        self.assertEqual(status_color("error"), BORDER_NOT_FOUND)

    def test_unknown_or_none_status_falls_back_to_idle(self):
        self.assertEqual(status_color(None), BORDER_IDLE)
        self.assertEqual(status_color("processing"), BORDER_IDLE)
        self.assertEqual(status_color("something_unexpected"), BORDER_IDLE)


class BuildCardTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.image_path = self.root / "sample.png"
        arr = (np.random.RandomState(0).rand(200, 300, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(self.image_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_returns_correct_size_image(self):
        img = build_card(None)
        self.assertEqual(img.size, CARD_SIZE)

    def test_no_image_path_renders_placeholder_without_crashing(self):
        img = build_card(None, border_color=BORDER_IDLE)
        self.assertEqual(img.mode, "RGB")

    def test_missing_file_falls_back_to_placeholder_without_crashing(self):
        missing = self.root / "does_not_exist.png"
        img = build_card(missing, border_color=BORDER_FOUND)
        self.assertEqual(img.size, CARD_SIZE)

    def test_valid_image_path_is_composited_in(self):
        placeholder = build_card(None)
        with_image = build_card(self.image_path)
        self.assertEqual(with_image.size, placeholder.size)
        # a placeholder csak a "?" jelet + üres hátteret tartalmazza, a
        # ténylegesen beillesztett fotó pixeleinek jelentősen el kell
        # térnie tőle -- ez a legegyszerűbb "tényleg történt valami" jel
        # anélkül, hogy pixelre pontos referenciaképet kellene tartani.
        diff = np.abs(
            np.array(placeholder, dtype=np.int16) - np.array(with_image, dtype=np.int16)
        )
        self.assertGreater(diff.mean(), 5.0)

    def test_badge_text_does_not_crash_and_changes_pixels(self):
        without_badge = build_card(self.image_path, border_color=BORDER_FOUND)
        with_badge = build_card(self.image_path, border_color=BORDER_FOUND,
                                 badge_text="FOUND", badge_color=BORDER_FOUND)
        diff = np.abs(
            np.array(without_badge, dtype=np.int16) - np.array(with_badge, dtype=np.int16)
        )
        self.assertGreater(diff.sum(), 0)

    def test_custom_size_is_respected(self):
        img = build_card(None, size=(100, 80))
        self.assertEqual(img.size, (100, 80))


if __name__ == "__main__":
    unittest.main()

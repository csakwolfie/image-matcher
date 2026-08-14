import unittest

from image_matcher.config import TUNABLE_KEYS
from image_matcher.gui.profile_fields import BASIC_KEYS, FIELD_SPECS


class FieldSpecsTest(unittest.TestCase):
    def test_every_tunable_key_has_exactly_one_spec(self):
        keys = [f.key for f in FIELD_SPECS]
        self.assertEqual(len(keys), len(set(keys)), "duplicate FieldSpec keys found")
        self.assertEqual(set(keys), set(TUNABLE_KEYS))

    def test_slider_specs_have_valid_ranges(self):
        for spec in FIELD_SPECS:
            if spec.kind == "slider":
                self.assertLess(spec.minimum, spec.maximum, f"{spec.key}: minimum >= maximum")

    def test_known_kinds_only(self):
        valid_kinds = {"bool", "slider", "int_entry", "float_entry", "detector_list"}
        for spec in FIELD_SPECS:
            self.assertIn(spec.kind, valid_kinds, f"{spec.key}: unknown kind '{spec.kind}'")

    def test_exactly_one_detector_list_field(self):
        detector_fields = [f for f in FIELD_SPECS if f.kind == "detector_list"]
        self.assertEqual([f.key for f in detector_fields], ["detector_priority"])


class BasicKeysTest(unittest.TestCase):
    def test_basic_keys_are_a_subset_of_tunable_keys(self):
        self.assertTrue(set(BASIC_KEYS) <= set(TUNABLE_KEYS))

    def test_basic_keys_have_no_duplicates(self):
        self.assertEqual(len(BASIC_KEYS), len(set(BASIC_KEYS)))

    def test_basic_keys_nonempty(self):
        self.assertTrue(BASIC_KEYS)


if __name__ == "__main__":
    unittest.main()

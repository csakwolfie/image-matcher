import tempfile
import unittest
from pathlib import Path

from image_matcher import i18n as i18n_module
from image_matcher.gui.argv_builder import build_argv, validate_form
from image_matcher.i18n import init_translator


class BuildArgvTest(unittest.TestCase):
    def test_only_reference_source_output_present(self):
        argv = build_argv({"reference": "ref", "source": "src", "output": "out"})
        self.assertEqual(argv, ["--reference", "ref", "--source", "src", "--output", "out"])

    def test_blank_limit_and_top_k_are_omitted(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out",
            "limit": "", "top_k": None, "workers": "",
        })
        self.assertNotIn("--limit", argv)
        self.assertNotIn("--top-k", argv)
        self.assertNotIn("--workers", argv)

    def test_numeric_fields_pass_through_as_strings(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out",
            "limit": "5", "workers": "4", "top_k": "10",
        })
        self.assertIn("--limit", argv)
        self.assertEqual(argv[argv.index("--limit") + 1], "5")
        self.assertIn("--workers", argv)
        self.assertEqual(argv[argv.index("--workers") + 1], "4")
        self.assertIn("--top-k", argv)
        self.assertEqual(argv[argv.index("--top-k") + 1], "10")

    def test_checkboxes_map_to_store_true_flags(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out",
            "no_cache": True, "rebuild_cache": True, "dry_run": True,
        })
        self.assertIn("--no-cache", argv)
        self.assertIn("--rebuild-cache", argv)
        self.assertIn("--dry-run", argv)

    def test_unchecked_checkboxes_omitted(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out",
            "no_cache": False, "rebuild_cache": False, "dry_run": False,
        })
        self.assertNotIn("--no-cache", argv)
        self.assertNotIn("--rebuild-cache", argv)
        self.assertNotIn("--dry-run", argv)

    def test_empty_profile_and_lang_omitted(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out",
            "profile": None, "lang": "",
        })
        self.assertNotIn("--profile", argv)
        self.assertNotIn("--lang", argv)

    def test_profile_and_lang_included_when_set(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out",
            "profile": "high_recall", "lang": "en",
        })
        self.assertIn("--profile", argv)
        self.assertEqual(argv[argv.index("--profile") + 1], "high_recall")
        self.assertIn("--lang", argv)
        self.assertEqual(argv[argv.index("--lang") + 1], "en")

    def test_cache_path_included_when_set(self):
        argv = build_argv({
            "reference": "ref", "source": "src", "output": "out", "cache": "my_cache_dir",
        })
        self.assertIn("--cache", argv)
        self.assertEqual(argv[argv.index("--cache") + 1], "my_cache_dir")


class ValidateFormTest(unittest.TestCase):
    """A hibaüzenetek tartalmát a hu.lang.json alapján ellenőrizzük –
    determinisztikus nyelv-állapot érdekében a singletont explicit "hu"-ra
    állítjuk setUp-ban, majd visszaállítjuk tearDown-ban (lásd
    tests/test_i18n.py azonos mintáját)."""

    def setUp(self):
        self.original_translator = i18n_module._translator
        init_translator("hu")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.existing_dir = Path(self.tmpdir.name)

    def tearDown(self):
        i18n_module._translator = self.original_translator
        self.tmpdir.cleanup()

    def test_all_fields_valid_returns_no_errors(self):
        errors = validate_form({
            "reference": str(self.existing_dir), "source": str(self.existing_dir),
            "output": "some/output/dir",
        })
        self.assertEqual(errors, [])

    def test_missing_reference_reports_error(self):
        errors = validate_form({"reference": "", "source": str(self.existing_dir), "output": "out"})
        self.assertEqual(len(errors), 1)
        self.assertIn("Referencia", errors[0])

    def test_missing_source_reports_error(self):
        errors = validate_form({"reference": str(self.existing_dir), "source": "", "output": "out"})
        self.assertEqual(len(errors), 1)
        self.assertIn("Source", errors[0])

    def test_missing_output_reports_error(self):
        errors = validate_form({"reference": str(self.existing_dir), "source": str(self.existing_dir), "output": ""})
        self.assertEqual(len(errors), 1)
        self.assertIn("Kimeneti", errors[0])

    def test_nonexistent_reference_dir_reports_error(self):
        missing = str(self.existing_dir / "nope")
        errors = validate_form({"reference": missing, "source": str(self.existing_dir), "output": "out"})
        self.assertEqual(len(errors), 1)
        self.assertIn(missing, errors[0])

    def test_all_missing_reports_three_errors(self):
        errors = validate_form({"reference": "", "source": "", "output": ""})
        self.assertEqual(len(errors), 3)


if __name__ == "__main__":
    unittest.main()

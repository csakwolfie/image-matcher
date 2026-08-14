import os
import tempfile
import unittest
from pathlib import Path

from image_matcher import config as config_module
from image_matcher.config import (
    apply_cli_overrides, apply_profile, get_default_language, list_profiles,
    load_default_config, resolve_config_path, resolve_profiles_dir,
    set_default_language,
)


class ConfigPrecedenceTest(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_default_config()
        self.assertEqual(cfg.min_good_matches, 200)
        self.assertEqual(cfg.min_inliers, 220)
        self.assertEqual(cfg.min_inlier_ratio, 0.93)
        self.assertEqual(cfg.score_uncertain, 0.97)
        self.assertEqual(cfg.detector_priority, ["SIFT", "AKAZE", "ORB", "BRISK"])

    def test_profile_overrides_only_its_own_keys(self):
        cfg = load_default_config()
        cfg2 = apply_profile(cfg, "high_recall")
        self.assertEqual(cfg2.min_good_matches, 180)
        self.assertEqual(cfg2.min_inliers, 190)
        self.assertEqual(cfg2.min_inlier_ratio, 0.90)
        self.assertEqual(cfg2.score_uncertain, 0.95)
        # a profilban nem szereplő kulcsok a config.yaml értékén maradnak
        self.assertEqual(cfg2.sift_nfeatures, cfg.sift_nfeatures)
        self.assertEqual(cfg2.cache_long_side, cfg.cache_long_side)

    def test_balanced_profile_matches_defaults(self):
        cfg = load_default_config()
        cfg2 = apply_profile(cfg, "balanced")
        self.assertEqual(cfg2.min_good_matches, cfg.min_good_matches)
        self.assertEqual(cfg2.min_inliers, cfg.min_inliers)

    def test_diagnostic_profile_disables_early_accept(self):
        cfg = load_default_config()
        cfg2 = apply_profile(cfg, "diagnostic")
        self.assertGreater(cfg2.score_accept, 10)
        self.assertGreater(cfg2.early_accept_score, 10)
        self.assertLess(cfg2.min_good_matches, cfg.min_good_matches)

    def test_cli_override_beats_profile_and_default(self):
        cfg = load_default_config()
        cfg = apply_profile(cfg, "high_recall")
        cfg = apply_cli_overrides(cfg, stage1_top_k=25)
        self.assertEqual(cfg.stage1_top_k, 25)

    def test_cli_override_none_is_ignored(self):
        cfg = load_default_config()
        cfg = apply_cli_overrides(cfg, stage1_top_k=25)
        cfg2 = apply_cli_overrides(cfg, stage1_top_k=None)
        self.assertEqual(cfg2.stage1_top_k, 25)

    def test_list_profiles_returns_the_three_seed_profiles(self):
        names = {name for name, _ in list_profiles()}
        self.assertEqual(names, {"balanced", "high_recall", "diagnostic"})
        for _, desc in list_profiles():
            self.assertTrue(desc)  # mindegyiknek van leírása

    def test_unknown_profile_raises(self):
        cfg = load_default_config()
        with self.assertRaises(FileNotFoundError):
            apply_profile(cfg, "nonexistent_profile_xyz")


class ConfigDiscoveryOrderTest(unittest.TestCase):
    """
    A resolve_config_path/resolve_profiles_dir precedenciája (első találat
    nyer): explicit > cwd > felhasználói mappa (~/.image_matcher) > a
    csomagba ágyazott gyári alapérték. A cwd-t és a USER_CONFIG_DIR-t
    valós, ideiglenes könyvtárakra állítjuk át teszt közben, majd
    tearDown-ban visszaállítjuk – nincs szükség mockolásra.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.original_cwd = Path.cwd()
        self.original_user_config_dir = config_module.USER_CONFIG_DIR

        self.cwd_dir = self.root / "cwd"
        self.user_dir = self.root / "user"
        self.cwd_dir.mkdir()
        self.user_dir.mkdir()
        os.chdir(self.cwd_dir)
        config_module.USER_CONFIG_DIR = self.user_dir

    def tearDown(self):
        os.chdir(self.original_cwd)
        config_module.USER_CONFIG_DIR = self.original_user_config_dir
        self.tmpdir.cleanup()

    def test_explicit_wins_over_everything(self):
        explicit = self.root / "explicit_config.yaml"
        result = resolve_config_path(explicit)
        self.assertEqual(result, explicit)

    def test_falls_back_to_packaged_default_when_nothing_else_exists(self):
        result = resolve_config_path()
        self.assertEqual(result, config_module.PACKAGED_DATA_DIR / "config.yaml")
        self.assertTrue(result.is_file())

    def test_user_dir_wins_over_packaged_default(self):
        (self.user_dir / "config.yaml").write_text("min_good_matches: 1\n", encoding="utf-8")
        result = resolve_config_path()
        self.assertEqual(result, self.user_dir / "config.yaml")

    def test_cwd_wins_over_user_dir_and_packaged_default(self):
        (self.user_dir / "config.yaml").write_text("min_good_matches: 1\n", encoding="utf-8")
        (self.cwd_dir / "config.yaml").write_text("min_good_matches: 2\n", encoding="utf-8")
        result = resolve_config_path()
        self.assertEqual(result, self.cwd_dir / "config.yaml")

    def test_profiles_dir_same_precedence(self):
        (self.user_dir / "profiles").mkdir()
        result = resolve_profiles_dir()
        self.assertEqual(result, self.user_dir / "profiles")

        (self.cwd_dir / "profiles").mkdir()
        result = resolve_profiles_dir()
        self.assertEqual(result, self.cwd_dir / "profiles")

    def test_profiles_dir_falls_back_to_packaged_default(self):
        result = resolve_profiles_dir()
        self.assertEqual(result, config_module.PACKAGED_DATA_DIR / "profiles")
        self.assertTrue(result.is_dir())

    def test_load_default_config_uses_resolved_path(self):
        (self.cwd_dir / "config.yaml").write_text(
            (config_module.PACKAGED_DATA_DIR / "config.yaml").read_text(encoding="utf-8")
            .replace("min_good_matches: 200", "min_good_matches: 111"),
            encoding="utf-8",
        )
        cfg = load_default_config()
        self.assertEqual(cfg.min_good_matches, 111)


class DefaultLanguageTest(unittest.TestCase):
    """
    get_default_language() / set_default_language() – a --lang kapcsoló
    nélküli alapértelmezett nyelv beolvasása/tartós elmentése a config.yaml
    default_language kulcsába. Ugyanaz a cwd/USER_CONFIG_DIR-patch minta,
    mint ConfigDiscoveryOrderTest-nél, mert set_default_language() a
    resolve_config_path() precedenciáját használja a célfájl kiválasztásához.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.original_cwd = Path.cwd()
        self.original_user_config_dir = config_module.USER_CONFIG_DIR

        self.cwd_dir = self.root / "cwd"
        self.user_dir = self.root / "user"
        self.cwd_dir.mkdir()
        self.user_dir.mkdir()
        os.chdir(self.cwd_dir)
        config_module.USER_CONFIG_DIR = self.user_dir

    def tearDown(self):
        os.chdir(self.original_cwd)
        config_module.USER_CONFIG_DIR = self.original_user_config_dir
        self.tmpdir.cleanup()

    def test_get_default_language_from_packaged_default(self):
        self.assertEqual(get_default_language(), "en")

    def test_get_default_language_falls_back_when_key_missing(self):
        (self.cwd_dir / "config.yaml").write_text(
            (config_module.PACKAGED_DATA_DIR / "config.yaml").read_text(encoding="utf-8")
            .replace("default_language: en\n", ""),
            encoding="utf-8",
        )
        self.assertNotIn("default_language",
                          (self.cwd_dir / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(get_default_language(), "en")  # i18n.DEFAULT_LANG fallback

    def test_set_default_language_creates_full_copy_when_no_override_exists(self):
        saved_path = set_default_language("hu")
        self.assertEqual(saved_path, self.user_dir / "config.yaml")
        self.assertEqual(get_default_language(), "hu")
        # Kritikus regressziós ellenőrzés: a mentés utáni fájl TELJES,
        # érvényes config.yaml marad – egy éles futtatás nem törik el
        # "hiányzó kulcsok" hibával.
        cfg = load_default_config()
        packaged_cfg = load_default_config(config_module.PACKAGED_DATA_DIR / "config.yaml")
        self.assertEqual(cfg.min_good_matches, packaged_cfg.min_good_matches)
        self.assertEqual(cfg.detector_priority, packaged_cfg.detector_priority)

    def test_set_default_language_updates_existing_cwd_override_in_place(self):
        (self.cwd_dir / "config.yaml").write_text(
            (config_module.PACKAGED_DATA_DIR / "config.yaml").read_text(encoding="utf-8")
            .replace("min_good_matches: 200", "min_good_matches: 111"),
            encoding="utf-8",
        )
        saved_path = set_default_language("hu")
        self.assertEqual(saved_path, self.cwd_dir / "config.yaml")
        self.assertFalse((self.user_dir / "config.yaml").exists())
        self.assertEqual(get_default_language(), "hu")
        self.assertEqual(load_default_config().min_good_matches, 111)

    def test_set_default_language_updates_existing_user_override_in_place(self):
        (self.user_dir / "config.yaml").write_text(
            (config_module.PACKAGED_DATA_DIR / "config.yaml").read_text(encoding="utf-8")
            .replace("min_good_matches: 200", "min_good_matches: 222"),
            encoding="utf-8",
        )
        saved_path = set_default_language("hu")
        self.assertEqual(saved_path, self.user_dir / "config.yaml")
        self.assertEqual(get_default_language(), "hu")
        self.assertEqual(load_default_config().min_good_matches, 222)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from pathlib import Path

from image_matcher import config as config_module
from image_matcher.config import (
    TUNABLE_KEYS, apply_profile, list_profiles, load_default_config,
    load_profile_overrides, save_profile,
)


class SaveAndLoadProfileTest(unittest.TestCase):
    """
    `save_profile`/`load_profile_overrides` – a felhasználói mappát
    (`USER_CONFIG_DIR`) egy tmpdir-re patcheljük, hogy a teszt SOSE írjon a
    fejlesztő valódi `~/.image_matcher/`-jébe (lásd
    ConfigDiscoveryOrderTest ugyanezen mintáját test_config.py-ban).
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

    def test_save_profile_writes_into_user_dir(self):
        saved_path = save_profile("my_profile", {"min_good_matches": 150}, "teszt profil")
        self.assertEqual(saved_path, self.user_dir / "profiles" / "my_profile.yaml")
        self.assertTrue(saved_path.is_file())

    def test_save_profile_never_touches_packaged_dir(self):
        save_profile("my_profile", {"min_good_matches": 150})
        packaged_dir = config_module.PACKAGED_DATA_DIR / "profiles"
        self.assertFalse((packaged_dir / "my_profile.yaml").is_file())

    def test_first_save_copies_builtin_profiles_so_they_stay_visible(self):
        # Az user-mappa létrejötte a resolve_profiles_dir() precedenciája
        # miatt (cwd > user > csomagolt, nem összefésülve) elrejtené a
        # beépített profilokat, HA a save_profile() nem másolná be őket
        # elsőként a user-mappába.
        save_profile("my_profile", {"min_good_matches": 150})
        names = {name for name, _ in list_profiles()}
        self.assertEqual(names, {"balanced", "high_recall", "diagnostic", "my_profile"})

    def test_second_save_does_not_reimport_or_overwrite_builtins(self):
        save_profile("first_profile", {"min_good_matches": 150})
        # kézzel "módosítjuk" a felhasználói mappába már bemásolt balanced
        # profilt, majd ellenőrizzük, hogy egy KÖVETKEZŐ save_profile()
        # hívás nem írja felül újra a csomagolt eredetivel
        balanced_path = self.user_dir / "profiles" / "balanced.yaml"
        balanced_path.write_text("min_good_matches: 1\ndescription: kézzel módosítva\n", encoding="utf-8")
        save_profile("second_profile", {"min_good_matches": 150})
        overrides, description = load_profile_overrides("balanced")
        self.assertEqual(overrides, {"min_good_matches": 1})
        self.assertEqual(description, "kézzel módosítva")

    def test_round_trip_overrides_and_description(self):
        overrides = {"min_good_matches": 150, "min_inlier_ratio": 0.8, "use_clahe": False,
                     "detector_priority": ["ORB", "SIFT", "AKAZE", "BRISK"]}
        save_profile("roundtrip_profile", overrides, "kerek-út teszt")
        loaded_overrides, description = load_profile_overrides("roundtrip_profile")
        self.assertEqual(loaded_overrides, overrides)
        self.assertEqual(description, "kerek-út teszt")

    def test_round_trip_without_description(self):
        save_profile("no_desc_profile", {"min_good_matches": 150})
        _, description = load_profile_overrides("no_desc_profile")
        self.assertEqual(description, "")

    def test_unknown_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            save_profile("bad_profile", {"not_a_real_key": 1})

    def test_saved_profile_is_usable_via_apply_profile(self):
        save_profile("applyable_profile", {"min_good_matches": 111})
        cfg = load_default_config()
        cfg2 = apply_profile(cfg, "applyable_profile")
        self.assertEqual(cfg2.min_good_matches, 111)

    def test_saved_profile_appears_in_list_profiles(self):
        save_profile("listed_profile", {"min_good_matches": 111}, "listázási teszt")
        names = {name for name, _ in list_profiles()}
        self.assertIn("listed_profile", names)

    def test_load_overrides_filters_out_unknown_yaml_keys(self):
        # kézzel írunk egy profilt egy nem-tunable extra kulccsal – a
        # load_profile_overrides csendben kihagyja (nem az ő dolga
        # figyelmeztetni, azt az apply_profile() teszi futáskor)
        profile_path = self.user_dir / "profiles" / "manual_profile.yaml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            "min_good_matches: 99\nsome_typo_key: 5\ndescription: kézi\n", encoding="utf-8")
        overrides, description = load_profile_overrides("manual_profile")
        self.assertEqual(overrides, {"min_good_matches": 99})
        self.assertEqual(description, "kézi")

    def test_save_profile_all_tunable_keys_round_trip(self):
        cfg = load_default_config()
        full_overrides = {k: getattr(cfg, k) for k in TUNABLE_KEYS}
        save_profile("full_profile", full_overrides)
        loaded_overrides, _ = load_profile_overrides("full_profile")
        self.assertEqual(loaded_overrides, full_overrides)


if __name__ == "__main__":
    unittest.main()

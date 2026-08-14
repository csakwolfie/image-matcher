import os
import tempfile
import unittest
from pathlib import Path

from image_matcher import i18n as i18n_module
from image_matcher.i18n import (
    Translator, init_translator, list_available_langs, load_translator,
    resolve_lang_dir, t,
)


class TranslatorTest(unittest.TestCase):
    def test_successful_lookup_and_format(self):
        translator = Translator({"greeting": "Hello, {name}!"})
        self.assertEqual(translator("greeting", name="World"), "Hello, World!")

    def test_lookup_without_placeholders(self):
        translator = Translator({"plain": "No placeholders here."})
        self.assertEqual(translator("plain"), "No placeholders here.")

    def test_missing_key_raises_key_error(self):
        translator = Translator({"greeting": "Hello, {name}!"})
        with self.assertRaises(KeyError):
            translator("nonexistent_key_xyz")

    def test_missing_placeholder_raises_key_error(self):
        translator = Translator({"greeting": "Hello, {name}!"})
        with self.assertRaises(KeyError):
            translator("greeting")


class LoadTranslatorTest(unittest.TestCase):
    def test_loads_packaged_hu_lang_file(self):
        translator = load_translator("hu")
        self.assertEqual(
            translator("main.no_profiles"),
            "Nincs elérhető profil a profiles/ mappában.",
        )

    def test_unknown_lang_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_translator("xx_nonexistent")

    def test_module_level_t_resolves_real_keys(self):
        self.assertTrue(t("main.no_profiles"))


class AvailableLangsAndInitTest(unittest.TestCase):
    """
    list_available_langs() felfedezése + init_translator() explicit (nem
    lusta) singleton-inicializálása, ami a --lang kapcsoló bekötésének
    alapja (cli.py). Minden teszt visszaállítja az i18n_module._translator
    singletont a tesztelőtti állapotára, hogy ne szennyezze a többi
    tesztet (lásd LangDirDiscoveryOrderTest ugyanezen mintáját).
    """

    def setUp(self):
        self.original_translator = i18n_module._translator

    def tearDown(self):
        i18n_module._translator = self.original_translator

    def test_lists_the_two_packaged_langs(self):
        self.assertEqual(list_available_langs(), ["en", "hu"])

    def test_returns_empty_list_for_nonexistent_dir(self):
        self.assertEqual(list_available_langs(Path("/nonexistent/lang/dir/xyz")), [])

    def test_init_translator_switches_active_language(self):
        init_translator("hu")
        self.assertEqual(t("main.no_profiles"),
                          "Nincs elérhető profil a profiles/ mappában.")
        init_translator("en")
        self.assertEqual(t("main.no_profiles"),
                          "No profiles available in the profiles/ folder.")

    def test_init_translator_unknown_lang_raises(self):
        with self.assertRaises(FileNotFoundError):
            init_translator("xx_nonexistent")


class LangDirDiscoveryOrderTest(unittest.TestCase):
    """
    A resolve_lang_dir precedenciája (első találat nyer): explicit > cwd >
    felhasználói mappa (~/.image_matcher) > a csomagba ágyazott gyári
    alapérték – ugyanaz a minta, mint a config.py-beli
    ConfigDiscoveryOrderTest-nél.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.original_cwd = Path.cwd()
        self.original_user_config_dir = i18n_module.USER_CONFIG_DIR

        self.cwd_dir = self.root / "cwd"
        self.user_dir = self.root / "user"
        self.cwd_dir.mkdir()
        self.user_dir.mkdir()
        os.chdir(self.cwd_dir)
        i18n_module.USER_CONFIG_DIR = self.user_dir

    def tearDown(self):
        os.chdir(self.original_cwd)
        i18n_module.USER_CONFIG_DIR = self.original_user_config_dir
        self.tmpdir.cleanup()

    def test_explicit_wins_over_everything(self):
        explicit = self.root / "explicit_lang_dir"
        result = resolve_lang_dir(explicit)
        self.assertEqual(result, explicit)

    def test_falls_back_to_packaged_default_when_nothing_else_exists(self):
        result = resolve_lang_dir()
        self.assertEqual(result, i18n_module.PACKAGED_DATA_DIR / "lang")
        self.assertTrue((result / "hu.lang.json").is_file())

    def test_user_dir_wins_over_packaged_default(self):
        (self.user_dir / "lang").mkdir()
        result = resolve_lang_dir()
        self.assertEqual(result, self.user_dir / "lang")

    def test_cwd_wins_over_user_dir_and_packaged_default(self):
        (self.user_dir / "lang").mkdir()
        (self.cwd_dir / "lang").mkdir()
        result = resolve_lang_dir()
        self.assertEqual(result, self.cwd_dir / "lang")


if __name__ == "__main__":
    unittest.main()

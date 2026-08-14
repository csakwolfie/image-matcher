import io
import unittest
from contextlib import redirect_stdout

from image_matcher import i18n as i18n_module
from image_matcher.cli import parse_args
from image_matcher.i18n import t


class LangFlagTest(unittest.TestCase):
    """
    A --lang kapcsoló bekötése (cli.py: _pre_parse_lang/_init_lang). A
    parser felépítése maga is t()-n keresztül készül, ezért a nyelvnek MÁR
    a build_parser() hívás előtt be kell állnia – ezt teszteljük itt azzal,
    hogy parse_args() után egy tetszőleges t() hívás a kért nyelven felel.
    Minden teszt visszaállítja a singletont, hogy ne szennyezze a többi
    tesztet (lásd tests/test_i18n.py azonos mintáját).
    """

    def setUp(self):
        self.original_translator = i18n_module._translator

    def tearDown(self):
        i18n_module._translator = self.original_translator

    def test_default_lang_is_hungarian(self):
        parse_args([])
        self.assertEqual(t("main.no_profiles"),
                          "Nincs elérhető profil a profiles/ mappában.")

    def test_lang_en_switches_translator(self):
        parse_args(["--lang", "en"])
        self.assertEqual(t("main.no_profiles"),
                          "No profiles available in the profiles/ folder.")

    def test_lang_hu_explicit(self):
        parse_args(["--lang", "hu"])
        self.assertEqual(t("main.no_profiles"),
                          "Nincs elérhető profil a profiles/ mappában.")

    def test_unknown_lang_exits_with_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(["--lang", "xx_nonexistent"])
        self.assertEqual(ctx.exception.code, 2)

    def test_lang_flag_survives_alongside_other_args(self):
        args = parse_args(["--lang", "en", "--reference", "ref", "--source", "src",
                            "--output", "out", "--workers", "3"])
        self.assertEqual(args.lang, "en")
        self.assertEqual(args.reference, "ref")
        self.assertEqual(args.workers, 3)
        self.assertEqual(t("main.no_profiles"),
                          "No profiles available in the profiles/ folder.")

    def test_help_text_localized_to_requested_lang(self):
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            parse_args(["--lang", "en", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Load a named profile", buf.getvalue())
        self.assertNotIn("mappából", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

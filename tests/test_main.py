import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_matcher import config as config_module
from image_matcher import i18n as i18n_module
from image_matcher.config import get_default_language
from image_matcher.main import main


class BareLangSetsDefaultTest(unittest.TestCase):
    """
    `--lang <kód>` ÖNMAGÁBAN (reference/source/output/list-profiles/dry-run
    nélkül) kiadva nem a "hiányzó kötelező kapcsolók" hibával áll le, hanem
    tartósan elmenti a nyelvet a config.yaml default_language kulcsába és
    0-val tér vissza. cwd/USER_CONFIG_DIR-t tmpdir-re patcheljük, hogy a
    teszt SOSE írjon a fejlesztő valódi ~/.image_matcher/ mappájába, és a
    globális fordító-singletont is visszaállítjuk (lásd test_i18n.py azonos
    mintáját).
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.original_cwd = Path.cwd()
        self.original_user_config_dir = config_module.USER_CONFIG_DIR
        self.original_translator = i18n_module._translator

        self.cwd_dir = self.root / "cwd"
        self.user_dir = self.root / "user"
        self.cwd_dir.mkdir()
        self.user_dir.mkdir()
        os.chdir(self.cwd_dir)
        config_module.USER_CONFIG_DIR = self.user_dir

    def tearDown(self):
        os.chdir(self.original_cwd)
        config_module.USER_CONFIG_DIR = self.original_user_config_dir
        i18n_module._translator = self.original_translator
        self.tmpdir.cleanup()

    def test_bare_lang_returns_zero_and_persists(self):
        exit_code = main(["--lang", "en"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(get_default_language(), "en")
        self.assertTrue((self.user_dir / "config.yaml").is_file())

    def test_subsequent_run_without_lang_uses_new_default(self):
        main(["--lang", "en"])
        # Egy KÖVETKEZŐ, --lang nélküli hívás (pl. --list-profiles) már az
        # imént elmentett "en" alapértelmezettet kell használja.
        main(["--list-profiles"])
        self.assertEqual(get_default_language(), "en")

    def test_lang_with_list_profiles_does_not_persist(self):
        # --lang egy VALÓDI művelet (--list-profiles) mellett csak az adott
        # futtatásra vonatkozik, nem menti el tartós alapértelmezettként.
        exit_code = main(["--lang", "en", "--list-profiles"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(get_default_language(), "hu")
        self.assertFalse((self.user_dir / "config.yaml").exists())

    def test_missing_required_args_error_unaffected_without_lang(self):
        exit_code = main([])
        self.assertEqual(exit_code, 1)
        self.assertFalse((self.user_dir / "config.yaml").exists())


class CancelAndPauseTest(unittest.TestCase):
    """
    A `main()`/`_search()` `cancel_event`/`pause_event` opcionális,
    kulcsszavas paraméterei – a jövőbeli GUI Szünet/Folytatás/Leállítás
    gombjainak alapja. A meglévő tesztek egy referenciás, kis szintetikus
    kép-halmazzal futnak – itt nem a találat MINŐSÉGE számít (lehet akár
    NOT_FOUND is), csak az, hogy a ciklus valóban blokkol/megszakad a
    megfelelő ponton.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.ref_dir = self.root / "ref"
        self.src_dir = self.root / "src"
        self.output_dir = self.root / "out"
        self.ref_dir.mkdir()
        self.src_dir.mkdir()

        img = (np.random.RandomState(1).rand(400, 400) * 255).astype(np.uint8)
        cv2.imwrite(str(self.ref_dir / "ref1.png"), img[50:300, 50:300])
        cv2.imwrite(str(self.src_dir / "source1.png"), img)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _argv(self):
        return ["--reference", str(self.ref_dir), "--source", str(self.src_dir),
                "--output", str(self.output_dir), "--workers", "1"]

    def test_cancel_event_set_before_loop_processes_zero_references(self):
        cancel_event = threading.Event()
        cancel_event.set()
        exit_code = main(self._argv(), cancel_event=cancel_event)
        self.assertEqual(exit_code, 0)
        results_csv = self.output_dir / "results.csv"
        self.assertTrue(results_csv.is_file())
        lines = [l for l in results_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)  # csak a fejléc, feldolgozott referencia-sor nélkül

    def test_pause_event_blocks_until_cleared(self):
        pause_event = threading.Event()
        pause_event.set()
        cancel_event = threading.Event()

        def clear_after_delay():
            time.sleep(0.3)
            pause_event.clear()

        threading.Thread(target=clear_after_delay, daemon=True).start()

        start = time.time()
        exit_code = main(self._argv(), cancel_event=cancel_event, pause_event=pause_event)
        elapsed = time.time() - start

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(elapsed, 0.3)
        results_csv = self.output_dir / "results.csv"
        lines = [l for l in results_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)  # fejléc + a feldolgozott 1 referencia

    def test_cancel_while_paused_stops_without_hanging(self):
        pause_event = threading.Event()
        pause_event.set()
        cancel_event = threading.Event()

        def cancel_after_delay():
            time.sleep(0.2)
            cancel_event.set()

        threading.Thread(target=cancel_after_delay, daemon=True).start()

        exit_code = main(self._argv(), cancel_event=cancel_event, pause_event=pause_event)
        self.assertEqual(exit_code, 0)
        results_csv = self.output_dir / "results.csv"
        lines = [l for l in results_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)  # a szünet alatt jött a cancel, 0 referencia dolgozódott fel


if __name__ == "__main__":
    unittest.main()

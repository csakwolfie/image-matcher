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
from image_matcher.main import ReferenceResult, main


def _make_textured_image(seed: int, size: int = 1200) -> np.ndarray:
    """Kellően "dús" szintetikus textúra, hogy SIFT sok stabil keypointot
    találjon (ugyanaz a minta, mint test_search_integration.py-ban)."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 255, (size, size), dtype=np.uint8)
    img = cv2.GaussianBlur(noise, (0, 0), sigmaX=3)
    for _ in range(60):
        x, y = rng.randint(60, size - 60, size=2)
        r = int(rng.randint(10, 45))
        color = int(rng.randint(0, 255))
        cv2.circle(img, (int(x), int(y)), r, color, -1)
    return img


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
        exit_code = main(["--lang", "hu", "--list-profiles"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(get_default_language(), "en")
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


class OnResultCallbackTest(unittest.TestCase):
    """
    Az `on_result` opcionális callback (lásd `main.ReferenceResult`) – a
    GUI élő találat-panelje ebből épül fel (`gui/worker.py` egy queue-ba
    teszi, `gui/app.py` a főszálon dolgozza fel). Itt közvetlenül
    `main()`-t hívjuk egy szintetikus, kontrollált adathalmazon, hogy
    minden `status` ág (processing/found/not_found/exists) legalább
    egyszer ténylegesen lefusson és a megfelelő mezőkkel érkezzen.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.ref_dir = self.root / "ref"
        self.src_dir = self.root / "src"
        self.output_dir = self.root / "out"
        self.ref_dir.mkdir()
        self.src_dir.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _argv(self):
        return ["--reference", str(self.ref_dir), "--source", str(self.src_dir),
                "--output", str(self.output_dir), "--workers", "2"]

    def test_found_case_emits_processing_then_found_with_details(self):
        true_source = _make_textured_image(seed=1)
        cv2.imwrite(str(self.src_dir / "source1.png"), true_source)
        cv2.imwrite(str(self.ref_dir / "ref1.png"), true_source[250:950, 250:950])

        events = []
        exit_code = main(self._argv(), on_result=events.append)

        self.assertEqual(exit_code, 0)
        statuses = [(e.reference, e.status) for e in events]
        self.assertIn(("ref1.png", "processing"), statuses)
        self.assertIn(("ref1.png", "found"), statuses)

        processing_event = next(e for e in events if e.status == "processing")
        self.assertEqual(processing_event.reference_path, self.ref_dir / "ref1.png")

        found_event = next(e for e in events if e.status == "found")
        self.assertEqual(found_event.matched_path, self.src_dir / "source1.png")
        self.assertIsNotNone(found_event.detector)
        self.assertGreater(found_event.good_matches, 0)
        self.assertGreater(found_event.inliers, 0)
        self.assertGreater(found_event.score, 0.0)

    def test_not_found_case_emits_processing_then_not_found(self):
        cv2.imwrite(str(self.src_dir / "source1.png"), _make_textured_image(seed=1))
        cv2.imwrite(str(self.ref_dir / "ref1.png"), _make_textured_image(seed=99))

        events = []
        exit_code = main(self._argv(), on_result=events.append)

        self.assertEqual(exit_code, 0)
        statuses = [(e.reference, e.status) for e in events]
        self.assertIn(("ref1.png", "processing"), statuses)
        self.assertIn(("ref1.png", "not_found"), statuses)

        not_found_event = next(e for e in events if e.status == "not_found")
        self.assertEqual(not_found_event.reference, "ref1.png")

    def test_exists_case_on_second_run_reports_saved_filename(self):
        true_source = _make_textured_image(seed=1)
        cv2.imwrite(str(self.src_dir / "source1.png"), true_source)
        cv2.imwrite(str(self.ref_dir / "ref1.png"), true_source[250:950, 250:950])

        first_events = []
        main(self._argv(), on_result=first_events.append)
        self.assertTrue(any(e.status == "found" for e in first_events))

        second_events = []
        exit_code = main(self._argv(), on_result=second_events.append)
        self.assertEqual(exit_code, 0)

        exists_event = next(e for e in second_events if e.status == "exists")
        self.assertEqual(exists_event.reference, "ref1.png")
        # a found/-ban a referencia tövére átnevezve (más kiterjesztéssel is végződhetne,
        # itt png-png, mert a szintetikus forrás is .png-ként lett elmentve)
        self.assertEqual(exists_event.matched_path, self.output_dir / "found" / "ref1.png")

    def test_on_result_none_by_default_does_not_error(self):
        cv2.imwrite(str(self.src_dir / "source1.png"), _make_textured_image(seed=1))
        cv2.imwrite(str(self.ref_dir / "ref1.png"), _make_textured_image(seed=99))
        exit_code = main(self._argv())
        self.assertEqual(exit_code, 0)

    def test_found_case_with_nested_subfolders_reports_correct_full_paths(self):
        """Regressziós teszt: `preprocessing.list_images()` REKURZÍVAN
        (almappákkal együtt) fedezi fel a képeket – a `matched_path`/
        `reference_path` mezőknek a TELJES, a mappa-mélységet is
        tartalmazó útvonalat kell adniuk, nem csak a puszta fájlnevet
        (amiből a GUI korábban tévesen `forrás_mappa/fájlnév`-et épített
        vissza, ami egy almappában lévő fájlra sosem létező útvonalra
        mutatott)."""
        true_source = _make_textured_image(seed=1)
        nested_src_dir = self.src_dir / "album" / "2026"
        nested_src_dir.mkdir(parents=True)
        cv2.imwrite(str(nested_src_dir / "source1.png"), true_source)

        nested_ref_dir = self.ref_dir / "crops"
        nested_ref_dir.mkdir(parents=True)
        cv2.imwrite(str(nested_ref_dir / "ref1.png"), true_source[250:950, 250:950])

        events = []
        exit_code = main(self._argv(), on_result=events.append)
        self.assertEqual(exit_code, 0)

        processing_event = next(e for e in events if e.status == "processing")
        self.assertEqual(processing_event.reference_path, nested_ref_dir / "ref1.png")
        self.assertTrue(processing_event.reference_path.is_file())

        found_event = next(e for e in events if e.status == "found")
        self.assertEqual(found_event.matched_path, nested_src_dir / "source1.png")
        self.assertTrue(found_event.matched_path.is_file())


if __name__ == "__main__":
    unittest.main()

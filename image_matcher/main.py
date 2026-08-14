"""Orchestráció: a teljes keresési folyamat összefűzése a modulokból."""

from __future__ import annotations

import glob
import shutil
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from .cache_disk import DescriptorCache
from .cli import parse_args
from .config import (
    Config, apply_cli_overrides, apply_profile, list_profiles, load_default_config,
    set_default_language,
)
from .detectors import ThreadLocalDetectors, create_detector
from .i18n import t
from .image_io import get_native_longer_side
from .preprocessing import list_images, preprocess_directory, to_cache_path
from .reporting import write_candidates_csv, write_results_csv
from .search import (
    find_best_match_for_reference, rank_candidates_by_score,
    retry_with_downscale, stage1_rank_candidates,
)

FOUND_DIR_NAME = "found"
RESULTS_CSV_NAME = "results.csv"
CANDIDATES_CSV_NAME = "results_candidates.csv"
CACHE_SUBDIR_REF = "reference"
CACHE_SUBDIR_SRC = "source"


@dataclass(frozen=True)
class ReferenceResult:
    """
    Egy referencia feldolgozásának strukturált eseménye – az `on_result`
    callback (lásd `main()`) ezt adja át, MINDIG a főszálon biztonságosan
    feldolgozható formában (a callback maga a keresési háttérszálon fut,
    ezért NEM szabad belőle közvetlenül Tkinter-widgetet módosítani – lásd
    `gui/worker.py`, ami ezt egy queue-ba teszi a `_poll_queue`-nak).

    `status`: "processing" (a referencia feldolgozása épp elkezdődött,
    még nincs eredmény) | "found" | "not_found" | "near_miss" |
    "exists" (korábbi futásból már megvolt) | "error".

    `matched`: a `status`-tól függően más mappában keresendő fájlnév –
    "found"/"near_miss"/"not_found" esetén a SOURCE mappában (az eredeti
    fájlnév), "exists" esetén a FOUND mappában (a `main.py` a mentéskor
    átnevezi a referencia tövére, lásd `_search()`), egyébként `None`.
    """
    reference: str
    status: str
    matched: Optional[str] = None
    detector: Optional[str] = None
    good_matches: int = 0
    inliers: int = 0
    score: float = 0.0


OnResultCallback = Callable[[ReferenceResult], None]


class _Tee:
    """Minden írást több stream-re (pl. konzol + logfájl) is továbbít."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def format_duration_compact(seconds: float) -> str:
    """Másodperc → rögzített szélességű 'MM:SS' (vagy 'HH:MM:SS', ha van óra) –
    az állapotsorhoz, ahol a folyamatos frissítés miatt fontos, hogy a szöveg
    hossza ne ugráljon."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# Azok a DecisionReason-ök, amik "közel volt, de nem ért fel a küszöbig"
# esetet jeleznek (RANSAC talált geometriailag plauzibilis illesztést, csak
# az inlier-szám/-arány vagy a score maradt a küszöb alatt) – ez adja a
# konzolon a NEAR MISS jelzést. Minden más elutasítási ok (nem volt elég
# good match, RANSAC nem talált homográfiát, "mágnes-kép" védelem tüzelt)
# NOT FOUND-nak számít, mert ott nem volt érdemi geometriai egyezés.
NEAR_MISS_DECISION_REASONS = frozenset({"REJECT_INLIER_RATIO", "REJECT_SCORE"})


class _StatusBar:
    """
    Egysoros, folyamatosan a helyén frissülő állapotsor a terminál alján
    (\\r-rel felülírva magát) – KIZÁRÓLAG a valódi konzolra ír
    (sys.__stdout__), nem a naplófájlba is toló _Tee-n keresztül, mert egy
    fájlban az ismétlődő \\r-es felülírás olvashatatlan lenne. Ha a kimenet
    nincs valódi terminálhoz kötve (átirányítva/pipe-olva), automatikusan
    kikapcsol, hogy ne szennyezze össze az átirányított tartalmat.
    """

    def __init__(self, total: int):
        self.total = total
        self.start_time = time.time()
        self.found = 0
        self.not_found = 0
        self.current = 0
        self._last_len = 0
        self._stream = sys.__stdout__
        is_tty = getattr(self._stream, "isatty", None)
        self._enabled = bool(self._stream) and callable(is_tty) and is_tty()

    def _render(self) -> str:
        elapsed = time.time() - self.start_time
        avg = elapsed / self.current if self.current else 0.0
        remaining = avg * (self.total - self.current)
        pct = 100.0 * self.current / self.total if self.total else 0.0
        return (f"Progress: {self.current}/{self.total} ({pct:.1f}%) | "
                f"{format_duration_compact(elapsed)} elapsed | "
                f"ETA {format_duration_compact(remaining)} | "
                f"FOUND {self.found} | NOT FOUND {self.not_found}")

    def refresh(self) -> None:
        if not self._enabled:
            return
        line = self._render()
        pad = max(0, self._last_len - len(line))
        self._stream.write("\r" + line + (" " * pad))
        self._stream.flush()
        self._last_len = len(line)

    def clear(self) -> None:
        """A sor kitörlése, mielőtt egy normál (soremeléses) blokk-üzenet íródna."""
        if not self._enabled or self._last_len == 0:
            return
        self._stream.write("\r" + (" " * self._last_len) + "\r")
        self._stream.flush()
        self._last_len = 0

    def advance(self, found: bool) -> None:
        self.current += 1
        if found:
            self.found += 1
        else:
            self.not_found += 1


def _fmt_idx(idx: int, total: int) -> str:
    width = len(str(total))
    return f"[{idx:0{width}d}/{total}]"


def _print_exists(idx: int, total: int, ref_name: str, existing_name: str) -> None:
    print(f"{_fmt_idx(idx, total)} {ref_name}  ✓ EXISTS  → {existing_name}")


def _print_found(idx: int, total: int, ref_name: str, matched_name: str, detector: str,
                  good: int, inliers: int, score: float, retry_note: str = "") -> None:
    print(f"{_fmt_idx(idx, total)} {ref_name}  ✓ FOUND{retry_note}")
    print(f"  → {matched_name}  (det={detector})")
    print(f"  good={good} | inliers={inliers} | score={score:.3f}")


def _print_near_miss(idx: int, total: int, ref_name: str, best_name: str,
                      good: int, inliers: int, score: float, reason: str) -> None:
    print(f"{_fmt_idx(idx, total)} {ref_name}  ~ NEAR MISS")
    print(f"  → {best_name}")
    print(f"  good={good} | inliers={inliers} | score={score:.3f}")
    print(f"  reason: {reason}")


def _print_not_found(idx: int, total: int, ref_name: str, candidate_count: int,
                      best_name: str, good: int, inliers: int, score: float, reason: str) -> None:
    print(f"{_fmt_idx(idx, total)} {ref_name}  ✗ NOT FOUND")
    if best_name:
        print(f"  candidates={candidate_count} | best={best_name} | good={good}")
        print(f"  inliers={inliers} | score={score:.3f}")
    else:
        print(f"  candidates={candidate_count}")
    print(f"  reason: {reason}")


def _print_error(idx: int, total: int, ref_name: str, message: str) -> None:
    print(f"{_fmt_idx(idx, total)} {ref_name}  ✗ ERROR")
    print(f"  reason: {message}")


def resolve_config(args) -> Config:
    cfg = load_default_config()
    if args.profile:
        cfg = apply_profile(cfg, args.profile)
    cfg = apply_cli_overrides(cfg, stage1_top_k=args.top_k)
    return cfg


def print_effective_config(cfg: Config, args) -> None:
    print(t("main.effective_settings_header"))
    print(f"  profile              : {args.profile or t('main.effective_profile_none')}")
    print(f"  detector_priority    : {cfg.detector_priority}")
    print(f"  min_good_matches     : {cfg.min_good_matches}")
    print(f"  min_inliers          : {cfg.min_inliers}")
    print(f"  min_inlier_ratio     : {cfg.min_inlier_ratio}")
    print(f"  score_uncertain      : {cfg.score_uncertain}")
    print(f"  score_accept         : {cfg.score_accept}")
    print(f"  early_accept_score   : {cfg.early_accept_score}")
    print(f"  cache_long_side      : {cfg.cache_long_side}")
    print(f"  stage1_top_k         : {cfg.stage1_top_k}")
    print(f"  use_clahe            : {cfg.use_clahe}")
    print(f"  use_homography_check : {cfg.use_homography_check}")
    print(f"  workers              : {args.workers}")


def _ensure_utf8_console() -> None:
    """
    Windows-on a konzol stream kódolása alapból a rendszer aktív kódlapja
    (pl. cp1250), NEM UTF-8 – ez a legtöbb ékezetes (á/é/ő/ű stb.) karaktert
    olvashatatlanná teszi, amint a kimenet nem közvetlenül egy natív
    konzolba megy (átirányítva, pipe-olva, más terminálba). A naplófájlt ez
    nem érinti (az explicit encoding="utf-8"-tal nyílik), csak a konzolra
    írt szöveget – ezért itt, a legelső print() előtt kényszerítjük UTF-8-ra.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and (getattr(stream, "encoding", "") or "").lower() != "utf-8":
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass  # nem kritikus – ha nem sikerül, marad a rendszer alapértelmezése


def main(argv=None, *, cancel_event: Optional[threading.Event] = None,
         pause_event: Optional[threading.Event] = None,
         on_result: Optional[OnResultCallback] = None) -> int:
    """
    `cancel_event`/`pause_event`/`on_result` opcionális, kulcsszavas-csak
    paraméterek – a CLI (`cli.py`/`run.py`) sosem ad át ilyet (marad
    `None`), tehát a parancssori viselkedés ettől függetlenül, bitre
    pontosan változatlan. A GUI (`image_matcher/gui/worker.py`) adja át
    őket: a Szünet/Folytatás/Leállítás gombok a `_search()` per-referencia
    ciklusát tudják vezérelni (lásd ott a pontos ellenőrzést), az
    `on_result` pedig referenciánként egy `ReferenceResult`-ot kap (lásd
    ott), a GUI élő találat-panelje ebből épül fel.
    """
    _ensure_utf8_console()
    args = parse_args(argv)

    # --lang kiadva ÖNMAGÁBAN (semmi olyan kapcsoló nélkül, ami tényleges
    # munkát végezne) nem futtat semmit – ehelyett tartósan elmenti a kért
    # nyelvet a config.yaml default_language kulcsába, hogy a --lang
    # nélküli, jövőbeli futtatások is ezt használják (lásd
    # config.set_default_language).
    if args.lang and not (args.reference or args.source or args.output
                           or args.list_profiles or args.dry_run):
        saved_path = set_default_language(args.lang)
        print(t("main.lang_default_set", lang=args.lang, path=saved_path))
        return 0

    if args.list_profiles:
        profiles = list_profiles()
        if not profiles:
            print(t("main.no_profiles"))
        else:
            print(t("main.profiles_header"))
            for name, desc in profiles:
                print(f"  {name:<15} {desc}")
        return 0

    try:
        cfg = resolve_config(args)
    except Exception as e:
        print(t("main.error_config_load", error=e))
        return 1

    if not args.reference or not args.source or not args.output:
        print(t("main.error_missing_required_args"))
        return 1

    ref_dir = Path(args.reference)
    src_dir = Path(args.source)
    output_dir = Path(args.output)

    if not ref_dir.is_dir():
        print(t("main.error_reference_dir_missing", ref_dir=ref_dir))
        return 1
    if not src_dir.is_dir():
        print(t("main.error_source_dir_missing", src_dir=src_dir))
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    found_dir = output_dir / FOUND_DIR_NAME
    results_path = output_dir / RESULTS_CSV_NAME
    candidates_path = output_dir / CANDIDATES_CSV_NAME

    log_path = output_dir / f"log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_file_handle = open(log_path, "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, log_file_handle)

    try:
        return _run(args, cfg, ref_dir, src_dir, output_dir, found_dir,
                     results_path, candidates_path, log_path,
                     cancel_event=cancel_event, pause_event=pause_event, on_result=on_result)
    finally:
        sys.stdout = real_stdout
        log_file_handle.close()


def _run(args, cfg: Config, ref_dir: Path, src_dir: Path, output_dir: Path,
          found_dir: Path, results_path: Path, candidates_path: Path, log_path: Path, *,
          cancel_event: Optional[threading.Event] = None,
          pause_event: Optional[threading.Event] = None,
          on_result: Optional[OnResultCallback] = None) -> int:
    print("=" * 70)
    print("  High-Accuracy Classical OpenCV Image Matcher")
    print(t("main.banner_subtitle"))
    print(t("main.banner_version", version=__version__))
    print(t("main.banner_workers", workers=args.workers))
    print("=" * 70)
    print(t("main.log_file", log_path=log_path))
    print(t("main.reference_dir", ref_dir=ref_dir))
    print(t("main.source_dir", src_dir=src_dir))
    print(t("main.output_dir", output_dir=output_dir))

    print_effective_config(cfg, args)

    no_cache = args.no_cache
    temp_cache_dir = None
    if no_cache:
        temp_cache_dir = Path(tempfile.mkdtemp(prefix="image_matcher_cache_"))
        cache_root = temp_cache_dir
        descriptor_persist_dir = None
    else:
        cache_root = Path(args.cache) if args.cache else (output_dir / "cache")
        descriptor_persist_dir = cache_root / "descriptors"

    ref_cache_root = cache_root / CACHE_SUBDIR_REF / str(cfg.cache_long_side)
    src_cache_root = cache_root / CACHE_SUBDIR_SRC / str(cfg.cache_long_side)

    print(t("main.cache_dir", cache_root=cache_root,
             temp_note=t("main.cache_dir_temp_note") if no_cache else "",
             cache_long_side=cfg.cache_long_side))
    print(t("main.found_dir", found_dir=found_dir))
    print(t("main.results_csv", results_path=results_path))
    print(t("main.candidates_csv", candidates_path=candidates_path))

    try:
        return _search(args, cfg, ref_dir, src_dir, found_dir, results_path,
                        candidates_path, cache_root, ref_cache_root, src_cache_root,
                        descriptor_persist_dir, cancel_event=cancel_event, pause_event=pause_event,
                        on_result=on_result)
    finally:
        if temp_cache_dir is not None:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)


def _search(args, cfg: Config, ref_dir: Path, src_dir: Path, found_dir: Path,
             results_path: Path, candidates_path: Path, cache_root: Path,
             ref_cache_root: Path, src_cache_root: Path, descriptor_persist_dir, *,
             cancel_event: Optional[threading.Event] = None,
             pause_event: Optional[threading.Event] = None,
             on_result: Optional[OnResultCallback] = None) -> int:
    max_workers = args.workers

    print(t("main.detectors_init"))
    detectors = {}
    for name in cfg.detector_priority:
        det = create_detector(name, cfg)
        if det is not None:
            detectors[name] = det
            print(t("main.detector_ok", name=name))
        else:
            print(t("main.detector_unavailable", name=name))

    if not detectors:
        print(t("main.error_no_detectors"))
        return 1

    first_det_name = cfg.detector_priority[0]
    if first_det_name not in detectors:
        first_det_name = next(n for n in cfg.detector_priority if n in detectors)

    if args.dry_run:
        ref_count = len(list_images(ref_dir))
        src_count = len(list_images(src_dir))
        if args.limit is not None:
            ref_count = min(ref_count, max(0, args.limit))
        print(t("main.reference_count", count=ref_count))
        print(t("main.source_count", count=src_count))
        print(t("main.dry_run_done"))
        return 0 if (ref_count and src_count) else 1

    print(t("main.preprocessing_start"))
    ref_cache_map = preprocess_directory(
        ref_dir, ref_cache_root, cfg.cache_long_side, "REFERENCE",
        cfg, max_workers, cfg.cache_jpeg_quality, force=args.rebuild_cache,
    )
    src_cache_map = preprocess_directory(
        src_dir, src_cache_root, cfg.cache_long_side, "SOURCE",
        cfg, max_workers, cfg.cache_jpeg_quality, force=args.rebuild_cache,
    )

    ref_files = sorted(ref_cache_map.values())
    src_cache_files = list(src_cache_map.keys())

    if args.limit is not None:
        ref_files = ref_files[:max(0, args.limit)]
        print(t("main.limit_applied", count=len(ref_files)))

    print(t("main.reference_count", count=len(ref_files)))
    print(t("main.source_count", count=len(src_cache_files)))

    if not ref_files:
        print(t("main.error_no_reference_images"))
        return 1
    if not src_cache_files:
        print(t("main.error_no_source_images"))
        return 1

    found_dir.mkdir(parents=True, exist_ok=True)

    thread_detectors = ThreadLocalDetectors(cfg)
    cache = DescriptorCache(cfg, thread_detectors, persist_dir=descriptor_persist_dir,
                             rebuild=args.rebuild_cache)
    if descriptor_persist_dir is not None:
        print(t("main.descriptor_cache_persistent", path=descriptor_persist_dir))

    print(t("main.stage1_cache_build", detector=first_det_name, workers=max_workers))
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(cache.get_descriptors, src, first_det_name) for src in src_cache_files]
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 100 == 0 or done == len(src_cache_files):
                print(f"  {done}/{len(src_cache_files)} ...")
    print(t("main.stage1_cache_done"))

    results = []
    candidates_detail = []
    found_count = 0
    skipped_count = 0

    print(t("main.search_start", count=len(ref_files)))
    print("-" * 70)

    bar = _StatusBar(len(ref_files))
    bar.refresh()

    for idx, ref_path in enumerate(ref_files, 1):
        if cancel_event is not None and cancel_event.is_set():
            bar.clear()
            print(t("main.cancelled", done=idx - 1, total=len(ref_files)))
            break
        if pause_event is not None and pause_event.is_set():
            bar.clear()
            print(t("main.paused", done=idx - 1, total=len(ref_files)))
            while pause_event.is_set() and not (cancel_event is not None and cancel_event.is_set()):
                time.sleep(0.1)
            if cancel_event is not None and cancel_event.is_set():
                print(t("main.cancelled", done=idx - 1, total=len(ref_files)))
                break
            print(t("main.resumed"))
            bar.refresh()

        ref_name = ref_path.name
        if on_result is not None:
            on_result(ReferenceResult(reference=ref_name, status="processing"))

        try:
            already_found = list(found_dir.glob(glob.escape(ref_path.stem) + ".*"))
            if already_found:
                bar.clear()
                _print_exists(idx, len(ref_files), ref_name, already_found[0].name)
                skipped_count += 1
                results.append({
                    "reference": ref_name, "matched": None,
                    "saved_as": already_found[0].name,
                    "good_matches": 0, "inliers": 0, "score": 0.0, "skipped": True,
                })
                if on_result is not None:
                    on_result(ReferenceResult(reference=ref_name, status="exists",
                                               matched=already_found[0].name))
                bar.advance(found=True)
                bar.refresh()
                continue

            ref_cache_path = to_cache_path(ref_dir, ref_path, ref_cache_root)

            candidate_cache_scored, stage1_diag = stage1_rank_candidates(
                ref_cache_path, src_cache_files, first_det_name, cache,
                cfg.stage1_top_k, cfg, thread_detectors, max_workers,
            )

            if not candidate_cache_scored:
                bar.clear()
                _print_not_found(idx, len(ref_files), ref_name, 0, "", 0, 0, 0.0, stage1_diag)
                results.append({
                    "reference": ref_name, "matched": None,
                    "good_matches": 0, "inliers": 0, "score": 0.0,
                    "near_miss_file": "", "near_miss_good": "",
                    "near_miss_inliers": "", "near_miss_score": "",
                    "stage1_diag": stage1_diag, "stage1_candidate_count": 0,
                    "winning_detector": "", "decision_reason": "REJECT_NO_INLIERS",
                    "reject_reason": stage1_diag,
                })
                if on_result is not None:
                    on_result(ReferenceResult(reference=ref_name, status="not_found"))
                bar.advance(found=False)
                bar.refresh()
                continue

            stage1_info_by_original: Dict[Path, Tuple[int, float]] = {}
            candidate_originals = []
            for rank, (c, s1_score) in enumerate(candidate_cache_scored, start=1):
                orig = src_cache_map[c]
                candidate_originals.append(orig)
                stage1_info_by_original[orig] = (rank, s1_score)

            best_src, info, near_miss, all_results = find_best_match_for_reference(
                ref_path, candidate_originals, cache, detectors, cfg, max_workers,
            )

            # Fix 2: tartalék kör kicsi NATÍV felbontású referenciákra – csak
            # akkor fut, ha a normál 2. kör nem talált elfogadható egyezést
            # ÉS a referencia natív hosszabb oldala a küszöb alatt van.
            retry_results: List[Dict[str, Any]] = []
            used_downscale_retry = False
            normal_success = (best_src is not None and info["success"]
                               and info["score"] >= cfg.score_uncertain)
            if cfg.use_downscale_retry and not normal_success:
                ref_native = get_native_longer_side(ref_path)
                if ref_native is not None and ref_native < cfg.small_reference_native_threshold:
                    ranked = rank_candidates_by_score(all_results, candidate_originals)
                    retry_src, retry_info, retry_results = retry_with_downscale(
                        ref_path, ref_native, ranked, detectors, thread_detectors, cfg,
                    )
                    if retry_src is not None:
                        best_src, info = retry_src, retry_info
                        used_downscale_retry = True

            for r in all_results:
                orig = r["candidate"]
                rank, s1_score = stage1_info_by_original.get(orig, (None, None))
                is_winner = (best_src is not None and orig == best_src
                             and r["detector"] == info.get("detector") and r["success"])
                candidates_detail.append({
                    "reference": ref_name, "stage1_rank": rank, "stage1_score": s1_score,
                    "candidate": str(orig), "detector": r["detector"],
                    "good_matches": r["good_matches"], "inliers": r["inliers"],
                    "inlier_ratio": r["inlier_ratio"], "stage2_score": r["score"],
                    "success": r["success"], "reject_reason": r["reject_reason"],
                    "decision_reason": r.get("decision_reason", ""), "is_winner": is_winner,
                })

            for r in retry_results:
                orig = r["candidate"]
                is_winner = (best_src is not None and orig == best_src
                             and r["detector"] == info.get("detector") and r["success"])
                candidates_detail.append({
                    "reference": ref_name, "stage1_rank": None, "stage1_score": None,
                    "candidate": str(orig), "detector": r["detector"],
                    "good_matches": r["good_matches"], "inliers": r["inliers"],
                    "inlier_ratio": r["inlier_ratio"], "stage2_score": r["score"],
                    "success": r["success"], "reject_reason": r["reject_reason"],
                    "decision_reason": r.get("decision_reason", ""), "is_winner": is_winner,
                })

            if best_src is not None and info["success"] and info["score"] >= cfg.score_uncertain:
                dest_name = ref_path.stem + best_src.suffix
                dest = found_dir / dest_name
                if not dest.exists():
                    shutil.copy2(best_src, dest)
                found_count += 1
                retry_note = "  [downscale-retry]" if used_downscale_retry else ""
                bar.clear()
                _print_found(idx, len(ref_files), ref_name, best_src.name, info["detector"],
                             info["good_matches"], info["inliers"], info["score"], retry_note)
                results.append({
                    "reference": ref_name, "matched": best_src.name, "saved_as": dest_name,
                    "good_matches": info["good_matches"], "inliers": info["inliers"],
                    "score": info["score"], "stage1_candidate_count": len(candidate_originals),
                    "winning_detector": info["detector"], "reject_reason": "",
                    "decision_reason": info.get("decision_reason", ""),
                })
                if on_result is not None:
                    on_result(ReferenceResult(
                        reference=ref_name, status="found", matched=best_src.name,
                        detector=info["detector"], good_matches=info["good_matches"],
                        inliers=info["inliers"], score=info["score"],
                    ))
                bar.advance(found=True)
                bar.refresh()
            else:
                near_miss_name = near_miss["src_path"].name if near_miss["src_path"] is not None else ""
                if info.get("src_path") is not None:
                    top_reject_reason = info.get("reject_reason", "")
                    top_decision_reason = info.get("decision_reason", "")
                    top_name = info["src_path"].name
                    top_good, top_inliers, top_score = info["good_matches"], info["inliers"], info["score"]
                elif near_miss["src_path"] is not None:
                    top_reject_reason = near_miss.get("reject_reason", "")
                    top_decision_reason = near_miss.get("decision_reason", "")
                    top_name = near_miss_name
                    top_good, top_inliers, top_score = (
                        near_miss["good_matches"], near_miss["inliers"], near_miss["score"],
                    )
                else:
                    top_reject_reason = t("main.no_common_point")
                    top_decision_reason = "REJECT_NO_INLIERS"
                    top_name, top_good, top_inliers, top_score = "", 0, 0, 0.0

                bar.clear()
                is_near_miss = bool(top_name) and top_decision_reason in NEAR_MISS_DECISION_REASONS
                if is_near_miss:
                    _print_near_miss(idx, len(ref_files), ref_name, top_name,
                                      top_good, top_inliers, top_score, top_reject_reason)
                else:
                    _print_not_found(idx, len(ref_files), ref_name, len(candidate_originals),
                                      top_name, top_good, top_inliers, top_score, top_reject_reason)
                results.append({
                    "reference": ref_name, "matched": None,
                    "good_matches": 0, "inliers": 0, "score": 0.0,
                    "near_miss_file": near_miss_name,
                    "near_miss_good": near_miss["good_matches"] if near_miss["src_path"] is not None else "",
                    "near_miss_inliers": near_miss["inliers"] if near_miss["src_path"] is not None else "",
                    "near_miss_score": f"{near_miss['score']:.4f}" if near_miss["src_path"] is not None else "",
                    "stage1_candidate_count": len(candidate_originals), "winning_detector": "",
                    "reject_reason": top_reject_reason, "decision_reason": top_decision_reason,
                })
                if on_result is not None:
                    on_result(ReferenceResult(
                        reference=ref_name, status="near_miss" if is_near_miss else "not_found",
                        matched=top_name or None, good_matches=top_good,
                        inliers=top_inliers, score=top_score,
                    ))
                bar.advance(found=False)
                bar.refresh()

        except Exception as e:
            bar.clear()
            _print_error(idx, len(ref_files), ref_name, str(e))
            traceback.print_exc()
            results.append({
                "reference": ref_name, "matched": None,
                "good_matches": 0, "inliers": 0, "score": 0.0,
            })
            if on_result is not None:
                on_result(ReferenceResult(reference=ref_name, status="error"))
            bar.advance(found=False)
            bar.refresh()

    bar.clear()
    write_results_csv(results, results_path)
    write_candidates_csv(candidates_detail, candidates_path)

    print("-" * 70)
    print(t("main.done_header"))
    print(t("main.summary_found_this_run", found=found_count, total=len(ref_files)))
    print(t("main.summary_skipped", skipped=skipped_count))
    print(t("main.summary_total_in_found", total_found=found_count + skipped_count, total=len(ref_files)))
    print(t("main.summary_found_dir", found_dir=found_dir))
    print(t("main.summary_results_csv", results_path=results_path))
    print(t("main.summary_candidates_csv", candidates_path=candidates_path, rows=len(candidates_detail)))
    print(t("main.summary_cache_dir", cache_root=cache_root))
    if cache.persist_dir is not None:
        total = cache.persist_hits + cache.persist_misses
        pct = (100.0 * cache.persist_hits / total) if total else 0.0
        print(t("main.summary_descriptor_cache", hits=cache.persist_hits, total=total, pct=pct))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

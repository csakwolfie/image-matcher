"""
Kétlépcsős keresés:
  1. kör – gyors jelöltkeresés a leskálázott cache-képeken, egyetlen
     detektorral, kizárás nélkül (csak rangsorolás).
  2. kör – pontos, szigorú ellenőrzés az EREDETI, teljes felbontású
     fájlokon, minden elérhető detektorral, prioritási sorrendben,
     early-accept-tel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .cache_disk import DescriptorCache
from .config import Config
from .detectors import ThreadLocalDetectors, is_binary_descriptor
from .i18n import t
from .image_io import get_native_longer_side, load_image_smart
from .matching import match_pair


def stage1_rank_candidates(
    ref_cache_path: Path,
    source_cache_files: List[Path],
    detector_name: str,
    cache: DescriptorCache,
    top_k: int,
    cfg: Config,
    thread_detectors: ThreadLocalDetectors,
    max_workers: int,
) -> Tuple[List[Tuple[Path, float]], str]:
    """
    Gyors előszűrés a leskálázott cache-képeken, csak az első (legnagyobb
    prioritású) detektorral. NEM zár ki egyetlen forrásképet sem abszolút
    good-match küszöb alapján – minden forrást pontoz, és a top_k
    legjobbat viszi tovább. A végső, szigorú döntést a 2. kör
    (find_best_match_for_reference) hozza meg az EREDETI fájlokon.

    Visszaadja: ([(cache_path, stage1_score), ...] a top_k legjobb, score
    szerint csökkenő sorrendben, diagnosztikai szöveg).
    """
    ref_gray, _ = load_image_smart(
        ref_cache_path,
        max_size=cfg.cache_long_side,
        min_size=cfg.min_process_size,
        use_clahe=cfg.use_clahe,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_size=cfg.clahe_tile_size,
    )
    if ref_gray is None:
        return [], t("search.error_ref_cache_not_loadable", name=ref_cache_path.name)

    h, w = ref_gray.shape[:2]

    try:
        kp1, des1 = thread_detectors.get(detector_name).detectAndCompute(ref_gray, None)
    except Exception as e:
        return [], t("search.error_feature_computation_failed", error=e)

    n_kp1 = len(kp1) if kp1 else 0
    if des1 is None or n_kp1 < 4:
        return [], t("search.error_too_few_features", count=n_kp1, w=w, h=h)

    scored: List[Tuple[Path, float]] = []
    # A technikai minimumot el nem érő próbálkozásokat is nyomon követjük
    # (Fix 3: diagnosztikai bővítés) – ha végül SENKI nem éri el a küszöböt,
    # ebből tudunk egy nyers rangsort adni a "nincs jelölt" üzenet mellé,
    # ahelyett hogy csak egy generikus szöveget kapnánk.
    raw_leaderboard: List[Tuple[Path, int, int]] = []  # (src_cache, good_matches, src_keypoints)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                match_pair, kp1, des1, src_cache, detector_name, cache, cfg.stage1_ratio, cfg,
                cfg.stage1_min_good_matches, 0, 0.0, 0.0
            ): src_cache
            for src_cache in source_cache_files
        }
        for fut in as_completed(futures):
            src_cache = futures[fut]
            res = fut.result()
            raw_leaderboard.append((src_cache, res["good_matches"], res.get("src_keypoints", 0)))
            if res["good_matches"] >= cfg.stage1_min_good_matches:
                scored.append((src_cache, res["score"]))

    if not scored:
        raw_leaderboard.sort(key=lambda x: x[1], reverse=True)
        top = raw_leaderboard[:5]
        detail = "; ".join(f"{p.name}: good={g}, src_kp={k}" for p, g, k in top)
        return [], t("search.stage1_no_candidates", count=n_kp1,
                      min_good=cfg.stage1_min_good_matches, detail=detail)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], ""


def find_best_match_for_reference(
    ref_path: Path,
    source_files: List[Path],
    cache: DescriptorCache,
    detectors: Dict[str, cv2.Feature2D],
    cfg: Config,
    max_workers: int,
) -> Tuple[Optional[Path], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Több detektor prioritási sorrendben. Early accept, ha már nagyon jó a
    score. Bizonytalan esetben következő detektor.

    Visszaad egy (best_src, best_overall, best_raw, all_results) négyest:
      - best_overall: a ténylegesen elfogadható (score-alapú) legjobb találat
      - best_raw: a legmagasabb NYERS good-matches értékű jelölt, FÜGGETLENÜL
        attól, hogy elérte-e a min_good_matches küszöböt – near-miss
        diagnosztikához.
      - all_results: MINDEN ténylegesen kipróbált (jelölt, detektor) páros
        teljes eredménye – jelölt-szintű naplózáshoz.
    """
    empty_entry: Dict[str, Any] = {
        "src_path": None, "detector": None, "good_matches": 0, "inliers": 0,
        "inlier_ratio": 0.0, "score": 0.0, "success": False,
        "reject_reason": "", "decision_reason": "",
    }
    best_overall = dict(empty_entry)
    best_raw = dict(empty_entry)
    all_results: List[Dict[str, Any]] = []

    for det_name in cfg.detector_priority:
        if det_name not in detectors:
            continue
        ratio = cfg.ratio_test_sift if not is_binary_descriptor(det_name) else cfg.ratio_test_bin

        # Referencia descriptorok EGYSZER számolva ehhez a detektorhoz.
        ref_gray, _ = load_image_smart(
            ref_path,
            max_size=cfg.max_process_size,
            min_size=cfg.min_process_size,
            use_clahe=cfg.use_clahe,
            clahe_clip_limit=cfg.clahe_clip_limit,
            clahe_tile_size=cfg.clahe_tile_size,
        )
        if ref_gray is None:
            continue
        try:
            kp1, des1 = detectors[det_name].detectAndCompute(ref_gray, None)
        except Exception:
            continue

        candidates = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(match_pair, kp1, des1, src_path, det_name, cache, ratio, cfg): src_path
                for src_path in source_files
            }
            for fut in as_completed(futures):
                src_path = futures[fut]
                res = fut.result()
                all_results.append({
                    "candidate": src_path,
                    "detector": det_name,
                    "good_matches": res["good_matches"],
                    "inliers": res["inliers"],
                    "inlier_ratio": res["inlier_ratio"],
                    "score": res["score"],
                    "success": res["success"],
                    "reject_reason": res.get("reject_reason", ""),
                    "decision_reason": res.get("decision_reason", ""),
                })
                if res["success"] or res["score"] > best_overall["score"]:
                    candidates.append((src_path, res))
                # Near-miss nyomon követés: nyers good_matches alapján, a
                # belső min_good_matches kaputól függetlenül.
                if res["good_matches"] > best_raw["good_matches"]:
                    best_raw = {
                        "src_path": src_path, "detector": det_name,
                        "good_matches": res["good_matches"], "inliers": res["inliers"],
                        "inlier_ratio": res["inlier_ratio"], "score": res["score"],
                        "success": res["success"], "reject_reason": res.get("reject_reason", ""),
                        "decision_reason": res.get("decision_reason", ""),
                    }

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        top_src, top_res = candidates[0]

        if top_res["score"] > best_overall["score"]:
            best_overall = {
                "src_path": top_src, "detector": det_name,
                "good_matches": top_res["good_matches"], "inliers": top_res["inliers"],
                "inlier_ratio": top_res["inlier_ratio"], "score": top_res["score"],
                "success": top_res["success"], "reject_reason": top_res.get("reject_reason", ""),
                "decision_reason": top_res.get("decision_reason", ""),
            }

        # Early accept – nagyon erős találat
        if best_overall["score"] >= cfg.early_accept_score and best_overall["success"]:
            break

        # Ha már elfogadható, nem megyünk tovább a lassabb/erősebb detektorokra
        if best_overall["score"] >= cfg.score_accept and best_overall["success"]:
            break

    return best_overall["src_path"], best_overall, best_raw, all_results


# =============================================================================
#  FIX 2 – TARTALÉK KÖR KICSI NATÍV FELBONTÁSÚ REFERENCIÁKHOZ
# =============================================================================

def rank_candidates_by_score(
    all_results: List[Dict[str, Any]], fallback_order: List[Path],
) -> List[Path]:
    """
    A 2. körben már kiszámolt (jelölt, detektor) eredmények alapján
    rangsorolja a jelölteket (a legjobb score az összes kipróbált detektor
    közül), csökkenő sorrendben – a downscale-retry (retry_with_downscale)
    ezt használja jelölt-sorrendként, az 1. kör újrafuttatása nélkül.
    A `fallback_order`-ben szereplő, de `all_results`-ban nem szereplő
    jelöltek (pl. ha egyetlen detektor sem futott le rájuk) a lista végére
    kerülnek, a fallback_order sorrendjében.
    """
    best_by_candidate: Dict[Path, float] = {}
    for r in all_results:
        c = r["candidate"]
        if c not in best_by_candidate or r["score"] > best_by_candidate[c]:
            best_by_candidate[c] = r["score"]
    ranked = sorted(best_by_candidate, key=lambda p: best_by_candidate[p], reverse=True)
    missing = [p for p in fallback_order if p not in best_by_candidate]
    return ranked + missing


def retry_with_downscale(
    ref_path: Path,
    ref_native_longer_side: int,
    ranked_candidates: List[Path],
    detectors: Dict[str, cv2.Feature2D],
    thread_detectors: ThreadLocalDetectors,
    cfg: Config,
) -> Tuple[Optional[Path], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Tartalék kör kicsi NATÍV felbontású referenciákhoz (lásd DEVLOG "Fix 2").
    A JELÖLT forrást skálázza le szelektíven (nem a referenciát), hogy a
    forrás finomrészletei – amikhez a natívan kicsi referencián nincs
    megfelelő – ne dominálják a matchinget. Ugyanazokat a szigorú
    küszöböket/trade-off ágakat (match_pair) használja, mint a normál 2. kör
    – csak a bemenet változik, nem az elfogadási logika.

    A jelölteket a megadott (legvalószínűbbtől a legkevésbé valószínűig
    rangsorolt) sorrendben próbálja, mindegyikre a teljes detector_priority
    sorrendet, és megáll az ELSŐ sikeres találatnál. NEM használja a
    perzisztens descriptor cache-t – minden jelölthöz eltérő, egyszeri
    (ritkán újrahasznosítható) feldolgozási méret kell, ezért jelöltenként
    friss, csak-memóriás DescriptorCache-t épít (persist_dir=None).

    Visszaad egy (best_src, best_overall, all_retry_results) hármast, a
    find_best_match_for_reference-hez hasonló szerkezetben (best_raw nélkül,
    mivel ez csak egy célzott, néhány jelöltes tartalék kör, nem a teljes
    korpuszt átfogó rangsorolás).
    """
    empty_entry: Dict[str, Any] = {
        "src_path": None, "detector": None, "good_matches": 0, "inliers": 0,
        "inlier_ratio": 0.0, "score": 0.0, "success": False,
        "reject_reason": "", "decision_reason": "",
    }
    all_retry_results: List[Dict[str, Any]] = []

    upscale_ratio = cfg.min_process_size / float(ref_native_longer_side)
    divisor = cfg.downscale_retry_factor * upscale_ratio
    if divisor <= 0:
        return None, dict(empty_entry), all_retry_results

    # Referencia descriptorok detektoronként EGYSZER, a jelölt-ciklus előtt –
    # a referencia feldolgozása NEM változik ebben a javításban, csak a
    # jelölt forrásé.
    ref_descriptors: Dict[str, Tuple[List, Any]] = {}
    for det_name in cfg.detector_priority:
        if det_name not in detectors:
            continue
        ref_gray, _ = load_image_smart(
            ref_path, max_size=cfg.max_process_size, min_size=cfg.min_process_size,
            use_clahe=cfg.use_clahe, clahe_clip_limit=cfg.clahe_clip_limit,
            clahe_tile_size=cfg.clahe_tile_size,
        )
        if ref_gray is None:
            continue
        try:
            kp1, des1 = detectors[det_name].detectAndCompute(ref_gray, None)
        except Exception:
            continue
        ref_descriptors[det_name] = (kp1, des1)

    for candidate_path in ranked_candidates:
        candidate_native = get_native_longer_side(candidate_path)
        if candidate_native is None:
            continue
        candidate_processing_size = min(candidate_native, cfg.max_process_size)
        target_size = candidate_processing_size / divisor
        target_size = int(round(max(cfg.min_process_size, min(cfg.max_process_size, target_size))))

        retry_cfg = cfg.replace(max_process_size=target_size)
        retry_cache = DescriptorCache(retry_cfg, thread_detectors, persist_dir=None)

        for det_name in cfg.detector_priority:
            if det_name not in ref_descriptors:
                continue
            kp1, des1 = ref_descriptors[det_name]
            ratio = cfg.ratio_test_sift if not is_binary_descriptor(det_name) else cfg.ratio_test_bin

            res = match_pair(kp1, des1, candidate_path, det_name, retry_cache, ratio, retry_cfg)
            all_retry_results.append({
                "candidate": candidate_path, "detector": det_name,
                "good_matches": res["good_matches"], "inliers": res["inliers"],
                "inlier_ratio": res["inlier_ratio"], "score": res["score"],
                "success": res["success"], "reject_reason": res.get("reject_reason", ""),
                "decision_reason": res.get("decision_reason", ""),
                "downscale_target_size": target_size,
            })
            if res["success"]:
                overall = {
                    "src_path": candidate_path, "detector": det_name,
                    "good_matches": res["good_matches"], "inliers": res["inliers"],
                    "inlier_ratio": res["inlier_ratio"], "score": res["score"],
                    "success": True,
                    # A downscale-retry tényét a reject_reason mezőbe írjuk
                    # (sikeres találatnál amúgy is üres) – így egyszerre
                    # látszik MELYIK trade-off ág (decision_reason) fogadta
                    # el, ÉS hogy kellett-e hozzá butítás, új CSV-oszlop
                    # nélkül.
                    "reject_reason": f"via_downscale_retry(target={target_size}px)",
                    "decision_reason": res.get("decision_reason", ""),
                }
                return candidate_path, overall, all_retry_results

    return None, dict(empty_entry), all_retry_results

"""
Descriptor matching, RANSAC geometriai ellenőrzés, homográfia-plauzibilitás,
score-számítás és a DecisionReason kategorizálás.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .cache_disk import DescriptorCache
from .config import Config
from .detectors import is_binary_descriptor


def match_descriptors(
    des1: Optional[np.ndarray],
    des2: Optional[np.ndarray],
    detector_name: str,
    ratio: float
) -> List[cv2.DMatch]:
    """KNN matching + Lowe ratio test. Binary → Hamming, float → L2."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    binary = is_binary_descriptor(detector_name)

    try:
        if binary:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Float (SIFT) → FLANN, gyorsabb nagy descriptor-halmazoknál
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=64)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # des1 = reference, des2 = source
        knn = matcher.knnMatch(des1, des2, k=2)

        good = []
        for pair in knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
            elif len(pair) == 1:
                good.append(pair[0])  # ritka eset

        return good

    except Exception:
        try:
            norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
            matcher = cv2.BFMatcher(norm, crossCheck=True)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches[:max(50, len(matches) // 3)]
        except Exception:
            return []


def is_homography_plausible(H: Optional[np.ndarray], cfg: Config) -> Tuple[bool, str]:
    """
    Durva plauzibilitás-ellenőrzés a RANSAC-kal becsült homográfiára –
    védelem periodikus/repetitív mintázatú ("mágnes-kép") hamis geometriai
    illesztései ellen. Visszaad egy (plauzibilis?, ok) párt – az ok "" ha
    plauzibilis, egyébként "scale" vagy "shear".
    """
    if H is None:
        return False, "shear"
    try:
        A = H[:2, :2]
        det = float(np.linalg.det(A))
        if det <= 0:
            return False, "shear"  # tükrözés

        s = np.linalg.svd(A, compute_uv=False)
        if s[1] <= 1e-9:
            return False, "shear"  # elfajult transzformáció

        if s[0] < cfg.min_homography_scale or s[0] > cfg.max_homography_scale:
            return False, "scale"
        if s[1] < cfg.min_homography_scale or s[1] > cfg.max_homography_scale:
            return False, "scale"

        shear_ratio = s[0] / s[1]
        if shear_ratio > cfg.max_homography_shear_ratio:
            return False, "shear"

        return True, ""
    except Exception:
        return False, "shear"


def geometric_verification(
    kp1: List, kp2: List, good_matches: List[cv2.DMatch], cfg: Config,
    reproj_thresh: Optional[float] = None, min_good_matches: Optional[int] = None,
) -> Tuple[int, float, Optional[np.ndarray], str]:
    """
    RANSAC homográfia. Visszaadja: (inlier_count, inlier_ratio, mask, reason).
    A `reason` üres string sikeres illesztésnél, egyébként a bukás oka.
    """
    reproj = cfg.ransac_reproj if reproj_thresh is None else reproj_thresh
    mgm = cfg.min_good_matches if min_good_matches is None else min_good_matches
    if len(good_matches) < mgm:
        return 0, 0.0, None, "too_few_good_matches"

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj,
            maxIters=cfg.ransac_max_iters,
            confidence=cfg.ransac_conf
        )
        if mask is None:
            return 0, 0.0, None, "ransac_no_homography"

        if cfg.use_homography_check:
            plausible, implausible_sub_reason = is_homography_plausible(H, cfg)
            if not plausible:
                return 0, 0.0, None, f"implausible_homography_{implausible_sub_reason}"

        inliers = int(mask.sum())
        ratio = inliers / float(len(good_matches))
        return inliers, ratio, mask, ""
    except Exception:
        return 0, 0.0, None, "ransac_exception"


def compute_score(good_matches: int, inliers: int, inlier_ratio: float) -> float:
    """
    Kombinált score a döntéshez.
    - good_matches normalizálva ~50-re
    - inlier_ratio erősen súlyozva
    """
    gm_norm = min(good_matches / 50.0, 1.5)  # 50 good match ≈ 1.0
    score = 0.40 * gm_norm + 0.60 * inlier_ratio
    if inliers >= 25:
        score += 0.08
    elif inliers >= 15:
        score += 0.04
    return float(score)


def classify_decision(
    geo_reason: str, inliers: int, min_inliers: int,
    inlier_ratio: float, min_inlier_ratio: float,
    score: float, score_threshold: float, success: bool,
    decision_strong_ratio: float,
    accept_branch: Optional[str] = None,
) -> str:
    """
    Tömör, gépileg is szűrhető kategória a részletes (szöveges)
    reject_reason mellé. `accept_branch` ("A"/"B"/"C"/None) jelzi, melyik
    trade-off ág (lásd match_pair) fogadta el a találatot – ez adja a
    ACCEPT_RATIO_COMPENSATED / ACCEPT_INLIER_COMPENSATED megkülönböztetést.
    """
    if success:
        if accept_branch == "B":
            return "ACCEPT_RATIO_COMPENSATED"
        if accept_branch == "C":
            return "ACCEPT_INLIER_COMPENSATED"
        return "ACCEPT_STRONG_GEOMETRY" if inlier_ratio >= decision_strong_ratio else "ACCEPT_INLIER"

    if geo_reason in ("too_few_good_matches", "ransac_no_homography", "ransac_exception"):
        return "REJECT_NO_INLIERS"
    if geo_reason == "implausible_homography_scale":
        return "REJECT_SCALE"
    if geo_reason == "implausible_homography_shear":
        return "REJECT_HOMOGRAPHY"
    if inliers == 0:
        return "REJECT_NO_INLIERS"
    if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return "REJECT_INLIER_RATIO"
    if score < score_threshold:
        return "REJECT_SCORE"
    return "REJECT_UNKNOWN"


def match_pair(
    kp1: List,
    des1: Optional[np.ndarray],
    src_path: Path,
    detector_name: str,
    cache: DescriptorCache,
    ratio: float,
    cfg: Config,
    min_good_matches: Optional[int] = None,
    min_inliers: Optional[int] = None,
    min_inlier_ratio: Optional[float] = None,
    score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Egy detektorral összehasonlít egy (előre kiszámolt) referencia-descriptor
    halmazt egy source-képpel. A min_*/score_threshold paraméterek
    felülírhatók lazábbra a gyors, kis-felbontású 1. körös jelölt-kereséshez
    (lásd search.stage1_rank_candidates); alapból cfg szigorú, globális
    küszöbeit használja (2. kör, pontos döntés).
    """
    mgm = cfg.min_good_matches if min_good_matches is None else min_good_matches
    mi = cfg.min_inliers if min_inliers is None else min_inliers
    mir = cfg.min_inlier_ratio if min_inlier_ratio is None else min_inlier_ratio
    st = cfg.score_uncertain if score_threshold is None else score_threshold

    result: Dict[str, Any] = {
        "detector": detector_name,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "score": 0.0,
        "success": False,
        "reject_reason": "",
        "decision_reason": "REJECT_NO_INLIERS",
        "src_keypoints": 0,
    }

    if des1 is None or len(kp1) < 4:
        result["reject_reason"] = "ref_no_features"
        return result

    kp2, des2, _ = cache.get_descriptors(src_path, detector_name)
    result["src_keypoints"] = len(kp2) if kp2 else 0
    if des2 is None or len(kp2) < 4:
        result["reject_reason"] = "src_no_features"
        return result

    good = match_descriptors(des1, des2, detector_name, ratio)
    result["good_matches"] = len(good)

    if len(good) < mgm:
        result["reject_reason"] = f"good_matches {len(good)} < MIN_GOOD_MATCHES {mgm}"
        return result

    inliers, inlier_ratio, _, geo_reason = geometric_verification(
        kp1, kp2, good, cfg, min_good_matches=mgm
    )
    result["inliers"] = inliers
    result["inlier_ratio"] = inlier_ratio
    result["score"] = compute_score(len(good), inliers, inlier_ratio)

    # Három elfogadási ág, bármelyik teljesülése elég (a score-kapu mindegyikben
    # kötelező) – lásd DEVLOG "Fix 1": a korábbi egyetlen ÉS-kapu (csak A ág)
    # kizárt olyan igazolt valós találatokat, ahol a score már rég a
    # score_uncertain fölött volt, csak az inliers/inlier_ratio nyers értéke
    # bukott egy hajszállal. A B/C ág küszöbei mindig cfg-ből jönnek (NEM az
    # mi/mir felülírt paraméterekből), így a stage1 laza hívásai nem
    # érintettek általuk (ott mi=0/mir=0.0 miatt az A ág amúgy is triviálisan
    # teljesül, ha van RANSAC-siker).
    branch_a = inliers >= mi and inlier_ratio >= mir
    branch_b = inlier_ratio >= cfg.ratio_compensation_bar and inliers >= cfg.relaxed_min_inliers
    branch_c = inliers >= cfg.inlier_compensation_bar and inlier_ratio >= cfg.relaxed_min_inlier_ratio

    fired_branch: Optional[str] = None
    if branch_a:
        fired_branch = "A"
    elif branch_b:
        fired_branch = "B"
    elif branch_c:
        fired_branch = "C"

    if fired_branch is not None and result["score"] >= st:
        result["success"] = True
        result["reject_reason"] = ""
    else:
        fired_branch = None
        reasons = []
        if geo_reason:
            reasons.append(f"geometry:{geo_reason}")
        if inliers < mi:
            reasons.append(f"inliers {inliers} < MIN_INLIERS {mi}")
        if inlier_ratio < mir:
            reasons.append(f"inlier_ratio {inlier_ratio:.3f} < MIN_INLIER_RATIO {mir}")
        if result["score"] < st:
            reasons.append(f"score {result['score']:.3f} < SCORE_UNCERTAIN {st}")
        result["reject_reason"] = "; ".join(reasons) if reasons else "unknown"

    result["decision_reason"] = classify_decision(
        geo_reason, inliers, mi, inlier_ratio, mir, result["score"], st,
        result["success"], cfg.decision_strong_ratio, accept_branch=fired_branch,
    )

    return result

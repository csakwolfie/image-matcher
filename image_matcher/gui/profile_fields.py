"""
`FieldSpec`/`FIELD_SPECS`: a `config.TUNABLE_KEYS` mind a 46 kulcsához egy
kézzel kurátort widget-specifikáció (csúszka vs. beviteli mező, és ha
csúszka, milyen tartománnyal) — ez dönti el a profilszerkesztő ablak
(`profile_editor.py`) mező-elrendezését. A tartományok forrása a meglévő
`image_matcher/data/config.yaml` inline kommentjei (pl. a
`downscale_retry_factor`-nál dokumentált 0.3–3.0 sweep-tartomány).

Besorolási elv:
  - 0–1 közeli, jól értelmezhető arány/küszöb → "slider"
  - szűk, dokumentált tartományú egyéb float/kis int → "slider"
  - nyitott végű/nagy számosság (feature-szám, iterációszám, pixelméret) →
    "int_entry"/"float_entry" (csúszkánál itt nem lenne ésszerű felső
    korlát)
  - log-skálájú/extrém dinamikatartományú float (pl. `akaze_threshold`) →
    "float_entry" (csúszka félrevezető lenne)
  - bool → "bool"
  - `detector_priority` (lista) → "detector_list" (sorrendezhető lista)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import TUNABLE_KEYS


@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: str  # "bool" | "slider" | "int_entry" | "float_entry" | "detector_list"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01


FIELD_SPECS: List[FieldSpec] = [
    # --- Kép-előfeldolgozás ---
    FieldSpec("max_process_size", kind="int_entry"),
    FieldSpec("min_process_size", kind="int_entry"),
    FieldSpec("use_clahe", kind="bool"),
    FieldSpec("clahe_clip_limit", kind="slider", minimum=0.5, maximum=8.0, step=0.1),
    FieldSpec("clahe_tile_size", kind="slider", minimum=2, maximum=32, step=1),

    # --- Homográfia-plauzibilitás ---
    FieldSpec("use_homography_check", kind="bool"),
    FieldSpec("max_homography_shear_ratio", kind="float_entry"),
    FieldSpec("min_homography_scale", kind="float_entry"),
    FieldSpec("max_homography_scale", kind="float_entry"),
    FieldSpec("decision_strong_ratio", kind="slider", minimum=0.0, maximum=1.0, step=0.01),

    # --- Detektor prioritás + paraméterek ---
    FieldSpec("detector_priority", kind="detector_list"),
    FieldSpec("sift_nfeatures", kind="int_entry"),
    FieldSpec("sift_contrast", kind="slider", minimum=0.01, maximum=0.10, step=0.001),
    FieldSpec("sift_edge", kind="slider", minimum=1, maximum=50, step=1),
    FieldSpec("sift_sigma", kind="slider", minimum=0.8, maximum=3.0, step=0.1),
    FieldSpec("akaze_threshold", kind="float_entry"),
    FieldSpec("akaze_n_octaves", kind="slider", minimum=1, maximum=8, step=1),
    FieldSpec("akaze_n_layers", kind="slider", minimum=1, maximum=8, step=1),
    FieldSpec("orb_nfeatures", kind="int_entry"),
    FieldSpec("orb_scale_factor", kind="slider", minimum=1.01, maximum=2.0, step=0.01),
    FieldSpec("orb_n_levels", kind="slider", minimum=1, maximum=16, step=1),
    FieldSpec("brisk_thresh", kind="slider", minimum=1, maximum=100, step=1),
    FieldSpec("brisk_octaves", kind="slider", minimum=0, maximum=8, step=1),

    # --- Matching / elfogadási küszöbök ---
    FieldSpec("ratio_test_sift", kind="slider", minimum=0.0, maximum=1.0, step=0.01),
    FieldSpec("ratio_test_bin", kind="slider", minimum=0.0, maximum=1.0, step=0.01),
    FieldSpec("min_good_matches", kind="int_entry"),
    FieldSpec("min_inliers", kind="int_entry"),
    FieldSpec("min_inlier_ratio", kind="slider", minimum=0.0, maximum=1.0, step=0.01),
    FieldSpec("ratio_compensation_bar", kind="slider", minimum=0.0, maximum=1.0, step=0.01),
    FieldSpec("relaxed_min_inliers", kind="int_entry"),
    FieldSpec("inlier_compensation_bar", kind="int_entry"),
    FieldSpec("relaxed_min_inlier_ratio", kind="slider", minimum=0.0, maximum=1.0, step=0.01),
    FieldSpec("ransac_reproj", kind="slider", minimum=0.5, maximum=10.0, step=0.1),
    FieldSpec("ransac_max_iters", kind="int_entry"),
    FieldSpec("ransac_conf", kind="slider", minimum=0.9, maximum=1.0, step=0.001),
    FieldSpec("score_accept", kind="slider", minimum=0.0, maximum=1.5, step=0.01),
    FieldSpec("score_uncertain", kind="slider", minimum=0.0, maximum=1.5, step=0.01),
    FieldSpec("early_accept_score", kind="slider", minimum=0.0, maximum=2.0, step=0.01),

    # --- Kétlépcsős keresés / cache ---
    FieldSpec("cache_long_side", kind="int_entry"),
    FieldSpec("cache_jpeg_quality", kind="slider", minimum=1, maximum=100, step=1),
    FieldSpec("stage1_ratio", kind="slider", minimum=0.0, maximum=1.0, step=0.01),
    FieldSpec("stage1_min_good_matches", kind="slider", minimum=1, maximum=50, step=1),
    FieldSpec("stage1_top_k", kind="slider", minimum=1, maximum=50, step=1),

    # --- Downscale-retry (Fix 2) ---
    FieldSpec("use_downscale_retry", kind="bool"),
    FieldSpec("small_reference_native_threshold", kind="int_entry"),
    FieldSpec("downscale_retry_factor", kind="slider", minimum=0.1, maximum=3.0, step=0.05),
]

assert {f.key for f in FIELD_SPECS} == set(TUNABLE_KEYS), (
    "FIELD_SPECS és TUNABLE_KEYS eltér – minden tunable kulcshoz pontosan "
    "egy FieldSpec kell, se hiány, se felesleg."
)

# A README "Legfontosabb kulcsok" táblázatával megegyező, kurátort lista –
# ez látszik alapból az "Egyszerű" nézetben; a maradék csak "Haladó" módban.
BASIC_KEYS = (
    "min_good_matches", "min_inliers", "min_inlier_ratio",
    "score_uncertain", "score_accept", "early_accept_score",
    "ratio_test_sift", "ratio_test_bin",
    "cache_long_side", "stage1_top_k", "stage1_min_good_matches",
    "use_clahe", "use_homography_check", "detector_priority",
    "max_process_size", "min_process_size",
)

assert set(BASIC_KEYS) <= set(TUNABLE_KEYS)

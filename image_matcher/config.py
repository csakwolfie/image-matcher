"""
config.yaml + profiles/*.yaml betöltése és a precedencia (explicit CLI >
--profile > config.yaml alapérték) feloldása.

A Config egy immutable (frozen dataclass) pillanatkép a finomhangolási
konstansokról egyetlen futtatásra – nincsenek modul-szintű globálisok, amiket
egy profilváltás vagy CLI-felülírás módosíthatna útközben, és a hívó kódnak
mindig konzisztens, egyetlen forrásból (ez a példány) származó értékeket ad.
Ez strukturálisan kizárja azt a korábbi hibaosztályt, ahol egy globális
konstans (pl. CACHE_LONG_SIDE) és egy CLI-vel feloldott lokális változó
szétcsúszhatott egymástól.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .i18n import DEFAULT_LANG, t
from .paths import PACKAGED_DATA_DIR, USER_CONFIG_DIR, first_existing

# A config.yaml egy nem-tunable, opcionális kulcsa – NEM tartozik a
# TUNABLE_KEYS/Config közé (nem feature-matching finomhangolás, hanem a CLI
# --lang kapcsolójának alapértelmezettje), ezért a Config.from_dict szigorú
# "minden kulcs kötelező" ellenőrzése nem vonatkozik rá – hiányzó esetén
# egyszerűen a csomag i18n.DEFAULT_LANG értékére esik vissza.
DEFAULT_LANGUAGE_KEY = "default_language"

# Kulcsok, amiket a config.yaml / profiles/*.yaml megadhat. Bármely más
# kulcs egy profil-fájlban FIGYELMEZTETÉST vált ki (elgépelt/régi kulcsnév
# ne tudja csendben, észrevétlenül figyelmen kívül hagyni magát).
TUNABLE_KEYS = (
    "max_process_size", "min_process_size",
    "use_clahe", "clahe_clip_limit", "clahe_tile_size",
    "use_homography_check", "max_homography_shear_ratio",
    "min_homography_scale", "max_homography_scale",
    "decision_strong_ratio",
    "detector_priority",
    "sift_nfeatures", "sift_contrast", "sift_edge", "sift_sigma",
    "akaze_threshold", "akaze_n_octaves", "akaze_n_layers",
    "orb_nfeatures", "orb_scale_factor", "orb_n_levels",
    "brisk_thresh", "brisk_octaves",
    "ratio_test_sift", "ratio_test_bin",
    "min_good_matches", "min_inliers", "min_inlier_ratio",
    "ratio_compensation_bar", "relaxed_min_inliers",
    "inlier_compensation_bar", "relaxed_min_inlier_ratio",
    "ransac_reproj", "ransac_max_iters", "ransac_conf",
    "score_accept", "score_uncertain", "early_accept_score",
    "cache_long_side", "cache_jpeg_quality",
    "stage1_ratio", "stage1_min_good_matches", "stage1_top_k",
    "use_downscale_retry", "small_reference_native_threshold", "downscale_retry_factor",
)

# Profil-fájlokban megengedett, de NEM finomhangolási kulcs – csak a
# --list-profiles kimenethez használt egysoros leírás.
PROFILE_META_KEYS = ("description",)


@dataclasses.dataclass(frozen=True)
class Config:
    max_process_size: int
    min_process_size: int
    use_clahe: bool
    clahe_clip_limit: float
    clahe_tile_size: int
    use_homography_check: bool
    max_homography_shear_ratio: float
    min_homography_scale: float
    max_homography_scale: float
    decision_strong_ratio: float
    detector_priority: List[str]
    sift_nfeatures: int
    sift_contrast: float
    sift_edge: int
    sift_sigma: float
    akaze_threshold: float
    akaze_n_octaves: int
    akaze_n_layers: int
    orb_nfeatures: int
    orb_scale_factor: float
    orb_n_levels: int
    brisk_thresh: int
    brisk_octaves: int
    ratio_test_sift: float
    ratio_test_bin: float
    min_good_matches: int
    min_inliers: int
    min_inlier_ratio: float
    ratio_compensation_bar: float
    relaxed_min_inliers: int
    inlier_compensation_bar: int
    relaxed_min_inlier_ratio: float
    ransac_reproj: float
    ransac_max_iters: int
    ransac_conf: float
    score_accept: float
    score_uncertain: float
    early_accept_score: float
    cache_long_side: int
    cache_jpeg_quality: int
    stage1_ratio: float
    stage1_min_good_matches: int
    stage1_top_k: int
    use_downscale_retry: bool
    small_reference_native_threshold: int
    downscale_retry_factor: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        missing = [k for k in TUNABLE_KEYS if k not in data]
        if missing:
            raise ValueError(t("config.error_missing_keys", keys=", ".join(missing)))
        return cls(**{k: data[k] for k in TUNABLE_KEYS})

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **overrides: Any) -> "Config":
        return dataclasses.replace(self, **overrides)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(t("config.error_yaml_root_not_object", path=path))
    return data


def resolve_config_path(explicit: Optional[Path] = None) -> Path:
    """
    A config.yaml elérési útjának feloldása precedencia szerint (az első
    találat nyer):
      1. `explicit` – ha a hívó direkt megadta (pl. jövőbeli --config-dir
         CLI kapcsoló).
      2. `./config.yaml` a jelenlegi munkakönyvtárban – ez teszi lehetővé,
         hogy git checkout-ból, telepítés nélkül futtatva a repó gyökeréből
         induló felhasználó a sajátjával írhassa felül a gyári alapértéket,
         és hogy egy telepített csomagnál is legyen egyszerű, projekt-
         szintű felülírási lehetőség.
      3. `~/.image_matcher/config.yaml` – felhasználói szintű felülírás,
         akkor is elérhető, ha nem a projekt könyvtárából futtatunk.
      4. a csomagba ágyazott gyári alapérték (`image_matcher/data/config.yaml`)
         – ez mindig létezik, végső biztonsági háló, akkor is működik, ha a
         csomagot pip install-lal telepítették (nincs külön "repó gyökér").
    """
    if explicit is not None:
        return explicit
    found = first_existing(
        Path.cwd() / "config.yaml",
        USER_CONFIG_DIR / "config.yaml",
    )
    return found or (PACKAGED_DATA_DIR / "config.yaml")


def resolve_profiles_dir(explicit: Optional[Path] = None) -> Path:
    """Ugyanaz a precedencia, mint resolve_config_path-nál, a profiles/ mappára."""
    if explicit is not None:
        return explicit
    found = first_existing(
        Path.cwd() / "profiles",
        USER_CONFIG_DIR / "profiles",
    )
    return found or (PACKAGED_DATA_DIR / "profiles")


def load_default_config(config_path: Optional[Path] = None) -> Config:
    """A config.yaml (gyári alapértékek) betöltése egy Config példánybe."""
    path = resolve_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(t("config.error_config_not_found", path=path))
    data = _load_yaml(path)
    return Config.from_dict(data)


def get_default_language(config_path: Optional[Path] = None) -> str:
    """
    A --lang kapcsoló nélküli alapértelmezett nyelv az éppen érvényes
    config.yaml `default_language` kulcsából. Türelmes olvasás – ha a fájl
    nem létezik, vagy a kulcs hiányzik belőle (pl. régebbi, e funkció előtti
    saját config.yaml felülírás), az i18n csomag DEFAULT_LANG értékére esik
    vissza, NEM hibázik (ellentétben a Config.from_dict szigorú
    kulcs-ellenőrzésével – ez a kulcs opcionális, nem feature-matching
    tunable).
    """
    path = resolve_config_path(config_path)
    if not path.is_file():
        return DEFAULT_LANG
    data = _load_yaml(path)
    return str(data.get(DEFAULT_LANGUAGE_KEY) or DEFAULT_LANG)


def set_default_language(lang: str) -> Path:
    """
    Az alapértelmezett nyelv tartós elmentése – a `--lang <kód>` kapcsoló
    önmagában (más kapcsoló nélkül) kiadva hívja. Mindig a ténylegesen
    ÉRVÉNYES config.yaml-t frissíti (ugyanaz a resolve_config_path
    precedencia, mint betöltéskor), hogy a mentés sose legyen "árnyékolva"
    egy magasabb precedenciájú fájl által:
      - ha már van cwd- vagy felhasználói szintű felülírás, azt frissíti
        helyben (a többi kulcsát megőrizve),
      - ha még nincs (a csomagba ágyazott gyári alapérték az érvényes),
        a felhasználói mappában (`~/.image_matcher/config.yaml`) hoz létre
        egy TELJES másolatot a gyári alapértékekről, a default_language
        felülírásával – a config.yaml MINDIG teljes pillanatkép kell legyen
        (nincs részleges felülírás a profilokon kívül), különben a
        Config.from_dict a következő éles futtatásnál "hiányzó kulcsok"
        hibával állna le.
    A YAML-fájl PyYAML-lel kerül visszaírásra, ami NEM őrzi meg a gyári
    fájl kommenteit – ez a mentett override-fájl elfogadható, tudott
    korlátja (a kommentekkel ellátott referencia változatlanul a csomagba
    ágyazott image_matcher/data/config.yaml).
    """
    effective_path = resolve_config_path()
    data = _load_yaml(effective_path)
    if effective_path == PACKAGED_DATA_DIR / "config.yaml":
        target_path = USER_CONFIG_DIR / "config.yaml"
    else:
        target_path = effective_path
    data[DEFAULT_LANGUAGE_KEY] = lang
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return target_path


def profile_path_for(name: str, profiles_dir: Optional[Path] = None) -> Path:
    directory = resolve_profiles_dir(profiles_dir)
    return directory / f"{name}.yaml"


def list_profiles(profiles_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
    """
    [(profil_név, leírás), ...] – ábécésorrendben, a profiles/ mappából. A
    leírás elsőforrása a lang-rendszer (`profiles.<név>.description` kulcs,
    a beépített balanced/high_recall/diagnostic profilokhoz) – ha egy
    (pl. felhasználó által létrehozott, saját) profilhoz nincs ilyen kulcs,
    a YAML saját `description:` mezője marad az érvényes forrás.
    """
    directory = resolve_profiles_dir(profiles_dir)
    if not directory.is_dir():
        return []
    result = []
    for f in sorted(directory.glob("*.yaml")):
        try:
            data = _load_yaml(f)
        except Exception as e:
            result.append((f.stem, t("config.error_profile_load_failed", error=e)))
            continue
        try:
            description = t(f"profiles.{f.stem}.description")
        except KeyError:
            description = str(data.get("description", ""))
        result.append((f.stem, description))
    return result


def apply_profile(config: Config, profile_name: str, profiles_dir: Optional[Path] = None) -> Config:
    """
    Egy NÉVVEL ELLÁTOTT profil betöltése és alkalmazása a config.yaml
    alapértékei fölé. A profil csak azokat a kulcsokat írja felül, amiket
    tartalmaz – a többi a config.yaml értékén marad.
    """
    path = profile_path_for(profile_name, profiles_dir)
    if not path.is_file():
        available = ", ".join(n for n, _ in list_profiles(profiles_dir)) or t("config.no_profiles_available")
        raise FileNotFoundError(
            t("config.error_profile_not_found", profile=profile_name, path=path, available=available)
        )
    data = _load_yaml(path)

    overrides: Dict[str, Any] = {}
    unknown: List[str] = []
    for k, v in data.items():
        if k in PROFILE_META_KEYS:
            continue
        if k in TUNABLE_KEYS:
            overrides[k] = v
        else:
            unknown.append(k)

    if unknown:
        print(t("config.warning_unknown_profile_keys", profile=profile_name, keys=", ".join(unknown)))

    return config.replace(**overrides)


def load_profile_overrides(name: str, profiles_dir: Optional[Path] = None
                            ) -> Tuple[Dict[str, Any], str]:
    """
    Egy MEGLÉVŐ profil nyers (`TUNABLE_KEYS`-re szűrt) felülírásai +
    `description`-je – a GUI profilszerkesztője ezzel tölti fel a formot
    szerkesztéskor. Ismeretlen kulcsokat csendben kihagyja (nem ez a hely a
    figyelmeztetésre – azt az `apply_profile()` már megteszi futáskor).
    """
    path = profile_path_for(name, profiles_dir)
    data = _load_yaml(path)
    overrides = {k: v for k, v in data.items() if k in TUNABLE_KEYS}
    description = str(data.get("description", ""))
    return overrides, description


def save_profile(name: str, overrides: Dict[str, Any], description: str = "") -> Path:
    """
    Egy profil (RÉSZLEGES felülírás) elmentése `~/.image_matcher/profiles/
    <name>.yaml`-ba. A profil-fájlok – a config.yaml-tól eltérően – ELEVE
    részleges-felülírás szemantikájúak (lásd `apply_profile`), ezért itt
    NEM kell a teljes gyári config.yaml-t másolni, mint
    `set_default_language`-nél – elég a ténylegesen bejelölt kulcsokat
    kiírni. Mindig a felhasználói szintű mappába ment (ugyanaz az elv, mint
    a nyelvi alapértéknél) – nincs cwd/user-dir választás.

    Ha a felhasználói `profiles/` mappa MÉG NEM létezik, első alkalommal a
    csomagba ágyazott beépített profilokat (balanced/high_recall/
    diagnostic) is bemásolja bele, MIELŐTT az újat kiírná – különben a
    `resolve_profiles_dir()` precedenciája (cwd > user > csomagolt, NEM
    összefésülve) a user-mappa létrejötte után teljesen elrejtené a
    beépített profilokat a `--list-profiles`/GUI elől.
    """
    unknown = [k for k in overrides if k not in TUNABLE_KEYS]
    if unknown:
        raise ValueError(t("config.error_unknown_profile_keys_on_save", keys=", ".join(unknown)))
    target_dir = USER_CONFIG_DIR / "profiles"
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True, exist_ok=True)
        packaged_dir = PACKAGED_DATA_DIR / "profiles"
        if packaged_dir.is_dir():
            for builtin in packaged_dir.glob("*.yaml"):
                shutil.copy2(builtin, target_dir / builtin.name)
    target_path = target_dir / f"{name}.yaml"
    data: Dict[str, Any] = dict(overrides)
    if description:
        data["description"] = description
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return target_path


def apply_cli_overrides(config: Config, **overrides: Any) -> Config:
    """
    Explicit CLI kapcsolók alkalmazása – ezek nyerik a legmagasabb
    precedenciát (explicit CLI > --profile > config.yaml alapérték).
    A None értékű overrides-ok figyelmen kívül maradnak (a felhasználó nem
    adta meg explicit módon a kapcsolót).
    """
    effective = {k: v for k, v in overrides.items() if v is not None}
    return config.replace(**effective)

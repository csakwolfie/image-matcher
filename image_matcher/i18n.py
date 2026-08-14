"""
Egyszerű, JSON-alapú szövegkivonatolási/lokalizációs mechanizmus. A nyelvet a
CLI `--lang` kapcsolója választja ki (lásd `cli.py`); új nyelv hozzáadása
pusztán egy új `<nyelv>.lang.json` adatfájl, a hívó kód (a
`t(kulcs, **helyettesítők)` hívások) módosítása nélkül.

A `./config.yaml`/`./profiles/` felfedezési precedenciájával megegyező
módon: munkakönyvtár > felhasználói mappa (`~/.image_matcher/lang/`) >
csomagba ágyazott gyári alapérték (`image_matcher/data/lang/`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import PACKAGED_DATA_DIR, USER_CONFIG_DIR, first_existing

DEFAULT_LANG = "en"


def resolve_lang_dir(explicit: Optional[Path] = None) -> Path:
    """Ugyanaz a precedencia, mint resolve_config_path/resolve_profiles_dir-nél (config.py)."""
    if explicit is not None:
        return explicit
    found = first_existing(
        Path.cwd() / "lang",
        USER_CONFIG_DIR / "lang",
    )
    return found or (PACKAGED_DATA_DIR / "lang")


def list_available_langs(lang_dir: Optional[Path] = None) -> List[str]:
    """A felfedezett nyelvi könyvtárban elérhető nyelvkódok (`<kód>.lang.json`
    fájlnevekből), ábécésorrendben. A `--lang` kapcsoló súgójához és a
    hibás/ismeretlen nyelv esetén kiírt üzenethez kell."""
    directory = lang_dir or resolve_lang_dir()
    if not directory.is_dir():
        return []
    return sorted(p.name[: -len(".lang.json")] for p in directory.glob("*.lang.json"))


class Translator:
    """
    Egy betöltött nyelvi szótár köré épülő lookup+format. Hiányzó kulcsnál
    vagy a sablonhoz nem illő helyettesítőknél HANGOSAN hibázik (nem ad
    csendben csonka/hibás szöveget) – ugyanaz az elv, mint a
    Config.from_dict hiányzó kulcs-ellenőrzésénél.
    """

    def __init__(self, strings: Dict[str, str]):
        self._strings = strings

    def __call__(self, key: str, **kwargs: Any) -> str:
        try:
            template = self._strings[key]
        except KeyError as e:
            raise KeyError(f"Missing translation key: '{key}'") from e
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError) as e:
            raise KeyError(f"Invalid/missing placeholder for key '{key}': {e}") from e


def load_translator(lang: str = DEFAULT_LANG, lang_dir: Optional[Path] = None) -> Translator:
    directory = lang_dir or resolve_lang_dir()
    path = directory / f"{lang}.lang.json"
    if not path.is_file():
        raise FileNotFoundError(f"Language file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        strings = json.load(f)
    return Translator(strings)


_translator: Optional[Translator] = None


def init_translator(lang: str = DEFAULT_LANG, lang_dir: Optional[Path] = None) -> Translator:
    """Explicit (nem lusta) inicializálás – a CLI induláskor hívja, MÉG a
    parancssori parser felépítése (és ezzel az első `t()` hívás) ELŐTT, hogy
    a `--lang` kapcsoló a súgószövegeket is a kért nyelven jelenítse meg.
    `FileNotFoundError`-t dob, ha a kért nyelvhez nincs `<lang>.lang.json`."""
    global _translator
    _translator = load_translator(lang, lang_dir)
    return _translator


def get_translator() -> Translator:
    """A modulszintű fordító. Ha a CLI még nem inicializálta explicit módon
    (`init_translator`) – pl. programozott/teszt-használatnál –, lustán
    betölti az alapértelmezett nyelvet. A nyelv egy futáson belül állandó –
    nincs ok, hogy egy folyamaton belül eltérjen, szemben pl. a Config-gal,
    aminek több, eltérő értékű példánya is élhet egyszerre (lásd
    downscale-retry retry_cfg)."""
    global _translator
    if _translator is None:
        _translator = load_translator()
    return _translator


def t(key: str, **kwargs: Any) -> str:
    return get_translator()(key, **kwargs)

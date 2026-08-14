"""
Közös elérési út-feloldási segédletek. Mind a `config.py` (config.yaml/
profiles/), mind az `i18n.py` (nyelvi fájlok) ugyanazt a felfedezési
precedenciát használja (munkakönyvtár > felhasználói mappa > csomagba
ágyazott gyári alapérték) – ez a pár primitív ezért egy közös, mindkettő
által importált modulban él, hogy a config.py és i18n.py ne importálja
körkörösen egymást (config.py-nak fordítási üzenetekhez kell az i18n.py,
az i18n.py-nak pedig ehhez a felfedezési logikához kellene a config.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_DATA_DIR = PACKAGE_ROOT / "data"
USER_CONFIG_DIR = Path.home() / ".image_matcher"


def first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None

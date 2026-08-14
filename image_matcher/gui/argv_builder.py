"""
`build_argv`/`validate_form`: a GUI form-állapotát (egyszerű `dict`) a
`main.main(argv)`-nak megfelelő argv-listává, illetve hibaüzenet-listává
alakító TISZTA függvények – szándékosan widget-mentesek (nem importálnak
`tkinter`-t), hogy unittest-tel Tkinter nélkül tesztelhetők legyenek.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..i18n import t


def _has_value(form: Dict[str, Any], key: str) -> bool:
    value = form.get(key)
    return value is not None and str(value).strip() != ""


def build_argv(form: Dict[str, Any]) -> List[str]:
    """
    A form mezőit `main.main(argv)`-kompatibilis argv-listává alakítja.
    Csak a kitöltött/bejelölt mezőket veszi fel – üres `limit`/`top_k`/
    `cache`/`profile`/`lang` esetén az adott kapcsoló egyszerűen kimarad
    (a `config.yaml`/profil értéke érvényesül, ugyanúgy, mintha a CLI-n se
    adtuk volna meg).
    """
    argv: List[str] = []

    if _has_value(form, "lang"):
        argv += ["--lang", str(form["lang"])]
    if _has_value(form, "profile"):
        argv += ["--profile", str(form["profile"])]
    if _has_value(form, "reference"):
        argv += ["--reference", str(form["reference"])]
    if _has_value(form, "source"):
        argv += ["--source", str(form["source"])]
    if _has_value(form, "output"):
        argv += ["--output", str(form["output"])]
    if _has_value(form, "limit"):
        argv += ["--limit", str(int(form["limit"]))]
    if _has_value(form, "workers"):
        argv += ["--workers", str(int(form["workers"]))]
    if _has_value(form, "top_k"):
        argv += ["--top-k", str(int(form["top_k"]))]
    if _has_value(form, "cache"):
        argv += ["--cache", str(form["cache"])]
    if form.get("no_cache"):
        argv.append("--no-cache")
    if form.get("rebuild_cache"):
        argv.append("--rebuild-cache")
    if form.get("dry_run"):
        argv.append("--dry-run")

    return argv


def validate_form(form: Dict[str, Any]) -> List[str]:
    """
    Futtatás előtti ellenőrzés – ugyanazok a szabályok, mint amiket
    `main.py` a CLI-n a kötelező mezőkre/mappa-létezésre elvégez
    (`main.py`: hiányzó `--reference`/`--source`/`--output`, illetve nem
    létező referencia-/source-mappa), csak GUI-specifikus, egyenként
    megjelenő hibaüzenetekkel, MIELŐTT a háttérszál egyáltalán elindulna.
    Üres lista = a form futtatható.
    """
    errors: List[str] = []

    reference = str(form.get("reference") or "").strip()
    source = str(form.get("source") or "").strip()
    output = str(form.get("output") or "").strip()

    if not reference:
        errors.append(t("gui.error_reference_required"))
    elif not Path(reference).is_dir():
        errors.append(t("gui.error_reference_dir_missing", path=reference))

    if not source:
        errors.append(t("gui.error_source_required"))
    elif not Path(source).is_dir():
        errors.append(t("gui.error_source_dir_missing", path=source))

    if not output:
        errors.append(t("gui.error_output_required"))

    return errors

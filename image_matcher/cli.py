"""
Parancssori felület – csoportosított kapcsolók:
  Profilkezelés / Útvonalak / Futtatásvezérlés / Cache / Futtatás.

Precedencia: explicit CLI kapcsoló > --profile fájl > config.yaml alapérték.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .config import get_default_language
from .i18n import init_translator, list_available_langs, t


def _pre_parse_lang(argv) -> str:
    """Kinyeri a --lang értékét MÉG a teljes parser felépítése előtt – a
    build_parser() minden szövege t()-n keresztül készül, tehát a nyelvnek
    már ekkor ismertnek kell lennie, különben a --help is mindig az
    alapértelmezett nyelven jelenne meg, a --lang kapcsolótól függetlenül.
    parse_known_args-ot használ (add_help=False), hogy ne ütközzön a teljes
    parser -h/--help vagy egyéb kapcsolóival. Ha nincs --lang megadva, a
    config.yaml `default_language` kulcsára esik vissza (lásd
    config.get_default_language – ez teszi lehetővé, hogy egy korábbi
    `--lang <kód>` önmagában kiadva tartósan megváltoztassa az
    alapértelmezettet)."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--lang", default=None)
    ns, _ = pre_parser.parse_known_args(argv)
    return ns.lang or get_default_language()


def _init_lang(lang: str) -> None:
    """A fordítási rendszer inicializálása a kért nyelvre. Ismeretlen nyelv
    esetén a hibaüzenet szükségszerűen fix (nem t()-n keresztüli) szöveg,
    mert épp a fordítórendszer sikertelen betöltését jelezné vele – önmagát
    nem tudná lefordítani."""
    try:
        init_translator(lang)
    except FileNotFoundError:
        available = ", ".join(list_available_langs()) or "(none found)"
        sys.stderr.write(
            f"[ERROR] Unknown language: '{lang}'. Available languages: {available}\n"
        )
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-matcher",
        description=t("cli.description"),
        epilog=t("cli.epilog"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    available_langs = list_available_langs()
    parser.add_argument(
        "--lang", default=None, choices=available_langs or None,
        help=t("cli.help.lang", available=", ".join(available_langs), default=get_default_language()),
    )

    profile_group = parser.add_argument_group(t("cli.group.profile_management"))
    profile_group.add_argument(
        "--profile", default=None,
        help=t("cli.help.profile"),
    )

    paths_group = parser.add_argument_group(t("cli.group.paths"))
    paths_group.add_argument(
        "--reference", "-r", default=None,
        help=t("cli.help.reference"),
    )
    paths_group.add_argument(
        "--source", "-s", default=None,
        help=t("cli.help.source"),
    )
    paths_group.add_argument(
        "--output", "-o", default=None,
        help=t("cli.help.output"),
    )

    run_control_group = parser.add_argument_group(t("cli.group.run_control"))
    run_control_group.add_argument(
        "--limit", type=int, default=None,
        help=t("cli.help.limit"),
    )
    run_control_group.add_argument(
        "--workers", "-w", type=int, default=max(1, os.cpu_count() or 4),
        help=t("cli.help.workers"),
    )
    run_control_group.add_argument(
        "--top-k", type=int, default=None,
        help=t("cli.help.top_k"),
    )

    cache_group = parser.add_argument_group(t("cli.group.cache"))
    cache_group.add_argument(
        "--cache", default=None,
        help=t("cli.help.cache"),
    )
    cache_group.add_argument(
        "--no-cache", action="store_true",
        help=t("cli.help.no_cache"),
    )
    cache_group.add_argument(
        "--rebuild-cache", action="store_true",
        help=t("cli.help.rebuild_cache"),
    )

    run_group = parser.add_argument_group(t("cli.group.run"))
    run_group.add_argument(
        "--dry-run", action="store_true",
        help=t("cli.help.dry_run"),
    )
    run_group.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
        help=t("cli.help.version"),
    )
    run_group.add_argument(
        "--list-profiles", action="store_true",
        help=t("cli.help.list_profiles"),
    )

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    _init_lang(_pre_parse_lang(argv))
    return build_parser().parse_args(argv)

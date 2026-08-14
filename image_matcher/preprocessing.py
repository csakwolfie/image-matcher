"""
Előfeldolgozás: kicsi, 8-bit, szürkeárnyalatos JPEG cache a gyors 1. körhöz.
Inkrementális – csak az új képeket dolgozza fel újra.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from .config import Config
from .i18n import t
from .image_io import is_image_file, load_image_smart

_PRINT_LOCK = threading.Lock()


def list_images(directory: Path) -> List[Path]:
    """Rekurzív lista az összes támogatott képről, rendezve."""
    files = []
    for p in directory.rglob("*"):
        if p.is_file() and is_image_file(p):
            files.append(p)
    return sorted(files)


def to_cache_path(root: Path, original_path: Path, cache_root: Path) -> Path:
    """
    Az eredeti fájl relatív útvonalát megőrizve képezi le a cache-mappába.
    Az eredeti fájlnevet (kiterjesztéssel együtt) + '.jpg'-t használjuk, hogy
    azonos nevű, de eltérő kiterjesztésű fájlok ne ütközzenek egymással.
    """
    rel = original_path.relative_to(root)
    return cache_root / rel.parent / (rel.name + ".jpg")


def _preprocess_one(src_path: Path, cache_path: Path, long_side: int, cfg: Config,
                     jpeg_quality: int) -> bool:
    """Egy kép leskálázása + szürkeárnyalatos + 8-bit JPEG mentése a cache-be."""
    gray, _ = load_image_smart(
        src_path,
        max_size=long_side,
        min_size=cfg.min_process_size,
        use_clahe=cfg.use_clahe,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_size=cfg.clahe_tile_size,
    )
    if gray is None:
        return False
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return False
        buf.tofile(str(cache_path))  # Unicode-biztos írás Windows-on
        return True
    except Exception as e:
        with _PRINT_LOCK:
            print(t("preprocessing.error_cache_write_failed", name=src_path.name, error=e))
        return False


def preprocess_directory(
    root: Path, cache_root: Path, long_side: int, label: str,
    cfg: Config, max_workers: int, jpeg_quality: int, force: bool = False,
) -> Dict[Path, Path]:
    """
    Előfeldolgozza egy mappa összes képét a cache_root alá, a relatív
    mappaszerkezet megőrzésével. Inkrementális: a már meglévő
    cache-fájlokat nem generálja újra, kivéve, ha force=True
    (--rebuild-cache), ekkor mindent felülír.
    Visszaadja: {cache_path: eredeti_path} leképezés a SIKERESEN cache-elt
    képekre (a beolvasási hibával jártakat kihagyja).
    """
    files = list_images(root)
    mapping: Dict[Path, Path] = {}
    todo: List[Tuple[Path, Path]] = []

    for f in files:
        cache_path = to_cache_path(root, f, cache_root)
        if cache_path.exists() and not force:
            mapping[cache_path] = f
        else:
            todo.append((f, cache_path))

    if todo:
        print(t("preprocessing.progress_new_images", label=label, new_count=len(todo),
                 cached_count=len(files) - len(todo), workers=max_workers))
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_preprocess_one, f, cache_path, long_side, cfg, jpeg_quality): (f, cache_path)
                for f, cache_path in todo
            }
            for fut in as_completed(futures):
                f, cache_path = futures[fut]
                if fut.result():
                    mapping[cache_path] = f
                done += 1
                if done % 100 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)} ...")
    else:
        print(t("preprocessing.all_cached", label=label, count=len(files)))

    return mapping

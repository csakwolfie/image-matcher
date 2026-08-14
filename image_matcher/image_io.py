"""
Kép betöltés: Unicode-biztos beolvasás Windows-on, intelligens skálázás,
szürkeárnyalatos konverzió, opcionális CLAHE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .i18n import t

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def get_native_longer_side(path: Path) -> Optional[int]:
    """
    A kép NATÍV (feldolgozás előtti, skálázatlan) hosszabb oldalának gyors
    lekérdezése, lehetőleg teljes dekódolás nélkül (csak a fejlécet olvassa
    be PIL-lel) – ehhez nem kell a teljes pixeladatot dekódolni, csak a
    méret-metaadatot. Fallback: teljes cv2 dekódolás, ha a PIL fejléc-olvasás
    sikertelen (pl. egzotikus/sérült fájl).
    """
    try:
        with Image.open(path) as img:
            w, h = img.size
        return max(w, h)
    except Exception:
        img = _imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        return max(h, w)


def _imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """
    Unicode-biztos képbetöltés Windows-on. Az OpenCV cv2.imread() nem
    kezeli megbízhatóan az ékezetes / nem-ASCII útvonalakat. Megoldás:
      1) np.fromfile + cv2.imdecode  (gyors, működik JPG/PNG/TIFF nagy részén)
      2) klasszikus cv2.imread        (ASCII útvonalakon gyorsabb)
      3) Pillow fallback              (nagy / egzotikus TIFF-ekhez)
    """
    path_str = str(path)

    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass

    try:
        img = cv2.imread(path_str, flags)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        with Image.open(path) as pil_img:
            if getattr(pil_img, "n_frames", 1) > 1:
                pil_img.seek(0)
            if pil_img.mode not in ("L", "RGB", "RGBA"):
                pil_img = pil_img.convert("RGB")
            img = np.array(pil_img)
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
    except Exception:
        pass

    return None


def load_image_smart(
    path: Path,
    max_size: int,
    min_size: int = 400,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Intelligens, Unicode-biztos betöltés: nagy képek leskálázása, túl kicsi
    képek felskálázása, szürkeárnyalatos konverzió, opcionális CLAHE.
    Visszaadja: (szürkeárnyalatos kép vagy None, skálázási faktor).
    """
    try:
        img = _imread_unicode(path, cv2.IMREAD_UNCHANGED)

        if img is None or img.size == 0:
            print(t("image_io.warning_read_failed", name=path.name))
            return None, 1.0

        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        h, w = gray.shape[:2]
        scale = 1.0

        longer = max(h, w)
        if longer > max_size:
            scale = max_size / float(longer)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif longer < min_size and longer > 0:
            scale = min_size / float(longer)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if use_clahe and gray is not None and gray.size > 0:
            # Új CLAHE-példány hívásonként – a cv2.CLAHE objektum nem
            # feltétlenül szálbiztos, ez viszont stateless és olcsó.
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                     tileGridSize=(clahe_tile_size, clahe_tile_size))
            gray = clahe.apply(gray)

        return gray, scale

    except Exception as e:
        print(t("image_io.error_load_failed", name=path.name, error=e))
        return None, 1.0

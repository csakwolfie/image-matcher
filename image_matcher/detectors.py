"""
Detektor gyár (SIFT/AKAZE/ORB/BRISK/KAZE) + szálankénti (thread-local)
detector-példányok, mert egy cv2.Feature2D objektumot nem biztonságos
több szálból egyszerre hívni.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import cv2

from .config import Config
from .i18n import t


def create_detector(name: str, cfg: Config) -> Optional[cv2.Feature2D]:
    """
    Detektor létrehozása biztonságosan. Ha egy algoritmus nincs benne a
    telepített OpenCV-ben, None-t ad vissza.
    """
    name = name.upper()
    try:
        if name == "SIFT":
            if hasattr(cv2, "SIFT_create"):
                return cv2.SIFT_create(
                    nfeatures=cfg.sift_nfeatures,
                    contrastThreshold=cfg.sift_contrast,
                    edgeThreshold=cfg.sift_edge,
                    sigma=cfg.sift_sigma
                )
            if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
                return cv2.xfeatures2d.SIFT_create(
                    nfeatures=cfg.sift_nfeatures,
                    contrastThreshold=cfg.sift_contrast,
                    edgeThreshold=cfg.sift_edge,
                    sigma=cfg.sift_sigma
                )
            print(t("detectors.warning_not_available", name=name))
            return None

        elif name == "AKAZE":
            if hasattr(cv2, "AKAZE_create"):
                return cv2.AKAZE_create(
                    threshold=cfg.akaze_threshold,
                    nOctaves=cfg.akaze_n_octaves,
                    nOctaveLayers=cfg.akaze_n_layers
                )
            # Egyes OpenCV buildeken (pl. bizonyos opencv-contrib-python
            # verziók) az AKAZE nem a fő cv2 modulban, hanem a
            # cv2.xfeatures2d alatt érhető el – ugyanaz a helyzet, mint a
            # SIFT-nél régebbi buildeken (lásd fent).
            if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "AKAZE_create"):
                return cv2.xfeatures2d.AKAZE_create(
                    threshold=cfg.akaze_threshold,
                    nOctaves=cfg.akaze_n_octaves,
                    nOctaveLayers=cfg.akaze_n_layers
                )
            print(t("detectors.warning_not_available", name=name))
            return None

        elif name == "ORB":
            if hasattr(cv2, "ORB_create"):
                return cv2.ORB_create(
                    nfeatures=cfg.orb_nfeatures,
                    scaleFactor=cfg.orb_scale_factor,
                    nlevels=cfg.orb_n_levels
                )
            print(t("detectors.warning_not_available", name=name))
            return None

        elif name == "BRISK":
            if hasattr(cv2, "BRISK_create"):
                return cv2.BRISK_create(
                    thresh=cfg.brisk_thresh,
                    octaves=cfg.brisk_octaves
                )
            # Lásd az AKAZE-nél fenti megjegyzést – egyes buildeken a BRISK is
            # a cv2.xfeatures2d alatt van, nem a fő cv2 modulban.
            if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "BRISK_create"):
                return cv2.xfeatures2d.BRISK_create(
                    thresh=cfg.brisk_thresh,
                    octaves=cfg.brisk_octaves
                )
            print(t("detectors.warning_not_available", name=name))
            return None

        elif name == "KAZE":
            if hasattr(cv2, "KAZE_create"):
                return cv2.KAZE_create()
            if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "KAZE_create"):
                return cv2.xfeatures2d.KAZE_create()
            print(t("detectors.warning_not_available", name=name))
            return None

        else:
            print(t("detectors.warning_unknown_detector", name=name))
            return None

    except AttributeError as e:
        print(t("detectors.warning_detector_unavailable_exception", name=name, error=e))
        return None
    except Exception as e:
        print(t("detectors.error_creation_failed", name=name, error=e))
        return None


def is_binary_descriptor(name: str) -> bool:
    return name.upper() in ("ORB", "BRISK", "AKAZE")


class ThreadLocalDetectors:
    """
    Minden szál saját detector-példányait tárolja (thread-local), mert egy
    cv2.Feature2D objektumot több szálból egyszerre hívni nem biztonságos.
    Egy példány EGYETLEN Config-hoz (egyetlen futtatáshoz) tartozik – a
    config a teljes futás alatt nem változik, így a szálankénti cache
    biztonságosan csak a detektor NEVE alapján kulcsolható.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._local = threading.local()

    def get(self, name: str) -> Optional[cv2.Feature2D]:
        cache: Optional[Dict[str, Optional[cv2.Feature2D]]] = getattr(self._local, "detectors", None)
        if cache is None:
            cache = {}
            self._local.detectors = cache
        if name not in cache:
            cache[name] = create_detector(name, self._cfg)
        return cache[name]

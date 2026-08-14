"""
DescriptorCache: memóriában ÉS opcionálisan lemezen (perzisztensen) tartja
a képek feature-descriptorait, hogy egy képet csak egyszer kelljen
feldolgozni detektoronként – futtatásokon ÁT is, ha van perzisztens cache.

A cache-kulcs tartalmazza a feldolgozási beállítások ("fingerprint": CLAHE,
detektor-paraméterek, feldolgozási méret) hash-ét is – ha ezek változnak
(pl. profilváltás miatt), a cache automatikusan érvénytelenné válik, nem ad
hallgatólagosan elavult eredményt.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .detectors import ThreadLocalDetectors
from .i18n import t
from .image_io import load_image_smart

_PRINT_LOCK = threading.Lock()


def processing_fingerprint(cfg: Config) -> str:
    """
    Rövid hash a kép-előfeldolgozást ÉS feature-detektálást befolyásoló
    beállításokból. A perzisztens descriptor cache ezt beépíti a
    fájlnevébe, hogy a beállítások futtatások közötti megváltoztatása ne
    okozzon hallgatólagosan elavult, más paraméterekkel számolt eredményt.
    """
    raw = (
        f"clahe={cfg.use_clahe};clip={cfg.clahe_clip_limit};tile={cfg.clahe_tile_size};"
        f"maxproc={cfg.max_process_size};minproc={cfg.min_process_size};"
        f"sift={cfg.sift_nfeatures},{cfg.sift_contrast},{cfg.sift_edge},{cfg.sift_sigma};"
        f"akaze={cfg.akaze_threshold},{cfg.akaze_n_octaves},{cfg.akaze_n_layers};"
        f"orb={cfg.orb_nfeatures},{cfg.orb_scale_factor},{cfg.orb_n_levels};"
        f"brisk={cfg.brisk_thresh},{cfg.brisk_octaves}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _keypoints_to_array(kps: List) -> np.ndarray:
    """cv2.KeyPoint lista → Nx7 float32 tömb (pt.x, pt.y, size, angle, response, octave, class_id)."""
    if not kps:
        return np.zeros((0, 7), dtype=np.float32)
    return np.array(
        [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kps],
        dtype=np.float32
    )


def _array_to_keypoints(arr: np.ndarray) -> List:
    """Nx7 float32 tömb → cv2.KeyPoint lista (a fenti inverze)."""
    import cv2
    return [
        cv2.KeyPoint(x=float(r[0]), y=float(r[1]), size=float(r[2]), angle=float(r[3]),
                     response=float(r[4]), octave=int(r[5]), class_id=int(r[6]))
        for r in arr
    ]


class DescriptorCache:
    """
    Egy futtatáshoz tartozó descriptor cache. persist_dir=None esetén csak
    memóriában (a futtatáson belül) cache-el; egyébként a lemezre is ír,
    így ismételt (pl. küszöb-hangolási) futtatásoknál a drága
    feature-detektálást nem kell újra elvégezni.
    """

    def __init__(self, cfg: Config, thread_detectors: ThreadLocalDetectors,
                 persist_dir: Optional[Path] = None, rebuild: bool = False):
        self.cfg = cfg
        self.thread_detectors = thread_detectors
        self.persist_dir = persist_dir
        self.rebuild = rebuild
        # key: (source_path_str, detector_name) → (keypoints, descriptors, scale)
        self._cache: Dict[Tuple[str, str], Tuple[List, Optional[np.ndarray], float]] = {}
        self._images_loaded: Dict[str, Tuple[Optional[np.ndarray], float]] = {}
        self._lock = threading.Lock()
        self.persist_hits = 0
        self.persist_misses = 0

    def get_image(self, path: Path) -> Tuple[Optional[np.ndarray], float]:
        key = str(path)
        with self._lock:
            cached = self._images_loaded.get(key)
        if cached is not None:
            return cached
        gray, scale = load_image_smart(
            path,
            max_size=self.cfg.max_process_size,
            min_size=self.cfg.min_process_size,
            use_clahe=self.cfg.use_clahe,
            clahe_clip_limit=self.cfg.clahe_clip_limit,
            clahe_tile_size=self.cfg.clahe_tile_size,
        )
        with self._lock:
            self._images_loaded[key] = (gray, scale)
        return gray, scale

    def _persist_file(self, path: Path, detector_name: str) -> Optional[Path]:
        if self.persist_dir is None:
            return None
        fingerprint = processing_fingerprint(self.cfg)
        digest = hashlib.sha1(f"{path}|{detector_name}|{fingerprint}".encode("utf-8")).hexdigest()
        return self.persist_dir / detector_name / f"{digest}.npz"

    def _load_from_disk(self, path: Path, detector_name: str
                         ) -> Optional[Tuple[List, Optional[np.ndarray], float]]:
        if self.rebuild:
            return None
        pf = self._persist_file(path, detector_name)
        if pf is None or not pf.exists():
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        try:
            with np.load(pf, allow_pickle=False) as data:
                # Érvényesség ellenőrzése: ha a fájl azóta módosult, a
                # cache-bejegyzést eldobjuk – véd a hallgatólagos, elavult
                # eredmények ellen.
                if int(data["fsize"]) != st.st_size:
                    return None
                if abs(float(data["mtime"]) - st.st_mtime) > 1.0:
                    return None
                kp_arr = data["keypoints"]
                des = data["descriptors"]
                if des.size == 0:
                    des = None
                scale = float(data["scale"])
                kps = _array_to_keypoints(kp_arr)
                return kps, des, scale
        except Exception:
            return None

    def _save_to_disk(self, path: Path, detector_name: str,
                       kps: List, des: Optional[np.ndarray], scale: float):
        pf = self._persist_file(path, detector_name)
        if pf is None:
            return
        try:
            st = path.stat()
            pf.parent.mkdir(parents=True, exist_ok=True)
            kp_arr = _keypoints_to_array(kps)
            des_arr = des if des is not None else np.zeros((0,), dtype=np.float32)
            # np.savez_compressed automatikusan ".npz"-re egészíti ki a
            # fájlnevet – a temp fájl nevének is .npz-vel KELL végződnie.
            tmp = pf.with_name(pf.stem + ".tmp.npz")
            np.savez_compressed(
                tmp, keypoints=kp_arr, descriptors=des_arr,
                scale=np.float64(scale), fsize=np.int64(st.st_size),
                mtime=np.float64(st.st_mtime)
            )
            os.replace(tmp, pf)  # atomikus – megszakadt futásnál ne maradjon sérült cache-fájl
        except Exception:
            pass  # a perzisztens cache írása soha ne dobjon hibát a fő folyamatban

    def get_descriptors(self, path: Path, detector_name: str
                         ) -> Tuple[List, Optional[np.ndarray], float]:
        key = (str(path), detector_name)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self.persist_dir is not None:
            disk_result = self._load_from_disk(path, detector_name)
            if disk_result is not None:
                with self._lock:
                    self._cache[key] = disk_result
                    self.persist_hits += 1
                return disk_result
            with self._lock:
                self.persist_misses += 1

        gray, scale = self.get_image(path)
        if gray is None:
            result = ([], None, 1.0)
            with self._lock:
                self._cache[key] = result
            return result

        try:
            # A detektor NEM feltétlenül szálbiztos ugyanazon példányon
            # belül, ezért mindig az aktuális szál saját, thread-local
            # detector-példányát használjuk.
            detector = self.thread_detectors.get(detector_name)
            kps, des = detector.detectAndCompute(gray, None)
            if des is None:
                des = np.array([])
            result = (kps, des, scale)
            with self._lock:
                self._cache[key] = result
            if self.persist_dir is not None:
                self._save_to_disk(path, detector_name, kps, des if des.size else None, scale)
            return result
        except Exception as e:
            with _PRINT_LOCK:
                print(t("cache_disk.error_feature_computation", name=path.name, detector=detector_name, error=e))
            result = ([], None, 1.0)
            with self._lock:
                self._cache[key] = result
            return result

    def clear_images(self):
        """Memória felszabadítás – a descriptorok megmaradnak."""
        self._images_loaded.clear()

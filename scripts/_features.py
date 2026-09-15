"""Shared feature extraction over an ImageFolder-style split, with an on-disk cache.

Face Mesh over 400 photographs takes about half a minute, so results are cached next to
the results directory and reused by the calibration, evaluation and figure scripts.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gaze import config as cfg
from gaze.geometric import extract_features
from gaze.landmarks import FaceDetector

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class SplitFeatures:
    paths: list[Path]
    labels: np.ndarray
    yaw: np.ndarray
    iris: np.ndarray
    ear: np.ndarray
    detected: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def list_split(split_dir: Path) -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    for index, class_name in enumerate(cfg.CLASSES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
                labels.append(index)
    if not paths:
        raise FileNotFoundError(
            f"No images found under {split_dir}. Expected subfolders {cfg.CLASSES}."
        )
    return paths, np.array(labels)


def _cache_path(split_dir: Path) -> Path:
    key = hashlib.md5(str(split_dir.resolve()).encode()).hexdigest()[:10]
    return cfg.RESULTS_DIR / "cache" / f"features_{split_dir.name}_{key}.pkl"


def extract_split_features(split_dir: Path, use_cache: bool = True) -> SplitFeatures:
    split_dir = Path(split_dir)
    cache = _cache_path(split_dir)
    paths, labels = list_split(split_dir)

    if use_cache and cache.exists():
        payload = pickle.loads(cache.read_bytes())
        if payload.get("count") == len(paths):
            return SplitFeatures(paths=paths, labels=labels, **payload["arrays"])

    yaw, iris, ear, detected = [], [], [], []
    with FaceDetector(static_image_mode=True) as detector:
        for path in paths:
            image = cv2.imread(str(path))
            face = None if image is None else detector.detect(image)
            if face is None or not face.has_landmarks:
                yaw.append(np.nan)
                iris.append(np.nan)
                ear.append(np.nan)
                detected.append(False)
                continue
            features = extract_features(face.landmarks)
            yaw.append(features.yaw_score)
            iris.append(features.iris_score if features.iris_score is not None else np.nan)
            ear.append(features.ear if features.ear is not None else np.nan)
            detected.append(True)

    arrays = {
        "yaw": np.array(yaw, dtype=float),
        "iris": np.array(iris, dtype=float),
        "ear": np.array(ear, dtype=float),
        "detected": np.array(detected, dtype=bool),
    }
    # Undetected faces get a neutral score so downstream code can stay vectorised; the
    # ``detected`` mask records where that substitution happened.
    for key in ("yaw", "iris", "ear"):
        arrays[key] = np.nan_to_num(arrays[key], nan=0.0)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"count": len(paths), "arrays": arrays}))
    return SplitFeatures(paths=paths, labels=labels, **arrays)

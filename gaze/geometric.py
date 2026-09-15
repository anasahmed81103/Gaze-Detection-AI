"""Training-free geometric gaze estimation from Face Mesh landmarks.

The head-direction score is the normalised asymmetry between the nose tip and the two
cheek boundary landmarks. Turning the head shortens the distance to the cheek you turn
towards and lengthens the other, so a single scalar separates left / front / right once
two decision thresholds are calibrated. On our data this is both more accurate and far
cheaper than the fine-tuned CNN (see docs/REPORT.md).

The iris score is the horizontal displacement of each iris centre from the midpoint of
its eye corners, normalised by eye width. It captures eye-in-socket movement that the
head-pose score cannot see, but it is a much weaker signal in isolation.

Both scores are signed so that **positive means the ``left`` class** under the shipped
calibration; the thresholds themselves live in ``models/geometric_calibration.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg


@dataclass(frozen=True)
class Thresholds:
    """Decision boundaries for a signed score: ``> high`` -> left, ``< low`` -> right."""

    low: float
    high: float

    def classify(self, score: float) -> int:
        if score > self.high:
            return cfg.LEFT
        if score < self.low:
            return cfg.RIGHT
        return cfg.FRONT


@dataclass
class GeometricFeatures:
    yaw_score: float
    iris_score: Optional[float]
    ear: Optional[float]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def yaw_score(landmarks: np.ndarray) -> float:
    """Normalised nose/cheek asymmetry in ``[-1, 1]``; positive turns towards ``left``."""
    nose = landmarks[cfg.NOSE_TIP]
    d_left = _dist(nose, landmarks[cfg.CHEEK_LEFT])
    d_right = _dist(nose, landmarks[cfg.CHEEK_RIGHT])
    return (d_right - d_left) / (d_right + d_left + 1e-9)


def _eye_iris_offset(landmarks: np.ndarray, corners: tuple[int, int],
                     iris: tuple[int, ...]) -> Optional[float]:
    if landmarks.shape[0] <= max(iris):
        return None
    centre = landmarks[list(iris)].mean(axis=0)
    p1, p2 = landmarks[corners[0]], landmarks[corners[1]]
    span = _dist(p1, p2)
    if span < 1e-6:
        return None
    axis = (p2 - p1) / span
    return float(np.dot(centre - (p1 + p2) / 2.0, axis) / span)


def iris_score(landmarks: np.ndarray) -> Optional[float]:
    """Mean horizontal iris displacement, sign-matched to :func:`yaw_score`."""
    offsets = [
        _eye_iris_offset(landmarks, cfg.EYE_CORNERS_LEFT, cfg.IRIS_LEFT),
        _eye_iris_offset(landmarks, cfg.EYE_CORNERS_RIGHT, cfg.IRIS_RIGHT),
    ]
    usable = [o for o in offsets if o is not None]
    if not usable:
        return None
    return -float(np.mean(usable))


def _single_ear(landmarks: np.ndarray, ring: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in ring)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Mean eye-aspect-ratio over both eyes; low values indicate a closed eye."""
    return float(np.mean([_single_ear(landmarks, cfg.EAR_LEFT),
                          _single_ear(landmarks, cfg.EAR_RIGHT)]))


def extract_features(landmarks: np.ndarray) -> GeometricFeatures:
    return GeometricFeatures(
        yaw_score=yaw_score(landmarks),
        iris_score=iris_score(landmarks),
        ear=eye_aspect_ratio(landmarks),
    )


class GeometricEstimator:
    """Applies calibrated thresholds to the geometric scores."""

    def __init__(self, calibration_path: Path | str = cfg.CALIBRATION_PATH):
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self.meta = payload
        self.yaw = Thresholds(**payload["yaw"]["thresholds"])
        self.iris = Thresholds(**payload["iris"]["thresholds"])
        self.ear_closed_below = float(payload["ear"]["closed_below"])

    def predict(self, landmarks: np.ndarray) -> tuple[int, GeometricFeatures]:
        features = extract_features(landmarks)
        return self.yaw.classify(features.yaw_score), features

    def eyes_off_centre(self, features: GeometricFeatures) -> Optional[int]:
        """Iris-only verdict, useful when the head is frontal but the eyes are not."""
        if features.iris_score is None:
            return None
        return self.iris.classify(features.iris_score)

    def eyes_closed(self, features: GeometricFeatures) -> Optional[bool]:
        if features.ear is None:
            return None
        return features.ear < self.ear_closed_below

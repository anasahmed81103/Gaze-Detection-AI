"""Face localisation: MediaPipe Face Mesh with a Haar cascade fallback.

Face Mesh supplies the 478 refined landmarks (including irises) that the geometric
estimator needs. When it finds nothing -- roughly 2% of frames on our own data -- the
Haar cascade still yields a face box so the CNN branch can be given a sensible crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from . import config as cfg


@dataclass
class FaceObservation:
    """Everything the downstream estimators need about one detected face."""

    bbox: tuple[int, int, int, int]
    landmarks: Optional[np.ndarray]  # (478, 2) pixel coordinates, or None for Haar
    source: str  # "face_mesh" or "haar"

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None


class FaceDetector:
    """Wraps Face Mesh and the Haar cascade behind a single ``detect`` call."""

    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.3):
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
        )
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceObservation]:
        height, width = frame_bgr.shape[:2]
        result = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if result.multi_face_landmarks:
            mesh = result.multi_face_landmarks[0].landmark
            points = np.array([[p.x * width, p.y * height] for p in mesh], dtype=np.float32)
            face_points = points[:468]
            x1, y1 = face_points.min(axis=0)
            x2, y2 = face_points.max(axis=0)
            return FaceObservation(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                landmarks=points,
                source="face_mesh",
            )

        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return FaceObservation(bbox=(int(x), int(y), int(x + w), int(y + h)),
                               landmarks=None, source="haar")

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def eye_roi(frame_bgr: np.ndarray, landmarks: np.ndarray, corners: Sequence[int],
            margin_x: float = 0.4, margin_y: float = 1.2) -> Optional[np.ndarray]:
    """Crop the region around one eye, using the margins from the original notebooks."""
    height, width = frame_bgr.shape[:2]
    xs = [landmarks[i][0] for i in corners]
    ys = [landmarks[i][1] for i in corners]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mx, my = (x2 - x1) * margin_x, (y2 - y1) * margin_y
    a, b = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
    c, d = int(min(x2 + mx, width)), int(min(y2 + my, height))
    roi = frame_bgr[b:d, a:c]
    return roi if roi.size else None

"""The end-to-end gaze pipeline.

Cascade, in order:

1. **Face Mesh** locates the face and 478 landmarks (98.0% of our frames).
2. **Geometric estimator** turns those landmarks into the head-direction verdict. This is
   the primary path because it measured 94.5% +/- 1.7% under 5-fold cross-validation,
   against 84.0% for the fine-tuned CNN on the same held-out images.
3. **EfficientNetB0** takes over when no landmarks are available, since it only needs
   raw pixels. It can also be run alongside the geometric path for agreement analysis.
4. **Iris offset and eye-aspect-ratio** are reported as auxiliary signals: they describe
   eye-in-socket movement and eye closure, which head pose alone cannot express.

Every field of :class:`GazeResult` is optional-aware, so callers can always tell which
branch produced the answer instead of silently trusting a single number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg
from .geometric import GeometricEstimator, GeometricFeatures
from .landmarks import FaceDetector, FaceObservation


UNKNOWN = "unknown"


@dataclass
class GazeResult:
    """One frame's verdict.

    ``label`` is ``None`` when no face could be located and no fallback was available. We
    deliberately do not collapse that case into ``front``: a monitoring system needs to
    distinguish "looking at the screen" from "cannot tell".
    """

    label: Optional[int]
    label_name: str
    source: str  # "geometric", "cnn", or "none"
    face: Optional[FaceObservation] = None
    features: Optional[GeometricFeatures] = None
    cnn_proba: Optional[np.ndarray] = None
    cnn_label: Optional[int] = None
    iris_label: Optional[int] = None
    eyes_closed: Optional[bool] = None
    smoothed_label: Optional[int] = None
    latency_ms: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def face_found(self) -> bool:
        return self.face is not None

    @property
    def determined(self) -> bool:
        return self.label is not None

    @property
    def confidence(self) -> Optional[float]:
        if self.cnn_proba is not None:
            return float(self.cnn_proba.max())
        return None


class GazePipeline:
    """Composable front-end for images, video files and webcams."""

    def __init__(
        self,
        use_cnn: bool = True,
        cnn_weights: Path | str = cfg.CNN_FINETUNED,
        calibration: Path | str = cfg.CALIBRATION_PATH,
        static_image_mode: bool = False,
        always_run_cnn: bool = False,
    ):
        self.detector = FaceDetector(static_image_mode=static_image_mode)
        self.geometric = GeometricEstimator(calibration)
        self.always_run_cnn = always_run_cnn
        self.cnn = None
        if use_cnn:
            from .cnn import HeadDirectionCNN

            self.cnn = HeadDirectionCNN(cnn_weights)

    def __call__(self, frame_bgr: np.ndarray) -> GazeResult:
        return self.process(frame_bgr)

    def process(self, frame_bgr: np.ndarray) -> GazeResult:
        import time

        started = time.perf_counter()
        face = self.detector.detect(frame_bgr)

        label: Optional[int] = None
        source = "none"
        features = None
        iris_label = None
        eyes_closed = None

        if face is not None and face.has_landmarks:
            label, features = self.geometric.predict(face.landmarks)
            source = "geometric"
            iris_label = self.geometric.eyes_off_centre(features)
            eyes_closed = self.geometric.eyes_closed(features)

        cnn_proba = None
        cnn_label = None
        need_cnn = self.cnn is not None and (label is None or self.always_run_cnn)
        if need_cnn:
            cnn_label, cnn_proba = self.cnn.predict(frame_bgr)
            if label is None:
                label, source = cnn_label, "cnn"

        return GazeResult(
            label=label,
            label_name=cfg.CLASSES[label] if label is not None else UNKNOWN,
            source=source,
            face=face,
            features=features,
            cnn_proba=cnn_proba,
            cnn_label=cnn_label,
            iris_label=iris_label,
            eyes_closed=eyes_closed,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "GazePipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

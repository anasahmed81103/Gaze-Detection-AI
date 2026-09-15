"""EfficientNetB0 head-direction classifier.

The network was trained on whole frames rather than face crops, so inference must also
receive whole frames -- cropping to the face costs about four accuracy points, which we
measured rather than assumed (see docs/REPORT.md).

TensorFlow is imported lazily so that the geometric-only path never pays for loading the
16 MB Keras graph. Note that MediaPipe 0.10.x imports TensorFlow itself, so TensorFlow is
still a hard dependency of the project as a whole -- the saving here is startup time and
memory, not the requirement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg


class HeadDirectionCNN:
    """Thin wrapper around the saved Keras model with the correct preprocessing."""

    def __init__(self, weights: Path | str = cfg.CNN_FINETUNED):
        from tensorflow.keras.models import load_model  # noqa: PLC0415 - lazy on purpose

        self.weights_path = Path(weights)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. "
                "See models/README.md for how to obtain them."
            )
        self._model = load_model(self.weights_path, compile=False)
        self._preprocess = self._load_preprocess()
        # One warm-up pass so the first real frame is not penalised by graph tracing.
        self.predict_proba(np.zeros((*cfg.CNN_INPUT_SIZE, 3), dtype=np.uint8))

    @staticmethod
    def _load_preprocess():
        from tensorflow.keras.applications.efficientnet import preprocess_input

        return preprocess_input

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, cfg.CNN_INPUT_SIZE).astype("float32")
        return self._preprocess(resized)

    def predict_proba(self, frame_bgr: np.ndarray) -> np.ndarray:
        batch = self._prepare(frame_bgr)[None, ...]
        return self._model.predict(batch, verbose=0)[0]

    def predict_proba_batch(self, frames_bgr: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        batch = np.stack([self._prepare(f) for f in frames_bgr])
        return self._model.predict(batch, batch_size=batch_size, verbose=0)

    def predict(self, frame_bgr: np.ndarray) -> tuple[int, np.ndarray]:
        proba = self.predict_proba(frame_bgr)
        return int(np.argmax(proba)), proba

    @property
    def num_parameters(self) -> int:
        return int(self._model.count_params())

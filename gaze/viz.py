"""Overlay drawing for demos and for the figures in the README."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config as cfg
from .pipeline import GazeResult

COLOURS = {
    "front": (86, 201, 86),
    "left": (232, 168, 56),
    "right": (72, 156, 240),
    "unknown": (150, 150, 150),
}
ALERT_COLOUR = (60, 60, 232)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def annotate(
    frame_bgr: np.ndarray,
    result: GazeResult,
    fps: Optional[float] = None,
    alert: bool = False,
    show_landmarks: bool = True,
    scale_override: Optional[float] = None,
) -> np.ndarray:
    """Draw the verdict, the supporting scores and optional landmarks onto a copy."""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    scale = scale_override if scale_override else max(0.5, min(width, height) / 720.0)
    shown = result.smoothed_label if result.smoothed_label is not None else result.label
    label = cfg.CLASSES[shown] if shown is not None else "unknown"
    colour = ALERT_COLOUR if alert else COLOURS[label]

    if result.face is not None:
        x1, y1, x2, y2 = result.face.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, max(1, int(2 * scale)))

    if show_landmarks and result.face is not None and result.face.has_landmarks:
        for idx in (*cfg.IRIS_LEFT, *cfg.IRIS_RIGHT):
            if idx < len(result.face.landmarks):
                px, py = result.face.landmarks[idx]
                cv2.circle(canvas, (int(px), int(py)), max(1, int(2 * scale)), (255, 255, 255), -1)
        for idx in (cfg.NOSE_TIP, cfg.CHEEK_LEFT, cfg.CHEEK_RIGHT):
            px, py = result.face.landmarks[idx]
            cv2.circle(canvas, (int(px), int(py)), max(2, int(3 * scale)), colour, -1)
        nose = result.face.landmarks[cfg.NOSE_TIP]
        for cheek in (cfg.CHEEK_LEFT, cfg.CHEEK_RIGHT):
            pt = result.face.landmarks[cheek]
            cv2.line(canvas, (int(nose[0]), int(nose[1])), (int(pt[0]), int(pt[1])),
                     colour, max(1, int(1 * scale)))

    lines = [f"source: {result.source}"]
    if result.features is not None:
        lines.append(f"yaw score: {result.features.yaw_score:+.3f}")
        if result.features.iris_score is not None:
            lines.append(f"iris score: {result.features.iris_score:+.3f}")
        if result.eyes_closed is not None:
            lines.append(f"eyes: {'closed' if result.eyes_closed else 'open'}")
    if result.cnn_proba is not None:
        lines.append("cnn: " + " ".join(
            f"{c[:1]}={p:.2f}" for c, p in zip(cfg.CLASSES, result.cnn_proba)))
    if fps is not None:
        lines.append(f"{fps:.1f} FPS  ({result.latency_ms:.0f} ms)")

    pad = int(12 * scale)
    line_h = int(22 * scale)
    panel_h = pad * 2 + int(38 * scale) + line_h * len(lines)
    panel_w = int(330 * scale)
    _panel(canvas, pad, pad, panel_w, panel_h)

    cv2.putText(canvas, label.upper(), (pad * 2, pad + int(32 * scale)),
                FONT, 1.05 * scale, colour, max(2, int(2.4 * scale)), cv2.LINE_AA)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (pad * 2, pad + int(38 * scale) + line_h * (i + 1) - int(6 * scale)),
                    FONT, 0.46 * scale, (232, 232, 232), max(1, int(1 * scale)), cv2.LINE_AA)

    if alert:
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), ALERT_COLOUR, max(2, int(6 * scale)))
        text = "GAZE AWAY FROM SCREEN"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.8 * scale, max(1, int(2 * scale)))
        cv2.putText(canvas, text, ((width - tw) // 2, height - int(24 * scale)),
                    FONT, 0.8 * scale, ALERT_COLOUR, max(1, int(2 * scale)), cv2.LINE_AA)
    return canvas


def letterbox(image: np.ndarray, side: int, pad_value: int = 22) -> np.ndarray:
    """Fit an image into a ``side`` x ``side`` square without distorting it."""
    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(image, (max(1, int(round(width * scale))),
                                 max(1, int(round(height * scale)))))
    canvas = np.full((side, side, 3), pad_value, dtype=np.uint8)
    y = (side - resized.shape[0]) // 2
    x = (side - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def caption_bar(image: np.ndarray, truth: str, predicted: str, source: str = "") -> np.ndarray:
    """Add a footer stating the ground-truth and predicted class, colour-coded."""
    height = max(30, int(image.shape[0] * 0.085))
    correct = truth == predicted
    bar = np.full((height, image.shape[1], 3), (32, 32, 32), dtype=np.uint8)
    colour = COLOURS.get(predicted, COLOURS["unknown"]) if correct else ALERT_COLOUR
    scale = image.shape[1] / 420.0
    text = f"actual {truth}   ->   predicted {predicted}"
    if source:
        text += f"  [{source}]"
    cv2.putText(bar, text, (int(10 * scale), int(height * 0.68)),
                FONT, 0.44 * scale, colour, max(1, int(1 * scale)), cv2.LINE_AA)
    mark = "OK" if correct else "MISS"
    (tw, _), _ = cv2.getTextSize(mark, FONT, 0.44 * scale, 1)
    cv2.putText(bar, mark, (image.shape[1] - tw - int(12 * scale), int(height * 0.68)),
                FONT, 0.44 * scale, colour, max(1, int(1 * scale)), cv2.LINE_AA)
    return np.vstack([image, bar])

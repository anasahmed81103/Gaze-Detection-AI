"""Measure how each method degrades as the input image resolution falls.

Motivation: the geometric estimator reads relative landmark positions, which Face Mesh
recovers reliably even from small images, whereas the CNN was trained on multi-megapixel
phone captures and keys on fine appearance detail. This script quantifies that gap.

    python scripts/resolution_sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import config as cfg  # noqa: E402
from gaze.geometric import GeometricEstimator, yaw_score  # noqa: E402
from gaze.landmarks import FaceDetector  # noqa: E402
from scripts._features import list_split  # noqa: E402

DEFAULT_SIDES = (512, 448, 384, 320, 256, 192, 128)


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    # Re-encode so the measurement includes realistic JPEG artefacts, not just downsampling.
    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--sides", nargs="+", type=int, default=list(DEFAULT_SIDES))
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--out", type=Path, default=cfg.RESULTS_DIR / "resolution_sweep.json")
    args = parser.parse_args()

    paths: list[Path] = []
    labels: list[int] = []
    for split in args.splits:
        split_paths, split_labels = list_split(args.data / split)
        paths.extend(split_paths)
        labels.extend(split_labels.tolist())
    labels = np.array(labels)
    print(f"{len(paths)} images from {'+'.join(args.splits)}")

    estimator = GeometricEstimator()
    model = None
    if not args.no_cnn and cfg.CNN_FINETUNED.exists():
        from gaze.cnn import HeadDirectionCNN

        model = HeadDirectionCNN(cfg.CNN_FINETUNED)

    rows = []
    print(f"\n{'max side':>9} {'detected':>9} {'geometric':>10} {'cnn':>8} {'cascade':>9}")
    for max_side in sorted(args.sides, reverse=True):
        scores, detected, frames = [], [], []
        with FaceDetector(static_image_mode=True) as detector:
            for path in paths:
                frame = _resize(cv2.imread(str(path)), max_side)
                frames.append(frame)
                face = detector.detect(frame)
                if face is not None and face.has_landmarks:
                    scores.append(yaw_score(face.landmarks))
                    detected.append(True)
                else:
                    scores.append(0.0)
                    detected.append(False)
        scores = np.array(scores)
        detected = np.array(detected)

        geometric = np.full(len(labels), cfg.FRONT, dtype=int)
        geometric[scores > estimator.yaw.high] = cfg.LEFT
        geometric[scores < estimator.yaw.low] = cfg.RIGHT
        row = {
            "max_side": max_side,
            "detection_rate": float(detected.mean()),
            "geometric_accuracy": float((geometric == labels).mean()),
        }
        if model is not None:
            cnn = model.predict_proba_batch(frames).argmax(1)
            row["cnn_accuracy"] = float((cnn == labels).mean())
            row["cascade_accuracy"] = float((np.where(detected, geometric, cnn) == labels).mean())
        rows.append(row)
        print(f"{max_side:>9} {row['detection_rate'] * 100:8.1f}% "
              f"{row['geometric_accuracy'] * 100:9.2f}% "
              f"{row.get('cnn_accuracy', float('nan')) * 100:7.2f}% "
              f"{row.get('cascade_accuracy', float('nan')) * 100:8.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"splits": args.splits, "n_images": len(paths),
                                    "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

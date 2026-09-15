"""Fit the geometric decision thresholds and write models/geometric_calibration.json.

Thresholds are fitted on the **train** split only, so the accuracies reported for the
val/test splits stay genuinely held out. A 5-fold cross-validation over the whole set is
also run and stored, because 81 held-out images alone cannot support a tight estimate.

    python scripts/calibrate_geometric.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import config as cfg  # noqa: E402
from scripts._features import extract_split_features  # noqa: E402

GRID_POINTS = 80


def _predict(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.full(len(scores), cfg.FRONT, dtype=int)
    out[scores > high] = cfg.LEFT
    out[scores < low] = cfg.RIGHT
    return out


def fit_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Exhaustive search over threshold pairs drawn from the score quantiles."""
    grid = np.quantile(scores, np.linspace(0.02, 0.98, GRID_POINTS))
    best = (-1.0, 0.0, 0.0)
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            acc = float((_predict(scores, low, high) == labels).mean())
            if acc > best[0]:
                best = (acc, float(low), float(high))
    return best[1], best[2], best[0]


def cross_validate(scores: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 42) -> dict:
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accuracies, lows, highs = [], [], []
    out_of_fold = np.zeros(len(labels), dtype=int)
    for train_idx, test_idx in splitter.split(scores, labels):
        low, high, _ = fit_thresholds(scores[train_idx], labels[train_idx])
        predicted = _predict(scores[test_idx], low, high)
        out_of_fold[test_idx] = predicted
        accuracies.append(float((predicted == labels[test_idx]).mean()))
        lows.append(low)
        highs.append(high)
    return {
        "folds": folds,
        "seed": seed,
        "fold_accuracy": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "out_of_fold_accuracy": float((out_of_fold == labels).mean()),
        "threshold_low_mean": float(np.mean(lows)),
        "threshold_low_std": float(np.std(lows)),
        "threshold_high_mean": float(np.mean(highs)),
        "threshold_high_std": float(np.std(highs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--out", type=Path, default=cfg.CALIBRATION_PATH)
    args = parser.parse_args()

    splits = {s: extract_split_features(args.data / s) for s in ("train", "val", "test")}
    for name, split in splits.items():
        print(f"{name:>5}: n={len(split.labels):4d}  face detected on "
              f"{split.detected.mean() * 100:.1f}%")

    detection_rate = float(np.concatenate([s.detected for s in splits.values()]).mean())
    all_labels = np.concatenate([s.labels for s in splits.values()])
    payload = {
        "description": "Decision thresholds for the geometric gaze estimator. "
                       "Scores are signed so that positive means the 'left' class.",
        "classes": list(cfg.CLASSES),
        "fitted_on": "train split of data/personal_data_small",
        "dataset": {
            "images_total": int(len(all_labels)),
            "train": int(len(splits["train"].labels)),
            "val": int(len(splits["val"].labels)),
            "test": int(len(splits["test"].labels)),
            "face_mesh_detection_rate": detection_rate,
        },
    }

    for feature in ("yaw", "iris"):
        train = splits["train"]
        scores_train = getattr(train, feature)
        mask = train.detected & np.isfinite(scores_train)
        low, high, train_acc = fit_thresholds(scores_train[mask], train.labels[mask])

        held_out_scores = np.concatenate([getattr(splits[s], feature) for s in ("val", "test")])
        held_out_labels = np.concatenate([splits[s].labels for s in ("val", "test")])
        held_out_acc = float((_predict(held_out_scores, low, high) == held_out_labels).mean())

        every_score = np.concatenate([getattr(splits[s], feature) for s in ("train", "val", "test")])
        cv = cross_validate(every_score, all_labels)

        payload[feature] = {
            "thresholds": {"low": low, "high": high},
            "train_accuracy": train_acc,
            "held_out_accuracy": held_out_acc,
            "cross_validation": cv,
        }
        print(f"\n{feature}: low={low:+.4f} high={high:+.4f}")
        print(f"  train      {train_acc * 100:.2f}%")
        print(f"  held out   {held_out_acc * 100:.2f}%  (val+test, n={len(held_out_labels)})")
        print(f"  5-fold CV  {cv['mean_accuracy'] * 100:.2f}% +/- {cv['std_accuracy'] * 100:.2f}%")

    # No labelled closed-eye split exists locally, so the blink threshold is derived from
    # the observed open-eye distribution and is explicitly a heuristic, not a benchmark.
    ear = np.concatenate([splits[s].ear for s in ("train", "val", "test")])
    ear = ear[np.isfinite(ear) & (ear > 0)]
    payload["ear"] = {
        "closed_below": float(np.percentile(ear, 1.0)),
        "open_eye_mean": float(ear.mean()),
        "open_eye_std": float(ear.std()),
        "note": "Heuristic. Set to the 1st percentile of the open-eye EAR distribution in "
                "data/personal_data_small, which contains no labelled closed-eye images. "
                "Not a benchmarked blink classifier.",
    }
    print(f"\near: open-eye mean={ear.mean():.3f} -> closed_below={payload['ear']['closed_below']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

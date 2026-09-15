"""Leave-one-subject-out evaluation: does the calibration transfer to an unseen person?

The train/val/test folders were split randomly over all photographs, so every subject
appears in every split. Accuracy measured that way answers "does this work on new photos
of people it has already seen", which is the easier question. This script answers the
harder one by grouping images by subject and holding each subject out in turn.

Subjects are recovered from the capture filenames, which encode device and time:

    20250504_032*            subject A  (Galaxy S22, patterned wall)
    20250504_033*, _034*     subject B  (Galaxy S22, cream wall)
    PXL_*                    subject C  (Google Pixel, plain wall)

Only the geometric estimator is evaluated this way. The shipped CNN was fine-tuned on
photographs of all three subjects, so a per-subject score for it would not be a
generalisation estimate; reproducing this protocol for the CNN needs three re-trainings
and is listed as future work in docs/REPORT.md.

    python scripts/cross_subject_eval.py
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

SUBJECT_DESCRIPTIONS = {
    "A": "Galaxy S22, patterned wall",
    "B": "Galaxy S22, cream wall",
    "C": "Google Pixel, plain wall",
}


def subject_of(path: Path) -> str:
    name = path.name
    if name.startswith("PXL_"):
        return "C"
    stamp = name.split("_")[1][:3] if "_" in name else ""
    return "A" if stamp == "032" else "B"


def _predict(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.full(len(scores), cfg.FRONT, dtype=int)
    out[scores > high] = cfg.LEFT
    out[scores < low] = cfg.RIGHT
    return out


def fit(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    grid = np.quantile(scores, np.linspace(0.02, 0.98, 80))
    best = (-1.0, 0.0, 0.0)
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            acc = float((_predict(scores, low, high) == labels).mean())
            if acc > best[0]:
                best = (acc, float(low), float(high))
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--out", type=Path, default=cfg.RESULTS_DIR / "cross_subject.json")
    args = parser.parse_args()

    splits = [extract_split_features(args.data / s) for s in ("train", "val", "test")]
    paths = [p for s in splits for p in s.paths]
    labels = np.concatenate([s.labels for s in splits])
    yaw = np.concatenate([s.yaw for s in splits])
    detected = np.concatenate([s.detected for s in splits])
    subjects = np.array([subject_of(p) for p in paths])

    print(f"{len(labels)} images, subjects: "
          + ", ".join(f"{s}={int((subjects == s).sum())}" for s in sorted(set(subjects))) + "\n")

    rows = []
    for held_out in sorted(set(subjects)):
        test_mask = subjects == held_out
        train_mask = ~test_mask
        low, high = fit(yaw[train_mask], labels[train_mask])
        predicted = _predict(yaw[test_mask], low, high)
        accuracy = float((predicted == labels[test_mask]).mean())
        detect_rate = float(detected[test_mask].mean())
        rows.append({
            "held_out_subject": held_out,
            "description": SUBJECT_DESCRIPTIONS.get(held_out, ""),
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "thresholds": {"low": low, "high": high},
            "accuracy": accuracy,
            "face_mesh_detection_rate": detect_rate,
            "confusion_matrix": [[int(((labels[test_mask] == t) & (predicted == p)).sum())
                                  for p in range(3)] for t in range(3)],
        })
        print(f"hold out subject {held_out} ({SUBJECT_DESCRIPTIONS[held_out]})")
        print(f"  trained on {int(train_mask.sum())} images, tested on {int(test_mask.sum())}")
        print(f"  thresholds low={low:+.3f} high={high:+.3f}")
        print(f"  accuracy {accuracy * 100:.2f}%   face detected on {detect_rate * 100:.1f}%\n")

    accuracies = np.array([r["accuracy"] for r in rows])
    weights = np.array([r["n_test"] for r in rows], dtype=float)
    summary = {
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std()),
        "weighted_accuracy": float((accuracies * weights).sum() / weights.sum()),
        "min_accuracy": float(accuracies.min()),
    }
    print(f"leave-one-subject-out mean {summary['mean_accuracy'] * 100:.2f}% "
          f"+/- {summary['std_accuracy'] * 100:.2f}%   "
          f"(worst subject {summary['min_accuracy'] * 100:.2f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol": "leave-one-subject-out, geometric head-yaw",
                                    "n_images": int(len(labels)),
                                    "folds": rows, "summary": summary}, indent=2),
                        encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

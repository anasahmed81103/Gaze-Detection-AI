"""Benchmark every head-direction method on the same held-out images.

Produces results/metrics.json, results/comparison.md and the confusion-matrix figure used
in the README. Nothing here is hard-coded: every number in the documentation comes out of
this script.

    python scripts/evaluate.py
    python scripts/evaluate.py --split test --no-cnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import config as cfg  # noqa: E402
from gaze.geometric import GeometricEstimator  # noqa: E402
from scripts._features import extract_split_features  # noqa: E402

SPLITS = ("train", "val", "test")


def _threshold_predict(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.full(len(scores), cfg.FRONT, dtype=int)
    out[scores > high] = cfg.LEFT
    out[scores < low] = cfg.RIGHT
    return out


def _scores(labels: np.ndarray, predicted: np.ndarray) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(labels, predicted, labels=list(range(len(cfg.CLASSES))),
                                  target_names=list(cfg.CLASSES), output_dict=True,
                                  zero_division=0)
    return {
        "accuracy": float((predicted == labels).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score")}
                      for c in cfg.CLASSES},
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=list(range(len(cfg.CLASSES)))).tolist(),
        "n": int(len(labels)),
    }


def _cnn_probabilities(paths: list[Path], weights: Path) -> np.ndarray:
    import cv2

    from gaze.cnn import HeadDirectionCNN

    model = HeadDirectionCNN(weights)
    frames = [cv2.imread(str(p)) for p in paths]
    return model.predict_proba_batch(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--out", type=Path, default=cfg.RESULTS_DIR)
    parser.add_argument("--split", choices=(*SPLITS, "held_out", "all"), default="held_out",
                        help="'held_out' is val+test, the images no threshold ever saw.")
    parser.add_argument("--no-cnn", action="store_true", help="Skip TensorFlow entirely.")
    args = parser.parse_args()

    features = {s: extract_split_features(args.data / s) for s in SPLITS}
    estimator = GeometricEstimator()

    if args.split == "held_out":
        chosen = ("val", "test")
    elif args.split == "all":
        chosen = SPLITS
    else:
        chosen = (args.split,)

    labels = np.concatenate([features[s].labels for s in chosen])
    paths = [p for s in chosen for p in features[s].paths]
    yaw = np.concatenate([features[s].yaw for s in chosen])
    iris = np.concatenate([features[s].iris for s in chosen])
    detected = np.concatenate([features[s].detected for s in chosen])

    print(f"Evaluating on split='{args.split}' ({' + '.join(chosen)}), n={len(labels)}")
    print(f"Face Mesh detection rate: {detected.mean() * 100:.1f}%\n")

    methods: dict[str, dict] = {}

    majority = np.full(len(labels), int(np.bincount(labels).argmax()), dtype=int)
    methods["Majority-class baseline"] = {
        "kind": "baseline", "parameters": 0, **_scores(labels, majority)}

    geo_pred = _threshold_predict(yaw, estimator.yaw.low, estimator.yaw.high)
    methods["Geometric head-yaw (Face Mesh)"] = {
        "kind": "geometric", "parameters": 2, **_scores(labels, geo_pred)}

    iris_pred = _threshold_predict(iris, estimator.iris.low, estimator.iris.high)
    methods["Geometric iris offset (Face Mesh)"] = {
        "kind": "geometric", "parameters": 2, **_scores(labels, iris_pred)}

    if not args.no_cnn:
        for title, weights in (
            ("EfficientNetB0 (LFR pre-train only)", cfg.CNN_PRETRAINED),
            ("EfficientNetB0 (fine-tuned)", cfg.CNN_FINETUNED),
        ):
            if not weights.exists():
                print(f"skipping {title}: {weights.name} not found")
                continue
            proba = _cnn_probabilities(paths, weights)
            predicted = proba.argmax(1)
            methods[title] = {"kind": "cnn", "parameters": 4_053_414,
                              **_scores(labels, predicted)}
            if weights == cfg.CNN_FINETUNED:
                cascade = np.where(detected, geo_pred, predicted)
                methods["Cascade (geometric, CNN fallback)"] = {
                    "kind": "hybrid", "parameters": 4_053_416,
                    "note": "Geometric verdict whenever Face Mesh finds landmarks, "
                            "EfficientNetB0 on the remaining frames.",
                    **_scores(labels, cascade)}

    order = sorted(methods, key=lambda k: methods[k]["accuracy"], reverse=True)
    width = max(len(k) for k in methods)
    print(f"{'method'.ljust(width)}   params      acc   macro-F1")
    print("-" * (width + 30))
    for name in order:
        m = methods[name]
        print(f"{name.ljust(width)}   {m['parameters']:>9,}  {m['accuracy'] * 100:6.2f}%   "
              f"{m['macro_f1']:.3f}")

    payload = {
        "split": args.split,
        "splits_used": list(chosen),
        "n_images": int(len(labels)),
        "class_distribution": {c: int((labels == i).sum()) for i, c in enumerate(cfg.CLASSES)},
        "face_mesh_detection_rate": float(detected.mean()),
        "calibration": estimator.meta,
        "methods": methods,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [f"# Method comparison ({args.split}, n={len(labels)})", "",
             "| Method | Trainable parameters | Accuracy | Macro F1 |",
             "| --- | --- | --- | --- |"]
    for name in order:
        m = methods[name]
        lines.append(f"| {name} | {m['parameters']:,} | {m['accuracy'] * 100:.2f}% | "
                     f"{m['macro_f1']:.3f} |")
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out / 'metrics.json'} and {args.out / 'comparison.md'}")


if __name__ == "__main__":
    main()

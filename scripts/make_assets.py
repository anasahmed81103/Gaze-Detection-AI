"""Regenerate every figure used in the README and the report.

All figures are derived from results/*.json and from real images in data/, so they cannot
drift away from the measurements. Run the evaluation scripts first:

    python scripts/calibrate_geometric.py
    python scripts/evaluate.py
    python scripts/benchmark.py
    python scripts/resolution_sweep.py
    python scripts/make_assets.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import GazePipeline  # noqa: E402
from gaze import config as cfg  # noqa: E402
from gaze.viz import annotate, caption_bar, letterbox  # noqa: E402
from scripts._features import extract_split_features  # noqa: E402

INK = "#1c1f26"
MUTED = "#6b7280"
CLASS_COLOURS = {"front": "#56c956", "left": "#e8a838", "right": "#4a9cf0"}
ACCENT = "#4a9cf0"
ACCENT_ALT = "#e8734a"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d8dce3",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "#eceef2",
    "grid.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run the scripts listed in this file's docstring.")
    return json.loads(path.read_text(encoding="utf-8"))


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- hero grid
def hero_grid(out: Path, per_class: int = 4, tile: int = 420) -> None:
    """Annotated predictions on held-out images, one row per class.

    Uses the shipped default configuration -- geometry first, CNN fallback -- so the grid
    shows exactly what a reader gets when they run the demo, misses included.
    """
    chosen: dict[str, list[Path]] = {}
    for split in ("test", "val"):
        for class_name in cfg.CLASSES:
            folder = cfg.DATA_DIR / split / class_name
            if folder.is_dir():
                chosen.setdefault(class_name, []).extend(sorted(folder.glob("*.jpg")))

    rows = []
    with GazePipeline(use_cnn=True, static_image_mode=True) as pipeline:
        for class_name in cfg.CLASSES:
            row = []
            for path in chosen[class_name][:per_class]:
                frame = cv2.imread(str(path))
                result = pipeline.process(frame)
                canvas = annotate(frame, result, show_landmarks=True, scale_override=0.86)
                canvas = letterbox(canvas, tile)
                canvas = caption_bar(canvas, class_name, result.label_name, result.source)
                canvas = cv2.copyMakeBorder(canvas, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                            value=(210, 214, 220))
                row.append(canvas)
            rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  {out.name}  {grid.shape[1]}x{grid.shape[0]}")


def demo_gif(out: Path, width: int = 360, fps: int = 6, colours: int = 96) -> None:
    """Run the *video* path over held-out photos ordered into a continuous head sweep.

    This exercises the same code as ``run_video.py`` -- majority-vote smoothing, the dwell
    timer and the live FPS counter -- rather than re-rendering the still grid. The frames are
    photographs sorted by their own yaw score so they play as one left-to-right sweep, which
    is honest as long as it is described as such: it is not a webcam recording, and it is
    labelled that way in the README.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  (skipping demo.gif, Pillow not installed)")
        return

    from gaze.geometric import GeometricEstimator, extract_features
    from gaze.landmarks import FaceDetector
    from gaze.smoothing import DwellTracker, MajorityVoteSmoother

    paths: list[Path] = []
    for split in ("test", "val"):
        for class_name in cfg.CLASSES:
            folder = cfg.DATA_DIR / split / class_name
            if folder.is_dir():
                paths.extend(sorted(folder.glob("*.jpg")))

    # Order by the yaw score itself so consecutive frames are visually adjacent poses.
    scored: list[tuple[float, Path]] = []
    detector, estimator = FaceDetector(static_image_mode=True), GeometricEstimator()
    for path in paths:
        face = detector.detect(cv2.imread(str(path)))
        if face is not None and face.has_landmarks:
            scored.append((extract_features(face.landmarks).yaw_score, path))
    detector.close()
    scored.sort(key=lambda item: -item[0])  # left (positive) -> front -> right
    ordered = [path for _, path in scored[::2]][:30]

    frames, timestamp = [], 0.0
    smoother, dwell = MajorityVoteSmoother(window=5), DwellTracker(seconds=1.5)
    with GazePipeline(use_cnn=False, static_image_mode=False) as pipeline:
        for path in ordered:
            frame = cv2.imread(str(path))
            result = pipeline.process(frame)
            result.smoothed_label = smoother.update(result.label)
            timestamp += 1.0 / fps
            alert = dwell.update(result.smoothed_label, timestamp)
            canvas = annotate(frame, result, fps=1000.0 / max(result.latency_ms, 1e-6),
                              alert=alert, show_landmarks=True, scale_override=0.7)
            frames.append(Image.fromarray(_bgr_to_rgb(letterbox(canvas, width))))

    # One palette for the whole animation, otherwise per-frame adaptive palettes make the
    # label colours drift and flicker between frames.
    palette = frames[0].quantize(colors=colours, method=Image.MEDIANCUT)
    images = [frame.quantize(palette=palette, dither=Image.NONE) for frame in frames]
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  {out.name}  {len(images)} frames  {out.stat().st_size / 1e6:.2f} MB")


# ------------------------------------------------------------------ confusion matrices
def confusion_matrices(out: Path, metrics: dict) -> None:
    wanted = [name for name in ("Geometric head-yaw (Face Mesh)", "EfficientNetB0 (fine-tuned)",
                                "Cascade (geometric, CNN fallback)")
              if name in metrics["methods"]]
    fig, axes = plt.subplots(1, len(wanted), figsize=(4.1 * len(wanted), 3.9))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, wanted):
        entry = metrics["methods"][name]
        matrix = np.array(entry["confusion_matrix"], dtype=float)
        normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        color="white" if normalised[i, j] > 0.55 else INK,
                        fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(cfg.CLASSES)), cfg.CLASSES)
        ax.set_yticks(range(len(cfg.CLASSES)), cfg.CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        short = name.split(" (")[0]
        ax.set_title(f"{short}\n{entry['accuracy'] * 100:.1f}% accuracy", fontsize=11)
        ax.grid(False)
    fig.suptitle(f"Held-out confusion matrices  (n={metrics['n_images']} images, "
                 f"counts shown)", fontsize=12, y=1.04)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ------------------------------------------------------------------- method comparison
def method_comparison(out: Path, metrics: dict, benchmark: dict) -> None:
    latency = {
        "Geometric head-yaw (Face Mesh)": benchmark["stages"]["Full pipeline (geometric)"]["median_ms"],
        "EfficientNetB0 (fine-tuned)":
            benchmark["stages"].get("Full pipeline (geometric + CNN every frame)", {}).get("median_ms"),
        "Cascade (geometric, CNN fallback)":
            benchmark["stages"]["Full pipeline (geometric)"]["median_ms"],
    }
    items = sorted(metrics["methods"].items(), key=lambda kv: kv[1]["accuracy"])
    names = [k.replace(" (Face Mesh)", "").replace(" (", "\n(") for k, _ in items]
    values = [v["accuracy"] * 100 for _, v in items]
    colours = [ACCENT if v["kind"] in ("geometric", "hybrid") else
               (ACCENT_ALT if v["kind"] == "cnn" else "#c3c8d2") for _, v in items]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars = ax.barh(names, values, color=colours, height=0.66)
    for bar, (name, entry) in zip(bars, items):
        ms = latency.get(name)
        suffix = f"   ({ms:.0f} ms/frame)" if ms else ""
        ax.text(bar.get_width() + 1.1, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%{suffix}", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, 116)
    ax.set_xlabel("Accuracy on held-out images (%)")
    ax.set_title(f"Head-direction accuracy, {metrics['n_images']} held-out images\n"
                 "blue = landmark geometry, orange = deep CNN, grey = trivial baseline",
                 fontsize=11, loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ------------------------------------------------------------------------ separability
def yaw_separability(out: Path, calibration: dict) -> None:
    splits = {s: extract_split_features(cfg.DATA_DIR / s) for s in ("train", "val", "test")}
    scores = np.concatenate([splits[s].yaw for s in splits])
    labels = np.concatenate([splits[s].labels for s in splits])
    detected = np.concatenate([splits[s].detected for s in splits])
    low = calibration["yaw"]["thresholds"]["low"]
    high = calibration["yaw"]["thresholds"]["high"]

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    bins = np.linspace(-1, 1, 70)
    for index, class_name in enumerate(cfg.CLASSES):
        mask = detected & (labels == index)
        ax.hist(scores[mask], bins=bins, alpha=0.72, label=f"{class_name}  (n={mask.sum()})",
                color=CLASS_COLOURS[class_name], edgecolor="white", linewidth=0.4)
    for value, text in ((low, f"low = {low:+.3f}"), (high, f"high = {high:+.3f}")):
        ax.axvline(value, color=INK, linestyle="--", linewidth=1.3)
        ax.text(value, ax.get_ylim()[1] * 0.94, f" {text}", fontsize=9, color=INK)
    ax.set_xlabel("Geometric head-yaw score   (nose-to-cheek asymmetry, positive = looking left)")
    ax.set_ylabel("images")
    ax.set_title("One scalar separates the three classes; the two dashed thresholds are the "
                 "entire model", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ------------------------------------------------------------------ resolution + latency
def resolution_plot(out: Path, sweep: dict) -> None:
    rows = sorted(sweep["rows"], key=lambda r: r["max_side"])
    sides = [r["max_side"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.plot(sides, [r["geometric_accuracy"] * 100 for r in rows], "o-", color=ACCENT,
            linewidth=2.2, markersize=6, label="Geometric head-yaw")
    if "cnn_accuracy" in rows[0]:
        ax.plot(sides, [r["cnn_accuracy"] * 100 for r in rows], "s--", color=ACCENT_ALT,
                linewidth=2.2, markersize=6, label="EfficientNetB0 (fine-tuned)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sides, [str(s) for s in sides])
    ax.set_xlabel("Input image longest side (px)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.set_title(f"Landmark geometry is resolution-invariant, the CNN is not\n"
                 f"{sweep['n_images']} held-out images", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


def latency_plot(out: Path, benchmark: dict) -> None:
    stages = benchmark["stages"]
    names = list(stages)[::-1]
    medians = [stages[n]["median_ms"] for n in names]
    p95 = [stages[n]["p95_ms"] - stages[n]["median_ms"] for n in names]
    colours = [ACCENT_ALT if "CNN" in n or "EfficientNet" in n else ACCENT for n in names]

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    wrapped = [n.replace(" (", "\n(") for n in names]
    ax.barh(wrapped, medians, color=colours, height=0.62, label="median")
    ax.barh(wrapped, p95, left=medians, color=colours, alpha=0.35, height=0.62,
            label="median to p95")
    for y, (median, name) in enumerate(zip(medians, names)):
        ax.text(stages[name]["p95_ms"] + max(medians) * 0.015, y,
                f"{median:.1f} ms   ({stages[name]['implied_fps']:.0f} FPS)",
                va="center", fontsize=9.5, color=INK)
    env = benchmark["environment"]
    ax.set_xlim(0, max(stages[n]["p95_ms"] for n in names) * 1.5)
    ax.set_xlabel("Latency per frame (ms, lower is better)")
    cores = env.get("logical_cores")
    machine = env.get("cpu", "CPU") + (f", {cores} threads" if cores else "")
    ax.set_title(f"CPU-only latency at {benchmark['settings']['resolution']}\n{machine}",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.grid(axis="y", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


def generalisation_plot(out: Path, cross: dict, metrics: dict) -> None:
    """Contrast the easy random split against the honest leave-one-subject-out protocol."""
    folds = cross["folds"]
    names = [f"unseen subject {f['held_out_subject']}\n(n={f['n_test']})" for f in folds]
    values = [f["accuracy"] * 100 for f in folds]
    random_split = metrics["methods"]["Geometric head-yaw (Face Mesh)"]["accuracy"] * 100
    mean = cross["summary"]["mean_accuracy"] * 100

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bars = ax.bar([*names, "leave-one-subject-out\nmean"], [*values, mean],
                  color=[*["#93b7e0"] * len(folds), ACCENT], width=0.62)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.6,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.axhline(random_split, color=ACCENT_ALT, linestyle="--", linewidth=1.6)
    ax.text(len(folds) + 0.45, random_split + 1.4,
            f"random held-out split: {random_split:.1f}%",
            ha="right", fontsize=9.5, color=ACCENT_ALT)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Held-out photos of known people are easy; an unseen person is the real test",
                 fontsize=11, loc="left")
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=cfg.ASSETS_DIR)
    parser.add_argument("--results", type=Path, default=cfg.RESULTS_DIR)
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = _load(args.results / "metrics.json")
    benchmark = _load(args.results / "benchmark.json")
    sweep = _load(args.results / "resolution_sweep.json")
    cross = _load(args.results / "cross_subject.json")
    calibration = _load(cfg.CALIBRATION_PATH)

    print("writing figures:")
    hero_grid(args.out / "hero_grid.jpg")
    if not args.skip_gif:
        demo_gif(args.out / "demo.gif")
    confusion_matrices(args.out / "confusion_matrices.png", metrics)
    method_comparison(args.out / "method_comparison.png", metrics, benchmark)
    yaw_separability(args.out / "yaw_separability.png", calibration)
    resolution_plot(args.out / "resolution_robustness.png", sweep)
    latency_plot(args.out / "latency_breakdown.png", benchmark)
    generalisation_plot(args.out / "generalisation.png", cross, metrics)
    print(f"\nall assets -> {args.out}")


if __name__ == "__main__":
    main()

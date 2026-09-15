"""Run gaze detection on a single image or a folder of images.

    python scripts/run_image.py --source path/to/photo.jpg
    python scripts/run_image.py --source data/personal_data_small/test --out results/preds
    python scripts/run_image.py --source photo.jpg --no-cnn --show

``--no-cnn`` skips TensorFlow altogether and runs the geometric estimator only, which is
the more accurate path anyway and starts in well under a second.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import GazePipeline  # noqa: E402
from gaze import config as cfg  # noqa: E402
from gaze.viz import annotate  # noqa: E402

SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        found = sorted(p for p in source.rglob("*") if p.suffix.lower() in SUFFIXES)
        if not found:
            raise FileNotFoundError(f"No images found under {source}")
        return found
    raise FileNotFoundError(f"{source} does not exist")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--out", type=Path, default=cfg.RESULTS_DIR / "predictions",
                        help="Where annotated images are written.")
    parser.add_argument("--no-cnn", action="store_true", help="Geometric estimator only.")
    parser.add_argument("--show", action="store_true", help="Open a preview window.")
    parser.add_argument("--no-save", action="store_true", help="Do not write annotated images.")
    parser.add_argument("--json", type=Path, help="Optional path for a JSON summary.")
    args = parser.parse_args()

    images = collect(args.source)
    print(f"{len(images)} image(s) from {args.source}")

    records = []
    with GazePipeline(use_cnn=not args.no_cnn, static_image_mode=True,
                      always_run_cnn=not args.no_cnn) as pipeline:
        if not args.no_save:
            args.out.mkdir(parents=True, exist_ok=True)
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  !! unreadable: {path}")
                continue
            result = pipeline.process(frame)
            yaw = f"{result.features.yaw_score:+.3f}" if result.features else "   n/a"
            print(f"  {path.name:<40} {result.label_name:<8} via {result.source:<9} "
                  f"yaw={yaw}  {result.latency_ms:.0f} ms")
            records.append({
                "path": str(path),
                "label": result.label_name,
                "source": result.source,
                "yaw_score": None if result.features is None else result.features.yaw_score,
                "iris_score": None if result.features is None else result.features.iris_score,
                "eyes_closed": result.eyes_closed,
                "cnn_proba": None if result.cnn_proba is None else result.cnn_proba.tolist(),
                "latency_ms": result.latency_ms,
            })
            canvas = annotate(frame, result)
            if not args.no_save:
                cv2.imwrite(str(args.out / f"{path.stem}_{result.label_name}.jpg"), canvas)
            if args.show:
                preview = canvas
                height = preview.shape[0]
                if height > 900:
                    scale = 900 / height
                    preview = cv2.resize(preview, None, fx=scale, fy=scale)
                cv2.imshow("gaze", preview)
                if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                    break
        if args.show:
            cv2.destroyAllWindows()

    if not args.no_save:
        print(f"\nannotated images -> {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"summary -> {args.json}")


if __name__ == "__main__":
    main()

"""Measure per-stage CPU latency and throughput on the machine you run it on.

Every latency figure quoted in the README comes from this script. It reports the median
and the 95th percentile rather than the mean, because a mean hides the occasional slow
frame that a real-time system actually cares about.

    python scripts/benchmark.py
    python scripts/benchmark.py --frames 200 --resolution 1280x720 --no-cnn
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import config as cfg  # noqa: E402
from gaze.geometric import GeometricEstimator, extract_features  # noqa: E402
from gaze.landmarks import FaceDetector  # noqa: E402


def _percentiles(samples: list[float]) -> dict:
    array = np.array(samples, dtype=float)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "implied_fps": float(1000.0 / np.median(array)),
    }


def _sample_frames(resolution: tuple[int, int], count: int) -> list[np.ndarray]:
    """Use real photographs, cycled and resized, so Face Mesh has actual faces to find."""
    paths = sorted((cfg.DATA_DIR / "test").rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No benchmark images under {cfg.DATA_DIR / 'test'}")
    frames = []
    for index in range(count):
        image = cv2.imread(str(paths[index % len(paths)]))
        frames.append(cv2.resize(image, resolution))
    return frames


def _cpu_name() -> str:
    """Best-effort human-readable CPU model; falls back to the architecture string."""
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:  # noqa: BLE001 - best effort only
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--resolution", default="640x480", help="WIDTHxHEIGHT")
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--out", type=Path, default=cfg.RESULTS_DIR / "benchmark.json")
    args = parser.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    frames = _sample_frames((width, height), args.frames + args.warmup)
    stages: dict[str, dict] = {}

    detector = FaceDetector(static_image_mode=False)
    estimator = GeometricEstimator()

    detect_times, geometry_times, landmark_sets = [], [], []
    for index, frame in enumerate(frames):
        start = time.perf_counter()
        face = detector.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= args.warmup:
            detect_times.append(elapsed)
        landmark_sets.append(face.landmarks if face and face.has_landmarks else None)

    for index, landmarks in enumerate(landmark_sets):
        if landmarks is None:
            continue
        start = time.perf_counter()
        features = extract_features(landmarks)
        estimator.yaw.classify(features.yaw_score)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= args.warmup:
            geometry_times.append(elapsed)

    stages["Face Mesh landmark detection"] = _percentiles(detect_times)
    stages["Geometric scoring + decision"] = _percentiles(geometry_times)
    detector.close()

    if not args.no_cnn and cfg.CNN_FINETUNED.exists():
        from gaze.cnn import HeadDirectionCNN

        model = HeadDirectionCNN(cfg.CNN_FINETUNED)
        cnn_times = []
        for index, frame in enumerate(frames):
            start = time.perf_counter()
            model.predict_proba(frame)
            elapsed = (time.perf_counter() - start) * 1000.0
            if index >= args.warmup:
                cnn_times.append(elapsed)
        stages["EfficientNetB0 inference"] = _percentiles(cnn_times)

    from gaze import GazePipeline

    for label, use_cnn in (("Full pipeline (geometric)", False),
                           ("Full pipeline (geometric + CNN every frame)", True)):
        if use_cnn and (args.no_cnn or not cfg.CNN_FINETUNED.exists()):
            continue
        with GazePipeline(use_cnn=use_cnn, always_run_cnn=use_cnn) as pipeline:
            times = []
            for index, frame in enumerate(frames):
                start = time.perf_counter()
                pipeline.process(frame)
                elapsed = (time.perf_counter() - start) * 1000.0
                if index >= args.warmup:
                    times.append(elapsed)
        stages[label] = _percentiles(times)

    name_width = max(len(k) for k in stages)
    print(f"\nresolution {width}x{height}, {args.frames} timed frames "
          f"({args.warmup} warm-up), CPU only\n")
    print(f"{'stage'.ljust(name_width)}   median    p95     FPS")
    print("-" * (name_width + 27))
    for name, values in stages.items():
        print(f"{name.ljust(name_width)}  {values['median_ms']:7.1f} {values['p95_ms']:7.1f} "
              f"{values['implied_fps']:7.1f}")

    payload = {
        "environment": {
            "cpu": _cpu_name(),
            "logical_cores": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "device": "CPU",
        },
        "settings": {"resolution": f"{width}x{height}", "timed_frames": args.frames,
                     "warmup_frames": args.warmup},
        "stages": stages,
    }
    try:
        import tensorflow as tf

        payload["environment"]["tensorflow"] = tf.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import mediapipe as mp

        payload["environment"]["mediapipe"] = mp.__version__
    except Exception:  # noqa: BLE001
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

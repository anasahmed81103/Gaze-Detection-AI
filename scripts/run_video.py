"""Run gaze detection on a webcam or a video file.

    python scripts/run_video.py                          # default webcam, geometric only
    python scripts/run_video.py --source clip.mp4 --save results/clip_annotated.mp4
    python scripts/run_video.py --camera 1 --with-cnn --dwell 2.0

Press ``q`` or ``Esc`` to quit. Predictions are smoothed with a majority vote over a short
window, and a dwell timer raises an on-screen alert once gaze has stayed away from the
screen for longer than ``--dwell`` seconds.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze import DwellTracker, GazePipeline, MajorityVoteSmoother  # noqa: E402
from gaze import config as cfg  # noqa: E402
from gaze.viz import annotate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="Video file. Omit to use the webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--with-cnn", action="store_true",
                        help="Also load EfficientNetB0 (fallback when no face is found).")
    parser.add_argument("--window", type=int, default=7, help="Majority-vote window length.")
    parser.add_argument("--dwell", type=float, default=2.0,
                        help="Seconds away from centre before alerting. 0 disables.")
    parser.add_argument("--save", type=Path, help="Write an annotated video here.")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames.")
    parser.add_argument("--headless", action="store_true", help="Do not open a window.")
    parser.add_argument("--no-landmarks", action="store_true", help="Hide landmark overlay.")
    parser.add_argument("--mirror", action="store_true",
                        help="Flip horizontally first. Labels are defined in image space, so "
                             "use this if you want 'left' to mean your own left.")
    args = parser.parse_args()

    if args.source is not None:
        if not args.source.exists():
            raise FileNotFoundError(args.source)
        capture = cv2.VideoCapture(str(args.source))
        source_name = args.source.name
    else:
        capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        source_name = f"webcam {args.camera}"

    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source_name}. If this is a webcam, check that no "
                           "other application is using it.")

    fps_in = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"source: {source_name}  {width}x{height} @ {fps_in:.0f} FPS")
    print(f"model path: {'geometric + EfficientNetB0' if args.with_cnn else 'geometric only'}")

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                 min(fps_in, 30.0), (width, height))

    smoother = MajorityVoteSmoother(window=args.window)
    dwell = DwellTracker(seconds=args.dwell) if args.dwell > 0 else None
    recent = deque(maxlen=30)
    counts = {name: 0 for name in (*cfg.CLASSES, "unknown")}
    frames = 0
    started = None  # set after model load so startup cost is not charged to throughput

    try:
        with GazePipeline(use_cnn=args.with_cnn, static_image_mode=False) as pipeline:
            started = time.perf_counter()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                now = time.perf_counter()
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                result = pipeline.process(frame)
                result.smoothed_label = smoother.update(result.label)
                if result.smoothed_label is not None:
                    counts[cfg.CLASSES[result.smoothed_label]] += 1
                else:
                    counts["unknown"] += 1
                recent.append(now)
                fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                alert = bool(dwell and dwell.update(result.smoothed_label, now))

                canvas = annotate(frame, result, fps=fps, alert=alert,
                                  show_landmarks=not args.no_landmarks)
                if writer is not None:
                    writer.write(canvas)
                if not args.headless:
                    cv2.imshow("Gaze detection  -  press q to quit", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                frames += 1
                if args.max_frames and frames >= args.max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if frames and started is not None:
        elapsed = time.perf_counter() - started
        print(f"\n{frames} frames in {elapsed:.1f} s  ->  {frames / elapsed:.1f} FPS "
              "(capture + inference + overlay, model load excluded)")
        total = sum(counts.values())
        print("share of frames per class: " + "  ".join(
            f"{k}={v / total * 100:.0f}%" for k, v in counts.items()))
    if args.save:
        print(f"annotated video -> {args.save}")


if __name__ == "__main__":
    main()

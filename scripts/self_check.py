"""Verify a fresh checkout is working before you trust any of its output.

Checks that the committed artefacts are present, that every relative link in the
documentation resolves, and that the pipeline behaves correctly on a real image, on a blank
frame, and through the temporal helpers.

    python scripts/self_check.py

Exits non-zero if anything fails, so it is usable in CI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaze import CLASSES, DwellTracker, GazePipeline, MajorityVoteSmoother  # noqa: E402
from gaze import config as cfg  # noqa: E402

REQUIRED_FILES = (
    "models/geometric_calibration.json",
    "models/head_dir_effnetb0_ft.keras",
    "models/head_dir_effnetb0_lfr.keras",
    "requirements.txt",
    "LICENSE",
    "README.md",
    "docs/REPORT.md",
    "docs/MODEL_CARD.md",
    "docs/DLP_Report_Final.pdf",
    "results/metrics.json",
    "results/benchmark.json",
    "results/cross_subject.json",
    "results/resolution_sweep.json",
    "assets/hero_grid.jpg",
    "assets/demo.gif",
)
# Markdown links and images, plus the src of raw HTML <img> tags, which the README uses to
# size the demo GIF.
LINK_PATTERNS = (re.compile(r"!?\[[^\]]*\]\(([^)]+)\)"),
                 re.compile(r"<img[^>]*\ssrc=[\"']([^\"']+)[\"']"))


def check_files(failures: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).exists():
            failures.append(f"missing required file: {relative}")
    print(f"artefacts   {len(REQUIRED_FILES)} required files checked")


def check_links(failures: list[str]) -> None:
    documents = [p for p in ROOT.rglob("*.md") if "legacy" not in p.parts]
    total = 0
    for document in documents:
        text = document.read_text(encoding="utf-8")
        targets = [t for pattern in LINK_PATTERNS for t in pattern.findall(text)]
        for target in targets:
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            cleaned = target.split("#")[0]
            if not cleaned:
                continue
            total += 1
            if not (document.parent / cleaned).resolve().exists():
                failures.append(f"broken link in {document.relative_to(ROOT)}: {target}")
    print(f"links       {total} relative links across {len(documents)} documents")


def check_pipeline(failures: list[str]) -> None:
    images = sorted((cfg.DATA_DIR / "test" / "left").glob("*.jpg"))
    if not images:
        failures.append("no images under data/personal_data_small/test/left")
        return
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    with GazePipeline(use_cnn=False, static_image_mode=True) as pipeline:
        with_landmarks = 0
        for path in images:
            result = pipeline.process(cv2.imread(str(path)))
            if result.label_name not in (*CLASSES, "unknown"):
                failures.append(f"unexpected label {result.label_name!r} for {path.name}")
            if result.features is not None:
                with_landmarks += 1
        print(f"geometric   landmarks on {with_landmarks}/{len(images)} test/left images")
        if with_landmarks == 0:
            failures.append("Face Mesh produced no landmarks on any test image")

        result = pipeline.process(blank)
        if result.determined or result.label_name != "unknown":
            failures.append(f"blank frame should be 'unknown', got {result.label_name!r}")
        print(f"no-face     reports {result.label_name!r} via {result.source!r}")

    if cfg.CNN_FINETUNED.exists():
        with GazePipeline(use_cnn=True, static_image_mode=True) as pipeline:
            result = pipeline.process(blank)
            if result.source != "cnn":
                failures.append(f"expected CNN fallback on a blank frame, got {result.source!r}")
            print(f"cnn         fallback reports {result.label_name!r} via {result.source!r}")


def check_temporal(failures: list[str]) -> None:
    smoother = MajorityVoteSmoother(window=5)
    if smoother.update(None) is not None:
        failures.append("smoother should stay undecided before any real prediction")
    if smoother.update(1) != 1:
        failures.append("smoother should return the only observed label")
    if [smoother.update(2) for _ in range(3)] != [1, 2, 2]:
        failures.append("majority vote did not behave as expected")

    dwell = DwellTracker(seconds=1.0)
    if dwell.update(1, 0.0) or not dwell.update(1, 1.5) or dwell.update(0, 2.0):
        failures.append("dwell tracker did not alert on the expected schedule")
    print("temporal    majority vote and dwell timer behave as specified")


def main() -> int:
    failures: list[str] = []
    print(f"checking {ROOT}\n")
    check_files(failures)
    check_links(failures)
    check_pipeline(failures)
    check_temporal(failures)

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gaze and head-direction estimation for online proctoring and driver monitoring.

Quick start::

    import cv2
    from gaze import GazePipeline

    with GazePipeline() as pipeline:
        result = pipeline.process(cv2.imread("face.jpg"))
        print(result.label_name, result.source)
"""
from .config import CLASSES, FRONT, LEFT, RIGHT
from .geometric import GeometricEstimator, GeometricFeatures, Thresholds
from .landmarks import FaceDetector, FaceObservation
from .pipeline import GazePipeline, GazeResult
from .smoothing import DwellTracker, MajorityVoteSmoother

__version__ = "1.0.0"

__all__ = [
    "CLASSES",
    "FRONT",
    "LEFT",
    "RIGHT",
    "DwellTracker",
    "FaceDetector",
    "FaceObservation",
    "GazePipeline",
    "GazeResult",
    "GeometricEstimator",
    "GeometricFeatures",
    "MajorityVoteSmoother",
    "Thresholds",
]

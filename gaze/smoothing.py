"""Temporal smoothing for video and webcam streams.

Per-frame predictions flicker around threshold boundaries. A short majority vote over a
sliding window removes almost all of that flicker at the cost of a few frames of lag,
which is invisible at 25 FPS.
"""
from __future__ import annotations

from collections import Counter, deque

from . import config as cfg


class MajorityVoteSmoother:
    def __init__(self, window: int = 7):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._history: deque[int] = deque(maxlen=window)

    def update(self, label: int | None) -> int | None:
        """Undetermined frames are ignored so a dropped detection cannot flip the verdict."""
        if label is not None:
            self._history.append(label)
        if not self._history:
            return None
        return Counter(self._history).most_common(1)[0][0]

    @property
    def stability(self) -> float:
        """Fraction of the current window that agrees with the smoothed label."""
        if not self._history:
            return 0.0
        return Counter(self._history).most_common(1)[0][1] / len(self._history)

    def reset(self) -> None:
        self._history.clear()


class DwellTracker:
    """Raises an alert once gaze has stayed away from ``front`` for long enough.

    This is the piece that turns a frame classifier into something usable for proctoring
    or driver monitoring: a single glance away is normal, a sustained one is not.
    """

    def __init__(self, seconds: float = 2.0):
        self.seconds = seconds
        self._away_since: float | None = None
        self.away_for = 0.0
        self.alert = False

    def update(self, label: int | None, timestamp: float) -> bool:
        if label is None:
            return self.alert
        if label == cfg.FRONT:
            self._away_since = None
            self.away_for = 0.0
            self.alert = False
        else:
            if self._away_since is None:
                self._away_since = timestamp
            self.away_for = timestamp - self._away_since
            self.alert = self.away_for >= self.seconds
        return self.alert

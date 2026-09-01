from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def magnitude(x, y, z) -> np.ndarray:
    return np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2)


def make_highpass(cutoff_hz: float, fs: float, order: int = 2):
    nyq = fs / 2.0
    wn = min(0.99, max(1e-6, cutoff_hz / nyq))
    return butter(order, wn, btype="highpass", output="sos")


class StreamingFilter:
    """Stateful SOS filter so chunk boundaries do not create fake transients."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = None

    def __call__(self, chunk) -> np.ndarray:
        c = np.asarray(chunk, dtype=np.float64)
        if c.size == 0:
            return c.astype(np.float32)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * c[0]
        out, self.zi = sosfilt(self.sos, c, zi=self.zi)
        return out.astype(np.float32)


def rolling_rms(sig, win: int) -> np.ndarray:
    s = np.asarray(sig, dtype=np.float64)
    if win <= 1 or s.size == 0:
        return np.abs(s).astype(np.float32)
    p = np.pad(s**2, (win - 1, 0), mode="edge")
    c = np.cumsum(p)
    out = (c[win - 1 :] - np.concatenate([[0.0], c[:-win]])) / win
    return np.sqrt(np.maximum(out, 0.0)).astype(np.float32)


def mad(v) -> float:
    a = np.asarray(v, dtype=np.float64)
    m = np.median(a)
    return float(1.4826 * np.median(np.abs(a - m)))


@dataclass
class Trigger:
    fs: float
    rms_window_ms: float = 100.0
    baseline_seconds: float = 10.0
    k_mad: float = 6.0
    refractory_s: float = 2.0
    min_peak_mag: float = 3.0

    def __post_init__(self):
        self._win = max(1, int(self.fs * self.rms_window_ms / 1000.0))
        self._hist: list[float] = []
        self._hist_max = int(self.fs * self.baseline_seconds / max(self._win, 1)) + 1
        self._cooldown = 0.0

    def threshold(self) -> float:
        if len(self._hist) < 5:
            return float("inf")
        h = np.asarray(self._hist)
        return float(np.median(h) + self.k_mad * max(mad(h), 1e-3))

    def process(self, chunk) -> tuple[bool, float]:
        r = rolling_rms(chunk, self._win)
        peak = float(r.max()) if r.size else 0.0
        dt = (len(chunk) / self.fs) if self.fs else 0.0
        thr = self.threshold()
        fired = self._cooldown <= 0.0 and peak > thr and peak > self.min_peak_mag
        if fired:
            self._cooldown = self.refractory_s
        else:
            self._cooldown = max(0.0, self._cooldown - dt)
            if peak < thr or thr == float("inf"):
                self._hist.append(peak)
                if len(self._hist) > self._hist_max:
                    self._hist.pop(0)
        return fired, peak

from __future__ import annotations

import numpy as np


def decaying(fs: float, tau_s: float, f_hz: float, n_s: float = 2.0, peak: float = 8.0) -> np.ndarray:
    t = np.arange(int(fs * n_s)) / fs
    y = np.sin(2 * np.pi * f_hz * t) * np.exp(-t / max(tau_s, 1e-3))
    m = np.max(np.abs(y)) or 1.0
    return (y / m * peak).astype(np.float32)


def body_like(fs: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return decaying(
        fs,
        tau_s=float(rng.uniform(0.35, 0.70)),
        f_hz=float(rng.uniform(8.0, 16.0)),
        peak=float(rng.uniform(6.0, 14.0)),
    )


def click_like(fs: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return decaying(
        fs,
        tau_s=float(rng.uniform(0.03, 0.08)),
        f_hz=float(rng.uniform(28.0, 45.0)),
        peak=float(rng.uniform(8.0, 18.0)),
    )

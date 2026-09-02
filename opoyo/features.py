from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "peak",
    "rms",
    "crest",
    "rise_ms",
    "decay_ms",
    "low_ratio",
    "centroid_hz",
]


def extract(window: np.ndarray, fs: float) -> dict[str, float]:
    """window: high-passed acceleration magnitude, ~2 s. Never raises."""
    w = np.asarray(window, dtype=np.float64)
    if w.size < 8:
        return {k: 0.0 for k in FEATURE_NAMES}

    env = np.abs(w)
    peak = float(env.max())
    rms = float(np.sqrt((w**2).mean()))
    pk_i = int(env.argmax())
    crest = peak / rms if rms > 1e-9 else 0.0

    thr = 0.1 * peak
    pre = np.nonzero(env[: pk_i + 1] >= thr)[0]
    rise_ms = ((pk_i - pre[0]) / fs * 1000.0) if pre.size else 0.0

    post = np.nonzero(env[pk_i:] <= thr)[0]
    decay_ms = (post[0] / fs * 1000.0) if post.size else (env.size - pk_i) / fs * 1000.0

    spec = np.abs(np.fft.rfft(w * np.hanning(w.size)))
    freqs = np.fft.rfftfreq(w.size, d=1.0 / fs)
    total = float(spec.sum()) + 1e-12
    low_cut = min(20.0, 0.4 * (fs / 2.0))
    low_ratio = float(spec[freqs <= low_cut].sum() / total)
    centroid_hz = float((freqs * spec).sum() / total)

    return {
        "peak": peak,
        "rms": rms,
        "crest": crest,
        "rise_ms": rise_ms,
        "decay_ms": decay_ms,
        "low_ratio": low_ratio,
        "centroid_hz": centroid_hz,
    }


def to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)

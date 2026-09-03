"""Single source of truth for the vibration feature vector.

Both the trainer (``train.fit``) and the live edge classifier (``edge.infer``)
import from here, so train-time and serve-time features cannot drift apart.

The window is peak-normalised before the shape features are taken, which makes
them invariant to range and floor covering. Two consequences follow, and they
are why ``peak`` is not a feature and ``crest`` is measured before normalising:

  * absolute peak after normalisation is identically 1.0 and carries no
    information, so it is dropped;
  * crest factor is a dimensionless ratio, so it is computed on the *raw*
    window where it still means peak-to-RMS.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("rms", "crest", "decay_ms", "db_med", "low", "cent")


def peak_window(mag: np.ndarray, fs: float, pre_s: float = 0.5, post_s: float = 1.5) -> np.ndarray:
    """Cut a fixed-length clip centred on the largest excursion."""
    pre = int(fs * pre_s)
    post = int(fs * post_s)
    i = int(np.argmax(np.abs(mag))) if mag.size else 0
    a = max(0, i - pre)
    b = min(mag.size, i + post)
    w = mag[a:b]
    need = pre + post
    if w.size < need:
        w = np.pad(w, (0, need - w.size))
    return w[:need]


def peak_normalize(w: np.ndarray) -> np.ndarray:
    """Kill range / carpet gain. Shape only."""
    w = np.asarray(w, dtype=np.float64)
    pk = float(np.max(np.abs(w))) if w.size else 0.0
    if pk < 1e-12:
        return w
    return w / pk


def crest_factor(w: np.ndarray) -> float:
    """Peak-to-RMS of the RAW window. Dimensionless, so normalisation-free."""
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return 0.0
    rms = float(np.sqrt((w**2).mean()))
    return float(np.max(np.abs(w)) / rms) if rms > 1e-12 else 0.0


def spec_placeholder(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8 or fs <= 0:
        return np.zeros(0), np.zeros(0)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    hz = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return spec, hz


def vector(mag: np.ndarray, db: np.ndarray, fs: float) -> np.ndarray:
    """Feature vector for one window. Order matches FEATURE_NAMES."""
    raw = peak_window(np.asarray(mag, dtype=np.float64), fs)
    crest = crest_factor(raw)
    w = peak_normalize(raw)

    rms = float(np.sqrt((w**2).mean())) if w.size else 0.0

    env = np.abs(w)
    pk = int(env.argmax()) if env.size else 0
    thr = 0.1 * (float(env.max()) if env.size else 0.0)
    post = np.nonzero(env[pk:] <= thr)[0] if env.size else np.array([])
    decay_ms = (post[0] / fs * 1000.0) if post.size else 0.0

    spec, hz = spec_placeholder(w, fs)
    tot = float(spec.sum()) + 1e-12
    low = float(spec[hz <= 8.0].sum() / tot) if spec.size else 0.0
    cent = float((hz * spec).sum() / tot) if spec.size else 0.0

    db = np.asarray(db, dtype=np.float64)
    db_ok = db[db > -100]
    db_med = float(np.median(db_ok)) if db_ok.size else -160.0

    return np.array([rms, crest, decay_ms, db_med, low, cent], dtype=np.float64)


def clip_of(mag: np.ndarray, fs: float) -> np.ndarray:
    """Peak-normalised clip, for the augmentation / CNN experiments."""
    return peak_normalize(peak_window(mag, fs))

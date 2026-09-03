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
    """Cut a fixed-length clip centred on the largest excursion.

    Args:
        mag: Magnitude samples for the window.
        fs: Sample rate in Hz.
        pre_s: Seconds kept before the peak.
        post_s: Seconds kept after the peak.

    Returns:
        Clip of length ``pre_s + post_s``, zero-padded if the source is short.
    """
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
    """Divide by peak absolute value so range and carpet gain drop out.

    Args:
        w: Raw or windowed magnitude.

    Returns:
        Shape-only clip. Unchanged when the peak is numerically zero.
    """
    w = np.asarray(w, dtype=np.float64)
    pk = float(np.max(np.abs(w))) if w.size else 0.0
    if pk < 1e-12:
        return w
    return w / pk


def crest_factor(w: np.ndarray) -> float:
    """Peak-to-RMS of the raw window.

    Dimensionless, so this is computed before peak normalisation.

    Args:
        w: Un-normalised magnitude clip.

    Returns:
        Crest factor, or 0.0 for an empty or silent clip.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return 0.0
    rms = float(np.sqrt((w**2).mean()))
    return float(np.max(np.abs(w)) / rms) if rms > 1e-12 else 0.0


def spec_placeholder(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed rFFT magnitude and frequency axis."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8 or fs <= 0:
        return np.zeros(0), np.zeros(0)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    hz = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return spec, hz


def vector(mag: np.ndarray, db: np.ndarray, fs: float) -> np.ndarray:
    """Build the six-feature vector for one window.

    Args:
        mag: Magnitude samples.
        db: Sound-level samples. Values at or below -100 are treated as missing.
        fs: Sample rate in Hz.

    Returns:
        Array in ``FEATURE_NAMES`` order.
    """
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
    # Phones send -120 when the mic is unused; drop that sentinel from the median.
    db_ok = db[db > -100]
    db_med = float(np.median(db_ok)) if db_ok.size else -160.0

    return np.array([rms, crest, decay_ms, db_med, low, cent], dtype=np.float64)


def clip_of(mag: np.ndarray, fs: float) -> np.ndarray:
    """Return a peak-normalised clip for augmentation and CNN experiments."""
    return peak_normalize(peak_window(mag, fs))

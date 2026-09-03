"""Train-only augs on peak-normalized clips. Never touch the test fold."""

from __future__ import annotations

import numpy as np

from train.features import peak_normalize


def time_shift(w: np.ndarray, rng: np.random.Generator, max_shift: int = 4) -> np.ndarray:
    k = int(rng.integers(-max_shift, max_shift + 1))
    return peak_normalize(np.roll(w, k))


def add_noise(w: np.ndarray, rng: np.random.Generator, sigma: float = 0.05) -> np.ndarray:
    return peak_normalize(w + rng.normal(0.0, sigma, size=w.shape))


def stretch(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = float(rng.uniform(0.90, 1.10))
    n = w.size
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, max(8, int(round(n * factor))))
    y = np.interp(dst, src, w)
    if y.size >= n:
        y = y[:n]
    else:
        y = np.pad(y, (0, n - y.size))
    return peak_normalize(y)


def polarity(w: np.ndarray) -> np.ndarray:
    return peak_normalize(-w)


def mix_quiet(w: np.ndarray, quiet: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = float(rng.uniform(0.05, 0.20))
    q = quiet
    if q.size != w.size:
        q = np.interp(np.linspace(0, 1, w.size), np.linspace(0, 1, max(q.size, 1)), q)
    return peak_normalize((1.0 - a) * w + a * q)


def expand(
    clip: np.ndarray,
    y: int,
    rng: np.random.Generator,
    quiet: np.ndarray | None,
    copies: int | None = None,
) -> list[np.ndarray]:
    """Original plus copies. More copies for the rare heeldrop class."""
    n = copies if copies is not None else (6 if y == 1 else 2)
    out = [clip]
    for _ in range(n):
        kind = int(rng.integers(0, 5 if quiet is not None else 4))
        if kind == 0:
            out.append(time_shift(clip, rng))
        elif kind == 1:
            out.append(add_noise(clip, rng))
        elif kind == 2:
            out.append(stretch(clip, rng))
        elif kind == 3:
            out.append(polarity(clip))
        else:
            out.append(mix_quiet(clip, quiet, rng))
    return out

"""Tiny rfft for a 2 s window. Placeholder until a real WAV spectrum exists."""

from __future__ import annotations

import cmath
import math


def rfft_mag(x: list[float], hz: float) -> tuple[list[float], list[float]]:
    """Return real-FFT magnitudes and frequencies.

    Args:
        x: Time-domain samples.
        hz: Sample rate.

    Returns:
        ``(|X[k]| for k=0..n//2, frequency in Hz)``. Empty input yields empty
        lists.
    """
    n = len(x)
    if n < 8 or hz <= 0:
        return [], []
    spec: list[float] = []
    freqs: list[float] = []
    two_pi = 2.0 * math.pi
    half = n // 2
    for k in range(half + 1):
        acc = 0j
        for i, v in enumerate(x):
            acc += v * cmath.exp(-1j * two_pi * k * i / n)
        spec.append(abs(acc))
        freqs.append(k * hz / n)
    return spec, freqs

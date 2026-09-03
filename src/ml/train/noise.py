"""Adversarial acoustic conditions for a live pitch venue.

Nothing here is recorded in a quiet room. These are the things actually
present when fifty people are watching you demo.
"""
from __future__ import annotations
import numpy as np


def babble(n: int, fs: float, rng, voices: int = 24) -> np.ndarray:
    """Many overlapping talkers. Speech-shaped noise, syllable-rate modulated."""
    out = np.zeros(n)
    for _ in range(voices):
        x = rng.normal(0, 1, n)
        # speech-shaped: roll off above ~1 kHz
        X = np.fft.rfft(x)
        hz = np.fft.rfftfreq(n, 1 / fs)
        X *= 1.0 / (1.0 + (hz / 500.0) ** 1.2)
        v = np.fft.irfft(X, n)
        # syllabic envelope, 2-5 Hz
        f = rng.uniform(2, 5)
        env = 0.5 * (1 + np.sin(2 * np.pi * f * np.arange(n) / fs + rng.uniform(0, 6.28)))
        out += v * env
    return out / (np.std(out) + 1e-12)


def applause(n: int, fs: float, rng, rate_hz: float = 900.0) -> np.ndarray:
    """Dense broadband claps. The dangerous one: impulsive, like an impact."""
    out = np.zeros(n)
    k = max(1, int(rate_hz * n / fs))
    idx = rng.integers(0, n, size=k)
    out[idx] = rng.normal(0, 1, k)
    # each clap is a short broadband burst, not a click
    h = np.exp(-np.arange(int(0.004 * fs)) / (0.0012 * fs))
    out = np.convolve(out, h, mode="same")
    return out / (np.std(out) + 1e-12)


def hum(n: int, fs: float, rng) -> np.ndarray:
    """Air conditioning / PA hum."""
    t = np.arange(n) / fs
    out = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
              for f in (50, 100, 150, 220))
    out += 0.4 * rng.normal(0, 1, n)
    return out / (np.std(out) + 1e-12)


def chatter_and_shuffle(n: int, fs: float, rng) -> np.ndarray:
    """Babble plus occasional chair scrapes and footfalls in the room."""
    out = babble(n, fs, rng, voices=16)
    for _ in range(rng.integers(1, 4)):
        i = int(rng.integers(0, max(1, n - int(0.3 * fs))))
        d = int(rng.uniform(0.05, 0.25) * fs)
        s = rng.normal(0, 1, d) * np.hanning(d)
        out[i:i + d] += 2.5 * s
    return out / (np.std(out) + 1e-12)


VENUE = {"babble": babble, "applause": applause, "hum": hum, "shuffle": chatter_and_shuffle}


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise at a target SNR, measured on the clip's own energy."""
    c = np.asarray(clean, float)
    ns = np.asarray(noise, float)[: c.size]
    if ns.size < c.size:
        ns = np.pad(ns, (0, c.size - ns.size))
    pc = float((c ** 2).mean()) + 1e-20
    pn = float((ns ** 2).mean()) + 1e-20
    g = np.sqrt(pc / (pn * 10 ** (snr_db / 10.0)))
    out = c + g * ns
    m = float(np.max(np.abs(out))) or 1.0
    return out / m if m > 1.0 else out

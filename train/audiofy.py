"""Audio-fication: shift the 0-25 Hz floor signal into the band YAMNet knows.

Our vibration clip is ~100 samples at 50 Hz. Reinterpreting it at a higher
sample rate scales frequency up by K and divides duration by K, so the two
constraints fight:

    K = N / (fs_in * D)        duration D seconds after the shift

With N=100, fs_in=50: keeping D >= 0.96 s (YAMNet's patch) caps K at ~2, which
leaves everything under YAMNet's 125 Hz mel floor. Getting into the mel band
needs K ~ 50-300, which leaves a 5-40 ms burst that then sits in silence.

`pad` puts that burst in silence. `tile` repeats it to fill the patch, which
adds a periodicity artefact at the repeat rate -- both are tested.
"""
from __future__ import annotations

import numpy as np

SR = 16000


def audiofy(x: np.ndarray, fs_in: float, K: float, mode: str = "pad",
            patch_s: float = 0.96) -> np.ndarray:
    """Frequency-scale x by K and return a 16 kHz clip of at least patch_s."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return np.zeros(int(SR * patch_s), dtype=np.float32)
    x = x - x.mean()
    dur_out = x.size / (fs_in * K)                 # seconds after the shift
    m = max(8, int(round(dur_out * SR)))
    y = np.interp(np.linspace(0.0, 1.0, m), np.linspace(0.0, 1.0, x.size), x)
    need = int(SR * patch_s)
    if mode == "tile" and y.size < need:
        reps = int(np.ceil(need / y.size))
        y = np.tile(y * np.hanning(y.size), reps)[:need]
    elif y.size < need:                             # pad: burst then silence
        y = np.pad(y, (0, need - y.size))
    y = y[:max(need, y.size)]
    pk = float(np.max(np.abs(y))) or 1.0
    return np.clip(y / pk, -1.0, 1.0).astype(np.float32)

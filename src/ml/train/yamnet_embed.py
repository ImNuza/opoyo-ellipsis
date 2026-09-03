"""Frozen YAMNet: wav [-1, 1] @ 16 kHz → 1024-D mean embedding."""

from __future__ import annotations

import os

import numpy as np

from shared.paths import MODELS_DIR

_model = None


# Point TF Hub at the copy committed in models/tfhub. Without this the first
# inference downloads from tfhub.dev, so a slow venue network turns into a
# silent fallback to the vibration-only head in the middle of a demo.

_CACHE = MODELS_DIR / "tfhub"
if _CACHE.is_dir():
    os.environ.setdefault("TFHUB_CACHE_DIR", str(_CACHE))


def load_yamnet():
    global _model
    if _model is None:
        import tensorflow_hub as hub

        _model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _model


def to_16k(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    if fs <= 0 or abs(fs - 16000) < 1:
        y = x
    else:
        n = max(16, int(round(x.size * 16000 / fs)))
        y = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, x.size), x).astype(np.float32)
    pk = float(np.max(np.abs(y))) or 1.0
    y = y / pk
    return np.clip(y, -1.0, 1.0)


def embed(pcm: np.ndarray, fs: float = 16000.0) -> np.ndarray:
    w = to_16k(pcm, fs)
    if w.size < 16000:
        w = np.pad(w, (0, 16000 - w.size))
    _scores, emb, _spec = load_yamnet()(w)
    return np.mean(emb.numpy(), axis=0)

"""Fall classifiers. Live path is FusionCnn; StubCnn/FakeCnn are test doubles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import joblib
import numpy as np

from shared.features import vector as _mag_vector
from shared.paths import MODELS_DIR as MODELS
from shared.schemas import InferenceResult, SensorWindow

_YAMNET = None


class Classifier(Protocol):
    def infer(self, window: SensorWindow) -> InferenceResult: ...


def _result(window: SensorWindow, is_fall: bool, confidence: float) -> InferenceResult:
    return InferenceResult(
        inference_id=uuid4().hex[:12],
        timestamp=window.t_end_ms,
        node_id=window.node_id,
        room=window.room,
        is_fall=is_fall,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


class StubCnn:
    """Placeholder. Same output shape as a trained model; always no-fall."""

    def infer(self, window: SensorWindow) -> InferenceResult:
        return _result(window, False, 0.0)


class FakeCnn:
    """Deterministic classifier for e2e tests. Does not look at window mag."""

    def __init__(self, is_fall: bool = True, confidence: float = 0.95) -> None:
        self.is_fall = is_fall
        self.confidence = confidence

    def infer(self, window: SensorWindow) -> InferenceResult:
        return _result(window, self.is_fall, self.confidence)


# Point TF Hub at the copy committed in models/tfhub. Without this the first
# inference downloads from tfhub.dev, so a slow venue network turns into a
# silent fallback to the vibration-only head in the middle of a demo.
_CACHE = MODELS / "tfhub"
if _CACHE.is_dir():
    os.environ.setdefault("TFHUB_CACHE_DIR", str(_CACHE))


def _get_yamnet():
    global _YAMNET
    if _YAMNET is None:
        import tensorflow_hub as hub

        _YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
    return _YAMNET


def _to_16k(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    if fs <= 0 or abs(fs - 16000) < 1:
        y = x
    else:
        n = max(16, int(round(x.size * 16000 / fs)))
        y = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, x.size), x).astype(np.float32)
    pk = float(np.max(np.abs(y))) or 1.0
    y = y / pk
    return np.clip(y, -1.0, 1.0)


def _yamnet_embed(pcm: np.ndarray, fs: float = 16000.0) -> np.ndarray:
    w = _to_16k(pcm, fs)
    if w.size < 16000:
        w = np.pad(w, (0, 16000 - w.size))
    _scores, emb, _spec = _get_yamnet()(w)
    return np.mean(emb.numpy(), axis=0)


class JointCnn:
    """One logistic over [6 hand features | 1024-d YAMNet embedding].

    Replaces the two-stage stack, which fed a 2-feature logistic nothing but
    two out-of-fold probabilities and scored lower in every condition tested.
    Falls back to the vibration features alone if the window carries no PCM.
    """

    def __init__(self, models: Path | None = None) -> None:
        root = Path(models) if models is not None else MODELS
        self.joint = joblib.load(root / "joint_head.joblib")
        self.mag = joblib.load(root / "mag_head.joblib")

    def infer(self, window: SensorWindow) -> InferenceResult:
        mag = np.asarray(window.mag, dtype=np.float64)
        db = np.asarray(window.db, dtype=np.float64)
        fs = float(window.hz) or 50.0
        h = _mag_vector(mag, db, fs)
        pcm = np.asarray(window.pcm, dtype=np.float64).ravel()
        if pcm.size < 16:
            p = float(self.mag.predict_proba(h.reshape(1, -1))[0, 1])
            return _result(window, p >= 0.5, p)
        try:
            z = _yamnet_embed(pcm, float(window.pcm_hz) or 16000.0)
        except Exception:
            p = float(self.mag.predict_proba(h.reshape(1, -1))[0, 1])
            return _result(window, p >= 0.5, p)
        x = np.concatenate([h, z]).reshape(1, -1)
        c = float(self.joint.predict_proba(x)[0, 1])
        return _result(window, c >= 0.5, c)


class FusionCnn:
    """Peak-norm mag logistic + frozen YAMNet logistic, fused on [p_mag, p_yam].

    Mag is in g; wav is in [-1, 1]. Each clip is peak-normalized on its own.
    Each head has its own StandardScaler from training. YAMNet loads on first
    window that has PCM.
    """

    def __init__(self, models: Path | None = None) -> None:
        root = Path(models) if models is not None else MODELS
        self.mag = joblib.load(root / "mag_head.joblib")
        self.yam = joblib.load(root / "yamnet_head.joblib")
        self.fuse = joblib.load(root / "fuse_head.joblib")

    def infer(self, window: SensorWindow) -> InferenceResult:
        mag = np.asarray(window.mag, dtype=np.float64)
        db = np.asarray(window.db, dtype=np.float64)
        fs = float(window.hz) or 50.0
        p_mag = float(self.mag.predict_proba(_mag_vector(mag, db, fs).reshape(1, -1))[0, 1])

        pcm = np.asarray(window.pcm, dtype=np.float64).ravel()
        pcm_hz = float(window.pcm_hz) or 16000.0
        if pcm.size < 16:
            return _result(window, p_mag >= 0.5, p_mag)
        try:
            p_yam = float(self.yam.predict_proba(_yamnet_embed(pcm, pcm_hz).reshape(1, -1))[0, 1])
        except Exception:
            return _result(window, p_mag >= 0.5, p_mag)
        conf = float(self.fuse.predict_proba(np.array([[p_mag, p_yam]], dtype=np.float64))[0, 1])
        return _result(window, conf >= 0.5, conf)


class MagOnlyCnn:
    """Vibration head alone. This is what `python -m train.fit` reproduces
    from the shipped takes, since no paired audio is included."""

    def __init__(self, models: Path | None = None) -> None:
        root = Path(models) if models is not None else MODELS
        self.mag = joblib.load(root / "mag_head.joblib")

    def infer(self, window: SensorWindow) -> InferenceResult:
        mag = np.asarray(window.mag, dtype=np.float64)
        db = np.asarray(window.db, dtype=np.float64)
        fs = float(window.hz) or 50.0
        p = float(self.mag.predict_proba(_mag_vector(mag, db, fs).reshape(1, -1))[0, 1])
        return _result(window, p >= 0.5, p)


class NoModelError(RuntimeError):
    """models/ has no usable classifier."""


def load_runtime(allow_stub: bool | None = None) -> Classifier:
    """FusionCnn -> MagOnlyCnn -> hard failure.

    StubCnn never fires, so falling back to it silently makes a broken install
    look like a quiet flat. Set OPOYO_ALLOW_STUB=1 only for smoke tests.
    """
    if allow_stub is None:
        allow_stub = os.getenv("OPOYO_ALLOW_STUB") == "1"

    joint = (MODELS / "joint_head.joblib", MODELS / "mag_head.joblib")
    if all(p.exists() for p in joint):
        try:
            clf = JointCnn()
            print("[edge] JointCnn loaded (hand features + YAMNet, one head)")
            return clf
        except Exception as exc:
            print(f"[edge] JointCnn load failed ({exc}); trying the two-stage stack")

    fusion = (
        MODELS / "mag_head.joblib",
        MODELS / "yamnet_head.joblib",
        MODELS / "fuse_head.joblib",
    )
    if all(p.exists() for p in fusion):
        try:
            clf = FusionCnn()
            print("[edge] FusionCnn loaded (vibration + YAMNet)")
            return clf
        except Exception as exc:
            print(f"[edge] FusionCnn load failed ({exc}); trying vibration-only")

    if (MODELS / "mag_head.joblib").exists():
        try:
            clf = MagOnlyCnn()
            print("[edge] MagOnlyCnn loaded (vibration only; no audio head present)")
            return clf
        except Exception as exc:
            print(f"[edge] MagOnlyCnn load failed ({exc})")

    msg = (
        "no usable classifier in models/. "
        "Run:  python -m train.fit data/takes    "
        "(StubCnn never reports a fall, so the edge will not start with it "
        "unless OPOYO_ALLOW_STUB=1.)"
    )
    if allow_stub:
        print(f"[edge] {msg} -- OPOYO_ALLOW_STUB=1, using StubCnn")
        return StubCnn()
    raise NoModelError(msg)

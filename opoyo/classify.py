from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from opoyo.features import to_vector


class Classifier(Protocol):
    def score(self, feats: dict[str, float]) -> float: ...


class RuleClassifier:
    """Transparent baseline. Longer decay + more low-frequency energy → body-like."""

    def __init__(self, decay_ref_ms=180.0, low_ref=0.45, crest_ref=6.0):
        self.decay_ref_ms = decay_ref_ms
        self.low_ref = low_ref
        self.crest_ref = crest_ref

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def score(self, feats):
        d = feats.get("decay_ms", 0.0) / self.decay_ref_ms
        l = feats.get("low_ratio", 0.0) / self.low_ref
        c = self.crest_ref / max(feats.get("crest", 1e-6), 1e-6)
        z = 1.6 * (d - 1.0) + 1.4 * (l - 1.0) + 0.6 * (c - 1.0)
        return float(np.clip(self._sig(z), 0.0, 1.0))


class ModelClassifier:
    """Sklearn model on the 7-D feature vector. Linear probe lives here after calibrate."""

    def __init__(self, path):
        import joblib

        self.model = joblib.load(path)

    def score(self, feats):
        v = to_vector(feats).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(v)[0, 1])
        return float(self.model.decision_function(v)[0])


def load_classifier(cfg) -> Classifier:
    p = Path(cfg.classify.model_path)
    if not p.is_absolute():
        from opoyo.config import ROOT

        p = ROOT / p
    if p.exists():
        try:
            return ModelClassifier(p)
        except Exception as e:
            print(f"[classify] model load failed ({e}); using RuleClassifier")
    return RuleClassifier()

#!/usr/bin/env python3
"""Fit a linear probe on 7-D features. Group by file. Report precision/recall/F1, not accuracy."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opoyo.config import CFG, ROOT as PKG_ROOT  # noqa: E402
from opoyo.dsp import StreamingFilter, make_highpass  # noqa: E402
from opoyo.features import extract, to_vector, FEATURE_NAMES  # noqa: E402

POS = {"heeldrop", "bag", "fall", "knee"}
NEG = {"book", "pan", "door", "walk", "quiet", "tv", "rigid"}
KNOWN = tuple(sorted(POS | NEG, key=len, reverse=True))


def label_from_stem(stem: str) -> str:
    low = stem.lower()
    for name in KNOWN:
        if low == name or low.startswith(name + "_"):
            return name
    return low.split("_", 1)[0]


def load_csv(path: Path) -> np.ndarray:
    xs, ys, zs = [], [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(float(row.get("ax") or row.get("x") or 0))
            ys.append(float(row.get("ay") or row.get("y") or 0))
            zs.append(float(row.get("az") or row.get("z") or 0))
    mag = np.sqrt(np.asarray(xs) ** 2 + np.asarray(ys) ** 2 + np.asarray(zs) ** 2)
    return mag * float(CFG.sensor.g_to_ms2)


def window_around_peak(mag: np.ndarray, fs: float) -> np.ndarray:
    pre = int(fs * float(CFG.window.pre_s))
    post = int(fs * float(CFG.window.post_s))
    i = int(np.argmax(np.abs(mag)))
    a = max(0, i - pre)
    b = min(mag.size, i + post)
    w = mag[a:b]
    need = pre + post
    if w.size < need:
        w = np.pad(w, (0, need - w.size))
    return w[:need]


def main() -> None:
    fs = float(CFG.sensor.accel_rate_hz)
    data = PKG_ROOT / "data"
    files = sorted(p for p in data.glob("*.csv") if p.name != "metadata.csv")
    if not files:
        print("no data/*.csv — collect first")
        sys.exit(1)
    X, y, groups = [], [], []
    hp = StreamingFilter(make_highpass(float(CFG.trigger.highpass_hz), fs))
    for path in files:
        label = label_from_stem(path.stem)
        if label not in POS and label not in NEG:
            print(f"skip unknown label: {path.name}")
            continue
        mag = hp(load_csv(path))
        feats = extract(window_around_peak(mag, fs), fs)
        X.append(to_vector(feats))
        y.append(1 if label in POS else 0)
        groups.append(path.name)
        print(f"{path.name:20s} y={y[-1]} decay={feats['decay_ms']:.0f}ms crest={feats['crest']:.2f}")
    X = np.vstack(X)
    y = np.asarray(y)
    if len(set(y)) < 2:
        print("need both classes")
        sys.exit(1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import LeaveOneGroupOut

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y)
    for train, test in logo.split(X, y, groups):
        clf.fit(X[train], y[train])
        preds[test] = clf.predict(X[test])
    print(classification_report(y, preds, target_names=["neg", "pos"], digits=3))
    print("confusion [[tn fp],[fn tp]]")
    print(confusion_matrix(y, preds))
    clf.fit(X, y)
    out = PKG_ROOT / CFG.classify.model_path
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(clf, out)
    print(f"wrote {out}  features={FEATURE_NAMES}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen YAMNet embeddings + logistic last layer. Same 5-fold as band logistic."""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))
from shared.paths import REPO_ROOT as ROOT  # noqa: E402

from train.features import peak_normalize  # noqa: E402
from train.labels import y_of  # noqa: E402


def read_wav(p: Path):
    wf = wave.open(str(p), "rb")
    rate = wf.getframerate()
    ch = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    return x, float(rate)


def around_peak(x, fs, pre=0.4, post=1.2):
    i = int(np.argmax(np.abs(x))) if x.size else 0
    a = max(0, i - int(pre * fs))
    b = min(x.size, i + int(post * fs))
    w = x[a:b]
    need = int((pre + post) * fs)
    if w.size < need:
        w = np.pad(w, (0, need - w.size))
    return w[:need].astype(np.float32)


def resample_16k(x, fs):
    if abs(fs - 16000) < 1:
        return x
    n = max(16, int(round(x.size * 16000 / fs)))
    t0 = np.linspace(0, 1, x.size)
    t1 = np.linspace(0, 1, n)
    return np.interp(t1, t0, x).astype(np.float32)


def load(root: Path):
    wavs, y, labs = [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        yi = y_of(folder.name)
        if yi is None:
            continue
        for csv_p in sorted(folder.glob("*.csv")):
            wav_p = csv_p.with_suffix(".wav")
            if not wav_p.exists():
                continue
            pcm, wr = read_wav(wav_p)
            clip = around_peak(pcm, wr)
            wavs.append(resample_16k(clip, wr))
            y.append(yi)
            labs.append(folder.name)
    return wavs, np.asarray(y), np.array(labs)


def embed(model, wavs):
    Z = []
    for w in wavs:
        if w.size < 16000:
            w = np.pad(w, (0, 16000 - w.size))
        w = np.clip(w, -1.0, 1.0).astype(np.float32)
        _scores, emb, _spec = model(w)
        Z.append(np.mean(emb.numpy(), axis=0))
    return np.vstack(Z)


def kfold(X, y, title):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = np.zeros_like(y)
    f1s = []
    for tr, te in skf.split(X, y):
        clf = Pipeline(
            [
                ("s", StandardScaler()),
                ("c", LogisticRegression(max_iter=400, class_weight="balanced")),
            ]
        )
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        f1s.append(f1_score(y[te], pred[te], pos_label=1, zero_division=0))
    print(title, "mean F1", round(float(np.mean(f1s)), 3), [round(f, 2) for f in f1s])
    print(classification_report(y, pred, target_names=["neg", "heeldrop"], digits=3, zero_division=0))
    print("confusion", confusion_matrix(y, pred).tolist())
    print()


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes"
    print("loading YAMNet (frozen)…")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    wavs, y, labs = load(root)
    print("n", len(y), "pos", int(y.sum()), "embed each wav → 1024-D mean")
    Z = embed(model, wavs)
    print("Z", Z.shape)
    kfold(Z, y, "YAMNet frozen + logistic  vs all")
    mask = np.isin(labs, ["heeldrop", "bag", "bottle", "chair", "key"])
    kfold(Z[mask], y[mask], "YAMNet frozen + logistic  vs bag/bottle/chair/key")
    mask = np.isin(labs, ["heeldrop", "noise"])
    kfold(Z[mask], y[mask], "YAMNet frozen + logistic  vs noise")


if __name__ == "__main__":
    main()

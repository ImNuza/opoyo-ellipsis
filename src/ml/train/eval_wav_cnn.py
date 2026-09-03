#!/usr/bin/env python3
"""Wav CNN vs band-logistic. No TF/YAMNet here — numpy 1-D conv + STFT logistic."""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))
from shared.paths import REPO_ROOT as ROOT  # noqa: E402
sys.path.insert(0, str(ROOT.parent / "opoyo-pipeline-clean"))

from opoyo.cnn import CnnClassifier, _sgd_step  # noqa: E402
from train.features import peak_normalize  # noqa: E402
from train.labels import y_of  # noqa: E402


def read_csv(p: Path):
    t, mag = [], []
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]))
            mag.append(float(r["mag"]))
    t = np.asarray(t)
    mag = np.asarray(mag)
    fs = float(1000 / np.median(np.diff(t))) if t.size > 1 else 50.0
    return mag, fs


def read_wav(p: Path):
    wf = wave.open(str(p), "rb")
    rate = wf.getframerate()
    ch = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
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
    return peak_normalize(w[:need])


def downsample(x, fs, target=4000.0):
    n = max(8, int(round(x.size * target / fs)))
    t0 = np.linspace(0, 1, x.size)
    t1 = np.linspace(0, 1, n)
    return np.interp(t1, t0, x), target


def stft_vec(w, fs, n_fft=512, hop=256, n_bands=8):
    w = np.asarray(w, float)
    if w.size < n_fft:
        w = np.pad(w, (0, n_fft - w.size))
    frames = []
    for i in range(0, w.size - n_fft + 1, hop):
        spec = np.abs(np.fft.rfft(w[i : i + n_fft] * np.hanning(n_fft)))
        edges = np.linspace(0, spec.size, n_bands + 1).astype(int)
        bands = [float(spec[edges[b] : edges[b + 1]].sum()) for b in range(n_bands)]
        frames.append(np.log1p(bands))
    F = np.asarray(frames)
    return np.concatenate([F.mean(0), F.std(0), F.max(0)])


def band_vec(w, fs):
    w = np.asarray(w, float)
    pk = float(np.max(np.abs(w))) or 1e-12
    n = w / pk
    spec = np.abs(np.fft.rfft(n * np.hanning(n.size)))
    hz = np.fft.rfftfreq(n.size, d=1 / fs)
    tot = spec.sum() + 1e-12
    bands = [
        float(spec[(hz >= lo) & (hz < hi)].sum() / tot)
        for lo, hi in [(0, 150), (150, 400), (400, 1000), (1000, 3000), (3000, 8000)]
    ]
    return np.array(bands + [float((hz * spec).sum() / tot)])


def load(root: Path):
    wavs, y, labs = [], [], []
    fs_w = 16000.0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        yi = y_of(folder.name)
        if yi is None:
            continue
        for csv_p in sorted(folder.glob("*.csv")):
            wav_p = csv_p.with_suffix(".wav")
            if not wav_p.exists():
                continue
            pcm, wr = read_wav(wav_p)
            fs_w = wr
            wavs.append(around_peak(pcm, wr))
            y.append(yi)
            labs.append(folder.name)
    n = max(w.size for w in wavs)
    wavs = np.stack([np.pad(w, (0, n - w.size))[:n] for w in wavs])
    return wavs, np.asarray(y), np.array(labs), fs_w


def logit_kfold(X, y, name):
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
    print(name, "mean F1", round(float(np.mean(f1s)), 3), [round(f, 2) for f in f1s])
    print(classification_report(y, pred, target_names=["neg", "heeldrop"], digits=3, zero_division=0))


def cnn_kfold(wavs, y, fs):
    """Train tiny 1-D CNN on downsampled wav each fold. Slow-ish, numpy SGD."""
    ds = []
    for w in wavs:
        x, _ = downsample(w, fs, 4000.0)
        ds.append(peak_normalize(x[:6400] if x.size > 6400 else np.pad(x, (0, 6400 - x.size))))
    ds = np.stack(ds)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = np.zeros_like(y)
    f1s = []
    for fold, (tr, te) in enumerate(skf.split(ds, y), 1):
        rng = np.random.default_rng(fold)
        clf = CnnClassifier.random(rng)
        idx = np.array(tr)
        for _ in range(8):
            rng.shuffle(idx)
            for i in idx:
                _sgd_step(clf, ds[i], float(y[i]), 0.04)
        scores = np.array([clf.score_clip(ds[i], 4000.0) for i in te])
        pred[te] = (scores >= 0.5).astype(int)
        f1s.append(f1_score(y[te], pred[te], pos_label=1, zero_division=0))
        print(f"  cnn fold {fold} F1 {f1s[-1]:.2f}")
    print("wav 1-D CNN mean F1", round(float(np.mean(f1s)), 3), [round(f, 2) for f in f1s])
    print(classification_report(y, pred, target_names=["neg", "heeldrop"], digits=3, zero_division=0))


def subset(wavs, y, labs, keep):
    m = np.isin(labs, keep)
    return wavs[m], y[m]


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes"
    wavs, y, labs, fs = load(root)
    print("n", len(y), "pos", int(y.sum()), "fs", fs, "len", wavs.shape[1])
    print("YAMNet: not installed (no tensorflow). Frozen last-layer skipped.")
    print()
    for title, keep in [
        ("vs all", None),
        ("vs objects", ["heeldrop", "bag", "bottle", "chair", "key"]),
    ]:
        if keep is None:
            W, Y = wavs, y
        else:
            W, Y = subset(wavs, y, labs, keep)
        print(f"=== {title} n={len(Y)} pos={int(Y.sum())} ===")
        logit_kfold(np.vstack([band_vec(w, fs) for w in W]), Y, "wav bands logistic")
        logit_kfold(np.vstack([stft_vec(w, fs) for w in W]), Y, "wav STFT logistic")
        cnn_kfold(W, Y, fs)
        print()


if __name__ == "__main__":
    main()

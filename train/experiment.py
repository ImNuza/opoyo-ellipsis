#!/usr/bin/env python3
"""Honest comparison of feature sets on a csv+wav take tree.

Repeated stratified CV, out-of-fold predictions only, no augmentation of the
test fold and no statistic computed across the split. Reports average
precision against the chance rate, because with this many positives F1 at a
fixed 0.5 threshold is noisy.

    python -m train.experiment data/takes-v2
"""
from __future__ import annotations

import csv as _csv
import sys
import wave
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.features import peak_normalize, peak_window, vector  # noqa: E402

POS = {"heeldrop"}


def read_csv(p: Path):
    t, mag, db = [], [], []
    for r in _csv.DictReader(p.open(encoding="utf-8")):
        try:
            t.append(float(r["t"])); mag.append(float(r["mag"]))
        except (KeyError, ValueError):
            continue
        db.append(float(r.get("db") or -120))
    t = np.asarray(t, float)
    fs = float(1000 / np.median(np.diff(t))) if t.size > 1 else 50.0
    return np.asarray(mag, float), np.asarray(db, float), fs


def read_wav(p: Path):
    wf = wave.open(str(p), "rb")
    rate, ch = wf.getframerate(), wf.getnchannels()
    raw = wf.readframes(wf.getnframes()); wf.close()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    return x, float(rate)


BANDS = [(0, 150), (150, 400), (400, 1000), (1000, 3000), (3000, 8000)]
WAV_NAMES = [f"b{lo}_{hi}" for lo, hi in BANDS] + ["cent_hz", "decay_ms", "flat"]


def wav_vector(x: np.ndarray, fs: float) -> np.ndarray:
    """Acoustic shape around the impact. ~8 dims, physically motivated:
    where the energy sits, how bright it is, how fast it dies."""
    i = int(np.argmax(np.abs(x))) if x.size else 0
    a, b = max(0, i - int(0.05 * fs)), min(x.size, i + int(0.45 * fs))
    w = x[a:b]
    need = int(0.5 * fs)
    if w.size < need:
        w = np.pad(w, (0, need - w.size))
    w = peak_normalize(w[:need])
    if w.size < 64:
        return np.zeros(len(WAV_NAMES))
    spec = np.abs(np.fft.rfft(w * np.hanning(w.size)))
    hz = np.fft.rfftfreq(w.size, d=1 / fs)
    tot = float(spec.sum()) + 1e-12
    bands = [float(spec[(hz >= lo) & (hz < hi)].sum() / tot) for lo, hi in BANDS]
    cent = float((hz * spec).sum() / tot)
    env = np.abs(w)
    pk = int(env.argmax())
    post = np.nonzero(env[pk:] <= 0.1)[0]
    decay_ms = (post[0] / fs * 1000.0) if post.size else (env.size - pk) / fs * 1000.0
    gm = float(np.exp(np.mean(np.log(spec + 1e-12))))
    flat = gm / (float(spec.mean()) + 1e-12)          # spectral flatness: tonal vs noisy
    return np.array(bands + [cent, decay_ms, flat], dtype=np.float64)


def load(root: Path):
    Xm, Xw, y, lab = [], [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for c in sorted(folder.glob("*.csv")):
            w = c.with_suffix(".wav")
            if not w.exists():
                continue
            mag, db, fs = read_csv(c)
            if mag.size < 16:
                continue
            pcm, wr = read_wav(w)
            Xm.append(vector(mag, db, fs))
            Xw.append(wav_vector(pcm, wr))
            y.append(1 if folder.name.lower() in POS else 0)
            lab.append(folder.name)
    return np.vstack(Xm), np.vstack(Xw), np.asarray(y), np.array(lab)


def pipe(C=1.0):
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, C=C, class_weight="balanced"))])


def evaluate(X, y, title, C=1.0, reps=10):
    """Each repeat is its own 5-fold partition; AP is averaged over repeats."""
    aps, ps = [], []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        p = cross_val_predict(pipe(C), X, y, cv=cv, method="predict_proba")[:, 1]
        aps.append(average_precision_score(y, p)); ps.append(p)
    p = np.mean(ps, axis=0)
    yhat = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    chance = y.mean(); ap = float(np.mean(aps))
    print(f"  {title:<34} AP {ap:.3f}+-{np.std(aps):.3f} (chance {chance:.3f}, lift {ap/chance:4.2f}x)  "
          f"P {pr:.3f}  R {rc:.3f}  F1 {f1:.3f}   TP{tp} FN{fn} FP{fp} TN{tn}")
    return ap, p


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes-v2"
    Xm, Xw, y, lab = load(root)
    print(f"{len(y)} takes with paired wav | pos={int(y.sum())} neg={int((1-y).sum())}")
    print(f"labels: " + ", ".join(f"{l}={int((lab==l).sum())}" for l in sorted(set(lab))))
    print("\nRepeatedStratifiedKFold(5 splits x 10 repeats), out-of-fold probabilities:\n")
    evaluate(Xm, y, "vibration only (6 feats)")
    evaluate(Xw, y, "audio bands only (8 feats)")
    ap_b, p_b = evaluate(np.hstack([Xm, Xw]), y, "vibration + audio (14 feats)")
    print("\nregularisation sweep on the combined set:")
    for C in (0.05, 0.1, 0.3, 1.0, 3.0):
        evaluate(np.hstack([Xm, Xw]), y, f"  C={C}", C=C, reps=6)
    print("\nper-label mean out-of-fold score (combined, C=1):")
    for l in sorted(set(lab)):
        m = lab == l
        print(f"  {l:<10} {p_b[m].mean():.3f}   (n={int(m.sum())}{', POSITIVE' if l in POS else ''})")


if __name__ == "__main__":
    main()

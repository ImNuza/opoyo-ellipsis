#!/usr/bin/env python3
"""5-fold with train-only aug on a folder-labelled csv+wav tree."""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.augment import add_noise, expand, mix_quiet, polarity, stretch  # noqa: E402
from train.features import peak_normalize  # noqa: E402
from train.labels import y_of  # noqa: E402


def mag_vec(w, fs):
    from train.features import vector_clip as vc

    db = np.full(8, -40.0)
    return vc(w, db, fs)


def wav_vec(w, fs):
    w = np.asarray(w, float)
    if w.size < 64:
        return np.zeros(7)
    pk = float(np.max(np.abs(w))) or 1e-12
    n = w / pk
    spec = np.abs(np.fft.rfft(n * np.hanning(n.size)))
    hz = np.fft.rfftfreq(n.size, d=1 / fs)
    tot = spec.sum() + 1e-12
    bands = [
        float(spec[(hz >= lo) & (hz < hi)].sum() / tot)
        for lo, hi in [(0, 150), (150, 400), (400, 1000), (1000, 3000), (3000, 8000)]
    ]
    return np.array(bands + [float((hz * spec).sum() / tot), pk])


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


def expand_wav(clip, y, rng, quiet):
    n = 6 if y == 1 else 2
    out = [clip]
    for _ in range(n):
        kind = int(rng.integers(0, 5 if quiet is not None else 4))
        if kind == 0:
            k = int(rng.integers(-int(0.03 * clip.size), int(0.03 * clip.size) + 1))
            out.append(peak_normalize(np.roll(clip, k)))
        elif kind == 1:
            out.append(add_noise(clip, rng, sigma=0.04))
        elif kind == 2:
            out.append(stretch(clip, rng))
        elif kind == 3:
            out.append(polarity(clip))
        else:
            out.append(mix_quiet(clip, quiet, rng))
    return out


def load(root: Path):
    mag, wav, y, labs = [], [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        yi = y_of(folder.name)
        if yi is None:
            continue
        for csv_p in sorted(folder.glob("*.csv")):
            wav_p = csv_p.with_suffix(".wav")
            if not wav_p.exists():
                continue
            m, fs = read_csv(csv_p)
            pcm, wr = read_wav(wav_p)
            mag.append(around_peak(m, fs))
            wav.append(around_peak(pcm, wr))
            y.append(yi)
            labs.append(folder.name)
    n_m = max(c.size for c in mag)
    n_w = max(c.size for c in wav)
    mag = np.stack([np.pad(c, (0, n_m - c.size))[:n_m] for c in mag])
    wav = np.stack([np.pad(c, (0, n_w - c.size))[:n_w] for c in wav])
    return mag, wav, np.asarray(y), labs


def feats_mag(clips, fs=50.0):
    return np.vstack([mag_vec(c, fs) for c in clips])


def feats_wav(clips, fs=16000.0):
    return np.vstack([wav_vec(c, fs) for c in clips])


def run(Xm, Xw, y, title, rng):
    print(f"=== {title}  n={len(y)} pos={int(y.sum())} neg={int((1-y).sum())} ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    q_m = Xm[y == 0].mean(0) if np.any(y == 0) else None
    q_w = Xw[y == 0].mean(0) if np.any(y == 0) else None

    def kfold(kind):
        pred = np.zeros_like(y)
        f1s = []
        for tr, te in skf.split(Xm, y):
            tr_m, tr_w, tr_y = [], [], []
            for i in tr:
                m_augs = expand(Xm[i], int(y[i]), rng, q_m)
                w_augs = expand_wav(Xw[i], int(y[i]), rng, q_w)
                # pair shortest
                k = min(len(m_augs), len(w_augs))
                for a, b in zip(m_augs[:k], w_augs[:k]):
                    tr_m.append(a)
                    tr_w.append(b)
                    tr_y.append(y[i])
            tr_m, tr_w, tr_y = np.stack(tr_m), np.stack(tr_w), np.asarray(tr_y)
            if kind == "mag":
                Xtr, Xte = feats_mag(tr_m), feats_mag(Xm[te])
            elif kind == "wav":
                Xtr, Xte = feats_wav(tr_w), feats_wav(Xw[te])
            else:
                Xtr = np.hstack([feats_mag(tr_m), feats_wav(tr_w)])
                Xte = np.hstack([feats_mag(Xm[te]), feats_wav(Xw[te])])
            clf = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("c", LogisticRegression(max_iter=500, class_weight="balanced")),
                ]
            )
            clf.fit(Xtr, tr_y)
            pred[te] = clf.predict(Xte)
            f1s.append(f1_score(y[te], pred[te], pos_label=1, zero_division=0))
        print(kind, "mean F1", round(float(np.mean(f1s)), 3), [round(f, 2) for f in f1s])
        print(classification_report(y, pred, target_names=["neg", "heeldrop"], digits=3, zero_division=0))
        print("confusion", confusion_matrix(y, pred).tolist())
        print()

    kfold("mag")
    kfold("wav")
    kfold("both")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes"
    rng = np.random.default_rng(0)
    mag, wav, y, labs = load(root)
    labs = np.array(labs)
    run(mag, wav, y, "heeldrop vs all", rng)
    mask = np.isin(labs, ["heeldrop", "bag", "bottle", "chair", "key"])
    run(mag[mask], wav[mask], y[mask], "heeldrop vs bag/bottle/chair/key", rng)
    mask = np.isin(labs, ["heeldrop", "noise"])
    run(mag[mask], wav[mask], y[mask], "heeldrop vs noise", rng)


if __name__ == "__main__":
    main()

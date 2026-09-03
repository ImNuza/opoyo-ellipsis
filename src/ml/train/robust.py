#!/usr/bin/env python3
"""Does a clean-trained model survive a noisy venue? And does augmentation fix it?

Three regimes, all with train/test separation inside every fold:
  A  train clean  -> test clean   (what we currently report)
  B  train clean  -> test noisy   (what actually happens on stage)
  C  train noisy  -> test noisy   (augmentation as the mitigation)

Noise is applied per-fold to the correct side only. Never to both.
"""
from __future__ import annotations
import sys, wave
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))
from shared.paths import REPO_ROOT as ROOT  # noqa: E402
import train.experiment as E                       # noqa: E402
from shared.features import vector                 # noqa: E402
from train.noise import VENUE, mix_at_snr          # noqa: E402
from train.yamnet_embed import embed               # noqa: E402

POS = {"heeldrop", "jump"}


def read_wav(p: Path):
    wf = wave.open(str(p), "rb"); r, ch = wf.getframerate(), wf.getnchannels()
    raw = wf.readframes(wf.getnframes()); wf.close()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return (x.reshape(-1, 2).mean(1) if ch == 2 else x), float(r)


def load(root: Path):
    out = []
    for f in sorted(p for p in root.iterdir() if p.is_dir()):
        for c in sorted(f.glob("*.csv")):
            w = c.with_suffix(".wav")
            if not w.exists():
                continue
            mag, db, fs = E.read_csv(c)
            if mag.size < 16:
                continue
            x, fr = read_wav(w)
            out.append((f.name, mag, db, fs, x, fr))
    return out


def pipe(C=1.0):
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, C=C, class_weight="balanced"))])


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes"
    items = load(root)
    y = np.array([1 if n in POS else 0 for n, *_ in items])
    Xm = np.vstack([vector(m, db, fs) for _, m, db, fs, _, _ in items])
    print(f"{len(items)} clips, {int(y.sum())} positive\n")

    print("embedding clean audio ...")
    Zc = np.vstack([embed(x, fr) for _, _, _, _, x, fr in items])

    # pre-embed every noisy variant once
    rng = np.random.default_rng(0)
    noisy = {}
    for name, fn in VENUE.items():
        for snr in (10, 0):
            print(f"embedding {name} @ {snr} dB ...")
            noisy[(name, snr)] = np.vstack(
                [embed(mix_at_snr(x, fn(x.size, fr, rng), snr), fr) for _, _, _, _, x, fr in items]
            )

    cv = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xm, y))

    def run(train_Z, test_Z, C=1.0):
        p = np.zeros(len(y))
        for tr, te in cv:
            Xtr = np.hstack([Xm[tr], train_Z[tr]])
            Xte = np.hstack([Xm[te], test_Z[te]])
            m = pipe(C).fit(Xtr, y[tr])
            p[te] = m.predict_proba(Xte)[:, 1]
        pr, rc, f1, _ = precision_recall_fscore_support(y, (p >= .5).astype(int),
                                                       average="binary", zero_division=0)
        return average_precision_score(y, p), pr, rc

    def run_aug(test_Z, C=1.0):
        """Train fold gets clean + all noisy copies; test fold gets one condition."""
        p = np.zeros(len(y))
        for tr, te in cv:
            Xtr = [np.hstack([Xm[tr], Zc[tr]])]
            ytr = [y[tr]]
            for Z in noisy.values():
                Xtr.append(np.hstack([Xm[tr], Z[tr]])); ytr.append(y[tr])
            m = pipe(C).fit(np.vstack(Xtr), np.concatenate(ytr))
            p[te] = m.predict_proba(np.hstack([Xm[te], test_Z[te]]))[:, 1]
        pr, rc, f1, _ = precision_recall_fscore_support(y, (p >= .5).astype(int),
                                                       average="binary", zero_division=0)
        return average_precision_score(y, p), pr, rc

    ap, pr, rc = run(Zc, Zc)
    print(f"\n{'condition':<24}{'A clean-train':>26}{'C noise-augmented train':>28}")
    print(f"{'':<24}{'AP':>8}{'P':>9}{'R':>9}{'AP':>10}{'P':>9}{'R':>9}")
    print(f"{'clean test (baseline)':<24}{ap:>8.3f}{pr:>9.3f}{rc:>9.3f}" + " " * 28)
    for key, Z in noisy.items():
        a1, p1, r1 = run(Zc, Z)
        a2, p2, r2 = run_aug(Z)
        label = f"{key[0]} @ {key[1]}dB"
        print(f"{label:<24}{a1:>8.3f}{p1:>9.3f}{r1:>9.3f}{a2:>10.3f}{p2:>9.3f}{r2:>9.3f}")

    print("\nvibration channel alone (immune to airborne noise by construction):")
    p = np.zeros(len(y))
    for tr, te in cv:
        m = pipe().fit(Xm[tr], y[tr]); p[te] = m.predict_proba(Xm[te])[:, 1]
    print(f"  AP {average_precision_score(y, p):.3f}  -- unchanged under every condition above")


if __name__ == "__main__":
    main()

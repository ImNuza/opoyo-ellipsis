#!/usr/bin/env python3
"""Borrow real fall recordings from the Tarkett smart-floor data set.

Our own takes contain no recording of a person going down flat: the positives
are heel-drops and jump landings, which are proxies for body-weight impact.
The Tarkett set (Truong, Atiq, Minvielle, Serra, Mougeot & Vayatis, IPOL 13
(2023) 183-197) has 563 annotated falls, 154 of them unconstrained falls by
nursing-home residents, so it is the only source of genuine fall signal
available to us.

Source data
-----------
    https://plmbox.math.cnrs.fr/f/a05ad8fbe7674392962b/?dl=1   (FallData.tar.gz, 330 MB)
    article: https://www.ipol.im/pub/art/2023/389/
    code:    https://github.com/deepcharles/fall-data

Each trial is a .csv of eight integer channels (piezo voltage through a 12-bit
ADC, 0..4096) sampled at 100 Hz, plus a .json carrying the fall index.

    FallData/Controlled/      409 acted falls, ~20 s each, FallEventStart/End given
    FallData/ControlledNoFall/ 333 non-fall activities, ~25 s each
    FallData/Unconstrained/   154 real falls, ~1 h each, single FallEvent index

Bringing it into our take format
--------------------------------
Our phone reports ``mag``, the magnitude of the gravity-free acceleration
vector: non-negative, ~0 at rest. The floor analogue is the norm of the
vibration across the eight channels once the DC pedestal (~2665 counts) is
removed, so a window is built as

    ac[c] = x[c] - median(x[c] over a local context)
    mag   = sqrt(sum_c ac[c]**2)

and decimated 100 -> 50 Hz. Absolute scale never survives into the features
(``shared.features`` peak-normalises the window, and ``crest`` is a
dimensionless ratio), so the counts-vs-g unit mismatch is harmless as long as
the DC pedestal is gone -- which is why the median subtraction is local rather
than global, and why a drifting hour-long unconstrained trial is detrended
around each extracted window rather than as a whole.

There is no microphone in the floor, so ``db`` is written as -120 (the
sentinel our loader already uses for "no audio"). ``ax/ay/az`` are 0: the
floor unit has no axes to report.

The label collision
-------------------
Tarkett's positive class is *a person falling*. Ours is *body-weight impact
through the feet*, and Tarkett explicitly files Jump and Run as NON-falls. So
the two label sets disagree on exactly the events our positives are made of.
Both readings are evaluated below:

    native  -- positive = Tarkett's falls only (what the data set says)
    impact  -- positive = falls + Jump + Run (what our detector is actually for)

Usage
-----
    python -m train.tarkett convert --src /path/to/FallData
    python -m train.tarkett eval
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))
from shared.paths import REPO_ROOT as ROOT  # noqa: E402

from shared.features import peak_normalize, peak_window, vector  # noqa: E402
from train.labels import y_of  # noqa: E402
from train.load import iter_takes  # noqa: E402

OUT = ROOT / "data" / "tarkett"
CACHE = OUT / "_cache.npz"

SRC_FS = 100.0           # Tarkett piezo sample rate
FS = 50.0                # our target rate
WIN_S = 4.0              # exported clip length, event sits at LEAD_S
LEAD_S = 1.5             # seconds of run-up before the annotated event
CTX_S = 10.0             # context used for the local DC estimate

# Positive under each labelling. Folder names are written by convert().
FALL_LABELS = {"fall_acted", "fall_real"}
IMPACT_EXTRA = {"jump", "run"}


# ----------------------------------------------------------------- conversion

def _mag(block: np.ndarray) -> np.ndarray:
    """Eight DC-removed channels -> one non-negative vibration magnitude."""
    ac = block.astype(np.float64) - np.median(block.astype(np.float64), axis=0, keepdims=True)
    return np.sqrt((ac**2).sum(axis=1))


def _decimate(x: np.ndarray) -> np.ndarray:
    """100 -> 50 Hz with an anti-alias filter, clamped back to non-negative."""
    from scipy.signal import decimate

    if x.size < 30:
        return np.maximum(x[::2], 0.0)
    return np.maximum(decimate(x, 2, ftype="fir", zero_phase=True), 0.0)


def _window(sig: np.ndarray, idx: int) -> np.ndarray | None:
    """Clip WIN_S seconds around sample ``idx`` of a raw 8-channel trial."""
    lead = int(LEAD_S * SRC_FS)
    need = int(WIN_S * SRC_FS)
    ctx = int(CTX_S * SRC_FS)
    a = idx - lead
    b = a + need
    if a < 0 or b > sig.shape[0]:
        a = max(0, min(a, sig.shape[0] - need))
        b = a + need
        if a < 0 or b > sig.shape[0]:
            return None
    # DC estimated on a wider context so an hour of drift cannot bias the clip
    ca, cb = max(0, a - ctx), min(sig.shape[0], b + ctx)
    ac = sig[ca:cb].astype(np.float64)
    ac = ac - np.median(ac, axis=0, keepdims=True)
    m = np.sqrt((ac**2).sum(axis=1))[a - ca : a - ca + need]
    return _decimate(m)


def _write(label: str, name: str, mag: np.ndarray) -> Path:
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.csv"
    t0 = 0.0
    step = 1000.0 / FS
    with p.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "mag", "db"])
        for i, v in enumerate(mag):
            w.writerow([f"{t0 + i * step:.0f}", "0.0000", "0.0000", "0.0000",
                        f"{v:.4f}", "-120.00"])
    return p


def _peaks(m: np.ndarray, k: int, sep: int, block: tuple[int, int] | None) -> list[int]:
    """Top-k local maxima of ``m``, at least ``sep`` apart, avoiding ``block``."""
    order = np.argsort(m)[::-1]
    out: list[int] = []
    for i in order:
        if block and block[0] <= i <= block[1]:
            continue
        if any(abs(int(i) - j) < sep for j in out):
            continue
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def convert(src: Path, neg_per_hour: int = 10) -> None:
    """Read FallData, write data/tarkett/<label>/<name>.csv and a cache."""
    need = int(WIN_S * SRC_FS)
    rows: list[tuple[str, str, np.ndarray]] = []

    def meta(p: Path) -> dict:
        return json.loads(p.with_suffix(".json").read_text())

    # --- acted falls -------------------------------------------------------
    for p in sorted((src / "Controlled").glob("*.csv")):
        md = meta(p)
        sig = np.loadtxt(p, delimiter=",", dtype=np.int16)
        idx = int(md["FallEvent"])
        w = _window(sig, idx)
        if w is not None:
            rows.append(("fall_acted", md["Code"], w))

    # --- non-fall activities ----------------------------------------------
    for p in sorted((src / "ControlledNoFall").glob("*.csv")):
        md = meta(p)
        sig = np.loadtxt(p, delimiter=",", dtype=np.int16)
        act = str(md["ActivityType"]).lower()
        dev = str(md["WalkingDevice"])
        label = act if dev in ("None", "N/C") else f"walk_{dev.lower()}"
        # no annotated instant here, so centre on the loudest moment
        full = _mag(sig)
        idx = int(np.argmax(full))
        w = _window(sig, idx)
        if w is not None:
            rows.append((label, md["Code"], w))

    # --- real falls + mined daily-activity negatives ------------------------
    for p in sorted((src / "Unconstrained").glob("*.csv")):
        md = meta(p)
        sig = np.loadtxt(p, delimiter=",", dtype=np.int16)
        idx = int(md["FallEvent"])
        w = _window(sig, idx)
        if w is not None:
            rows.append(("fall_real", md["Code"], w))
        # negatives: loud moments well away from the annotated fall. The paper
        # states the rest of the hour is ordinary daily activity by residents,
        # caretakers and visitors, so these are legitimate hard negatives.
        full = _mag(sig)
        guard = int(30 * SRC_FS)
        blocked = (max(0, idx - guard), min(full.size, idx + guard))
        for k, j in enumerate(_peaks(full, neg_per_hour, int(20 * SRC_FS), blocked)):
            w = _window(sig, j)
            if w is not None:
                rows.append(("daily", f"{md['Code']}-n{k}", w))

    for label, name, w in rows:
        _write(label, name, w)

    labels = np.array([r[0] for r in rows])
    names = np.array([r[1] for r in rows])
    mags = np.vstack([r[2][:int(WIN_S * FS)] for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, labels=labels, names=names, mags=mags)

    print(f"wrote {len(rows)} clips to {OUT}")
    for l in sorted(set(labels)):
        print(f"  {l:<20} {int((labels == l).sum())}")


# --------------------------------------------------------------------- loading

def load_tarkett(scheme: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, clips, y) for the Tarkett clips under a labelling scheme."""
    if not CACHE.exists():
        raise SystemExit(f"no cache at {CACHE}; run `python -m train.tarkett convert` first")
    z = np.load(CACHE, allow_pickle=False)
    labels, mags = z["labels"], z["mags"]
    pos = set(FALL_LABELS) | (IMPACT_EXTRA if scheme == "impact" else set())
    y = np.array([1 if l in pos else 0 for l in labels], dtype=int)
    db = np.full(mags.shape[1], -120.0)
    X = np.vstack([vector(m, db, FS) for m in mags])
    clips = np.vstack([_clip50(m, FS) for m in mags])
    return X, clips, y


def _clip50(mag: np.ndarray, fs: float) -> np.ndarray:
    """Fixed 100-sample (2 s @ 50 Hz) peak-normalised clip, whatever fs came in."""
    if abs(fs - FS) > 0.5:
        n = max(2, int(round(mag.size * FS / fs)))
        mag = np.interp(np.linspace(0, mag.size - 1, n), np.arange(mag.size), mag)
    w = peak_window(np.asarray(mag, float), FS)
    return peak_normalize(w)[:100]


def load_ours() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(features, clips, y, label) for our 93 takes. Features use the take's own fs."""
    X, C, y, lab = [], [], [], []
    for l, _p, a in iter_takes():
        fs = float(a["fs"])
        X.append(vector(a["mag"], a["db"], fs))
        C.append(_clip50(a["mag"], fs))
        y.append(y_of(l))
        lab.append(l)
    return np.vstack(X), np.vstack(C), np.asarray(y), np.array(lab)


# ------------------------------------------------------------------ evaluation

def pipe(C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))])


def cv_ap(X: np.ndarray, y: np.ndarray, reps: int = 10, C: float = 1.0) -> tuple[float, float]:
    """Out-of-fold AP on our takes. No Tarkett anywhere in here."""
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    aps = []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        p = cross_val_predict(pipe(C), X, y, cv=cv, method="predict_proba")[:, 1]
        aps.append(average_precision_score(y, p))
    return float(np.mean(aps)), float(np.std(aps))


def cv_ap_pooled(Xo: np.ndarray, yo: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
                 reps: int = 10, C: float = 1.0, w: float = 1.0) -> tuple[float, float]:
    """Train on Tarkett + the training fold, score the held-out fold of ours.

    Tarkett rows join the *training* half of every split only, so no test take
    is ever contaminated. ``w`` weights each Tarkett row relative to one of
    ours, since Tarkett outnumbers us ~20:1.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    aps = []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        oof = np.zeros(len(yo))
        for tr, te in cv.split(Xo, yo):
            X = np.vstack([Xo[tr], Xt])
            y = np.concatenate([yo[tr], yt])
            sw = np.concatenate([np.ones(len(tr)), np.full(len(yt), w)])
            m = pipe(C)
            m.fit(X, y, c__sample_weight=sw)
            oof[te] = m.predict_proba(Xo[te])[:, 1]
        aps.append(average_precision_score(yo, oof))
    return float(np.mean(aps)), float(np.std(aps))


def _z(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / np.where(sd < 1e-9, 1.0, sd)


def cv_ap_coral(Xo: np.ndarray, yo: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
                reps: int = 10, C: float = 1.0, w: float = 0.1) -> tuple[float, float]:
    """Pooled training with each domain standardised against itself.

    The two sensors do not agree on absolute values -- Tarkett's spectral
    centroid sits ~3 Hz below ours, and it has no microphone at all, so its
    ``db_med`` is a constant. Z-scoring each domain by its own statistics puts
    them on a common footing and turns Tarkett's dead ``db_med`` into a
    harmless zero. Our statistics come from the training fold only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    mt, st = Xt.mean(0), Xt.std(0)
    Zt = _z(Xt, mt, st)
    aps = []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        oof = np.zeros(len(yo))
        for tr, te in cv.split(Xo, yo):
            mo, so = Xo[tr].mean(0), Xo[tr].std(0)
            X = np.vstack([_z(Xo[tr], mo, so), Zt])
            y = np.concatenate([yo[tr], yt])
            sw = np.concatenate([np.ones(len(tr)), np.full(len(yt), w)])
            m = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            m.fit(X, y, sample_weight=sw)
            oof[te] = m.predict_proba(_z(Xo[te], mo, so))[:, 1]
        aps.append(average_precision_score(yo, oof))
    return float(np.mean(aps)), float(np.std(aps))


def cv_ap_stack(Xo: np.ndarray, yo: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
                reps: int = 10, C: float = 1.0) -> tuple[float, float]:
    """Frozen Tarkett detector as one extra column, head refit per fold.

    This is the hand-feature analogue of freezing the extractor: a model is fit
    once on Tarkett alone -- it never sees a take of ours -- and its opinion
    becomes a seventh feature. Our z-scoring is refit on the training fold each
    time, so no test take touches any fitted statistic.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    mt, st = Xt.mean(0), Xt.std(0)
    frozen = LogisticRegression(max_iter=2000, class_weight="balanced")
    frozen.fit(_z(Xt, mt, st), yt)

    aps = []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        oof = np.zeros(len(yo))
        for tr, te in cv.split(Xo, yo):
            mo, so = Xo[tr].mean(0), Xo[tr].std(0)
            def aug(idx):
                z = _z(Xo[idx], mo, so)
                return np.hstack([z, frozen.decision_function(z).reshape(-1, 1)])
            m = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            m.fit(aug(tr), yo[tr])
            oof[te] = m.predict_proba(aug(te))[:, 1]
        aps.append(average_precision_score(yo, oof))
    return float(np.mean(aps)), float(np.std(aps))


def build_cnn(n_in: int, seed: int = 0):
    """Small 1-D conv net over the peak-normalised clip.

    Everything up to ``embed`` is the feature extractor that gets frozen; the
    single dense unit after it is the head that gets refit on our takes.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(n_in, 1))
    x = inp
    for f, k in ((16, 9), (32, 7), (32, 5)):
        x = tf.keras.layers.Conv1D(f, k, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="embed")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def pretrain_embed(clips_t: np.ndarray, yt: np.ndarray, clips_o: np.ndarray,
                   seeds: int = 3, epochs: int = 40) -> np.ndarray:
    """Pre-train on Tarkett, freeze, and embed our clips. Averaged over seeds."""
    import tensorflow as tf

    embs = []
    for s in range(seeds):
        m = build_cnn(clips_t.shape[1], seed=s)
        cw = {0: 1.0, 1: float((yt == 0).sum() / max(1, (yt == 1).sum()))}
        m.fit(clips_t[..., None], yt, epochs=epochs, batch_size=64, verbose=0,
              class_weight=cw, shuffle=True)
        enc = tf.keras.Model(m.input, m.get_layer("embed").output)
        embs.append(enc.predict(clips_o[..., None], verbose=0))
    return np.hstack(embs)


def headline(reps: int = 30) -> None:
    """The comparison the decision actually rests on.

    Two controls are run alongside, because a gain from bolting 2433 extra
    rows onto 93 could be nothing but regularisation:

      * shuffled Tarkett labels -- if the gain survives label destruction it
        was never knowledge transfer;
      * a C sweep on our takes alone -- if shrinking the coefficients gets to
        the same place for free, Tarkett is not earning its keep.
    """
    rng = np.random.default_rng(0)
    Xo, _Co, yo, _lab = load_ours()
    Xt, _Ct, yt = load_tarkett("native")

    print(f"ours: {len(yo)} takes, {int(yo.sum())} positive ({yo.mean():.3f} chance)")
    print(f"tarkett: {len(yt)} clips, {int(yt.sum())} falls\n")

    print(f"{'config':<38}{'AP':>16}   paired vs same-C baseline")
    rows = {}
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        b = np.array([_ap_seed(Xo, yo, r, C=C) for r in range(reps)])
        t = np.array([_ap_seed_coral(Xo, yo, Xt, yt, r, C=C, w=0.1) for r in range(reps)])
        rows[C] = (b, t)
        d = t - b
        print(f"  C={C:<7} ours only              {b.mean():.3f} +- {b.std():.3f}")
        print(f"  C={C:<7} + Tarkett (w=0.1)      {t.mean():.3f} +- {t.std():.3f}"
              f"   {d.mean():+.3f}  won {(d > 0).mean():.0%} of {reps} seeds")

    bC = max(rows, key=lambda c: rows[c][0].mean())
    tC = max(rows, key=lambda c: rows[c][1].mean())
    b, t = rows[bC][0], rows[tC][1]
    print(f"\n  best tuned baseline   C={bC:<6}      {b.mean():.3f} +- {b.std():.3f}")
    print(f"  best with Tarkett     C={tC:<6}      {t.mean():.3f} +- {t.std():.3f}"
          f"   {(t - b).mean():+.3f}  won {((t - b) > 0).mean():.0%}")

    sh = np.mean([[_ap_seed_coral(Xo, yo, Xt, rng.permutation(yt), r, C=tC, w=0.1)
                   for r in range(reps)] for _ in range(5)], axis=0)
    print(f"\n  CONTROL: Tarkett with SHUFFLED labels  {sh.mean():.3f} +- {sh.std():.3f}"
          f"   {(sh - rows[tC][0]).mean():+.3f}  (must be <= 0 for the transfer to be real)")


def evaluate() -> None:
    Xo, Co, yo, lab = load_ours()
    print(f"ours: {len(yo)} takes, {int(yo.sum())} positive ({yo.mean():.3f} chance)")

    base, sd = cv_ap(Xo, yo)
    print(f"\nBASELINE   vibration hand features        AP {base:.3f} +- {sd:.3f}")

    for scheme in ("native", "impact"):
        Xt, Ct, yt = load_tarkett(scheme)
        print(f"\n--- Tarkett labelling: {scheme} "
              f"({len(yt)} clips, {int(yt.sum())} positive) ---")

        # sanity: does a Tarkett-trained hand-feature model transfer at all?
        m = pipe().fit(Xt, yt)
        from sklearn.metrics import average_precision_score
        p = m.predict_proba(Xo)[:, 1]
        print(f"  zero-shot Tarkett -> ours               AP "
              f"{average_precision_score(yo, p):.3f}")

        for w in (0.02, 0.1, 1.0):
            ap, s = cv_ap_pooled(Xo, yo, Xt, yt, w=w)
            print(f"  pooled train (Tarkett weight {w:<5})     AP {ap:.3f} +- {s:.3f}"
                  f"   {ap - base:+.3f}")
        for w in (0.02, 0.1, 0.3, 1.0):
            ap, s = cv_ap_coral(Xo, yo, Xt, yt, w=w)
            print(f"  pooled + per-domain z (weight {w:<5})    AP {ap:.3f} +- {s:.3f}"
                  f"   {ap - base:+.3f}")
        ap, s = cv_ap_stack(Xo, yo, Xt, yt)
        print(f"  frozen Tarkett score as 7th feature     AP {ap:.3f} +- {s:.3f}"
              f"   {ap - base:+.3f}")

        emb = pretrain_embed(Ct, yt, Co)
        ap, s = cv_ap(emb, yo)
        print(f"  frozen CNN embedding only               AP {ap:.3f} +- {s:.3f}"
              f"   {ap - base:+.3f}")
        ap, s = cv_ap(np.hstack([Xo, emb]), yo)
        print(f"  hand features + frozen CNN embedding    AP {ap:.3f} +- {s:.3f}"
              f"   {ap - base:+.3f}")


def ablate(reps: int = 30) -> None:
    """Which part of Tarkett earns the gain, and does the gain survive?

    The comparison is paired: every configuration sees the same 5-fold
    partition for a given seed, so the per-seed difference is meaningful and
    the spread of that difference is the thing worth reading.
    """
    z = np.load(CACHE, allow_pickle=False)
    labels, mags = z["labels"], z["mags"]
    db = np.full(mags.shape[1], -120.0)
    Xall = np.vstack([vector(m, db, FS) for m in mags])
    Xo, _Co, yo, lab = load_ours()

    base = np.array([_ap_seed(Xo, yo, r) for r in range(reps)])
    print(f"baseline (ours only)                     AP {base.mean():.3f} +- {base.std():.3f}")

    pos_all = set(FALL_LABELS) | IMPACT_EXTRA
    subsets = {
        "acted falls vs controlled non-falls": (
            {"fall_acted"}, {"jump", "run", "walk", "objectfalling", "objectinteraction",
                             "walk_walkerfoot", "walk_walkerwheels", "walk_walkingstick",
                             "walk_wheelchair"}),
        "acted falls vs mined daily activity": ({"fall_acted"}, {"daily"}),
        "REAL falls only vs mined daily": ({"fall_real"}, {"daily"}),
        "all falls vs all non-falls (native)": (FALL_LABELS, None),
        "impact labelling (falls+jump+run)": (pos_all, None),
    }
    for title, (pos, neg) in subsets.items():
        keep = np.array([l in pos or (neg is None and l not in pos) or
                         (neg is not None and l in neg) for l in labels])
        Xt = Xall[keep]
        yt = np.array([1 if l in pos else 0 for l in labels[keep]], dtype=int)
        got = np.array([_ap_seed_coral(Xo, yo, Xt, yt, r) for r in range(reps)])
        d = got - base
        win = (d > 0).mean()
        print(f"{title:<40} AP {got.mean():.3f} +- {got.std():.3f}"
              f"   {d.mean():+.3f} (won {win:.0%} of {reps} seeds,"
              f" n_t={len(yt)} pos={int(yt.sum())})")


def _ap_seed(Xo, yo, r, C=1.0):
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    p = cross_val_predict(pipe(C), Xo, yo, cv=cv, method="predict_proba")[:, 1]
    return average_precision_score(yo, p)


def _ap_seed_coral(Xo, yo, Xt, yt, r, C=1.0, w=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold
    mt, st = Xt.mean(0), Xt.std(0)
    Zt = _z(Xt, mt, st)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    oof = np.zeros(len(yo))
    for tr, te in cv.split(Xo, yo):
        mo, so = Xo[tr].mean(0), Xo[tr].std(0)
        X = np.vstack([_z(Xo[tr], mo, so), Zt])
        y = np.concatenate([yo[tr], yt])
        sw = np.concatenate([np.ones(len(tr)), np.full(len(yt), w)])
        m = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        m.fit(X, y, sample_weight=sw)
        oof[te] = m.predict_proba(_z(Xo[te], mo, so))[:, 1]
    return average_precision_score(yo, oof)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", type=Path, required=True, help="path to the FallData folder")
    c.add_argument("--neg-per-hour", type=int, default=10)
    sub.add_parser("eval")
    a2 = sub.add_parser("ablate")
    a2.add_argument("--reps", type=int, default=30)
    a3 = sub.add_parser("headline")
    a3.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.src, a.neg_per_hour)
    elif a.cmd == "ablate":
        ablate(a.reps)
    elif a.cmd == "headline":
        headline(a.reps)
    else:
        evaluate()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen Oxford SSL accelerometer ResNet as a vibration feature extractor.

Model: Yuan et al., "Self-supervised learning for human activity recognition
using 700,000 person-days of wearable data", npj Digital Medicine 2024.
ResNet-18-style 1-D CNN pretrained by multi-task SSL on UK Biobank wrist
accelerometry. Weights ship inside the repo, so we vendor them:

    https://github.com/OxWearables/ssl-wearables
    ssl-wearables/model_check_point/mtl_best.mdl    (harnet10, 3 x 300 @ 30 Hz)
    ssl-wearables/model_check_point/mtl_5_best.mdl  (harnet5,  3 x 150 @ 30 Hz)

copied to models/ssl_wearables/ together with sslearning/models/accNet.py so
nothing here needs torch.hub or a network at run time.

Domain gap, stated up front because it drives every result below: the UK
Biobank data is wrist-worn, gravity-included, +-8 g, and the classes are
postures and locomotion held for ten seconds. Ours is a phone flat on a floor
measuring the slab ring after an impact -- gravity already removed, peak
amplitude 0.005-0.03 g, and the event lasts under 200 ms. So we sweep the two
choices that matter for that gap:

  window : how a 1-5 s clip is stretched to the 300-sample input
           `pad`  zero-pad around the peak (silence either side)
           `tile` repeat the clip until it fills the window
  scale  : `raw`  leave it in g, three orders of magnitude below train range
           `norm` peak-normalise to +-1 g, which puts it where the frozen
                  BatchNorm running statistics expect the signal to be

Result, 93 takes / 28 positives, 5-fold x 10 repeats, out-of-fold AP:

    hand features (6 dims)                  0.430 +- 0.017
    frozen harnet10, tile + raw (1024)      0.643 +- 0.045    <- ship this
    same architecture, weights NOT loaded   0.394 +- 0.034
    raw amplitude only (2 dims)             0.507 +- 0.036
    clip duration only (1 dim)              0.283 +- 0.007
    embedding + hand features (1030)        0.617 +- 0.046

The three controls are the point. The random-init row uses the identical
architecture and the identical input, so the 0.643 - 0.394 gap is what the
pretrained weights bought and not what the convolutions bought. The amplitude
row matters because `raw` beats `norm` by 0.17, which says the net is partly
reading how hard the floor was hit -- something the hand features throw away
on purpose -- but amplitude alone only reaches 0.507, so it is not the whole
story. The duration row is below the 0.301 chance rate, so tiling is not
smuggling the hand-trimmed clip length in as a repetition period.

Concatenating the hand features costs 0.026. Six correlated dims cannot help a
1024-dim space with 93 samples, so the embedding ships alone.

    python -m train.ssl_wearables
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))
from shared.paths import MODELS_DIR, REPO_ROOT as ROOT  # noqa: E402

from shared.features import vector  # noqa: E402
from train.labels import y_of  # noqa: E402
from train.load import iter_takes  # noqa: E402

WEIGHTS = MODELS_DIR / "ssl_wearables"
TARGET_HZ = 30.0
REPS = 10


# ---------------------------------------------------------------- model


def _resnet_class():
    """Import the vendored accNet.py without needing the sslearning package."""
    spec = importlib.util.spec_from_file_location("accNet", WEIGHTS / "accNet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Resnet


def load_harnet(epoch_len: int = 10, pretrained: bool = True):
    """Frozen feature extractor. Returns (module, n_samples, n_loaded, n_total).

    The checkpoint was written by a DataParallel job, so every key carries a
    `module.` prefix that has to come off. The four keys we deliberately drop
    are the EvaClassifier head -- it predicts the SSL pretext tasks, not
    anything we want, and a logistic head replaces it.
    """
    ckpt = {10: "mtl_best.mdl", 5: "mtl_5_best.mdl"}[epoch_len]
    model = _resnet_class()(output_size=2, is_eva=True, epoch_len=epoch_len, resnet_version=1)
    if not pretrained:
        # Control: identical architecture, weights left at their init. Any gap
        # between this and the loaded model is what the 700,000 person-days
        # actually bought us; no gap means we are reading a random projection.
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model.feature_extractor, epoch_len * int(TARGET_HZ), 0, len(model.state_dict())
    raw = torch.load(WEIGHTS / ckpt, map_location="cpu", weights_only=False)
    stripped = {".".join(k.split(".")[1:]): v for k, v in raw.items()}
    have = model.state_dict()
    keep = {k: v for k, v in stripped.items() if k in have and k.split(".")[0] != "classifier"}
    have.update(keep)
    model.load_state_dict(have)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.feature_extractor, epoch_len * int(TARGET_HZ), len(keep), len(model.state_dict())


# ---------------------------------------------------------------- windows


def resample_30hz(t_ms: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """(3, n) at the phone's ~52.6 Hz -> (3, m) on a uniform 30 Hz grid.

    Interpolates against the real timestamps rather than assuming a nominal
    rate, so sample jitter and the 52.6-vs-50 discrepancy both wash out.
    """
    if t_ms.size < 2:
        return np.repeat(axes[:, :1], 2, axis=1)
    t = (t_ms - t_ms[0]) / 1000.0
    span = float(t[-1])
    m = max(4, int(round(span * TARGET_HZ)))
    grid = np.linspace(0.0, span, m)
    return np.vstack([np.interp(grid, t, a) for a in axes])


def fit_window(x3: np.ndarray, n: int, mode: str) -> np.ndarray:
    """(3, m) -> (3, n), centred on the impact.

    `pad`  keeps one copy of the event and surrounds it with zeros, which is
           honest about there being no signal there but hands the net a step
           discontinuity and a mostly-dead input.
    `tile` repeats the clip to fill the window, so the net sees a periodic
           train of impacts at roughly the clip rate. Nothing in the SSL
           pretraining saw that either, but the activations stay alive.
    """
    m = x3.shape[1]
    if m >= n:
        pk = int(np.argmax(np.linalg.norm(x3, axis=0)))
        a = int(np.clip(pk - n // 2, 0, m - n))
        return x3[:, a : a + n]
    if mode == "tile":
        reps = int(np.ceil(n / m)) + 1
        tiled = np.tile(x3, reps)
        pk = int(np.argmax(np.linalg.norm(tiled[:, : tiled.shape[1] - n], axis=0)))
        a = int(np.clip(pk - n // 2, 0, tiled.shape[1] - n))
        return tiled[:, a : a + n]
    pad = n - m
    left = pad // 2
    return np.pad(x3, ((0, 0), (left, pad - left)))


def clips(mode: str, scale: str, n: int):
    """Every take as a (3, n) window at 30 Hz, plus labels and hand features."""
    X, y, lab, hand = [], [], [], []
    for label, _path, a in iter_takes():
        yi = y_of(label)
        if yi is None or a["t"].size < 16:
            continue
        axes = np.vstack([a["ax"], a["ay"], a["az"]])
        w = fit_window(resample_30hz(a["t"], axes), n, mode)
        if scale == "norm":
            pk = float(np.max(np.linalg.norm(w, axis=0)))
            if pk > 1e-12:
                w = w / pk
        X.append(w)
        y.append(yi)
        lab.append(label)
        hand.append(vector(a["mag"], a["db"], float(a["fs"])))
    return np.stack(X), np.asarray(y), np.array(lab), np.vstack(hand)


def duration() -> np.ndarray:
    """Control: clip length only, 1 dim.

    `tile` writes the clip length into the window as a repetition period, and
    a 1-D CNN reads periodicity for free. Our takes are hand-trimmed, so clip
    length is an artefact of recording, not of physics -- if length alone
    scores near the embedding, the embedding is reading the trim.
    """
    out = []
    for _label, _path, a in iter_takes():
        if y_of(_label) is None or a["t"].size < 16:
            continue
        out.append([a["t"].size / float(a["fs"])])
    return np.asarray(out, dtype=np.float64)


def amplitude(mode: str = "pad") -> np.ndarray:
    """Control: absolute loudness only, 2 dims.

    shared.features.vector() peak-normalises before it measures anything, so
    the 6 hand features are deliberately blind to how hard the floor was hit.
    The SSL embedding at `raw` scale is not. This isolates that one difference
    so the comparison is not silently a comparison against a handicap.
    """
    out = []
    for _label, _path, a in iter_takes():
        if y_of(_label) is None or a["t"].size < 16:
            continue
        m = np.abs(np.asarray(a["mag"], dtype=np.float64))
        out.append([np.log10(float(m.max()) + 1e-9),
                    np.log10(float(np.sqrt((m**2).mean())) + 1e-9)])
    return np.asarray(out, dtype=np.float64)


def embed(fe, X: np.ndarray, batch: int = 16) -> np.ndarray:
    """Frozen forward pass -> (N, 1024). The last layer already pools to
    length 1, so the flatten is the embedding, no extra pooling needed."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            t = torch.from_numpy(X[i : i + batch]).float()
            out.append(fe(t).flatten(1).numpy())
    return np.vstack(out)


# ---------------------------------------------------------------- eval


def evaluate(X, y, C=1.0, reps=REPS):
    """Same protocol as train.experiment: each repeat is its own 5-fold
    partition, AP over pooled out-of-fold probabilities, mean +- sd."""
    aps = []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        pipe = Pipeline(
            [
                ("s", StandardScaler()),
                ("c", LogisticRegression(max_iter=2000, C=C, class_weight="balanced")),
            ]
        )
        p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        aps.append(average_precision_score(y, p))
    return float(np.mean(aps)), float(np.std(aps))


def report(title, X, y, Cs=(0.001, 0.01, 0.1, 1.0)):
    """Sweep C and print the best. With 93 x 1024 the regularisation is not a
    detail -- it is most of the model, so the sweep is reported, not hidden."""
    best = None
    row = []
    for C in Cs:
        ap, sd = evaluate(X, y, C=C)
        row.append(f"C={C:<6g} {ap:.3f}+-{sd:.3f}")
        if best is None or ap > best[0]:
            best = (ap, sd, C)
    print(f"  {title:<38} best AP {best[0]:.3f}+-{best[1]:.3f} @C={best[2]:g}")
    print(f"    {'  '.join(row)}")
    return best


def main():
    print("Oxford ssl-wearables (harnet10) frozen features -> logistic head")
    print(f"weights: {WEIGHTS}\n")

    fe, n, loaded, total = load_harnet(10)
    print(f"harnet10 loaded {loaded}/{total} tensors "
          f"({total - loaded} classifier-head tensors dropped), window {n} @ {TARGET_HZ:g} Hz")

    _X0, y, lab, hand = clips("pad", "raw", n)
    print(f"{len(y)} takes | pos={int(y.sum())} neg={int((1 - y).sum())} chance={y.mean():.3f}")
    print("labels: " + ", ".join(f"{l}={int((lab == l).sum())}" for l in sorted(set(lab))))

    amp = amplitude()
    print("\nbaselines and controls, same protocol:")
    report("hand features (6 dims)", hand, y, Cs=(0.1, 1.0, 10.0))
    report("raw amplitude only (2 dims)", amp, y, Cs=(0.1, 1.0, 10.0))
    report("clip duration only (1 dim)", duration(), y, Cs=(0.1, 1.0, 10.0))
    report("hand + amplitude (8 dims)", np.hstack([hand, amp]), y, Cs=(0.1, 1.0, 10.0))

    print("\nfrozen SSL embedding (1024 dims):")
    results = {}
    for mode in ("pad", "tile"):
        for scale in ("raw", "norm"):
            X, y2, _l, _h = clips(mode, scale, n)
            Z = embed(fe, X)
            assert (y2 == y).all()
            results[(mode, scale)] = (report(f"{mode} + {scale}", Z, y), Z)

    (best_key, ((ap, sd, C), Zb)) = max(results.items(), key=lambda kv: kv[1][0][0])
    print(f"\nbest embedding config: {best_key[0]} + {best_key[1]}  AP {ap:.3f}+-{sd:.3f}")

    print("\nembedding + hand features concatenated:")
    report(f"{best_key[0]}+{best_key[1]} (1024) + hand (6)", np.hstack([Zb, hand]), y)
    report(f"{best_key[0]}+{best_key[1]} (1024) + hand + amp (8)", np.hstack([Zb, hand, amp]), y)

    print("\ncontrol -- same net, weights NOT loaded (random init):")
    torch.manual_seed(0)
    fe_r, n_r, _l, _t = load_harnet(10, pretrained=False)
    Xr, _y, _lb, _h = clips(best_key[0], best_key[1], n_r)
    report(f"random-init {best_key[0]}+{best_key[1]} (1024)", embed(fe_r, Xr), y)

    print("\nharnet5 (5 s / 150 samples, closer to our 1-5 s clips):")
    fe5, n5, loaded5, total5 = load_harnet(5)
    print(f"  loaded {loaded5}/{total5} tensors, window {n5} @ {TARGET_HZ:g} Hz")
    for mode in ("pad", "tile"):
        for scale in ("raw", "norm"):
            X5, _y5, _l, _h = clips(mode, scale, n5)
            report(f"harnet5 {mode} + {scale} (512)", embed(fe5, X5), y)


if __name__ == "__main__":
    main()

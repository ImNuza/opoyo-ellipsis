#!/usr/bin/env python3
"""Frozen LIMU-BERT as a vibration encoder, evaluated out-of-fold.

SOURCE
    github.com/dapowan/LIMU-BERT-Public (Xu et al., SenSys 2021). The four
    pretrained checkpoints ship inside that repository under ``saved/``; they
    are copied here to ``models/limu_bert/pretrain_base_<ds>_20_120/<ds>.pt``.
    Architecture is ``base_v1`` from the repo's ``config/limu_bert.json``:
    6 input channels, hidden 72, ff 144, 4 layers (weight-shared), 4 heads,
    seq_len 120, sampled at 20 Hz. The encoder module below is a transcription
    of ``models.py`` in that repo so the state dicts load key-for-key.

WHAT HAD TO BE BENT TO FIT OUR DATA -- read before trusting a number
    * We have 3-axis accelerometer only. LIMU-BERT base_v1 wants 6 channels
      (accel + gyro). Channels 3:6 are ZERO-FILLED. The gyro half of the input
      projection therefore contributes only its bias.
    * We sample at ~52.6 Hz, the model at 20 Hz. Two resamplings are offered:
        - ``pad``     resample the 2 s peak window to true 20 Hz (40 samples)
                      and zero-pad to 120. Faithful in frequency, but 2/3 of
                      the sequence is padding the model never saw in training.
        - ``stretch`` resample the same 2 s window to all 120 positions. Fills
                      the sequence, but time-stretches 3x, so our 0-26 Hz band
                      is presented to the model as 0-8.8 Hz.
    * Our accelerometer is gravity-removed and the impacts are tiny (peak
      |a| ~ 0.006-0.019 in file units, vs the ~1 g excursions LIMU-BERT saw
      after its /9.8 normalisation). Feeding the raw scale drives the input
      projection to ~0 and every window embeds to the same point, so the
      default here is PEAK-NORMALISED per window. ``--scale raw`` reproduces
      the /9.8 path for comparison.

EVALUATION
    StratifiedKFold(5, shuffle=True) over random_state 0..R-1, out-of-fold
    probabilities only, Pipeline(StandardScaler, LogisticRegression(
    class_weight="balanced")), average_precision_score. Baseline to beat is
    the 6 hand features from shared/features.py.

    python -m train.limu_bert            # primary config
    python -m train.limu_bert --grid     # full sweep, 64 cells
    python -m train.limu_bert --controls # pretrained vs random-init vs raw

RESULT -- NO SHIP
    The checkpoints load key-for-key and the embeddings are well conditioned,
    but the pretraining contributes nothing here. With the design frozen to
    pad / peak / mean and C=0.1, out-of-fold AP over 20 repeats is

        hand features (6)                 0.420 +- 0.024
        pretrained encoder, 4 checkpoints 0.454   (range 0.385 - 0.529)
        random-init encoder, 5 seeds      0.441   (range 0.386 - 0.493)
        pretrained + hand, 4 checkpoints  0.467   (range 0.396 - 0.560)
        random-init + hand, 6 seeds       0.440   (range 0.394 - 0.481)

    A frozen encoder with random weights scores the same as the pretrained
    one, so the apparent lift over the 0.397 hand-feature baseline is what any
    random 72-dim nonlinear projection of the waveform buys, plus the effect
    of re-tuning C (the same hand features reach 0.430 at C=0.1). The single
    best cell in the grid, shoaib/pad/peak/mean + hand at 0.567, is the
    argmax of ~130 evaluations on 28 positives; its random-weight counterpart
    reaches 0.481, so most of that gap is selection.

    The likely cause is domain: LIMU-BERT was pretrained on 20 Hz body-worn
    accelerometer + gyro during locomotion, at ~1 g. We feed it a floor-
    mounted phone recording millig structural vibration whose discriminative
    energy sits above the 10 Hz Nyquist of a 20 Hz model. Resampling to 20 Hz
    destroys the impact signature before the encoder ever sees it.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.features import peak_window, vector  # noqa: E402
from train.labels import y_of  # noqa: E402
from train.load import iter_takes  # noqa: E402

WEIGHTS = ROOT / "models" / "limu_bert"
DATASETS = ("hhar", "uci", "motion", "shoaib")

SEQ_LEN = 120
SR = 20.0
FEATURE_NUM = 6
HIDDEN = 72
HIDDEN_FF = 144
N_LAYERS = 4
N_HEADS = 4


# --------------------------------------------------------------------------
# encoder: transcribed from LIMU-BERT-Public/models.py so keys match exactly
# --------------------------------------------------------------------------
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FEATURE_NUM, HIDDEN)
        self.pos_embed = nn.Embedding(SEQ_LEN, HIDDEN)
        self.norm = LayerNorm(HIDDEN)

    def forward(self, x):
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), x.size(1))
        e = self.norm(self.lin(x))  # emb_norm defaults to True in config.py
        return self.norm(e + self.pos_embed(pos))


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(HIDDEN, HIDDEN)
        self.proj_k = nn.Linear(HIDDEN, HIDDEN)
        self.proj_v = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        b, s, _ = x.shape
        w = HIDDEN // N_HEADS
        q, k, v = (p(x).view(b, s, N_HEADS, w).transpose(1, 2)
                   for p in (self.proj_q, self.proj_k, self.proj_v))
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(w), dim=-1)
        return (scores @ v).transpose(1, 2).contiguous().view(b, s, HIDDEN)


class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, HIDDEN_FF)
        self.fc2 = nn.Linear(HIDDEN_FF, HIDDEN)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """Weight-shared blocks -- the repo runs one block n_layers times."""

    def __init__(self):
        super().__init__()
        self.embed = Embeddings()
        self.attn = MultiHeadedSelfAttention()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.norm1 = LayerNorm(HIDDEN)
        self.pwff = PositionWiseFeedForward()
        self.norm2 = LayerNorm(HIDDEN)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(N_LAYERS):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LIMUBert(nn.Module):
    """Pretrain wrapper. Only ``transformer.*`` is used; the reconstruction
    head is kept so the checkpoint loads with strict=True."""

    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.norm = LayerNorm(HIDDEN)
        self.decoder = nn.Linear(HIDDEN, FEATURE_NUM)

    def forward(self, x):
        return self.transformer(x)


def random_encoder(seed: int = 0) -> LIMUBert:
    """Same architecture, no pretraining. The control that decides whether the
    LIMU-BERT weights carry anything or the gain is just a random nonlinear
    projection of the waveform."""
    torch.manual_seed(seed)
    model = LIMUBert()
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def load_encoder(dataset: str = "hhar") -> LIMUBert:
    path = WEIGHTS / f"pretrain_base_{dataset}_20_120" / f"{dataset}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Copy it from LIMU-BERT-Public/saved/ "
            "(git clone https://github.com/dapowan/LIMU-BERT-Public)."
        )
    model = LIMUBert()
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    del missing, unexpected
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------
# our windows -> (120, 6)
# --------------------------------------------------------------------------
def _resample(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == n:
        return x
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, x.size), x)


def to_limu_input(a: dict, time: str = "stretch", scale: str = "peak") -> np.ndarray:
    """(120, 6) float32. Channels 3:6 are the zero-filled gyro."""
    mag = a["mag"]
    fs = float(a["fs"])
    axes = np.stack([a["ax"], a["ay"], a["az"]], axis=1)

    # cut the same 2 s peak window the hand features use, on all three axes
    i = int(np.argmax(np.abs(mag))) if mag.size else 0
    pre, post = int(fs * 0.5), int(fs * 1.5)
    lo, hi = max(0, i - pre), min(len(mag), i + post)
    w = axes[lo:hi]
    if w.shape[0] < pre + post:
        w = np.pad(w, ((0, pre + post - w.shape[0]), (0, 0)))
    w = w[: pre + post]

    if time == "stretch":                       # 2 s -> all 120 positions
        r = np.stack([_resample(w[:, c], SEQ_LEN) for c in range(3)], axis=1)
    elif time == "pad":                         # true 20 Hz, then zero-pad
        n20 = max(2, int(round(w.shape[0] / fs * SR)))
        r = np.stack([_resample(w[:, c], n20) for c in range(3)], axis=1)
        r = np.pad(r, ((0, max(0, SEQ_LEN - r.shape[0])), (0, 0)))[:SEQ_LEN]
    else:
        raise ValueError(time)

    if scale == "peak":
        pk = float(np.max(np.abs(r)))
        r = r / pk if pk > 1e-12 else r
    elif scale == "raw":
        r = r / 9.8                             # the repo's Preprocess4Normalization
    else:
        raise ValueError(scale)

    out = np.zeros((SEQ_LEN, FEATURE_NUM), dtype=np.float32)
    out[:, :3] = r.astype(np.float32)
    return out


def pool(h: np.ndarray, how: str) -> np.ndarray:
    """h is (n, 120, 72)."""
    if how == "mean":
        return h.mean(axis=1)
    if how == "meanstd":
        return np.hstack([h.mean(axis=1), h.std(axis=1)])
    if how == "meanmaxstd":
        return np.hstack([h.mean(axis=1), h.max(axis=1), h.std(axis=1)])
    raise ValueError(how)


# --------------------------------------------------------------------------
# data + evaluation
# --------------------------------------------------------------------------
def load_takes():
    takes, y, lab = [], [], []
    for label, _path, a in iter_takes():
        if a["mag"].size < 16:
            continue
        takes.append(a)
        y.append(y_of(label))
        lab.append(label)
    return takes, np.asarray(y), np.array(lab)


def hand_features(takes) -> np.ndarray:
    return np.vstack([vector(a["mag"], a["db"], float(a["fs"])) for a in takes])


def embed_all(takes, model, time, scale) -> np.ndarray:
    X = np.stack([to_limu_input(a, time, scale) for a in takes])
    with torch.no_grad():
        h = model(torch.from_numpy(X)).numpy()
    return h


def evaluate(X, y, C=0.1, reps=10):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, C=C,
                                              class_weight="balanced"))])
    aps = []
    for r in range(reps):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        aps.append(average_precision_score(y, p))
    return float(np.mean(aps)), float(np.std(aps))


def line(name, X, y, C, reps=10):
    ap, sd = evaluate(X, y, C=C, reps=reps)
    print(f"  {name:<44} d={X.shape[1]:<4} C={C:<5} AP {ap:.3f} +- {sd:.3f}")
    return ap


def controls(takes, y, Xh, reps):
    """Is the pretraining doing the work, or is any 120x72 projection enough?

    Design is frozen to pad / peak / mean -- chosen on principle (pad keeps the
    true frequency axis, peak-norm puts our millig impacts in the range the
    model was trained on), not by reading the grid."""
    T, S, P = "pad", "peak", "mean"
    Xin = np.stack([to_limu_input(a, T, S) for a in takes])[:, :, :3]
    flat = Xin.reshape(len(takes), -1)

    print("\ncontrols, design frozen to pad/peak/mean")
    for C in (0.01, 0.1):
        line("raw 20 Hz waveform, flattened (120x3)", flat, y, C, reps)
    binned = np.hstack([
        Xin.reshape(len(takes), 20, 6, 3).mean(2).reshape(len(takes), -1),
        Xin.reshape(len(takes), 20, 6, 3).std(2).reshape(len(takes), -1)])
    for C in (0.01, 0.1):
        line("raw waveform, 20 time-bins mean+std", binned, y, C, reps)

    print("\n  -- random-init encoder (no pretraining), 5 seeds --")
    rnd = []
    for seed in range(5):
        Xe = pool(embed_all(takes, random_encoder(seed), T, S), P)
        for C in (0.01, 0.1):
            a = line(f"random-init seed {seed}", Xe, y, C, reps)
            rnd.append((C, a))

    print("\n  -- pretrained checkpoints, same design --")
    pre = []
    for ds in DATASETS:
        Xe = pool(embed_all(takes, load_encoder(ds), T, S), P)
        for C in (0.01, 0.1):
            a = line(f"pretrained {ds}", Xe, y, C, reps)
            pre.append((C, a))
        line(f"pretrained {ds} + hand", np.hstack([Xe, Xh]), y, 0.1, reps)

    for C in (0.01, 0.1):
        r = [a for c, a in rnd if c == C]
        q = [a for c, a in pre if c == C]
        print(f"\n  C={C}: random-init {np.mean(r):.3f} +- {np.std(r):.3f} (n=5)"
              f"   pretrained {np.mean(q):.3f} +- {np.std(q):.3f} (n=4)")


def main():
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--grid", action="store_true", help="full sweep")
    ap_arg.add_argument("--controls", action="store_true",
                        help="pretrained vs random-init vs raw waveform")
    ap_arg.add_argument("--reps", type=int, default=10)
    args = ap_arg.parse_args()

    takes, y, lab = load_takes()
    Xh = hand_features(takes)
    print(f"{len(y)} takes | pos={int(y.sum())} neg={int((1 - y).sum())} "
          f"chance={y.mean():.3f}")
    print("labels: " + ", ".join(f"{l}={int((lab == l).sum())}"
                                for l in sorted(set(lab))))

    print("\nbaseline")
    for C in (0.1, 1.0):
        line("hand features (6)", Xh, y, C, args.reps)

    if args.controls:
        controls(takes, y, Xh, args.reps)
        return

    combos = ([(d, t, s, p)
               for d in DATASETS for t in ("stretch", "pad")
               for s in ("peak", "raw") for p in ("mean", "meanstd")]
              if args.grid else
              [(d, "pad", "peak", p) for d in DATASETS for p in ("mean", "meanstd")])

    print("\nfrozen LIMU-BERT embedding")
    cache = {}
    best = (-1.0, None)
    for ds, t, s, p in combos:
        key = (ds, t, s)
        if key not in cache:
            cache[key] = embed_all(takes, load_encoder(ds), t, s)
        Xe = pool(cache[key], p)
        if np.allclose(Xe.std(axis=0).max(), 0.0):
            print(f"  {ds}/{t}/{s}/{p}: embedding is constant, skipped")
            continue
        for C in (0.01, 0.1):
            a = line(f"{ds}/{t}/{s}/{p}", Xe, y, C, args.reps)
            if a > best[0]:
                best = (a, (ds, t, s, p, C))

    print("\nembedding + hand features")
    for ds, t, s, p in combos:
        Xe = pool(cache[(ds, t, s)], p)
        if np.allclose(Xe.std(axis=0).max(), 0.0):
            continue
        Xc = np.hstack([Xe, Xh])
        for C in (0.01, 0.1):
            line(f"{ds}/{t}/{s}/{p} + hand", Xc, y, C, args.reps)

    print(f"\nbest embedding-only cell: AP {best[0]:.3f} at {best[1]}")
    print("NOTE: that maximum is taken over the whole grid above, so it is an "
          "optimistic estimate. Judge the primary cell, not the argmax.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PANNs CNN14 (AudioSet) frozen embeddings vs the frozen-YAMNet baseline.

CNN14 gives a 2048-D embedding, YAMNet 1024-D. Both are frozen feature
extractors in front of the same logistic head on the same folds, so the only
thing that changes between two rows is the backbone.

Two things matter for reading the numbers:

  * Our wavs are recorded at **16 kHz**. The headline CNN14 is a 32 kHz model
    with a mel bank running to 14 kHz, so upsampling leaves the top half of its
    input empty. `Cnn14_16k` is the same architecture retrained at 16 kHz
    (fmax 8 kHz) and is the fair one to quote here; both are reported.
  * `train/fit.py` embeds the **whole** wav, `train/eval_yamnet.py` a 1.6 s
    window around the peak. That choice moves AP more than the backbone does,
    so every backbone is run under both.

Weights (github.com/qiuqiangkong/audioset_tagging_cnn, hosted on Zenodo record
3987831), expected in models/panns/:
  Cnn14_mAP=0.431.pth       327 MB   32 kHz
  Cnn14_16k_mAP=0.438.pth   327 MB   16 kHz

The architecture is reimplemented here rather than pulled from
`panns-inference` so the repo keeps one new dependency (torch) instead of the
librosa/matplotlib stack. The mel filterbank is read out of the checkpoint
(`logmel_extractor.melW`), which is what makes that safe: the front-end cannot
drift from the one the weights were trained with.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))
from shared.paths import MODELS_DIR, REPO_ROOT as ROOT  # noqa: E402

from shared.features import vector  # noqa: E402
from train.labels import y_of  # noqa: E402
from train.load import load_csv  # noqa: E402

PANNS = MODELS_DIR / "panns"
# name -> (checkpoint, sample rate, n_fft, hop)
VARIANTS = {
    "CNN14 32k": (PANNS / "Cnn14_mAP=0.431.pth", 32000, 1024, 320),
    "CNN14 16k": (PANNS / "Cnn14_16k_mAP=0.438.pth", 16000, 512, 160),
}
SEEDS = tuple(range(10))
C_REG = 0.05


# --------------------------------------------------------------------------
# CNN14
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """Trunk only: everything up to and including fc1, which is the embedding."""

    def __init__(self, mel_bins: int = 64, classes_num: int = 527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        chans = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
        for i, (a, b) in enumerate(chans, start=1):
            setattr(self, f"conv_block{i}", ConvBlock(a, b))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        """logmel: (B, 1, T, mel) -> (B, 2048)."""
        x = self.bn0(logmel.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, pool_size=(2, 2) if i < 6 else (1, 1))
        x = torch.mean(x, dim=3)  # collapse frequency
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return F.relu_(self.fc1(x1 + x2))


class Frontend:
    """Checkpoint-supplied log-mel, matching torchlibrosa's defaults exactly."""

    def __init__(self, mel_w: torch.Tensor, n_fft: int, hop: int):
        self.mel_w, self.n_fft, self.hop = mel_w, n_fft, hop
        self.window = torch.hann_window(n_fft, periodic=True)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
            window=self.window, center=True, pad_mode="reflect", return_complex=True,
        )
        power = (spec.real**2 + spec.imag**2).transpose(1, 2)  # (B, T, freq)
        mel = torch.matmul(power, self.mel_w)
        return (10.0 * torch.log10(torch.clamp(mel, min=1e-10))).unsqueeze(1)


def load_cnn14(path: Path, n_fft: int, hop: int):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    mel_w = sd["logmel_extractor.melW"].float()  # (n_fft//2+1, mel_bins)
    trunk = {
        k: v for k, v in sd.items()
        if not k.startswith(("spectrogram_extractor.", "logmel_extractor."))
    }
    model = Cnn14(mel_bins=mel_w.shape[1])
    missing, unexpected = model.load_state_dict(trunk, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model, Frontend(mel_w, n_fft, hop)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def read_wav(p: Path):
    with wave.open(str(p), "rb") as wf:
        rate, ch = wf.getframerate(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    return x, float(rate)


def around_peak(x, fs, pre=0.4, post=1.2):
    """The 1.6 s window train/eval_yamnet.py uses."""
    i = int(np.argmax(np.abs(x))) if x.size else 0
    a = max(0, i - int(pre * fs))
    b = min(x.size, i + int(post * fs))
    w = x[a:b]
    need = int((pre + post) * fs)
    if w.size < need:
        w = np.pad(w, (0, need - w.size))
    return w[:need].astype(np.float32)


def norm(x: np.ndarray) -> np.ndarray:
    pk = float(np.max(np.abs(x))) or 1.0
    return np.clip(x / pk, -1.0, 1.0).astype(np.float32)


def to_rate(x: np.ndarray, fs: float, target: int) -> np.ndarray:
    if abs(fs - target) < 1:
        return x.astype(np.float32)
    g = np.gcd(int(round(fs)), target)
    return resample_poly(x, target // g, int(round(fs)) // g).astype(np.float32)


def load_takes(root: Path):
    """One row per CSV with a matching WAV. Returns raw clips at native rate."""
    full, peak, rates, vib, y, labs = [], [], [], [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        yi = y_of(folder.name)
        if yi is None:
            continue
        for csv_p in sorted(folder.glob("*.csv")):
            wav_p = csv_p.with_suffix(".wav")
            if not wav_p.exists():
                continue
            pcm, rate = read_wav(wav_p)
            full.append(norm(pcm))
            peak.append(norm(around_peak(pcm, rate)))
            rates.append(rate)
            d = load_csv(csv_p)
            vib.append(vector(d["mag"], d["db"], float(d["fs"])))
            y.append(yi)
            labs.append(folder.name)
    return full, peak, rates, np.vstack(vib), np.asarray(y), np.array(labs)


def panns_embed(clips, rates, model, front: Frontend, sr: int) -> np.ndarray:
    """One clip at a time: clips differ in length and zero-padding would leak
    into the avg-pool half of the CNN14 head."""
    out = []
    with torch.no_grad():
        for x, fs in zip(clips, rates):
            w = to_rate(x, fs, sr)
            if w.size < sr:
                w = np.pad(w, (0, sr - w.size))
            wt = torch.from_numpy(w).float().unsqueeze(0)
            out.append(model(front(wt)).numpy()[0])
    return np.vstack(out)


def yamnet_embed(clips, rates) -> np.ndarray | None:
    """Recompute the baseline here so both backbones share window and folds."""
    import os

    cache = MODELS_DIR / "tfhub"
    if cache.is_dir():
        os.environ.setdefault("TFHUB_CACHE_DIR", str(cache))
    try:
        import tensorflow_hub as hub
    except Exception as exc:  # pragma: no cover - optional baseline
        print("  (YAMNet unavailable:", exc, ")")
        return None
    m = hub.load("https://tfhub.dev/google/yamnet/1")
    out = []
    for x, fs in zip(clips, rates):
        w = to_rate(x, fs, 16000)
        if w.size < 16000:
            w = np.pad(w, (0, 16000 - w.size))
        _s, emb, _sp = m(w.astype(np.float32))
        out.append(np.mean(emb.numpy(), axis=0))
    return np.vstack(out)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def oof_ap(X, y, seeds=SEEDS, c=C_REG):
    aps = []
    for r in seeds:
        p = np.zeros(len(y), dtype=float)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            clf = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("c", LogisticRegression(max_iter=5000, class_weight="balanced", C=c)),
                ]
            )
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y, p))
    return float(np.mean(aps)), float(np.std(aps))


def row(name, X, y, c=C_REG, seeds=SEEDS):
    m, s = oof_ap(X, y, seeds=seeds, c=c)
    print(f"  {name:<32s} d={X.shape[1]:<5d} AP {m:.3f} +- {s:.3f}")
    return m


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes"
    full, peak, rates, vib, y, labs = load_takes(root)
    print(f"takes {len(y)}  positives {int(y.sum())}  wav rate {sorted(set(rates))}")

    Z: dict[str, np.ndarray] = {}
    for name, (ckpt, sr, n_fft, hop) in VARIANTS.items():
        if not ckpt.exists():
            print(f"skip {name}: missing {ckpt}")
            continue
        model, front = load_cnn14(ckpt, n_fft, hop)
        for win, clips in (("peak1.6s", peak), ("fullwav", full)):
            t0 = time.time()
            Z[f"{name} {win}"] = panns_embed(clips, rates, model, front, sr)
            print(f"  {name} {win}: {Z[f'{name} {win}'].shape} in {time.time() - t0:.0f}s")
        del model

    for win, clips in (("peak1.6s", peak), ("fullwav", full)):
        zy = yamnet_embed(clips, rates)
        if zy is not None:
            Z[f"YAMNet {win}"] = zy

    order = sorted(Z, key=lambda k: ("YAMNet" in k, k))
    print(f"\nout-of-fold AP, StratifiedKFold(5) x {len(SEEDS)} seeds, "
          f"StandardScaler + LogisticRegression(balanced, C={C_REG})\n")
    print("audio only:")
    for k in order:
        row(k, Z[k], y)
    print("audio + 6 vibration features:")
    for k in order:
        row(k + " + vib", np.hstack([Z[k], vib]), y)
    row("vibration only", vib, y)

    print("\nfit.py's own setting (whole wav, C=1.0, single split random_state=0):")
    for k in order:
        if k.endswith("fullwav"):
            row(k, Z[k], y, c=1.0, seeds=(0,))

    # Sweep C for every backbone, not just PANNs: a 2048-D embedding needs more
    # shrinkage than a 1024-D one, so tuning only one side would manufacture a win.
    print("\nregularisation sweep, whole wav, audio only:")
    fulls = [k for k in order if k.endswith("fullwav")]
    print("      " + "".join(f"{k:<22s}" for k in fulls))
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        cells = []
        for k in fulls:
            m, s = oof_ap(Z[k], y, c=c)
            cells.append(f"{m:.3f}+-{s:.3f}".ljust(22))
        print(f"  C={c:<5g}" + "".join(cells))

    # Paired per-seed comparison at each backbone's own best C: 10 shared
    # partitions, so "PANNs wins" means it won on the same data YAMNet saw.
    def best_c(k):
        return max((0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), key=lambda c: oof_ap(Z[k], y, c=c)[0])

    pa, ya = "CNN14 32k fullwav", "YAMNet fullwav"
    if pa in Z and ya in Z:
        cp, cy = best_c(pa), best_c(ya)
        print(f"\npaired per-seed, PANNs C={cp} vs YAMNet C={cy}:")
        wins = 0
        for r in range(10):
            ap_p = oof_ap(Z[pa], y, seeds=(r,), c=cp)[0]
            ap_y = oof_ap(Z[ya], y, seeds=(r,), c=cy)[0]
            wins += ap_p > ap_y
            print(f"  seed {r}: PANNs {ap_p:.3f}   YAMNet {ap_y:.3f}   {'PANNs' if ap_p > ap_y else 'YAMNet'}")
        print(f"  PANNs wins {wins}/10 shared partitions")


if __name__ == "__main__":
    main()

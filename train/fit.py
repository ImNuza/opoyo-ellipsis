#!/usr/bin/env python3
"""Train heads into models/. Mag and wav are scaled separately, then fused.

python -m train.fit [path-to-takes]
Writes:
  models/mag_head.joblib      vibration (peak-norm clip → scaler → logistic)
  models/yamnet_head.joblib   frozen YAMNet embed → scaler → logistic
  models/fuse_head.joblib     [p_mag, p_yam] → logistic
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.features import FEATURE_NAMES, vector  # noqa: E402
from train.labels import y_of  # noqa: E402
from train.load import iter_takes  # noqa: E402
from train.noise import VENUE, mix_at_snr  # noqa: E402

MODELS = ROOT / "models"


def _pipe_c(C):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced")),
        ]
    )


def _pipe():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )


def _read_wav(path: Path) -> tuple[np.ndarray, float] | None:
    wav = path.with_suffix(".wav")
    if not wav.exists():
        return None
    wf = wave.open(str(wav), "rb")
    rate = wf.getframerate()
    ch = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    return x, float(rate)


def _report(title: str, y, p, names=None, X=None) -> None:
    """Precision / recall / F1 / AP and the confusion matrix. Never accuracy."""
    import numpy as _np

    yhat = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0
    )
    ap = average_precision_score(y, p) if len(set(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    print(f"\n--- {title} ---")
    print(f"  n={len(y)}  pos={int(_np.sum(y))}  neg={len(y)-int(_np.sum(y))}")
    print(f"  precision {pr:.3f}   recall {rc:.3f}   F1 {f1:.3f}   AP {ap:.3f}")
    print(f"  confusion  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    if int(_np.sum(y)) < 20:
        print(f"  !! only {int(_np.sum(y))} positives -- these numbers carry a very wide interval")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "takes"
    Xm, y, names = [], [], []
    wavs: list[tuple[np.ndarray, float] | None] = []
    for label, path, arr in iter_takes(root):
        yi = y_of(label)
        if yi is None:
            continue
        fs = float(arr["fs"])
        Xm.append(vector(arr["mag"], arr["db"], fs))
        y.append(yi)
        names.append(path)
        wavs.append(_read_wav(path))
    if len(set(y)) < 2:
        print("need heeldrop + at least one other folder")
        sys.exit(1)
    Xm = np.vstack(Xm)
    y = np.asarray(y)
    MODELS.mkdir(parents=True, exist_ok=True)

    n_pos = int(y.sum())
    n_splits = max(2, min(5, n_pos))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # out-of-fold probabilities: the only honest numbers, and what the fuse
    # head must be trained on (in-sample p_mag would make it over-trust the
    # vibration head).
    p_mag_oof = cross_val_predict(_pipe(), Xm, y, cv=cv, method="predict_proba")[:, 1]
    _report("vibration head (out-of-fold)", y, p_mag_oof, FEATURE_NAMES, Xm)

    mag = _pipe()
    mag.fit(Xm, y)
    joblib.dump(mag, MODELS / "mag_head.joblib")
    p_mag = p_mag_oof
    print(f"wrote models/mag_head.joblib  n={len(y)} pos={n_pos} feats={len(FEATURE_NAMES)}")

    have = [i for i, w in enumerate(wavs) if w is not None]
    if have:
        from train.yamnet_embed import embed

        print("embedding wavs with frozen YAMNet…")
        Z = np.vstack([embed(wavs[i][0], wavs[i][1]) for i in have])

        # Venue-noise augmentation. A clean-trained audio head loses roughly
        # half its AP when the room is as loud as the impact (babble @0 dB:
        # 0.932 -> 0.489). Training on noisy copies recovers most of that
        # (-> 0.871) and costs nothing on clean audio (0.932 -> 0.939).
        # Augmented rows are TRAINING rows only; the CV report below still
        # scores against clean out-of-fold predictions.
        n_aug = int(os.getenv("OPOYO_NOISE_AUG", "3"))
        Z_aug, y_aug = [Z], [y[have]]
        if n_aug > 0:
            rng = np.random.default_rng(1)
            names = list(VENUE)
            print(f"augmenting audio head with {n_aug} venue-noise passes…")
            for k in range(n_aug):
                fn = VENUE[names[k % len(names)]]
                snr = float(rng.choice([10.0, 5.0, 0.0]))
                Z_aug.append(np.vstack([
                    embed(mix_at_snr(wavs[i][0], fn(wavs[i][0].size, wavs[i][1], rng), snr),
                          wavs[i][1]) for i in have]))
                y_aug.append(y[have])
            print(f"  audio training rows: {len(have)} clean + {n_aug * len(have)} noisy")
        cv_y = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        # out-of-fold, exactly as for p_mag. Using yam.predict_proba(Z) here
        # would be in-sample from a 1024-dim model on <100 rows -- it separates
        # perfectly and the fuse head then reports AP 1.000, which is a leak.
        p_yam_oof = cross_val_predict(_pipe(), Z, y[have], cv=cv_y, method="predict_proba")[:, 1]
        _report("audio head (out-of-fold)", y[have], p_yam_oof)
        yam = _pipe()
        yam.fit(np.vstack(Z_aug), np.concatenate(y_aug))
        joblib.dump(yam, MODELS / "yamnet_head.joblib")
        p_yam = np.full(len(y), np.nan)
        p_yam[have] = p_yam_oof
        print(f"wrote models/yamnet_head.joblib  n_wav={len(have)} dim={Z.shape[1]}")
        both = [i for i in have]
        fuse_x = np.column_stack([p_mag[both], p_yam[both]])
        _report("fusion (out-of-fold p_mag)", y[both],
                cross_val_predict(_pipe(), fuse_x, y[both],
                                  cv=StratifiedKFold(n_splits=max(2, min(5, int(y[both].sum()))),
                                                     shuffle=True, random_state=0),
                                  method="predict_proba")[:, 1])
        fuse = _pipe()
        fuse.fit(fuse_x, y[both])
        joblib.dump(fuse, MODELS / "fuse_head.joblib")

        # Joint head: the 6 hand features concatenated with the 1024-d YAMNet
        # embedding, one logistic on the lot. Beats the two-stage stack in
        # every condition we measured -- clean +0.038, and +0.04 to +0.07
        # under babble / applause / hum / shuffle at 0 dB -- because the stack
        # throws away the embedding and lets a 2-feature logistic see only two
        # noisy out-of-fold probabilities.
        Xj = np.hstack([Xm[have], Z])
        _report("joint head, hand+embedding (out-of-fold)", y[have],
                cross_val_predict(_pipe_c(0.05), Xj, y[have], cv=cv_y,
                                  method="predict_proba")[:, 1])
        joint = _pipe_c(0.05)
        joint.fit(np.vstack([Xj] + [np.hstack([Xm[have], A]) for A in Z_aug[1:]]),
                  np.concatenate(y_aug))
        joblib.dump(joint, MODELS / "joint_head.joblib")
        print("wrote models/joint_head.joblib  (this is what the edge now uses)")
        print("wrote models/fuse_head.joblib  (p_mag, p_yam each already in [0,1], then scaled)")
    else:
        print("no wavs — mag head only")


if __name__ == "__main__":
    main()

"""End-to-end: real csv+wav → SensorWindow → FusionCnn → InferenceResult."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from edge.infer import FusionCnn, MODELS
from shared.paths import DATA_DIR
from shared.schemas import SensorWindow

DATA = DATA_DIR / "takes"


def _csv(path: Path):
    t, mag, db = [], [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            mag.append(float(row["mag"]))
            db.append(float(row.get("db") or -120))
    t = np.asarray(t)
    fs = float(1000 / np.median(np.diff(t))) if t.size > 1 else 50.0
    return mag, db, fs, t


def _wav(path: Path):
    wf = wave.open(str(path), "rb")
    rate = wf.getframerate()
    ch = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    return x.tolist(), float(rate)


def _window(folder: str) -> SensorWindow:
    d = DATA / folder
    csvs = sorted(d.glob("*.csv"))
    assert csvs, f"no csv in {d}"
    csv_p = next(p for p in csvs if p.with_suffix(".wav").exists())
    mag, db, fs, t = _csv(csv_p)
    pcm, pcm_hz = _wav(csv_p.with_suffix(".wav"))
    n = min(len(mag), 100)
    mag, db = mag[:n], db[:n]
    # keep pcm aligned: first n/fs seconds
    n_pcm = int(round(n / fs * pcm_hz))
    pcm = pcm[:n_pcm]
    return SensorWindow(
        node_id="Phone 1",
        room=1,
        t_start_ms=int(t[0]),
        t_end_ms=int(t[n - 1]),
        hz=fs,
        mag=mag,
        ax=[0.0] * n,
        ay=[0.0] * n,
        az=[0.0] * n,
        db=db,
        pcm=pcm,
        pcm_hz=pcm_hz,
    )


@pytest.mark.skipif(not (MODELS / "yamnet_head.joblib").exists(), reason="no trained heads")
def test_heeldrop_scores_higher_than_key():
    clf = FusionCnn(MODELS)
    pos = clf.infer(_window("heeldrop"))
    neg = clf.infer(_window("key"))
    print("heeldrop", pos.is_fall, round(pos.confidence, 3))
    print("key", neg.is_fall, round(neg.confidence, 3))
    assert 0.0 <= pos.confidence <= 1.0
    assert 0.0 <= neg.confidence <= 1.0
    assert pos.confidence > neg.confidence

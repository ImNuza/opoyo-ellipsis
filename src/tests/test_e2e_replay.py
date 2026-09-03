"""Replay a recorded csv+wav through Hub join + FusionCnn.

This is the live path without a phone: same JSON samples + 16 kHz PCM
frames the iOS client would send. Android collect files are the source.
"""

from __future__ import annotations

import asyncio
import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from edge.app import Hub
from edge.gate import EscalationGate, RecordingCloudClient
from edge.infer import FusionCnn, MODELS
from edge.log import InferenceLog
from shared.paths import DATA_DIR
from shared.pcm import pack_frame, unpack_frame

DATA = DATA_DIR / "takes"
PHONE = "11111111-1111-1111-1111-111111111111"
FRAME_N = 320  # 20 ms @ 16 kHz


def _take(folder: str):
    d = DATA / folder
    csv_p = None
    for p in sorted(d.glob("*.csv")):
        if not p.with_suffix(".wav").exists():
            continue
        with p.open(encoding="utf-8") as f:
            n = sum(1 for _ in csv.DictReader(f))
        if n >= 100:
            csv_p = p
            break
    if csv_p is None:
        raise FileNotFoundError(f"no csv+wav with >=100 rows in {d}")
    rows = []
    with csv_p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    wf = wave.open(str(csv_p.with_suffix(".wav")), "rb")
    rate = wf.getframerate()
    ch = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    pcm = np.frombuffer(raw, dtype="<i2")
    if ch == 2:
        pcm = pcm.reshape(-1, 2).mean(1).astype(np.int16)
    return rows, pcm, float(rate)


def _packets(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        mag = float(row["mag"])
        out.append(
            {
                "v": 2,
                "id": PHONE,
                "_nid": PHONE,
                "model": "Android",
                "t": int(float(row["t"])),
                "ax": float(row.get("ax") or 0.0),
                "ay": float(row.get("ay") or 0.0),
                "az": float(row.get("az") or 0.0),
                "mag": mag,
                "db": float(row.get("db") or -120.0),
                "from": "127.0.0.1:9",
            }
        )
    return out


def _pcm_frames(pcm: np.ndarray, t0_ms: int, rate: float) -> list:
    frames = []
    seq = 0
    i = 0
    while i + FRAME_N <= pcm.size:
        t_ms = t0_ms + int(round(i * 1000.0 / rate))
        packed = pack_frame(PHONE, seq=seq, t_ms=t_ms, samples=pcm[i : i + FRAME_N].tobytes())
        frames.append(unpack_frame(packed))
        seq += 1
        i += FRAME_N
    return frames


def _score(tmp_path: Path, folder: str) -> dict:
    rows, pcm, rate = _take(folder)
    packets = _packets(rows)
    frames = _pcm_frames(pcm, int(float(rows[0]["t"])), rate)
    store = InferenceLog(tmp_path / f"{folder}.jsonl")
    gate = EscalationGate(threshold=0.90, client=RecordingCloudClient(), store=store)
    hub = Hub(classifier=FusionCnn(MODELS), gate=gate, store=store)

    async def run() -> None:
        for frame in frames:
            await hub.ingest_pcm(frame.node_id, seq=frame.seq, t_ms=frame.t_ms, pcm=frame.pcm)
        for packet in packets:
            await hub.ingest(packet)

    asyncio.run(run())
    latest = hub.latest_inference
    assert latest is not None
    pcm_meta = hub.last_pcm.get(PHONE) or {}
    return {
        "is_fall": latest["is_fall"],
        "confidence": latest["confidence"],
        "coverage": float(pcm_meta.get("coverage") or 0.0),
        "samples": int(pcm_meta.get("samples") or 0),
    }


@pytest.mark.skipif(not (MODELS / "fuse_head.joblib").exists(), reason="no trained heads")
@pytest.mark.skipif(not (DATA / "heeldrop").exists(), reason="no telegram dataset")
def test_replay_heeldrop_beats_key(tmp_path: Path):
    pos = _score(tmp_path / "pos", "heeldrop")
    neg = _score(tmp_path / "neg", "key")
    print("heeldrop", pos)
    print("key", neg)
    assert pos["coverage"] >= 0.8
    assert neg["coverage"] >= 0.8
    assert pos["confidence"] > neg["confidence"]
    assert pos["is_fall"] is True
    assert pos["confidence"] >= 0.90
    assert neg["confidence"] < 0.90

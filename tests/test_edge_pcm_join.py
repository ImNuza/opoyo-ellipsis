from __future__ import annotations

import asyncio

import numpy as np

from edge.app import Hub
from edge.gate import EscalationGate, RecordingCloudClient
from edge.infer import FakeCnn
from shared.schemas import InferenceResult, SensorWindow
from edge.log import InferenceLog
from phone.fake_phone import make_packet
from shared.pcm import pack_frame, unpack_frame

PHONE_A = "11111111-1111-1111-1111-111111111111"
PHONE_B = "22222222-2222-2222-2222-222222222222"
T0 = 1_000_000
HZ = 50.0
N = 100
DT = 20  # ms
FRAME_N = 320


def _hub(tmp_path) -> Hub:
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(threshold=0.90, client=RecordingCloudClient(), store=store)
    return Hub(classifier=FakeCnn(is_fall=True, confidence=0.95), gate=gate, store=store)


def _json_samples(node: str, n: int = N, t0: int = T0) -> list[dict]:
    out = []
    for i in range(n):
        elapsed = i / HZ
        out.append(make_packet(node, t_ms=t0 + i * DT, elapsed=elapsed, kind="knee"))
    return out


def _pcm_bytes(n: int = FRAME_N, value: int = 1000) -> bytes:
    return np.full(n, value, dtype=np.int16).tobytes()


def _pcm_frames(node: str, t_start: int, t_end: int) -> list[tuple]:
    """20 ms frames covering [t_start, t_end) on the phone clock."""
    frames = []
    seq = 0
    t = t_start
    while t < t_end:
        packed = pack_frame(node, seq=seq, t_ms=t, samples=_pcm_bytes())
        frames.append(unpack_frame(packed))
        seq += 1
        t += DT
    return frames


async def _ingest_json(hub: Hub, packets: list[dict]) -> None:
    for packet in packets:
        await hub.ingest(packet)


async def _ingest_pcm(hub: Hub, frames) -> None:
    for frame in frames:
        await hub.ingest_pcm(frame.node_id, seq=frame.seq, t_ms=frame.t_ms, pcm=frame.pcm)


def test_join_covers_window_timestamps(tmp_path):
    hub = _hub(tmp_path)
    packets = _json_samples(PHONE_A)
    t_end = T0 + (N - 1) * DT
    frames = _pcm_frames(PHONE_A, T0, t_end + DT)

    async def run() -> None:
        await _ingest_pcm(hub, frames)
        await _ingest_json(hub, packets)

    asyncio.run(run())
    got = hub.last_pcm[PHONE_A]
    span_ms = got["t_end_ms"] - got["t_start_ms"]
    assert got["t_start_ms"] == T0
    assert abs(got["samples"] - span_ms * 16) <= 16
    assert got["coverage"] >= 0.95
    assert hub.latest_inference is not None
    assert hub.latest_inference["is_fall"] is True


def test_pcm_for_other_id_does_not_cover(tmp_path):
    hub = _hub(tmp_path)
    packets = _json_samples(PHONE_A)
    t_end = T0 + (N - 1) * DT
    frames = _pcm_frames(PHONE_B, T0, t_end + DT)

    async def run() -> None:
        await hub.ingest(packets[0])  # register A
        await _ingest_pcm(hub, frames)
        await _ingest_json(hub, packets[1:])

    asyncio.run(run())
    got = hub.last_pcm[PHONE_A]
    assert got["coverage"] == 0.0


def test_join_uses_phone_t_not_ingest_order(tmp_path):
    hub = _hub(tmp_path)
    packets = _json_samples(PHONE_A)
    t_end = T0 + (N - 1) * DT
    frames = _pcm_frames(PHONE_A, T0, t_end + DT)

    async def run() -> None:
        await _ingest_json(hub, packets[:50])
        await _ingest_pcm(hub, frames)
        await _ingest_json(hub, packets[50:])

    asyncio.run(run())
    got = hub.last_pcm[PHONE_A]
    assert got["coverage"] >= 0.95
    span_ms = got["t_end_ms"] - got["t_start_ms"]
    assert abs(got["samples"] - span_ms * 16) <= 16


class _CaptureCnn:
    def __init__(self) -> None:
        self.windows: list[SensorWindow] = []

    def infer(self, window: SensorWindow) -> InferenceResult:
        self.windows.append(window)
        return InferenceResult(
            inference_id="cap",
            timestamp=window.t_end_ms,
            node_id=window.node_id,
            room=window.room,
            is_fall=False,
            confidence=0.0,
        )


def test_join_attaches_pcm_to_window(tmp_path):
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(threshold=0.90, client=RecordingCloudClient(), store=store)
    clf = _CaptureCnn()
    hub = Hub(classifier=clf, gate=gate, store=store)
    packets = _json_samples(PHONE_A)
    t_end = T0 + (N - 1) * DT
    frames = _pcm_frames(PHONE_A, T0, t_end + DT)

    async def run() -> None:
        await _ingest_pcm(hub, frames)
        await _ingest_json(hub, packets)

    asyncio.run(run())
    assert clf.windows
    pcm = clf.windows[0].pcm
    assert len(pcm) > 1000
    assert clf.windows[0].pcm_hz == 16000.0
    assert max(abs(x) for x in pcm) <= 1.0 + 1e-6

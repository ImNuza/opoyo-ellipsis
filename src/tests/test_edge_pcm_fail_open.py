from __future__ import annotations

import asyncio

from edge.app import Hub, MAX_NODES
from edge.gate import EscalationGate, RecordingCloudClient
from edge.infer import FakeCnn
from edge.log import InferenceLog
from phone.fake_phone import make_packet
from shared.pcm import unpack_frame

PHONE_A = "11111111-1111-1111-1111-111111111111"
PHONE_UNKNOWN = "99999999-9999-9999-9999-999999999999"
T0 = 1_000_000
N = 100
DT = 20
HZ = 50.0


def _hub(tmp_path) -> tuple[Hub, RecordingCloudClient]:
    store = InferenceLog(tmp_path / "inference.jsonl")
    client = RecordingCloudClient()
    gate = EscalationGate(threshold=0.90, client=client, store=store)
    hub = Hub(classifier=FakeCnn(is_fall=True, confidence=0.95), gate=gate, store=store)
    return hub, client


def test_json_only_still_classifies(tmp_path):
    hub, client = _hub(tmp_path)

    async def run() -> None:
        for i in range(N):
            await hub.ingest(
                make_packet(PHONE_A, t_ms=T0 + i * DT, elapsed=i / HZ, kind="knee")
            )

    asyncio.run(run())
    got = hub.last_pcm[PHONE_A]
    assert got["coverage"] == 0.0
    assert got["samples"] == 0
    assert hub.latest_inference is not None
    assert hub.latest_inference["is_fall"] is True
    assert len(client.posted) == 1


def test_unknown_pcm_does_not_create_node(tmp_path):
    hub, _client = _hub(tmp_path)
    pcm = (b"\x00\x00") * 320

    async def run() -> None:
        await hub.ingest_pcm(PHONE_UNKNOWN, seq=0, t_ms=T0, pcm=pcm)

    asyncio.run(run())
    assert PHONE_UNKNOWN not in hub.nodes
    assert len(hub.nodes) == 0
    assert hub.dropped == 0
    assert PHONE_UNKNOWN in hub.pcm_rings


def test_max_nodes_still_from_json_not_pcm(tmp_path):
    hub, _client = _hub(tmp_path)
    pcm = (b"\x00\x00") * 320

    async def run() -> None:
        for i in range(MAX_NODES):
            nid = f"11111111-1111-1111-1111-11111111111{i}"
            await hub.ingest(make_packet(nid, t_ms=T0, elapsed=0.0))
        await hub.ingest_pcm(PHONE_UNKNOWN, seq=0, t_ms=T0, pcm=pcm)

    asyncio.run(run())
    assert len(hub.nodes) == MAX_NODES
    assert PHONE_UNKNOWN not in hub.nodes


def test_unpack_garbage_is_none():
    assert unpack_frame(b"not-a-frame") is None
    assert unpack_frame(b'{"v":2,"id":"x"}') is None

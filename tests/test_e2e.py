from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from edge.app import create_app as create_edge
from edge.gate import RecordingCloudClient
from edge.infer import FakeCnn
from edge.log import InferenceLog
from phone.fake_phone import sample_stream
from server.app import create_app as create_cloud
from server.decision_tree import DecisionTree
from tests.fakes import FakeClock, FakeSender

PHONE_A = "11111111-1111-1111-1111-111111111111"
PHONE_B = "22222222-2222-2222-2222-222222222222"
KIN = "kin-chat"
SECONDARY = "sec-chat"
SENIOR = "senior-chat"
SAMPLES = 120


async def _feed(hub, packets) -> None:
    for packet in packets:
        await hub.ingest(packet)


def _edge(*, tmp_path: Path, cloud_client, classifier=None):
    return create_edge(
        enable_udp=False,
        log_path=tmp_path / "inference.jsonl",
        cloud_client=cloud_client,
        classifier=classifier,
    )


def test_fake_phones_ingest_and_log(tmp_path: Path):
    cloud = RecordingCloudClient()
    app = _edge(tmp_path=tmp_path, cloud_client=cloud)
    packets = sample_stream([PHONE_A, PHONE_B], n=SAMPLES, kind="knee")
    asyncio.run(_feed(app.state.hub, packets))

    nodes = app.state.hub.nodes
    assert set(nodes) == {PHONE_A, PHONE_B}
    assert {n.name for n in nodes.values()} == {"Phone 1", "Phone 2"}
    assert all(n.packets == SAMPLES for n in nodes.values())

    store = InferenceLog(tmp_path / "inference.jsonl")
    rows = store.tail(50)
    assert len(rows) >= 2
    by_node = {row.node_id for row in rows}
    assert by_node == {"Phone 1", "Phone 2"}
    assert {row.room for row in rows} == {1, 2}
    assert all(row.is_fall is False for row in rows)
    assert all(row.confidence == 0.0 for row in rows)
    assert cloud.posted == []

    with TestClient(app) as client:
        state = client.get("/api/state").json()
    live = [s for s in state["slots"] if not s.get("empty")]
    assert {s["id"] for s in live} == {PHONE_A, PHONE_B}
    assert state["inference"]["latest"] is not None
    assert state["inference"]["latest"]["is_fall"] is False


def test_fake_phone_over_threshold_triggers_telegram(tmp_path: Path):
    cloud_recorder = RecordingCloudClient()
    app = _edge(
        tmp_path=tmp_path,
        cloud_client=cloud_recorder,
        classifier=FakeCnn(is_fall=True, confidence=0.95),
    )
    packets = sample_stream([PHONE_A], n=SAMPLES, kind="knee")
    asyncio.run(_feed(app.state.hub, packets))

    store = InferenceLog(tmp_path / "inference.jsonl")
    rows = store.tail(50)
    assert rows
    assert all(row.is_fall is True for row in rows)
    assert all(row.confidence == 0.95 for row in rows)
    assert cloud_recorder.posted
    event = cloud_recorder.posted[0]
    assert event.is_fall is True
    assert event.threshold == 0.90
    assert event.confidence == 0.95
    assert event.room == 1
    assert event.node_id == "Phone 1"

    telegram = FakeSender()
    tree = DecisionTree(
        clock=FakeClock(),
        telegram=telegram,
        next_of_kin_chat_id=KIN,
        secondary_chat_id=SECONDARY,
        senior_chat_id=SENIOR,
    )
    with TestClient(create_cloud(tree=tree)) as client:
        response = client.post("/events", json=event.model_dump())
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "rung1_dispatched"

    assert [dest for dest, _text in telegram.sent] == [KIN, SENIOR]
    family_text = telegram.sent[0][1]
    senior_text = telegram.sent[1][1]
    assert "OPOYO: fall" in family_text
    assert "Room 1" in family_text
    assert "0.95" in family_text
    assert "Reply yes" in senior_text

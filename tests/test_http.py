from __future__ import annotations

from fastapi.testclient import TestClient

from edge.app import create_app as create_edge
from server.app import create_app as create_cloud
from server.decision_tree import DecisionTree
from shared.schemas import AckEvent, FallEvent
from tests.fakes import FakeClock, FakeSender


def _fall() -> dict:
    return FallEvent(
        event_id="evt-http",
        inference_id="evt-http",
        timestamp=1735689602123,
        node_id="Phone 1",
        room=1,
        is_fall=True,
        confidence=0.95,
        threshold=0.9,
    ).model_dump()


def _cloud_tree() -> DecisionTree:
    return DecisionTree(
        clock=FakeClock(),
        telegram=FakeSender(),
        twilio=FakeSender(),
        next_of_kin_chat_id="kin",
        secondary_chat_id="sec",
        senior_phone="+6500000000",
    )


def test_cloud_post_events_returns_case():
    with TestClient(create_cloud(tree=_cloud_tree())) as client:
        response = client.post("/events", json=_fall())
        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "rung1_dispatched"
        assert body["case_id"]


def test_cloud_ack_moves_state():
    with TestClient(create_cloud(tree=_cloud_tree())) as client:
        created = client.post("/events", json=_fall()).json()
        case_id = created["case_id"]
        ack = AckEvent(
            case_id=case_id,
            actor="senior",
            outcome="fine",
            timestamp=1,
        )
        response = client.post(f"/cases/{case_id}/ack", json=ack.model_dump())
        assert response.status_code == 200
        assert response.json()["state"] == "false_alarm_closed"


def test_edge_has_no_raw_sample_cloud_dump():
    app = create_edge(enable_udp=False)
    paths = {getattr(route, "path", "") for route in app.routes}
    forbidden = [
        p for p in paths if "sample" in p.lower() and "cloud" in p.lower()
    ]
    assert forbidden == []
    assert "/events" not in paths
    assert "/api/edge/config" not in paths
    assert "/api/phones/{node_id}/name" not in paths

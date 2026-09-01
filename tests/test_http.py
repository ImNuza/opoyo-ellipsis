from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from edge.app import create_app as create_edge
from server.adapters import RecordingAdapter
from server.app import create_app as create_cloud
from server.tree import EscalationTree, FakeClock
from shared.schemas import AckEvent, FallEvent


def _fall() -> dict:
    return FallEvent(
        event_id="evt-http",
        inference_id="evt-http",
        timestamp=1735689602123,
        node_id="n1",
        room="Bathroom",
        is_fall=True,
        confidence=0.95,
        threshold=0.9,
    ).model_dump()


def test_cloud_post_events_returns_case():
    tree = EscalationTree(
        clock=FakeClock(),
        telegram=RecordingAdapter(),
        twilio=RecordingAdapter(),
        secondary=RecordingAdapter(),
        careline=RecordingAdapter(),
    )
    with TestClient(create_cloud(tree=tree)) as client:
        response = client.post("/events", json=_fall())
        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "rung1_dispatched"
        assert body["case_id"]


def test_cloud_ack_moves_state():
    tree = EscalationTree(
        clock=FakeClock(),
        telegram=RecordingAdapter(),
        twilio=RecordingAdapter(),
        secondary=RecordingAdapter(),
        careline=RecordingAdapter(),
    )
    with TestClient(create_cloud(tree=tree)) as client:
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


def test_edge_config_roundtrip(tmp_path: Path):
    app = create_edge(enable_udp=False, log_path=tmp_path / "inference.jsonl")
    with TestClient(app) as client:
        response = client.post(
            "/api/edge/config",
            json={"escalate_min_confidence": 0.5},
        )
        assert response.status_code == 200
        assert response.json()["cfg"]["escalate_min_confidence"] == 0.5


def test_edge_has_no_raw_sample_cloud_dump():
    app = create_edge(enable_udp=False)
    paths = {getattr(route, "path", "") for route in app.routes}
    forbidden = [p for p in paths if "sample" in p.lower() and "cloud" in p.lower()]
    assert forbidden == []
    assert "/events" not in paths

from __future__ import annotations

import pytest

from shared.schemas import InferenceResult
from edge.gate import EscalationGate, RecordingCloudClient, should_escalate
from edge.log import InferenceLog


def _result(is_fall: bool, confidence: float) -> InferenceResult:
    return InferenceResult(
        inference_id="inf1",
        timestamp=1735689602123,
        node_id="Phone 1",
        room=1,
        is_fall=is_fall,
        confidence=confidence,
    )


@pytest.mark.parametrize(
    "is_fall,confidence,threshold,posted",
    [
        (False, 0.99, 0.90, False),
        (True, 0.89, 0.90, False),
        (True, 0.90, 0.90, True),
        (True, 0.94, 0.95, False),
        (True, 0.94, 0.90, True),
    ],
)
def test_should_escalate_table(
    is_fall: bool, confidence: float, threshold: float, posted: bool
):
    assert should_escalate(_result(is_fall, confidence), threshold) is posted


def test_gate_posts_fall_event_with_threshold(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(threshold=0.90, client=client, store=store)
    event = gate.handle(_result(True, 0.94))
    assert event is not None
    assert len(client.posted) == 1
    body = client.posted[0]
    assert body.is_fall is True
    assert body.threshold == 0.90
    assert body.confidence == 0.94
    assert body.room == 1
    assert body.node_id == "Phone 1"
    assert len(store.tail(5)) == 1


def test_runtime_threshold_change(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(threshold=0.90, client=client, store=store)
    gate.handle(_result(True, 0.94))
    assert len(client.posted) == 1
    gate.threshold = 0.95
    gate.handle(
        InferenceResult(
            inference_id="inf2",
            timestamp=2,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.94,
        )
    )
    assert len(client.posted) == 1


def test_no_fall_still_logged(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(threshold=0.90, client=client, store=store)
    assert gate.handle(_result(False, 0.99)) is None
    assert client.posted == []
    assert store.tail(1)[0].is_fall is False

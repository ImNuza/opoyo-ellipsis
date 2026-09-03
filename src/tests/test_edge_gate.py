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


def test_cooldown_skips_second_post_same_node(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    t = {"now": 0.0}
    gate = EscalationGate(
        threshold=0.90,
        client=client,
        store=store,
        cooldown_s=3.0,
        clock=lambda: t["now"],
    )
    first = gate.handle(_result(True, 0.94))
    second = gate.handle(
        InferenceResult(
            inference_id="inf2",
            timestamp=2,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.94,
        )
    )
    assert first is not None
    assert second is None
    assert len(client.posted) == 1
    assert len(store.tail(5)) == 2


def test_cooldown_expires_after_three_seconds(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    t = {"now": 0.0}
    gate = EscalationGate(
        threshold=0.90,
        client=client,
        store=store,
        cooldown_s=3.0,
        clock=lambda: t["now"],
    )
    first = _result(True, 0.94)
    gate.handle(first)
    t["now"] = 3.0
    later = gate.handle(
        InferenceResult(
            inference_id="inf2",
            timestamp=first.timestamp + 3000,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.94,
        )
    )
    assert later is not None
    assert len(client.posted) == 2


class _SlowPostClient:
    """Advances the fake clock during POST, like a blocking Telegram send."""

    def __init__(self, t: dict[str, float], delay_s: float) -> None:
        self.posted: list = []
        self._t = t
        self._delay_s = delay_s

    def post(self, event) -> None:
        self._t["now"] += self._delay_s
        self.posted.append(event)


def test_cooldown_starts_after_post_not_before(tmp_path):
    t = {"now": 0.0}
    client = _SlowPostClient(t, delay_s=3.0)
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(
        threshold=0.90,
        client=client,
        store=store,
        cooldown_s=3.0,
        clock=lambda: t["now"],
    )
    first = gate.handle(_result(True, 0.94))
    second = gate.handle(
        InferenceResult(
            inference_id="inf2",
            timestamp=_result(True, 0.94).timestamp + 1000,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.94,
        )
    )
    assert first is not None
    assert second is None
    assert len(client.posted) == 1


def test_cooldown_holds_when_infer_is_slower_than_three_seconds(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    t = {"now": 0.0}
    gate = EscalationGate(
        threshold=0.50,
        client=client,
        store=store,
        cooldown_s=3.0,
        clock=lambda: t["now"],
    )
    t0 = 1_788_418_513_053
    first = gate.handle(
        InferenceResult(
            inference_id="inf1",
            timestamp=t0,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.84,
        )
    )
    t["now"] = 5.0
    second = gate.handle(
        InferenceResult(
            inference_id="inf2",
            timestamp=t0 + 1001,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.72,
        )
    )
    third = gate.handle(
        InferenceResult(
            inference_id="inf3",
            timestamp=t0 + 2003,
            node_id="Phone 1",
            room=1,
            is_fall=True,
            confidence=0.60,
        )
    )
    assert first is not None
    assert second is None
    assert third is None
    assert len(client.posted) == 1


def test_cooldown_is_per_node(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    t = {"now": 0.0}
    gate = EscalationGate(
        threshold=0.90,
        client=client,
        store=store,
        cooldown_s=3.0,
        clock=lambda: t["now"],
    )
    gate.handle(_result(True, 0.94))
    other = gate.handle(
        InferenceResult(
            inference_id="inf2",
            timestamp=2,
            node_id="Phone 2",
            room=2,
            is_fall=True,
            confidence=0.94,
        )
    )
    assert other is not None
    assert len(client.posted) == 2


def test_no_fall_still_logged(tmp_path):
    client = RecordingCloudClient()
    store = InferenceLog(tmp_path / "inference.jsonl")
    gate = EscalationGate(threshold=0.90, client=client, store=store)
    assert gate.handle(_result(False, 0.99)) is None
    assert client.posted == []
    assert store.tail(1)[0].is_fall is False

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import FallEvent, InferenceResult, SensorSample, fall_event_from_inference


PHONE = {
    "v": 2,
    "id": "c0a1",
    "model": "iPhone 17 Pro Max",
    "t": 1735689600123,
    "ax": 0.01,
    "ay": 0.0,
    "az": -0.02,
    "mag": 0.03,
    "db": -41.2,
}


def test_sensor_sample_accepts_v2_packet():
    sample = SensorSample.model_validate(PHONE)
    assert sample.id == "c0a1"
    assert sample.t == 1735689600123
    assert sample.mag == 0.03


@pytest.mark.parametrize("missing", ["id", "t", "mag"])
def test_sensor_sample_rejects_missing_required(missing: str):
    payload = dict(PHONE)
    payload.pop(missing)
    with pytest.raises(ValidationError):
        SensorSample.model_validate(payload)


def test_inference_confidence_bounds_and_bool():
    ok = InferenceResult(
        inference_id="abc",
        timestamp=1,
        node_id="n1",
        room="Phone 1",
        is_fall=False,
        confidence=0.12,
    )
    dumped = ok.model_dump()
    assert dumped["is_fall"] is False
    assert dumped["confidence"] == 0.12
    with pytest.raises(ValidationError):
        InferenceResult(
            inference_id="abc",
            timestamp=1,
            node_id="n1",
            room="Phone 1",
            is_fall=True,
            confidence=1.2,
        )


def test_fall_event_requires_is_fall_true():
    with pytest.raises(ValidationError):
        FallEvent(
            event_id="e1",
            inference_id="e1",
            timestamp=1,
            node_id="n1",
            room="Bathroom",
            is_fall=False,  # type: ignore[arg-type]
            confidence=0.94,
            threshold=0.9,
        )


def test_fall_event_from_inference_copies_fields():
    result = InferenceResult(
        inference_id="a1b2",
        timestamp=1735689602123,
        node_id="c0a1",
        room="Bathroom",
        is_fall=True,
        confidence=0.94,
    )
    event = fall_event_from_inference(result, 0.9)
    assert event.timestamp == result.timestamp
    assert event.node_id == result.node_id
    assert event.room == result.room
    assert event.confidence == result.confidence
    assert event.is_fall is True
    assert event.threshold == 0.9
    assert event.inference_id == result.inference_id

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    v: int
    id: str
    model: str = "iPhone"
    t: int
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    mag: float
    db: float = -120.0


class SensorWindow(BaseModel):
    node_id: str
    room: int
    t_start_ms: int
    t_end_ms: int
    hz: float
    mag: list[float]
    ax: list[float]
    ay: list[float]
    az: list[float]
    db: list[float]


class InferenceResult(BaseModel):
    inference_id: str
    timestamp: int
    node_id: str
    room: int
    is_fall: bool
    confidence: float = Field(ge=0.0, le=1.0)


class FallEvent(BaseModel):
    event_id: str
    inference_id: str
    timestamp: int
    node_id: str
    room: int
    is_fall: Literal[True]
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


AckActor = Literal["senior", "family", "secondary", "careline"]
AckOutcome = Literal["fine", "not_fine", "no_answer", "taken"]
EscalationRung = Literal["family_telegram", "senior_call", "secondary", "careline"]
CaseState = Literal[
    "rung1_dispatched",
    "false_alarm_closed",
    "awaiting_family",
    "family_handling",
    "secondary_alerted",
    "resolved",
    "careline_alerted",
]


class AckEvent(BaseModel):
    case_id: str
    actor: AckActor
    outcome: AckOutcome
    timestamp: int


class EscalationCommand(BaseModel):
    case_id: str
    rung: EscalationRung
    at_s: int
    event: FallEvent


class EscalationCase(BaseModel):
    case_id: str
    event: FallEvent
    state: CaseState
    started_at_s: float
    family_wait_started_at: float | None = None
    commands: list[EscalationCommand] = Field(default_factory=list)
    acks: list[AckEvent] = Field(default_factory=list)


def fall_event_from_inference(result: InferenceResult, threshold: float) -> FallEvent:
    return FallEvent(
        event_id=result.inference_id,
        inference_id=result.inference_id,
        timestamp=result.timestamp,
        node_id=result.node_id,
        room=result.room,
        is_fall=True,
        confidence=result.confidence,
        threshold=threshold,
    )

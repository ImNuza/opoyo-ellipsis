"""Wire and log models shared by phone packets, edge inference, and cloud cases.

Phones emit SensorSample. The edge builds SensorWindow → InferenceResult.
Only a gated InferenceResult becomes FallEvent and leaves the machine.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorSample(BaseModel):
    """One 50 Hz UDP packet from a phone. Extra keys are ignored."""

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
    """2 s / 50 Hz slice for the classifier. node_id is Phone N; room is slot 1–5.

    pcm is the time-aligned 16 kHz clip from PcmRing, in [-1, 1]. Empty if the
    WAV channel missed this window (classifier then uses mag only).
    """

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
    pcm: list[float] = Field(default_factory=list)
    pcm_hz: float = 16000.0


class InferenceResult(BaseModel):
    """Classifier output. Logged always; posted to the cloud only if gated."""

    inference_id: str
    timestamp: int
    node_id: str
    room: int
    is_fall: bool
    confidence: float = Field(ge=0.0, le=1.0)


class FallEvent(BaseModel):
    """Gated fall. is_fall is always True; threshold is the edge gate that fired."""

    event_id: str
    inference_id: str
    timestamp: int
    node_id: str
    room: int
    is_fall: Literal[True]
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


AckActor = Literal["senior", "family", "secondary", "careline"]
AckOutcome = Literal["yes", "fine", "not_fine", "no_answer", "taken"]
EscalationRung = Literal["family_telegram", "senior_telegram", "secondary", "careline"]
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
    """Human response that advances or closes an EscalationCase."""

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
    """Cloud case: FallEvent plus ladder state, commands, and acks."""

    case_id: str
    event: FallEvent
    state: CaseState
    started_at_s: float
    family_wait_started_at: float | None = None
    commands: list[EscalationCommand] = Field(default_factory=list)
    acks: list[AckEvent] = Field(default_factory=list)


def fall_event_from_inference(result: InferenceResult, threshold: float) -> FallEvent:
    """Copy inference identity into a FallEvent. Caller must already have gated."""
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

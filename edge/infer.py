"""Fall classifiers. Live path is StubCnn; FakeCnn is the test double that can fire."""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from shared.schemas import InferenceResult, SensorWindow


class Classifier(Protocol):
    def infer(self, window: SensorWindow) -> InferenceResult: ...


class StubCnn:
    """Placeholder 1D CNN. Same output shape as a trained model; always no-fall."""

    def infer(self, window: SensorWindow) -> InferenceResult:
        return InferenceResult(
            inference_id=uuid4().hex[:12],
            timestamp=window.t_end_ms,
            node_id=window.node_id,
            room=window.room,
            is_fall=False,
            confidence=0.0,
        )


class FakeCnn:
    """Deterministic classifier for e2e tests. Does not look at window mag."""

    def __init__(self, is_fall: bool = True, confidence: float = 0.95) -> None:
        self.is_fall = is_fall
        self.confidence = confidence

    def infer(self, window: SensorWindow) -> InferenceResult:
        return InferenceResult(
            inference_id=uuid4().hex[:12],
            timestamp=window.t_end_ms,
            node_id=window.node_id,
            room=window.room,
            is_fall=self.is_fall,
            confidence=self.confidence,
        )

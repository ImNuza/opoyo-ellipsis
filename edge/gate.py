"""Log every inference; POST FallEvent to the cloud only when gated."""

from __future__ import annotations

import time
from typing import Callable, Protocol

from shared.schemas import FallEvent, InferenceResult

ESCALATE_COOLDOWN_S = 3.0


class CloudClient(Protocol):
    def post(self, event: FallEvent) -> None: ...


class RecordingCloudClient:
    def __init__(self) -> None:
        self.posted: list[FallEvent] = []

    def post(self, event: FallEvent) -> None:
        self.posted.append(event)


class HttpCloudClient:
    """POST /events on the cloud. Failures are swallowed so ingest never blocks."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def post(self, event: FallEvent) -> None:
        import httpx

        try:
            httpx.post(
                f"{self.base_url}/events",
                json=event.model_dump(),
                timeout=8.0,
            )
        except Exception:
            return


def should_escalate(result: InferenceResult, threshold: float) -> bool:
    return bool(result.is_fall) and result.confidence >= threshold


class EscalationGate:
    """Always append to the log. Escalate iff is_fall and confidence >= threshold.

    After a POST for a node, further gated falls from that node are logged but
    not posted until ``cooldown_s`` has elapsed. Overlapping 2 s / 1 s windows
    otherwise open two Telegram ladders from one impact.
    """

    def __init__(
        self,
        threshold: float,
        client: CloudClient,
        store: object | None = None,
        cooldown_s: float = ESCALATE_COOLDOWN_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.client = client
        self.store = store
        self.cooldown_s = cooldown_s
        self._clock = clock or time.monotonic
        self._last_sent_at: dict[str, float] = {}

    def handle(self, result: InferenceResult) -> FallEvent | None:
        append = getattr(self.store, "append", None)
        if callable(append):
            append(result)
        if not should_escalate(result, self.threshold):
            return None
        now = self._clock()
        last = self._last_sent_at.get(result.node_id)
        if last is not None and (now - last) < self.cooldown_s:
            return None
        event = FallEvent(
            event_id=result.inference_id,
            inference_id=result.inference_id,
            timestamp=result.timestamp,
            node_id=result.node_id,
            room=result.room,
            is_fall=True,
            confidence=result.confidence,
            threshold=self.threshold,
        )
        self.client.post(event)
        self._last_sent_at[result.node_id] = now
        return event
    

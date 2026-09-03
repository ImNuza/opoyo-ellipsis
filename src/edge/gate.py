"""Log every inference; POST FallEvent to the cloud only when gated."""

from __future__ import annotations

import time
from typing import Callable, Protocol

from shared.config import CFG
from shared.schemas import FallEvent, InferenceResult

ESCALATE_COOLDOWN_S = CFG.alert.cooldown_s


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
                timeout=CFG.edge.cloud_post_timeout_s,
            )
        except Exception:
            return


def should_escalate(result: InferenceResult, threshold: float) -> bool:
    return bool(result.is_fall) and result.confidence >= threshold


class EscalationGate:
    """Always append to the log. Escalate iff is_fall and confidence >= threshold.

    Cooldown starts after a successful POST, not before it. Further gated falls
    from that node are logged but not posted until ``cooldown_s`` of wall time
    has passed *and* the new window is at least ``cooldown_s`` after the posted
    event. Overlapping 2 s / 1 s hops would otherwise open two Telegram ladders
    from one impact once YAMNet + Telegram take longer than the cooldown.
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
        self._last_event_ms: dict[str, int] = {}

    def _cooling_down(self, result: InferenceResult) -> bool:
        last_wall = self._last_sent_at.get(result.node_id)
        if last_wall is not None and (self._clock() - last_wall) < self.cooldown_s:
            return True
        last_ms = self._last_event_ms.get(result.node_id)
        if last_ms is not None:
            dt_ms = result.timestamp - last_ms
            if 0 <= dt_ms < self.cooldown_s * 1000.0:
                return True
        return False

    def handle(self, result: InferenceResult) -> FallEvent | None:
        append = getattr(self.store, "append", None)
        if callable(append):
            append(result)
        if not should_escalate(result, self.threshold):
            return None
        if self._cooling_down(result):
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
        # Stamp before POST so a slow Telegram send cannot open a second
        # window; stamp again after so the 3 s pause starts when it returns.
        self._last_sent_at[result.node_id] = self._clock()
        self._last_event_ms[result.node_id] = result.timestamp
        self.client.post(event)
        self._last_sent_at[result.node_id] = self._clock()
        return event 

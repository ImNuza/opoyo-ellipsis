from __future__ import annotations

from typing import Protocol

from shared.schemas import FallEvent, InferenceResult


class CloudClient(Protocol):
    def post(self, event: FallEvent) -> None: ...


class RecordingCloudClient:
    def __init__(self) -> None:
        self.posted: list[FallEvent] = []

    def post(self, event: FallEvent) -> None:
        self.posted.append(event)


class HttpCloudClient:
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


class EscalationGate:
    def __init__(
        self,
        threshold: float,
        client: CloudClient,
        store: object | None = None,
    ) -> None:
        self.threshold = threshold
        self.client = client
        self.store = store

    def handle(self, result: InferenceResult) -> FallEvent | None:
        append = getattr(self.store, "append", None)
        if callable(append):
            append(result)
        # Check if the result should be escalated based on the threshold
        if bool(result.is_fall) and result.confidence >= self.threshold:
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
            return event
        return None
    

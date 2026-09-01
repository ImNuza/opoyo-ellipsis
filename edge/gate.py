from __future__ import annotations

from typing import Protocol

from shared.schemas import FallEvent, InferenceResult, fall_event_from_inference


def should_escalate(result: InferenceResult, threshold: float) -> bool:
    return bool(result.is_fall) and result.confidence >= threshold


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
        if not should_escalate(result, self.threshold):
            return None
        event = fall_event_from_inference(result, self.threshold)
        self.client.post(event)
        return event

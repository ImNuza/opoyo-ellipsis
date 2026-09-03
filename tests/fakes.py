"""Test doubles for cloud adapters and clocks."""

from __future__ import annotations


class FakeClock:
    """Monotonic clock that tests can advance."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeSender:
    """Records ``send`` calls as ``(destination, message)`` pairs."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.markups: list[dict | None] = []

    def send(
        self,
        destination: str,
        message: str,
        reply_markup: dict | None = None,
    ) -> bool:
        self.sent.append((destination, message))
        self.markups.append(reply_markup)
        return True

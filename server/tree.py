from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from shared.schemas import AckEvent, EscalationCase, EscalationCommand, FallEvent
from server.adapters import Notifier


TERMINAL = frozenset(
    {
        "false_alarm_closed",
        "family_handling",
        "resolved",
        "careline_alerted",
    }
)

FAMILY_WAIT_S = 60.0
CARELINE_AT_S = 180.0


class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        import time

        return time.time()


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class EscalationTree:
    def __init__(
        self,
        clock: Clock,
        telegram: Notifier,
        twilio: Notifier,
        secondary: Notifier,
        careline: Notifier,
    ) -> None:
        self.clock = clock
        self.telegram = telegram
        self.twilio = twilio
        self.secondary = secondary
        self.careline = careline
        self.cases: dict[str, EscalationCase] = {}
        self._by_event: dict[str, str] = {}

    def ingest(self, event: FallEvent) -> EscalationCase:
        existing_id = self._by_event.get(event.event_id)
        if existing_id is not None:
            return self.cases[existing_id]
        now = self.clock.now()
        case = EscalationCase(
            case_id=uuid4().hex[:12],
            event=event,
            state="rung1_dispatched",
            started_at_s=now,
        )
        self.cases[case.case_id] = case
        self._by_event[event.event_id] = case.case_id
        self._fire(case, "family_telegram", 0, self.telegram)
        self._fire(case, "senior_call", 0, self.twilio)
        return case

    def on_ack(self, ack: AckEvent) -> EscalationCase | None:
        case = self.cases.get(ack.case_id)
        if case is None:
            return None
        if case.state in TERMINAL:
            return case
        case.acks.append(ack)
        if ack.actor == "senior" and ack.outcome == "fine":
            case.state = "false_alarm_closed"
        elif ack.actor == "senior" and ack.outcome in {"no_answer", "not_fine"}:
            if case.state == "rung1_dispatched":
                case.state = "awaiting_family"
                case.family_wait_started_at = self.clock.now()
        elif ack.actor == "family" and ack.outcome == "taken":
            if case.state == "awaiting_family":
                case.state = "family_handling"
        elif ack.actor == "secondary" and ack.outcome == "taken":
            if case.state == "secondary_alerted":
                case.state = "resolved"
        return case

    def on_tick(self, now: float | None = None) -> None:
        t = self.clock.now() if now is None else now
        for case in list(self.cases.values()):
            if case.state == "awaiting_family" and case.family_wait_started_at is not None:
                if t - case.family_wait_started_at >= FAMILY_WAIT_S:
                    self._fire(case, "secondary", 60, self.secondary)
                    case.state = "secondary_alerted"
            if case.state == "secondary_alerted":
                if t - case.started_at_s >= CARELINE_AT_S:
                    self._fire(case, "careline", 180, self.careline)
                    case.state = "careline_alerted"

    def _fire(
        self,
        case: EscalationCase,
        rung: str,
        at_s: int,
        adapter: Notifier,
    ) -> None:
        cmd = EscalationCommand(
            case_id=case.case_id,
            rung=rung,  # type: ignore[arg-type]
            at_s=at_s,
            event=case.event,
        )
        case.commands.append(cmd)
        adapter.notify(cmd)

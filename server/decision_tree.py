"""Figure A5 escalation: Telegram family + senior, then secondary, CareLine.

Twilio is not on this ladder. The adapter in ``server.adapters.twilio`` stays
as a plug-in point for a future voice path.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from server.adapters import Sender
from server.adapters.telegram import senior_ack_markup
from shared.schemas import AckEvent, EscalationCase, EscalationCommand, FallEvent


TERMINAL = frozenset(
    {
        "false_alarm_closed",
        "family_handling",
        "resolved",
        "careline_alerted",
    }
)

SENIOR_WAIT_S = 60.0
FAMILY_WAIT_S = 60.0
CARELINE_AT_S = 180.0


class Clock(Protocol):
    """Seconds since an arbitrary epoch."""

    def now(self) -> float:
        """Return the current time in seconds."""


class SystemClock:
    """Wall clock."""

    def now(self) -> float:
        return time.time()


def fall_message(event: FallEvent) -> str:
    """Build the standard fall alert body.

    Args:
        event: Gated fall from the edge.

    Returns:
        Human-readable alert including room, local time, and confidence.
    """
    ts = datetime.fromtimestamp(event.timestamp / 1000.0).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return (
        f"OPOYO: fall. Room {event.room}. {ts}. "
        f"confidence {event.confidence:.2f}."
    )


def senior_check_message(event: FallEvent) -> str:
    """Ask the senior over Telegram whether they are okay.

    A ``yes`` ack closes the case as all-clear.
    """
    ts = datetime.fromtimestamp(event.timestamp / 1000.0).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return (
        f"OPOYO: possible fall. Room {event.room}. {ts}. "
        f"Are you okay? Tap I'm fine or I need help "
        f"(or reply yes / no). "
        f"confidence {event.confidence:.2f}."
    )


class DecisionTree:
    """Timed escalation: next of kin + senior over Telegram, then secondary."""

    def __init__(
        self,
        clock: Clock,
        telegram: Sender,
        next_of_kin_chat_id: str,
        secondary_chat_id: str,
        senior_chat_id: str,
    ) -> None:
        self._clock = clock
        self._telegram = telegram
        self._next_of_kin_chat_id = next_of_kin_chat_id
        self._secondary_chat_id = secondary_chat_id
        self._senior_chat_id = senior_chat_id
        self.cases: dict[str, EscalationCase] = {}
        self._by_event: dict[str, str] = {}

    def ingest(self, event: FallEvent) -> EscalationCase:
        """Start a case and fire t+0 alerts.

        Args:
            event: Confirmed fall from the edge.

        Returns:
            The new or existing case for ``event.event_id``.
        """
        existing_id = self._by_event.get(event.event_id)
        if existing_id is not None:
            return self.cases[existing_id]
        now = self._clock.now()
        case = EscalationCase(
            case_id=uuid4().hex[:12],
            event=event,
            state="rung1_dispatched",
            started_at_s=now,
        )
        self.cases[case.case_id] = case
        self._by_event[event.event_id] = case.case_id
        self._dispatch(
            case,
            rung="family_telegram",
            at_s=0,
            sender=self._telegram,
            destination=self._next_of_kin_chat_id,
            message=fall_message(event),
        )
        self._dispatch(
            case,
            rung="senior_telegram",
            at_s=0,
            sender=self._telegram,
            destination=self._senior_chat_id,
            message=senior_check_message(event),
            reply_markup=senior_ack_markup(case.case_id),
        )
        return case

    def on_ack(self, ack: AckEvent) -> EscalationCase | None:
        """Apply a human response to an open case.

        Args:
            ack: Senior / family / secondary outcome.

        Returns:
            The updated case, or None if ``ack.case_id`` is unknown.
        """
        case = self.cases.get(ack.case_id)
        if case is None:
            return None
        if case.state in TERMINAL:
            return case
        case.acks.append(ack)
        if ack.actor == "senior" and ack.outcome in {"yes", "fine"}:
            case.state = "false_alarm_closed"
        elif ack.actor == "senior" and ack.outcome in {"no_answer", "not_fine"}:
            if case.state == "rung1_dispatched":
                case.state = "awaiting_family"
                case.family_wait_started_at = self._clock.now()
        elif ack.actor == "family" and ack.outcome == "taken":
            if case.state == "awaiting_family":
                case.state = "family_handling"
        elif ack.actor == "secondary" and ack.outcome == "taken":
            if case.state == "secondary_alerted":
                case.state = "resolved"
        return case

    def on_tick(self, now: float | None = None) -> None:
        """Advance timers and fire secondary / CareLine rungs.

        Args:
            now: Override clock reading (tests). Default uses ``clock.now()``.
        """
        t = self._clock.now() if now is None else now
        for case in list(self.cases.values()):
            if (
                case.state == "rung1_dispatched"
                and t - case.started_at_s >= SENIOR_WAIT_S
            ):
                self.on_ack(
                    AckEvent(
                        case_id=case.case_id,
                        actor="senior",
                        outcome="no_answer",
                        timestamp=int(t * 1000),
                    )
                )
            if (
                case.state == "awaiting_family"
                and case.family_wait_started_at is not None
                and t - case.family_wait_started_at >= FAMILY_WAIT_S
            ):
                self._dispatch(
                    case,
                    rung="secondary",
                    at_s=60,
                    sender=self._telegram,
                    destination=self._secondary_chat_id,
                    message=fall_message(case.event),
                )
                case.state = "secondary_alerted"
            if (
                case.state == "secondary_alerted"
                and t - case.started_at_s >= CARELINE_AT_S
            ):
                case.commands.append(
                    EscalationCommand(
                        case_id=case.case_id,
                        rung="careline",
                        at_s=180,
                        event=case.event,
                    )
                )
                case.state = "careline_alerted"

    def _dispatch(
        self,
        case: EscalationCase,
        rung: str,
        at_s: int,
        sender: Sender,
        destination: str,
        message: str,
        reply_markup: dict | None = None,
    ) -> None:
        cmd = EscalationCommand(
            case_id=case.case_id,
            rung=rung,  # type: ignore[arg-type]
            at_s=at_s,
            event=case.event,
        )
        case.commands.append(cmd)
        sender.send(destination, message, reply_markup=reply_markup)

from __future__ import annotations

from typing import Protocol

from shared.schemas import EscalationCommand
from server import notify


class Notifier(Protocol):
    def notify(self, cmd: EscalationCommand) -> bool: ...


class RecordingAdapter:
    def __init__(self) -> None:
        self.commands: list[EscalationCommand] = []

    def notify(self, cmd: EscalationCommand) -> bool:
        self.commands.append(cmd)
        return True


class TelegramAdapter:
    def notify(self, cmd: EscalationCommand) -> bool:
        text = notify.message_fall(
            cmd.event.room,
            cmd.event.timestamp,
            cmd.event.confidence,
        )
        return notify.send_sync(text)


TwilioAdapter = RecordingAdapter
SecondaryAdapter = RecordingAdapter
CareLineAdapter = RecordingAdapter

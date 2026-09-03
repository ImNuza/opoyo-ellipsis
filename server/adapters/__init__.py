"""Outbound message transports.

Adapters know how to deliver a string to a destination. They do not choose
recipients or compose alert copy. The live decision tree uses Telegram.
Twilio stays here so a voice/SMS path can plug in later without changing
the Sender protocol.
"""

from typing import Protocol


class Sender(Protocol):
    """Delivers a text payload to a destination identifier."""

    def send(
        self,
        destination: str,
        message: str,
        reply_markup: dict | None = None,
    ) -> bool:
        """Send ``message`` to ``destination``.

        Args:
            destination: Chat id, phone number, or other transport address.
            message: Already-formatted alert text.
            reply_markup: Optional transport-specific keyboard / buttons.

        Returns:
            True if the transport accepted the message.
        """

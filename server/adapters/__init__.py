"""Outbound message transports.

Adapters know how to deliver a string to a destination. They do not choose
recipients or compose alert copy.
"""

from typing import Protocol


class Sender(Protocol):
    """Delivers a text payload to a destination identifier."""

    def send(self, destination: str, message: str) -> bool:
        """Send ``message`` to ``destination``.

        Args:
            destination: Chat id, phone number, or other transport address.
            message: Already-formatted alert text.

        Returns:
            True if the transport accepted the message.
        """

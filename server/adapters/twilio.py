"""Twilio transport.

The live Voice API is not wired yet. ``send`` keeps the same signature as
Telegram so the decision tree can treat both adapters the same.
"""

from __future__ import annotations


class Twilio:
    """Stub voice/SMS sender. Always reports success."""

    def send(self, destination: str, message: str) -> bool:
        """Accept a call/SMS payload without hitting the network.

        Args:
            destination: Senior phone number (E.164).
            message: Alert body that would be spoken or texted.

        Returns:
            True. Replace with the Twilio REST client when credentials exist.
        """
        del destination, message
        return True

"""Twilio transport (not on the live ladder).

Kept so a voice/SMS path can plug in later with the same ``Sender`` signature
as Telegram. ``DecisionTree`` currently messages the senior over Telegram.
"""

from __future__ import annotations


class Twilio:
    """Stub voice/SMS sender. Always reports success."""

    def send(
        self,
        destination: str,
        message: str,
        reply_markup: dict | None = None,
    ) -> bool:
        """Accept a call/SMS payload without hitting the network.

        Args:
            destination: Senior phone number (E.164).
            message: Alert body that would be spoken or texted.
            reply_markup: Ignored. Voice/SMS has no inline keyboard.

        Returns:
            True. Replace with the Twilio REST client when credentials exist.
        """
        del destination, message, reply_markup
        return True

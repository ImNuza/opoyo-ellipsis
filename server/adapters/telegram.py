"""Telegram Bot API transport."""

from __future__ import annotations

import httpx


class Telegram:
    """Sends text to a Telegram chat via sendMessage."""

    def __init__(self, bot_token: str) -> None:
        self._bot_token = bot_token

    def send(self, destination: str, message: str) -> bool:
        """POST ``message`` to Telegram chat ``destination``.

        Args:
            destination: Telegram chat id.
            message: Alert body.

        Returns:
            True if the API responds HTTP 200.
        """
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        try:
            response = httpx.post(
                url,
                json={"chat_id": destination, "text": message},
                timeout=8.0,
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

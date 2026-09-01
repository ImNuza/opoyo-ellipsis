from __future__ import annotations

import httpx
import pytest

from server.adapters.telegram import Telegram


def test_send_posts_destination_and_message(monkeypatch):
    posted: list[dict] = []

    class FakeResponse:
        status_code = 200

    def fake_post(url, json, timeout):
        posted.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("server.adapters.telegram.httpx.post", fake_post)
    client = Telegram("bot-token")
    assert client.send("kin-chat", "hello kin") is True
    assert posted[0]["json"] == {"chat_id": "kin-chat", "text": "hello kin"}
    assert posted[0]["url"] == "https://api.telegram.org/botbot-token/sendMessage"


def test_send_returns_false_on_http_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("server.adapters.telegram.httpx.post", fake_post)
    client = Telegram("bot-token")
    assert client.send("kin-chat", "hello") is False

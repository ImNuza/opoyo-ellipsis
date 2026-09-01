from __future__ import annotations

from server.notify import chat_id, message_confirmed, message_suspect


def test_family_target_uses_group(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TARGET", "family")
    monkeypatch.setenv("TELEGRAM_CHAT_ID_FAMILY", "-100")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    assert chat_id() == "-100"


def test_personal_target(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TARGET", "personal")
    monkeypatch.setenv("TELEGRAM_CHAT_ID_FAMILY", "-100")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    assert chat_id() == "1"


def test_copy():
    assert "possible fall" in message_suspect("Phone 2")
    assert "Phone 2" in message_suspect("Phone 2")
    assert "still down" in message_confirmed("Phone 2")

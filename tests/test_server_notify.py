from __future__ import annotations

from server.notify import chat_id, message_confirmed, message_fall, message_suspect


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


def test_legacy_copy():
    assert "possible fall" in message_suspect("Phone 2")
    assert "Phone 2" in message_suspect("Phone 2")
    assert "still down" in message_confirmed("Phone 2")


def test_fall_copy_includes_room_time_confidence():
    text = message_fall("Bathroom", 1735689602123, 0.94)
    assert "Bathroom" in text
    assert "0.94" in text
    assert "2024" in text or "2025" in text or "2026" in text

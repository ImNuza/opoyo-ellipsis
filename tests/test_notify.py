from __future__ import annotations

import os

from opoyo.notify import chat_id, format_alert
from opoyo.schemas import FallEvent


def _patch_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # load_dotenv already ran in opoyo.config at import; secret() still calls os.getenv
    real = os.getenv

    def getenv(name, default=None):
        if name in env:
            return env[name]
        return real(name, default)

    monkeypatch.setattr(os, "getenv", getenv)


def test_family_target_uses_group(monkeypatch):
    _patch_env(
        monkeypatch,
        TELEGRAM_TARGET="family",
        TELEGRAM_CHAT_ID_FAMILY="-100",
        TELEGRAM_CHAT_ID="1",
    )
    assert chat_id() == "-100"


def test_personal_target(monkeypatch):
    _patch_env(
        monkeypatch,
        TELEGRAM_TARGET="personal",
        TELEGRAM_CHAT_ID_FAMILY="-100",
        TELEGRAM_CHAT_ID="1",
    )
    assert chat_id() == "1"


def test_format_alert_contains_room_and_score():
    from datetime import datetime

    ev = FallEvent.new("Kitchen", fall_score=0.87, peak_mag=9.2, ts_unix=1_700_000_000.0)
    text = format_alert(ev)
    assert "Kitchen" in text
    assert "0.87" in text
    assert datetime.fromtimestamp(ev.ts_unix).strftime("%H:%M:%S") in text

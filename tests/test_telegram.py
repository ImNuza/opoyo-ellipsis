from __future__ import annotations

import httpx
import pytest

from server.adapters.telegram import (
    Telegram,
    TelegramAck,
    parse_callback_data,
    parse_senior_text,
    parse_update,
    senior_ack_markup,
)
from server.app import _apply_telegram_ack, _newest_rung1
from server.decision_tree import DecisionTree
from shared.schemas import FallEvent
from tests.fakes import FakeClock, FakeSender


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


def test_send_includes_reply_markup(monkeypatch):
    posted: list[dict] = []

    class FakeResponse:
        status_code = 200

    def fake_post(url, json, timeout):
        posted.append(json)
        return FakeResponse()

    monkeypatch.setattr("server.adapters.telegram.httpx.post", fake_post)
    markup = senior_ack_markup("abc123")
    Telegram("bot-token").send("senior", "check", reply_markup=markup)
    assert posted[0]["reply_markup"] == markup
    assert posted[0]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == (
        "ack:abc123:yes"
    )


def test_send_returns_false_on_http_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("server.adapters.telegram.httpx.post", fake_post)
    client = Telegram("bot-token")
    assert client.send("kin-chat", "hello") is False


def test_send_returns_false_without_token():
    assert Telegram("").send("kin-chat", "hello") is False
    assert Telegram(None).send("kin-chat", "hello") is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", "yes"),
        ("YES", "yes"),
        (" y ", "yes"),
        ("ok", "yes"),
        ("okay", "yes"),
        ("fine", "yes"),
        ("I'm fine", "yes"),
        ("im fine", "yes"),
        ("no", "not_fine"),
        ("help", "not_fine"),
        ("not fine", "not_fine"),
        ("hurt", "not_fine"),
        ("hello", None),
        ("", None),
    ],
)
def test_parse_senior_text(text, expected):
    assert parse_senior_text(text) == expected


def test_parse_callback_data():
    assert parse_callback_data("ack:deadbeef:yes") == ("deadbeef", "yes")
    assert parse_callback_data("ack:deadbeef:not_fine") == ("deadbeef", "not_fine")
    assert parse_callback_data("ack::yes") is None
    assert parse_callback_data("nope") is None


def test_parse_update_callback():
    update = {
        "update_id": 1,
        "callback_query": {
            "id": "cb1",
            "data": "ack:case99:yes",
            "message": {"chat": {"id": 555}},
        },
    }
    got = parse_update(update)
    assert got == TelegramAck(
        chat_id="555",
        outcome="yes",
        case_id="case99",
        callback_query_id="cb1",
    )


def test_parse_update_text():
    update = {
        "update_id": 2,
        "message": {"chat": {"id": 555}, "text": "yes"},
    }
    got = parse_update(update)
    assert got == TelegramAck(chat_id="555", outcome="yes")


def test_parse_update_ignores_noise():
    assert parse_update({"update_id": 3, "message": {"chat": {"id": 1}, "text": "hi"}}) is None
    assert parse_update({}) is None


def test_get_updates_passes_offset(monkeypatch):
    seen: list[dict] = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"ok": True, "result": []}

    def fake_get(url, params, timeout):
        seen.append({"url": url, "params": params})
        return FakeResponse()

    monkeypatch.setattr("server.adapters.telegram.httpx.get", fake_get)
    client = Telegram("bot-token")
    client._drained = True
    client._offset = 9
    assert client.get_updates() == []
    assert seen[0]["url"].endswith("/getUpdates")
    assert seen[0]["params"]["offset"] == 9
    assert seen[0]["params"]["timeout"] == 0


def test_first_get_updates_drains_backlog(monkeypatch):
    rounds: list[list[dict]] = [
        [{"update_id": 10, "message": {"chat": {"id": 1}, "text": "yes"}}],
        [{"update_id": 11, "message": {"chat": {"id": "senior"}, "text": "yes"}}],
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return {"ok": True, "result": self._payload}

    def fake_get(url, params, timeout):
        return FakeResponse(rounds.pop(0) if rounds else [])

    monkeypatch.setattr("server.adapters.telegram.httpx.get", fake_get)
    client = Telegram("bot-token")
    assert client.get_updates() == []
    acks = client.poll_acks("senior")
    assert len(acks) == 1
    assert acks[0].outcome == "yes"
    assert acks[0].case_id == ""


def test_poll_acks_keeps_callbacks_and_senior_text(monkeypatch):
    payload = [
        {
            "update_id": 20,
            "callback_query": {
                "id": "cb",
                "data": "ack:c1:not_fine",
                "message": {"chat": {"id": 9}},
            },
        },
        {"update_id": 21, "message": {"chat": {"id": "other"}, "text": "yes"}},
        {"update_id": 22, "message": {"chat": {"id": "senior"}, "text": "no"}},
    ]

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"ok": True, "result": payload}

    monkeypatch.setattr(
        "server.adapters.telegram.httpx.get",
        lambda url, params, timeout: FakeResponse(),
    )
    client = Telegram("bot-token")
    client._drained = True
    acks = client.poll_acks("senior")
    assert [a.outcome for a in acks] == ["not_fine", "not_fine"]
    assert acks[0].case_id == "c1"
    assert acks[1].case_id == ""


def test_get_updates_returns_empty_on_http_error(monkeypatch):
    def fake_get(url, params, timeout):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("server.adapters.telegram.httpx.get", fake_get)
    client = Telegram("bot-token")
    client._drained = True
    assert client.get_updates() == []


def _event() -> FallEvent:
    return FallEvent(
        event_id="evt-tg",
        inference_id="evt-tg",
        timestamp=1,
        node_id="Phone 1",
        room=1,
        is_fall=True,
        confidence=0.95,
        threshold=0.9,
    )


def test_telegram_callback_yes_closes_case_without_http_ack(monkeypatch):
    tree = DecisionTree(
        clock=FakeClock(),
        telegram=FakeSender(),
        next_of_kin_chat_id="kin",
        secondary_chat_id="sec",
        senior_chat_id="senior",
    )
    case = tree.ingest(_event())
    posted: list[str] = []

    class FakeResponse:
        status_code = 200

    def fake_post(url, json, timeout):
        posted.append(url)
        return FakeResponse()

    monkeypatch.setattr("server.adapters.telegram.httpx.post", fake_post)
    inbox = Telegram("bot-token")
    item = TelegramAck(
        chat_id="senior",
        outcome="yes",
        case_id=case.case_id,
        callback_query_id="cb1",
    )
    updated = _apply_telegram_ack(tree, inbox, item)
    assert updated is not None
    assert updated.state == "false_alarm_closed"
    assert any(url.endswith("sendMessage") for url in posted)
    assert any(url.endswith("answerCallbackQuery") for url in posted)


def test_typed_yes_binds_newest_open_case(monkeypatch):
    tree = DecisionTree(
        clock=FakeClock(),
        telegram=FakeSender(),
        next_of_kin_chat_id="kin",
        secondary_chat_id="sec",
        senior_chat_id="senior",
    )
    case = tree.ingest(_event())
    assert _newest_rung1(tree) == case.case_id

    class FakeResponse:
        status_code = 200

    monkeypatch.setattr(
        "server.adapters.telegram.httpx.post",
        lambda url, json, timeout: FakeResponse(),
    )
    inbox = Telegram("bot-token")
    item = TelegramAck(chat_id="senior", outcome="yes", case_id="")
    updated = _apply_telegram_ack(tree, inbox, item)
    assert updated is not None
    assert updated.case_id == case.case_id
    assert updated.state == "false_alarm_closed"

"""Telegram Bot API transport for outbound alerts and inbound acks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx

from shared.config import CFG
from shared.schemas import AckActor, AckEvent, AckOutcome

TelegramOutcome = Literal["yes", "not_fine", "taken"]

_YES = frozenset({"yes", "y", "ok", "okay", "fine", "im fine", "i am fine", "i'm fine"})
_NO = frozenset({"no", "help", "hurt", "not fine", "notfine", "nope"})
_TAKEN = frozenset(
    {
        "taken",
        "on it",
        "i'm on it",
        "im on it",
        "i am on it",
        "got it",
        "handling",
        "i'll handle",
        "ill handle",
        "i will handle",
    }
)
_CALLBACK_OUTCOMES = frozenset({"yes", "not_fine", "taken"})


def _clean_text(text: str) -> tuple[str, str]:
    """Return spaced and compact lowercase forms for phrase matching."""
    raw = (text or "").strip().lower()
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch.isspace() or ch == "'")
    cleaned = " ".join(cleaned.split())
    compact = cleaned.replace(" ", "")
    return cleaned, compact


def _actor_for_outcome(outcome: TelegramOutcome) -> AckActor:
    """Map callback outcome to actor. ``taken`` is family; the rest are senior."""
    return "family" if outcome == "taken" else "senior"


@dataclass(frozen=True)
class TelegramAck:
    """One reply parsed from a Bot API update. Not a DecisionTree type."""

    chat_id: str
    outcome: TelegramOutcome
    case_id: str = ""
    callback_query_id: str | None = None
    actor: AckActor = "senior"

    def to_ack(self, case_id: str | None = None, timestamp: int | None = None) -> AckEvent:
        outcome: AckOutcome = self.outcome
        return AckEvent(
            case_id=case_id if case_id is not None else self.case_id,
            actor=self.actor,
            outcome=outcome,
            timestamp=int(time.time() * 1000) if timestamp is None else timestamp,
        )


def senior_ack_markup(case_id: str) -> dict:
    """Return the inline keyboard posted with the senior check-in.

    Args:
        case_id: Escalation case hex id, embedded in callback_data.
    """
    return {
        "inline_keyboard": [
            [
                {"text": "I'm fine", "callback_data": f"ack:{case_id}:yes"},
                {"text": "I need help", "callback_data": f"ack:{case_id}:not_fine"},
            ]
        ]
    }


def family_ack_markup(case_id: str) -> dict:
    """Return the inline keyboard posted with the family fall alert.

    Args:
        case_id: Escalation case hex id, embedded in callback_data.
    """
    return {
        "inline_keyboard": [
            [{"text": "I'm on it", "callback_data": f"ack:{case_id}:taken"}]
        ]
    }


def parse_senior_text(text: str) -> Literal["yes", "not_fine"] | None:
    """Map a typed reply to a senior outcome. Unknown text is ignored."""
    cleaned, compact = _clean_text(text)
    if not cleaned:
        return None
    if cleaned in _YES or compact in {"imfine", "iamfine"}:
        return "yes"
    if cleaned in _NO or compact == "notfine":
        return "not_fine"
    return None


def parse_family_text(text: str) -> Literal["taken"] | None:
    """Map a typed reply to family taken. Unknown text is ignored."""
    cleaned, compact = _clean_text(text)
    if not cleaned:
        return None
    if cleaned in _TAKEN or compact in {"imonit", "iamonit", "illhandle", "iwillhandle"}:
        return "taken"
    return None


def parse_callback_data(data: str) -> tuple[str, TelegramOutcome] | None:
    """Parse ``ack:{{case_id}}:yes|not_fine|taken``. Case ids are hex without colons."""
    if not data or not data.startswith("ack:"):
        return None
    rest = data[4:]
    idx = rest.rfind(":")
    if idx <= 0:
        return None
    case_id = rest[:idx]
    outcome = rest[idx + 1 :]
    if not case_id or outcome not in _CALLBACK_OUTCOMES:
        return None
    return case_id, outcome  # type: ignore[return-value]


def parse_update(update: dict) -> TelegramAck | None:
    """Turn one getUpdates payload into an ack, or None if it is unrelated."""
    if not isinstance(update, dict):
        return None
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        parsed = parse_callback_data(str(callback.get("data") or ""))
        if parsed is None:
            return None
        case_id, outcome = parsed
        message = callback.get("message") if isinstance(callback.get("message"), dict) else {}
        chat = message.get("chat") if isinstance(message, dict) else {}
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        if chat_id is None:
            sender = callback.get("from") if isinstance(callback.get("from"), dict) else {}
            chat_id = sender.get("id")
        return TelegramAck(
            chat_id=str(chat_id or ""),
            outcome=outcome,
            case_id=case_id,
            callback_query_id=str(callback.get("id") or "") or None,
            actor=_actor_for_outcome(outcome),
        )
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    text = str(message.get("text") or "")
    family = parse_family_text(text)
    senior = parse_senior_text(text) if family is None else None
    if family is None and senior is None:
        return None
    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    if family is not None:
        return TelegramAck(chat_id=str(chat_id), outcome=family, actor="family")
    if senior is None:
        return None
    return TelegramAck(chat_id=str(chat_id), outcome=senior, actor="senior")


class Telegram:
    """Sends text to a Telegram chat and polls senior / family acks via getUpdates."""

    def __init__(self, bot_token: str | None) -> None:
        self._bot_token = (bot_token or "").strip()
        self._offset: int | None = None
        self._drained = False

    @property
    def configured(self) -> bool:
        return bool(self._bot_token)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._bot_token}/{method}"

    def send(
        self,
        destination: str,
        message: str,
        reply_markup: dict | None = None,
    ) -> bool:
        """POST ``message`` to Telegram chat ``destination``.

        Args:
            destination: Telegram chat id.
            message: Alert body.
            reply_markup: Optional Bot API reply_markup (inline keyboard).

        Returns:
            True if the API responds HTTP 200.
        """
        if not self._bot_token or not destination:
            return False
        payload: dict = {"chat_id": destination, "text": message}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            response = httpx.post(
                self._url("sendMessage"),
                json=payload,
                timeout=CFG.server.telegram_timeout_s,
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def confirm_senior(self, chat_id: str, outcome: str) -> bool:
        """Short follow-up after a senior ack is applied to a case."""
        if outcome in {"yes", "fine"}:
            text = "Case closed — glad you're okay."
        else:
            text = "Help noted, family is on it."
        return self.send(chat_id, text)

    def confirm_family(self, chat_id: str) -> bool:
        """Short follow-up after family takes the case."""
        return self.send(chat_id, "Noted — you're handling this. Case closed.")

    def answer_callback(self, callback_query_id: str, text: str = "") -> bool:
        """Clear the inline-button spinner. Failures are swallowed."""
        if not self._bot_token or not callback_query_id:
            return False
        payload: dict = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        try:
            response = httpx.post(
                self._url("answerCallbackQuery"),
                json=payload,
                timeout=CFG.server.telegram_timeout_s,
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def drain_backlog(self) -> None:
        """Advance offset past unread updates without turning them into acks."""
        if self._drained:
            return
        updates = self._fetch_updates()
        if updates:
            last = updates[-1].get("update_id")
            if isinstance(last, int):
                self._offset = last + 1
        self._drained = True

    def get_updates(self) -> list[dict]:
        """Return new updates. First call drains the backlog and returns []."""
        if not self._drained:
            self.drain_backlog()
            return []
        updates = self._fetch_updates()
        if updates:
            last = updates[-1].get("update_id")
            if isinstance(last, int):
                self._offset = last + 1
        return updates

    def poll_acks(
        self,
        senior_chat_id: str,
        family_chat_id: str = "",
    ) -> list[TelegramAck]:
        """Button taps and typed replies from the senior and family chats.

        Callbacks already carry ``case_id``. Typed replies have empty
        ``case_id``; the cloud binds those to the newest matching open case.
        """
        senior = str(senior_chat_id or "")
        family = str(family_chat_id or "")
        found: list[TelegramAck] = []
        for update in self.get_updates():
            parsed = parse_update(update)
            if parsed is None:
                continue
            if parsed.callback_query_id:
                found.append(parsed)
                continue
            if parsed.actor == "senior" and senior and parsed.chat_id == senior:
                found.append(parsed)
            elif parsed.actor == "family" and family and parsed.chat_id == family:
                found.append(parsed)
        return found

    def _fetch_updates(self) -> list[dict]:
        if not self._bot_token:
            return []
        params: dict = {"timeout": 0}
        if self._offset is not None:
            params["offset"] = self._offset
        try:
            response = httpx.get(
                self._url("getUpdates"),
                params=params,
                timeout=CFG.server.telegram_timeout_s,
            )
        except httpx.HTTPError:
            return []
        if response.status_code != 200:
            return []
        try:
            body = response.json()
        except ValueError:
            return []
        if not isinstance(body, dict) or not body.get("ok"):
            return []
        result = body.get("result")
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

"""Cloud HTTP API: ingest fall events and drive the escalation ladder."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from server.adapters.telegram import Telegram, TelegramAck
from server.decision_tree import DecisionTree, SystemClock
from shared.config import CFG
from shared.paths import REPO_ROOT
from shared.schemas import AckEvent, EscalationCase, FallEvent


def load_cloud_env() -> dict[str, str]:
    """Load ``.env`` once for Telegram chat ids.

    Returns:
        Token and destination map used to construct the live tree.
    """
    load_dotenv(REPO_ROOT / ".env")
    return {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"),
        "next_of_kin_chat_id": os.environ.get("TELEGRAM_CHAT_ID_NEXT_OF_KIN"),
        "secondary_chat_id": os.environ.get("TELEGRAM_CHAT_ID_SECONDARY"),
        "senior_chat_id": os.environ.get("TELEGRAM_CHAT_ID_SENIOR"),
    }


def build_tree() -> DecisionTree:
    """Construct the live decision tree from process environment."""
    cfg = load_cloud_env()
    return DecisionTree(
        clock=SystemClock(),
        telegram=Telegram(cfg["telegram_bot_token"]),
        next_of_kin_chat_id=cfg["next_of_kin_chat_id"],
        secondary_chat_id=cfg["secondary_chat_id"],
        senior_chat_id=cfg["senior_chat_id"],
    )


def _newest_open(
    tree: DecisionTree,
    states: frozenset[str],
) -> str | None:
    """Return the newest case in ``states``.

    Used when a typed Telegram reply has no ``case_id`` on the callback.
    """
    open_cases = [case for case in tree.cases.values() if case.state in states]
    if not open_cases:
        return None
    open_cases.sort(key=lambda c: c.started_at_s, reverse=True)
    return open_cases[0].case_id


def _newest_rung1(tree: DecisionTree) -> str | None:
    """Newest open senior check-in. Used when the senior typed yes/no."""
    return _newest_open(tree, frozenset({"rung1_dispatched"}))


def _newest_family_open(tree: DecisionTree) -> str | None:
    """Newest case family can still take."""
    return _newest_open(tree, frozenset({"rung1_dispatched", "awaiting_family"}))


def _apply_telegram_ack(
    tree: DecisionTree,
    telegram: Telegram,
    item: TelegramAck,
) -> EscalationCase | None:
    """Bind a Telegram reply to a case and confirm back in chat.

    Args:
        tree: Live decision tree.
        telegram: Bot client used for confirmations.
        item: Parsed button tap or typed reply.

    Returns:
        Updated case, or None if no matching open case exists.
    """
    if item.actor == "family":
        case_id = item.case_id or _newest_family_open(tree)
    else:
        case_id = item.case_id or _newest_rung1(tree)
    if not case_id:
        return None
    case = tree.on_ack(item.to_ack(case_id=case_id))
    if case is None:
        return None
    if item.actor == "family":
        if case.state == "family_handling":
            telegram.confirm_family(item.chat_id)
            if item.callback_query_id:
                telegram.answer_callback(item.callback_query_id, "You're on it.")
        return case
    telegram.confirm_senior(item.chat_id, item.outcome)
    if item.callback_query_id:
        note = (
            "Glad you're okay."
            if item.outcome in {"yes", "fine"}
            else "Help is on the way."
        )
        telegram.answer_callback(item.callback_query_id, note)
    return case


def create_app(tree: DecisionTree | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        tree: Injected tree for tests. If omitted, env is loaded and a live
            tree is constructed.

    Returns:
        Configured FastAPI application.
    """
    live_telegram: Telegram | None = None
    senior_chat_id = ""
    family_chat_id = ""
    if tree is None:
        cfg = load_cloud_env()
        live_telegram = Telegram(cfg["telegram_bot_token"])
        senior_chat_id = cfg["senior_chat_id"] or ""
        family_chat_id = cfg["next_of_kin_chat_id"] or ""
        escalation = DecisionTree(
            clock=SystemClock(),
            telegram=live_telegram,
            next_of_kin_chat_id=cfg["next_of_kin_chat_id"],
            secondary_chat_id=cfg["secondary_chat_id"],
            senior_chat_id=cfg["senior_chat_id"],
        )
    else:
        escalation = tree
    app = FastAPI(title="OPOYO cloud")
    app.state.tree = escalation
    app.state.telegram = live_telegram
    app.state.senior_chat_id = senior_chat_id
    app.state.family_chat_id = family_chat_id

    @app.post("/events")
    async def ingest_event(event: FallEvent) -> JSONResponse:
        """Open (or reuse) a case. 202 + EscalationCase. Fires t+0 alerts."""
        case = escalation.ingest(event)
        return JSONResponse(case.model_dump(), status_code=202)

    @app.post("/cases/{case_id}/ack")
    async def ack_case(case_id: str, body: AckEvent) -> JSONResponse:
        payload = (
            body
            if body.case_id == case_id
            else body.model_copy(update={"case_id": case_id})
        )
        case = escalation.on_ack(payload)
        if case is None:
            return JSONResponse({"error": "unknown case"}, status_code=404)
        return JSONResponse(case.model_dump())

    @app.get("/cases/{case_id}")
    async def get_case(case_id: str) -> JSONResponse:
        case = escalation.cases.get(case_id)
        if case is None:
            return JSONResponse({"error": "unknown case"}, status_code=404)
        return JSONResponse(case.model_dump())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.on_event("startup")
    async def _ticks() -> None:
        # Ladder timers and Telegram polling live on this loop, not on an HTTP route.
        async def loop() -> None:
            while True:
                await asyncio.sleep(CFG.server.tick_s)
                try:
                    escalation.on_tick()
                    inbox = app.state.telegram
                    if inbox is not None and inbox.configured:
                        for item in inbox.poll_acks(
                            app.state.senior_chat_id or "",
                            app.state.family_chat_id or "",
                        ):
                            _apply_telegram_ack(escalation, inbox, item)
                except Exception:
                    continue

        asyncio.create_task(loop())

    return app


# uvicorn / fastapi CLI: `server.app:app`
app = create_app()


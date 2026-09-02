"""Cloud HTTP API: ingest fall events and drive the escalation ladder."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from server.adapters.telegram import Telegram
from server.adapters.twilio import Twilio
from server.decision_tree import DecisionTree, SystemClock
from shared.schemas import AckEvent, FallEvent


def load_cloud_env() -> dict[str, str]:
    """Load ``.env`` once and require Telegram + senior-phone settings.

    Returns:
        Token and destination map used to construct the live tree.
    """
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"),
        "next_of_kin_chat_id": os.environ.get("TELEGRAM_CHAT_ID_NEXT_OF_KIN"),
        "secondary_chat_id": os.environ.get("TELEGRAM_CHAT_ID_SECONDARY"),
        "senior_phone": os.environ.get("SENIOR_PHONE"),
    }


def build_tree() -> DecisionTree:
    """Construct the live decision tree from process environment."""
    cfg = load_cloud_env()
    return DecisionTree(
        clock=SystemClock(),
        telegram=Telegram(cfg["telegram_bot_token"]),
        twilio=Twilio(),
        next_of_kin_chat_id=cfg["next_of_kin_chat_id"],
        secondary_chat_id=cfg["secondary_chat_id"],
        senior_phone=cfg["senior_phone"],
    )


def create_app(tree: DecisionTree | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        tree: Injected tree for tests. If omitted, env is loaded and a live
            tree is constructed.

    Returns:
        Configured FastAPI application.
    """
    escalation = tree if tree is not None else build_tree()
    app = FastAPI(title="OPOYO cloud")
    app.state.tree = escalation

    @app.post("/events")
    async def ingest_event(event: FallEvent) -> JSONResponse:
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
        async def loop() -> None:
            while True:
                await asyncio.sleep(0.5)
                escalation.on_tick()

        asyncio.create_task(loop())

    return app


app = create_app()


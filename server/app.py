from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from shared.schemas import AckEvent, FallEvent
from server import notify
from server.adapters import RecordingAdapter, TelegramAdapter
from server.tree import EscalationTree, SystemClock


def build_tree() -> EscalationTree:
    return EscalationTree(
        clock=SystemClock(),
        telegram=TelegramAdapter(),
        twilio=RecordingAdapter(),
        secondary=RecordingAdapter(),
        careline=RecordingAdapter(),
    )


def create_app(tree: EscalationTree | None = None) -> FastAPI:
    notify.load_env()
    if tree is None:
        notify.require_live()
        escalation = build_tree()
    else:
        escalation = tree
    app = FastAPI(title="OPOYO cloud")
    app.state.tree = escalation

    @app.post("/events")
    async def ingest_event(event: FallEvent) -> JSONResponse:
        case = escalation.ingest(event)
        return JSONResponse(case.model_dump(), status_code=202)

    @app.post("/cases/{case_id}/ack")
    async def ack_case(case_id: str, body: AckEvent) -> JSONResponse:
        payload = body if body.case_id == case_id else body.model_copy(update={"case_id": case_id})
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


def __getattr__(name: str) -> FastAPI:
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

UDP_HOST = "0.0.0.0"
UDP_PORT = 9000
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_HISTORY = 400


class Hub:
    def __init__(self) -> None:
        self.latest: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.clients: set[WebSocket] = set()
        self.packets = 0
        self.last_packet_at: float | None = None
        self.lock = asyncio.Lock()

    async def ingest(self, packet: dict[str, Any]) -> None:
        now = time.time()
        packet["recv_t"] = int(now * 1000)
        async with self.lock:
            self.latest = packet
            self.history.append(packet)
            self.packets += 1
            self.last_packet_at = now
            clients = list(self.clients)
        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(packet)
            except Exception:
                stale.append(ws)
        if stale:
            async with self.lock:
                for ws in stale:
                    self.clients.discard(ws)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            rate = 0.0
            if self.last_packet_at is not None:
                window = [p for p in self.history if (time.time() - p.get("recv_t", 0) / 1000) <= 1.5]
                rate = len(window) / 1.5
            return {
                "latest": self.latest,
                "packets": self.packets,
                "hz": round(rate, 1),
                "history": list(self.history)[-80:],
            }

    async def add(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)


hub = Hub()
app = FastAPI(title="OPOYO receiver")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SampleProtocol(asyncio.DatagramProtocol):
    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload["from"] = f"{addr[0]}:{addr[1]}"
        loop = asyncio.get_running_loop()
        loop.create_task(hub.ingest(payload))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def state() -> dict[str, Any]:
    return await hub.snapshot()


@app.websocket("/ws")
async def ws_feed(ws: WebSocket) -> None:
    await ws.accept()
    await hub.add(ws)
    try:
        await ws.send_json(await hub.snapshot())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.remove(ws)
    except Exception:
        await hub.remove(ws)


@app.on_event("startup")
async def start_udp() -> None:
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(
        SampleProtocol,
        local_addr=(UDP_HOST, UDP_PORT),
    )

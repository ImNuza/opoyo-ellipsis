from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from opoyo.config import CFG, ROOT
from opoyo.notify import format_alert, send_telegram
from opoyo.pipeline import LivePipeline
from opoyo.schemas import FallEvent

UDP_HOST = "0.0.0.0"
UDP_PORT = 9000
DASH = ROOT / "dashboard"
DATA = ROOT / "data"
MAX_NODES = 5
MAX_HISTORY = 220
LIVE_S = 2.5
RATE_S = 1.5


def _num(packet: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(packet.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def _node_id(packet: dict[str, Any], addr: tuple[str, int]) -> str:
    raw = packet.get("id") or packet.get("node")
    if isinstance(raw, str):
        cleaned = raw.strip()[:64]
        if cleaned:
            return cleaned
    return addr[0]


class Node:
    def __init__(self, node_id: str, slot: int, model: str) -> None:
        self.id = node_id
        self.slot = slot
        self.model = model
        self.name_override: str | None = None
        self.from_addr = ""
        self.latest: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.packets = 0
        self.last_at: float | None = None

    @property
    def name(self) -> str:
        return self.name_override or f"Phone {self.slot}"

    def live(self, now: float) -> bool:
        return self.last_at is not None and (now - self.last_at) <= LIVE_S

    def hz(self, now: float) -> float:
        window = [p for p in self.history if (now - p.get("recv_t", 0) / 1000.0) <= RATE_S]
        return round(len(window) / RATE_S, 1)

    def public(self, now: float) -> dict[str, Any]:
        latest = self.latest or {}
        return {
            "id": self.id,
            "slot": self.slot,
            "empty": False,
            "name": self.name,
            "model": self.model,
            "live": self.live(now),
            "packets": self.packets,
            "hz": self.hz(now),
            "ax": _num(latest, "ax"),
            "ay": _num(latest, "ay"),
            "az": _num(latest, "az"),
            "mag": _num(latest, "mag"),
            "db": _num(latest, "db", -120.0),
        }


class Hub:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.log: deque[dict[str, Any]] = deque(maxlen=50)
        self.rms_hist: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.dropped = 0
        self.recording: dict[str, Any] | None = None
        self._rung2: asyncio.Task | None = None
        self.pipeline = LivePipeline(CFG, on_event=self._on_event)

    def _on_event(self, ev: FallEvent) -> None:
        payload = ev.public()
        self.log.appendleft(payload)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._broadcast({"k": "event", "event": payload}))
            if ev.state == "alert":
                loop.create_task(self._escalate(ev))

    async def _escalate(self, ev: FallEvent) -> None:
        dry = bool(CFG.escalate.dry_run)
        await send_telegram(format_alert(ev, 1), dry_run=dry)
        delay = float(CFG.escalate.rung2_delay_s)

        async def rung2():
            await asyncio.sleep(delay)
            await send_telegram(format_alert(ev, 2), dry_run=dry)

        if self._rung2 and not self._rung2.done():
            self._rung2.cancel()
        self._rung2 = asyncio.create_task(rung2())

    async def acknowledge(self, event_id: str = "") -> dict[str, Any]:
        if self._rung2 and not self._rung2.done():
            self._rung2.cancel()
        ev = self.pipeline.machine.acknowledge(event_id) if event_id else None
        if not event_id:
            self.pipeline.cancel()
        return {"ok": True, "detect": self.pipeline.public()}

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        async with self.lock:
            clients = list(self.clients)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        if stale:
            async with self.lock:
                for ws in stale:
                    self.clients.discard(ws)

    async def ingest(self, packet: dict[str, Any]) -> None:
        now = time.time()
        packet["recv_t"] = int(now * 1000)
        nid = packet.get("_nid")
        if not isinstance(nid, str) or not nid:
            return
        model = packet.get("model")
        if not isinstance(model, str) or not model.strip():
            model = "iPhone"
        model = model.strip()[:48]
        async with self.lock:
            node = self.nodes.get(nid)
            if node is None:
                if len(self.nodes) >= MAX_NODES:
                    self.dropped += 1
                    return
                node = Node(nid, len(self.nodes) + 1, model)
                self.nodes[nid] = node
            node.model = model
            node.from_addr = str(packet.get("from", ""))
            node.latest = packet
            node.history.append(packet)
            node.packets += 1
            node.last_at = now
            rec = self.recording
        self.pipeline.feed(
            _num(packet, "ax"),
            _num(packet, "ay"),
            _num(packet, "az"),
            db=_num(packet, "db", -120.0),
            mag_g=_num(packet, "mag"),
            node_id=nid,
        )
        pub = self.pipeline.public()
        self.rms_hist.append({"t": now, "rms": pub["rms"], "thr": pub["threshold"]})
        if rec is not None:
            rec["rows"].append(
                {
                    "t": packet.get("t"),
                    "ax": _num(packet, "ax"),
                    "ay": _num(packet, "ay"),
                    "az": _num(packet, "az"),
                    "mag": _num(packet, "mag"),
                    "db": _num(packet, "db", -120.0),
                }
            )
        await self._broadcast(
            {
                "k": "tick",
                "id": nid,
                "detect": pub,
                "mag": _num(packet, "mag"),
                "db": _num(packet, "db", -120.0),
                "ax": _num(packet, "ax"),
                "ay": _num(packet, "ay"),
                "az": _num(packet, "az"),
            }
        )

    async def snapshot(self) -> dict[str, Any]:
        now = time.time()
        async with self.lock:
            slots = []
            by_slot = {n.slot: n for n in self.nodes.values()}
            for slot in range(1, MAX_NODES + 1):
                node = by_slot.get(slot)
                slots.append({"slot": slot, "empty": True} if node is None else node.public(now))
        return {
            "k": "state",
            "slots": slots,
            "detect": self.pipeline.public(),
            "log": list(self.log),
            "rms_hist": list(self.rms_hist),
            "dropped": self.dropped,
            "recording": None
            if self.recording is None
            else {"label": self.recording["label"], "n": len(self.recording["rows"])},
        }


hub = Hub()
app = FastAPI(title="OPOYO")
app.mount("/static", StaticFiles(directory=str(DASH)), name="static")


class SampleProtocol(asyncio.DatagramProtocol):
    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload["from"] = f"{addr[0]}:{addr[1]}"
        payload["_nid"] = _node_id(payload, addr)
        asyncio.get_running_loop().create_task(hub.ingest(payload))


class RenameBody(BaseModel):
    name: str = ""


class RecordBody(BaseModel):
    label: str
    floor: str = ""
    room: str = ""
    distance_m: float | None = None
    notes: str = ""


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(DASH / "index.html")


@app.get("/api/state")
async def state() -> dict[str, Any]:
    return await hub.snapshot()


@app.post("/api/cancel")
async def cancel() -> dict[str, Any]:
    hub.pipeline.cancel()
    if hub._rung2 and not hub._rung2.done():
        hub._rung2.cancel()
    return {"detect": hub.pipeline.public()}


@app.post("/api/ack")
async def ack() -> dict[str, Any]:
    return await hub.acknowledge()


@app.post("/api/record/start")
async def record_start(body: RecordBody) -> dict[str, Any]:
    label = body.label.strip().lower().replace(" ", "")
    if not label:
        return JSONResponse({"error": "label required"}, status_code=400)
    hub.recording = {
        "label": label,
        "floor": body.floor,
        "room": body.room,
        "distance_m": body.distance_m,
        "notes": body.notes,
        "rows": [],
        "t0": time.time(),
    }
    return {"ok": True, "label": label}


@app.post("/api/record/stop")
async def record_stop() -> dict[str, Any]:
    rec = hub.recording
    hub.recording = None
    if rec is None:
        return JSONResponse({"error": "not recording"}, status_code=400)
    DATA.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = DATA / f"{rec['label']}_{n:02d}.csv"
        if not path.exists():
            break
        n += 1
    lines = ["t,ax,ay,az,mag,db\n"]
    for r in rec["rows"]:
        lines.append(
            f"{r['t']},{r['ax']},{r['ay']},{r['az']},{r['mag']},{r['db']}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    meta = DATA / "metadata.csv"
    if not meta.exists():
        meta.write_text(
            "filename,label,floor,room,distance_m,phone_model,case,footwear,object,notes\n",
            encoding="utf-8",
        )
    dist = rec["distance_m"] if rec["distance_m"] is not None else ""
    with meta.open("a", encoding="utf-8") as f:
        f.write(
            f"{path.name},{rec['label']},{rec['floor']},{rec['room']},{dist},,,,{rec['notes']}\n"
        )
    return {"ok": True, "path": str(path), "rows": len(rec["rows"])}


@app.websocket("/ws")
async def ws_feed(ws: WebSocket) -> None:
    await ws.accept()
    async with hub.lock:
        hub.clients.add(ws)
    try:
        await ws.send_json(await hub.snapshot())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    async with hub.lock:
        hub.clients.discard(ws)


@app.on_event("startup")
async def start_udp() -> None:
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(SampleProtocol, local_addr=(UDP_HOST, UDP_PORT))

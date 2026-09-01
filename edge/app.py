from __future__ import annotations

import asyncio
import json
import time
import os
from collections import deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from edge.gate import EscalationGate, HttpCloudClient
from edge.infer import Classifier, StubCnn
from edge.log import InferenceLog
from edge.window import WindowBuilder
from shared.schemas import EdgeConfig, SensorSample

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

UDP_HOST = "0.0.0.0"
UDP_PORT = 9000
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent / "data"
NAMES_PATH = DATA_DIR / "names.json"
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


def default_edge_config() -> EdgeConfig:
    return EdgeConfig(
        escalate_min_confidence=os.environ.get("EDGE_ESCALATE_MIN_CONFIDENCE"),
        cloud_url=os.environ.get("EDGE_CLOUD_URL"),
    )


class Node:
    def __init__(self, node_id: str, slot: int, model: str, short: str) -> None:
        self.id = node_id
        self.slot = slot
        self.model = model
        self.short = short
        self.name_override: str | None = None
        self.from_addr = ""
        self.latest: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.packets = 0
        self.last_at: float | None = None

    @property
    def name(self) -> str:
        if self.name_override:
            return self.name_override
        return f"Phone {self.slot}"

    def live(self, now: float) -> bool:
        return self.last_at is not None and (now - self.last_at) <= LIVE_S

    def hz(self, now: float) -> float:
        if self.last_at is None:
            return 0.0
        window = [
            p
            for p in self.history
            if (now - p.get("recv_t", 0) / 1000.0) <= RATE_S
        ]
        return round(len(window) / RATE_S, 1)

    def public(self, now: float) -> dict[str, Any]:
        latest = self.latest or {}
        return {
            "id": self.id,
            "slot": self.slot,
            "empty": False,
            "name": self.name,
            "model": self.model,
            "short": self.short,
            "from": self.from_addr,
            "live": self.live(now),
            "packets": self.packets,
            "hz": self.hz(now),
            "ax": _num(latest, "ax"),
            "ay": _num(latest, "ay"),
            "az": _num(latest, "az"),
            "mag": _num(latest, "mag"),
            "db": _num(latest, "db", -120.0),
            "t": latest.get("t"),
        }


class Hub:
    def __init__(
        self,
        cfg: EdgeConfig,
        classifier: Classifier,
        gate: EscalationGate,
        store: InferenceLog,
    ) -> None:
        self.cfg = cfg
        self.classifier = classifier
        self.gate = gate
        self.store = store
        self.nodes: dict[str, Node] = {}
        self.windows: dict[str, WindowBuilder] = {}
        self.combined_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.clients: set[WebSocket] = set()
        self.dropped = 0
        self.packets = 0
        self.lock = asyncio.Lock()
        self.names: dict[str, str] = {}
        self.latest_inference: dict[str, Any] | None = None
        self._load_names()

    def _load_names(self) -> None:
        if not NAMES_PATH.exists():
            return
        try:
            data = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.names = {str(k): str(v) for k, v in data.items() if str(v).strip()}

    def _save_names(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        NAMES_PATH.write_text(json.dumps(self.names, indent=2), encoding="utf-8")

    def _combined(self, now: float) -> dict[str, Any]:
        live = [n for n in self.nodes.values() if n.live(now) and n.latest]
        if not live:
            return {"mag": 0.0, "db": -120.0, "live": 0, "hz": 0.0, "packets": self.packets}
        mags = [_num(n.latest or {}, "mag") for n in live]
        dbs = [_num(n.latest or {}, "db", -120.0) for n in live]
        hz = sum(n.hz(now) for n in live)
        return {
            "mag": max(mags),
            "db": max(dbs),
            "live": len(live),
            "hz": round(hz, 1),
            "packets": self.packets,
        }

    def inference_public(self) -> dict[str, Any]:
        return {
            "k": "inference",
            "latest": self.latest_inference,
            "log": [row.model_dump() for row in self.store.tail(50)],
            "cfg": self.cfg.model_dump(),
        }

    def _builder(self, node_id: str, room: str) -> WindowBuilder:
        builder = self.windows.get(node_id)
        if builder is None:
            builder = WindowBuilder(
                window_s=self.cfg.window_s,
                hop_s=self.cfg.hop_s,
                hz=50.0,
                room=room,
            )
            self.windows[node_id] = builder
        return builder

    async def ingest(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        now = time.time()
        packet["recv_t"] = int(now * 1000)
        addr = packet.get("from", "")
        nid = packet.get("_nid")
        if not isinstance(nid, str) or not nid:
            return None
        try:
            sample = SensorSample.model_validate(packet)
        except ValidationError:
            return None
        model = packet.get("model")
        if not isinstance(model, str) or not model.strip():
            model = "iPhone"
        model = model.strip()[:48]
        short = nid.replace("-", "")[:4]

        async with self.lock:
            node = self.nodes.get(nid)
            if node is None:
                if len(self.nodes) >= MAX_NODES:
                    self.dropped += 1
                    return None
                slot = len(self.nodes) + 1
                node = Node(nid, slot, model, short)
                override = self.names.get(nid)
                if override:
                    node.name_override = override
                self.nodes[nid] = node
            node.model = model
            node.short = short
            node.from_addr = str(addr)
            node.latest = packet
            node.history.append(packet)
            node.packets += 1
            node.last_at = now
            self.packets += 1
            combined = self._combined(now)
            self.combined_history.append(
                {
                    "t": packet.get("t"),
                    "recv_t": packet["recv_t"],
                    "mag": combined["mag"],
                    "db": combined["db"],
                }
            )
            window = self._builder(nid, node.name).push(sample, room=node.name)
            if window is not None:
                result = self.classifier.infer(window)
                self.gate.handle(result)
                self.latest_inference = result.model_dump()
            inference = self.inference_public()
            tick = {
                "k": "tick",
                "id": nid,
                "name": node.name,
                "slot": node.slot,
                "model": node.model,
                "short": node.short,
                "from": node.from_addr,
                "ax": _num(packet, "ax"),
                "ay": _num(packet, "ay"),
                "az": _num(packet, "az"),
                "mag": _num(packet, "mag"),
                "db": _num(packet, "db", -120.0),
                "t": packet.get("t"),
                "packets": node.packets,
                "hz": node.hz(now),
                "combined": combined,
                "inference": inference,
            }
            clients = list(self.clients)

        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(tick)
            except Exception:
                stale.append(ws)
        if stale:
            async with self.lock:
                for ws in stale:
                    self.clients.discard(ws)
        return tick

    async def snapshot(self) -> dict[str, Any]:
        now = time.time()
        async with self.lock:
            slots: list[dict[str, Any]] = []
            by_slot = {n.slot: n for n in self.nodes.values()}
            histories: dict[str, list[dict[str, Any]]] = {}
            for slot in range(1, MAX_NODES + 1):
                node = by_slot.get(slot)
                if node is None:
                    slots.append({"slot": slot, "empty": True})
                    continue
                slots.append(node.public(now))
                histories[node.id] = [
                    {
                        "t": p.get("t"),
                        "ax": _num(p, "ax"),
                        "ay": _num(p, "ay"),
                        "az": _num(p, "az"),
                        "mag": _num(p, "mag"),
                        "db": _num(p, "db", -120.0),
                    }
                    for p in list(node.history)[-MAX_HISTORY:]
                ]
            return {
                "k": "state",
                "slots": slots,
                "combined": self._combined(now),
                "combined_history": list(self.combined_history)[-MAX_HISTORY:],
                "histories": histories,
                "dropped": self.dropped,
                "max": MAX_NODES,
                "inference": self.inference_public(),
            }

    async def rename(self, node_id: str, name: str) -> dict[str, Any] | None:
        cleaned = name.strip()[:32]
        async with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                return None
            if cleaned and cleaned.lower() != f"phone {node.slot}".lower():
                node.name_override = cleaned
                self.names[node_id] = cleaned
            else:
                node.name_override = None
                self.names.pop(node_id, None)
            self._save_names()
            now = time.time()
            public = node.public(now)
        return public

    async def set_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            current = self.cfg.model_dump()
            current.update({k: v for k, v in updates.items() if v is not None})
            self.cfg = EdgeConfig.model_validate(current)
            self.gate.threshold = self.cfg.escalate_min_confidence
            if hasattr(self.gate.client, "base_url") and updates.get("cloud_url"):
                self.gate.client.base_url = self.cfg.cloud_url.rstrip("/")
            payload = self.inference_public()
            clients = list(self.clients)
        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        if stale:
            async with self.lock:
                for ws in stale:
                    self.clients.discard(ws)
        return payload

    async def add(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)


class RenameBody(BaseModel):
    name: str = ""


class EdgeConfigBody(BaseModel):
    escalate_min_confidence: float | None = None
    window_s: float | None = None
    hop_s: float | None = None
    cloud_url: str | None = None


def create_app(
    *,
    classifier: Classifier | None = None,
    cloud_client: Any | None = None,
    log_path: Path | None = None,
    enable_udp: bool | None = None,
    cfg: EdgeConfig | None = None,
) -> FastAPI:
    config = cfg or default_edge_config()
    store = InferenceLog(log_path or (DATA_DIR / "inference.jsonl"))
    client = cloud_client or HttpCloudClient(config.cloud_url)
    gate = EscalationGate(threshold=config.escalate_min_confidence, client=client, store=store)
    hub = Hub(
        cfg=config,
        classifier=classifier or StubCnn(),
        gate=gate,
        store=store,
    )
    udp_on = os.environ.get("EDGE_ENABLE_UDP") if enable_udp is None else enable_udp

    app = FastAPI(title="OPOYO edge")
    app.state.hub = hub
    if STATIC_DIR.exists():
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
            payload["_nid"] = _node_id(payload, addr)
            loop = asyncio.get_running_loop()
            loop.create_task(hub.ingest(payload))

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return await hub.snapshot()

    @app.post("/api/phones/{node_id}/name")
    async def rename_phone(node_id: str, body: RenameBody) -> JSONResponse:
        public = await hub.rename(node_id, body.name)
        if public is None:
            return JSONResponse({"error": "unknown phone"}, status_code=404)
        return JSONResponse(public)

    @app.post("/api/edge/config")
    async def edge_config(body: EdgeConfigBody) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if body.escalate_min_confidence is not None:
            updates["escalate_min_confidence"] = min(
                1.0, max(0.0, float(body.escalate_min_confidence))
            )
        if body.window_s is not None:
            updates["window_s"] = float(body.window_s)
        if body.hop_s is not None:
            updates["hop_s"] = float(body.hop_s)
        if body.cloud_url is not None:
            updates["cloud_url"] = body.cloud_url.strip()
        return await hub.set_config(updates)

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
        if not udp_on:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.create_datagram_endpoint(
                SampleProtocol,
                local_addr=(UDP_HOST, UDP_PORT),
            )
        except OSError:
            return

    return app


def __getattr__(name: str) -> FastAPI:
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

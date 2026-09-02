"""Edge FastAPI: UDP ingest, windowed inference, dashboard, gated POST to cloud.

HTTP :8000 serves the dashboard and /ws ticks. Phones send SensorSample JSON to
UDP :9000. Raw axes never leave this process.
"""

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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from edge.gate import EscalationGate, HttpCloudClient
from edge.infer import Classifier, StubCnn
from edge.log import InferenceLog
from edge.window import WindowBuilder
from shared.schemas import SensorSample

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

UDP_HOST = "0.0.0.0"
UDP_PORT = 9000
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_NODES = 5
MAX_HISTORY = 220
LIVE_S = 2.5  # phone drops off the dashboard after this many seconds of silence
RATE_S = 1.5  # packet-rate window for the Hz readout
ESCALATE_MIN_CONFIDENCE = 0.90
WINDOW_S = 2.0
HOP_S = 1.0
CLOUD_URL = "http://127.0.0.1:8001"


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
    """One physical phone. UUID is the hub key; slot is Phone N / room N."""

    def __init__(self, node_id: str, slot: int, model: str, short: str) -> None:
        self.id = node_id
        self.slot = slot
        self.model = model
        self.short = short
        self.from_addr = ""
        self.latest: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.packets = 0
        self.last_at: float | None = None

    @property
    def name(self) -> str:
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
    """In-memory coordinator: nodes, windows, classifier, gate, WebSocket clients."""

    def __init__(
        self,
        classifier: Classifier,
        gate: EscalationGate,
        store: InferenceLog,
    ) -> None:
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
        self.latest_inference: dict[str, Any] | None = None

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
            "cfg": {
                "escalate_min_confidence": ESCALATE_MIN_CONFIDENCE,
                "window_s": WINDOW_S,
                "hop_s": HOP_S,
                "cloud_url": CLOUD_URL,
            },
        }

    def _builder(self, key: str, node: Node) -> WindowBuilder:
        builder = self.windows.get(key)
        if builder is None:
            builder = WindowBuilder(
                window_s=WINDOW_S,
                hop_s=HOP_S,
                hz=50.0,
                node_id=node.name,
                room=node.slot,
            )
            self.windows[key] = builder
        return builder

    async def ingest(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        """Validate a sample, update the node, maybe classify, fan out a WS tick.

        Requires ``_nid`` (UDP sets this from packet ``id``). Missing/invalid
        samples are dropped with no log line.
        """
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
            # Inference identity is Phone N / room slot, not the device UUID.
            window = self._builder(nid, node).push(
                sample, node_id=node.name, room=node.slot
            )
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

    async def add(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)


def create_app(
    *,
    classifier: Classifier | None = None,
    cloud_client: Any | None = None,
    log_path: Path | None = None,
    enable_udp: bool | None = None,
    udp_host: str | None = None,
    udp_port: int | None = None,
) -> FastAPI:
    """Build the edge app. Tests inject classifier, cloud_client, log_path, UDP."""
    store = InferenceLog(log_path or (DATA_DIR / "inference.jsonl"))
    client = cloud_client or HttpCloudClient(CLOUD_URL)
    gate = EscalationGate(threshold=ESCALATE_MIN_CONFIDENCE, client=client, store=store)
    hub = Hub(
        classifier=classifier or StubCnn(),
        gate=gate,
        store=store,
    )
    # Unset → listen. Explicit 0/false/no → off. Tests pass enable_udp=False.
    if enable_udp is None:
        raw = os.environ.get("EDGE_ENABLE_UDP")
        if raw is None:
            udp_on = True
        else:
            udp_on = raw.strip().lower() not in {"0", "false", "no", ""}
    else:
        udp_on = bool(enable_udp)
    bind_host = UDP_HOST if udp_host is None else udp_host
    bind_port = UDP_PORT if udp_port is None else udp_port

    app = FastAPI(title="OPOYO edge")
    app.state.hub = hub
    app.state.udp_port = None
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
            # Hub.ingest requires _nid; phones only send id.
            payload["_nid"] = _node_id(payload, addr)
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
        # First frame is a full snapshot; later frames are per-packet ticks.
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
            print("[edge] UDP disabled (set EDGE_ENABLE_UDP=1)")
            return
        loop = asyncio.get_running_loop()
        transport, _protocol = await loop.create_datagram_endpoint(
            SampleProtocol,
            local_addr=(bind_host, bind_port),
        )
        sockname = transport.get_extra_info("sockname")
        app.state.udp_port = int(sockname[1]) if sockname else bind_port
        print(f"[edge] UDP listening on {bind_host}:{app.state.udp_port}")

    return app


# uvicorn / fastapi CLI look up this name: `edge.app:app`
app = create_app()


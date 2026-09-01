from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.detect import Detector, DetectorConfig
from server import notify

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
notify.load_env()

UDP_HOST = "0.0.0.0"
UDP_PORT = 9000
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent / "data"
NAMES_PATH = DATA_DIR / "names.json"
DETECT_PATH = DATA_DIR / "detect.json"
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def detector_config() -> DetectorConfig:
    return DetectorConfig(
        peak_g=_env_float("FUSION_PEAK_G", 0.4),
        quiet_g=_env_float("FUSION_QUIET_G", 0.05),
        rise_ms=_env_float("FUSION_RISE_MS", 80.0),
        decay_min_ms=_env_float("FUSION_DECAY_MIN_MS", 150.0),
        decay_max_ms=_env_float("FUSION_DECAY_MAX_MS", 400.0),
        decay_hold_ms=_env_float("FUSION_DECAY_HOLD_MS", 80.0),
        recover_g=_env_float("FUSION_RECOVER_G", 0.15),
        recover_after_ms=_env_float("FUSION_RECOVER_AFTER_MS", 500.0),
        escalate_s=_env_float("ESCALATE_S", 10.0),
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
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.combined_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.clients: set[WebSocket] = set()
        self.dropped = 0
        self.packets = 0
        self.lock = asyncio.Lock()
        self.names: dict[str, str] = {}
        self.detector = Detector(detector_config())
        self._load_names()
        self._load_detect()

    def _load_names(self) -> None:
        if not NAMES_PATH.exists():
            return
        try:
            data = json.loads(NAMES_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.names = {str(k): str(v) for k, v in data.items() if str(v).strip()}

    def _save_names(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        NAMES_PATH.write_text(json.dumps(self.names, indent=2))

    def _load_detect(self) -> None:
        if not DETECT_PATH.exists():
            return
        try:
            data = json.loads(DETECT_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.detector.update_cfg(**data)

    def _save_detect(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cfg = self.detector.cfg
        DETECT_PATH.write_text(
            json.dumps(
                {
                    "peak_g": cfg.peak_g,
                    "quiet_g": cfg.quiet_g,
                    "min_peak_samples": cfg.min_peak_samples,
                    "rise_ms": cfg.rise_ms,
                    "decay_min_ms": cfg.decay_min_ms,
                    "decay_max_ms": cfg.decay_max_ms,
                    "decay_hold_ms": cfg.decay_hold_ms,
                    "recover_g": cfg.recover_g,
                    "recover_after_ms": cfg.recover_after_ms,
                    "escalate_s": cfg.escalate_s,
                },
                indent=2,
            )
        )

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

    def _name_for(self, node_id: str) -> str:
        node = self.nodes.get(node_id)
        if node is not None:
            return node.name
        return node_id[:8] or "phone"

    async def ingest(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        now = time.time()
        packet["recv_t"] = int(now * 1000)
        addr = packet.get("from", "")
        nid = packet.get("_nid")
        if not isinstance(nid, str) or not nid:
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
            changes = self.detector.tick(now, combined["mag"], nid)
            detect = self.detector.public(now)
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
                "detect": detect,
            }
            clients = list(self.clients)
            name_for = self._name_for

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
        if changes:
            await _emit_detect(changes, name_for)
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
                "detect": self.detector.public(now),
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

    async def cancel(self) -> dict[str, Any]:
        now = time.time()
        async with self.lock:
            self.detector.cancel(now)
            detect = self.detector.public(now)
            clients = list(self.clients)
        payload = {"k": "detect", **detect}
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
        return detect

    async def set_detect_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        async with self.lock:
            self.detector.update_cfg(**updates)
            self._save_detect()
            detect = self.detector.public(now)
            clients = list(self.clients)
        payload = {"k": "detect", **detect}
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
        return detect

    async def watch_detect(self) -> None:
        while True:
            await asyncio.sleep(0.15)
            async with self.lock:
                now = time.time()
                changes = self.detector.tick(now)
                detect = self.detector.public(now)
                clients = list(self.clients)
                name_for = self._name_for
            if changes or detect.get("state") == "suspect":
                payload = {"k": "detect", **detect}
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
            if changes:
                await _emit_detect(changes, name_for)

    async def add(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)


async def _emit_detect(changes: list[dict[str, Any]], name_for) -> None:
    for change in changes:
        state = change.get("state")
        node_id = str(change.get("node_id") or "")
        name = name_for(node_id)
        if state == "suspect":
            asyncio.create_task(notify.send(notify.message_suspect(name)))
        elif state == "confirmed":
            asyncio.create_task(notify.send(notify.message_confirmed(name)))


class RenameBody(BaseModel):
    name: str = ""


class DetectConfigBody(BaseModel):
    peak_g: float | None = None
    decay_min_ms: float | None = None
    decay_max_ms: float | None = None
    rise_ms: float | None = None
    min_peak_samples: int | None = None


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


@app.post("/api/cancel")
async def cancel_event() -> dict[str, Any]:
    return await hub.cancel()


@app.post("/api/detect/config")
async def detect_config(body: DetectConfigBody) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if body.peak_g is not None:
        updates["peak_g"] = min(3.0, max(0.05, float(body.peak_g)))
    if body.decay_min_ms is not None:
        updates["decay_min_ms"] = min(400.0, max(20.0, float(body.decay_min_ms)))
    if body.decay_max_ms is not None:
        updates["decay_max_ms"] = min(2000.0, max(80.0, float(body.decay_max_ms)))
    if body.rise_ms is not None:
        updates["rise_ms"] = min(400.0, max(20.0, float(body.rise_ms)))
    if body.min_peak_samples is not None:
        updates["min_peak_samples"] = min(5, max(1, int(body.min_peak_samples)))
    return await hub.set_detect_config(updates)


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
    loop.create_task(hub.watch_detect())

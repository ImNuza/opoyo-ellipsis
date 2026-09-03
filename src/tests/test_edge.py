from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import uvicorn
from websockets.sync.client import connect

from edge.app import DATA_DIR, create_app as create_edge
from edge.gate import RecordingCloudClient
from edge.infer import FakeCnn
from edge.log import InferenceLog
from phone.fake_phone import sample_stream, send_udp

PHONE_A = "11111111-1111-1111-1111-111111111111"
PHONE_B = "22222222-2222-2222-2222-222222222222"
SAMPLES = 120


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for edge server")


def _print_server(http_port: int, udp_port: int, state: dict) -> None:
    print(f"[edge] server http://127.0.0.1:{http_port}  udp 127.0.0.1:{udp_port}")
    slots = [s for s in state.get("slots", []) if not s.get("empty")]
    print(f"[edge] server slots={len(slots)} dropped={state.get('dropped')} packets={state.get('combined', {}).get('packets')}")
    for slot in slots:
        print(
            f"[edge]   {slot.get('name')} id={slot.get('id')[:8]} "
            f"live={slot.get('live')} mag={slot.get('mag')} db={slot.get('db')} "
            f"hz={slot.get('hz')} pkt={slot.get('packets')}"
        )
    latest = (state.get("inference") or {}).get("latest")
    if latest:
        print(
            f"[edge] server latest inference room={latest.get('room')} "
            f"is_fall={latest.get('is_fall')} conf={latest.get('confidence')}"
        )


def test_edge_udp_ws_logs_activity(capsys):
    log_path = DATA_DIR / "inference.jsonl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    cloud = RecordingCloudClient()
    app = create_edge(
        enable_udp=True,
        udp_host="127.0.0.1",
        udp_port=0,
        log_path=log_path,
        cloud_client=cloud,
        classifier=FakeCnn(is_fall=True, confidence=0.95),
    )
    http_port = _free_tcp_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=http_port,
            log_level="info",
            ws="websockets",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    with capsys.disabled():
        thread.start()
        try:
            _wait(lambda: server.started)
            _wait(lambda: app.state.udp_port is not None)
            udp_port = int(app.state.udp_port)
            print(f"[edge] listening http://127.0.0.1:{http_port}  ws://127.0.0.1:{http_port}/ws")
            print(f"[edge] UDP phones -> 127.0.0.1:{udp_port}")
            print(f"[edge] log file {log_path}")

            packets = sample_stream([PHONE_A, PHONE_B], n=SAMPLES, kind="knee")
            with connect(f"ws://127.0.0.1:{http_port}/ws", open_timeout=5) as ws:
                snapshot = json.loads(ws.recv(timeout=5))
                assert snapshot.get("k") == "state"
                print("[edge] ws snapshot received")

                sent = send_udp(packets, "127.0.0.1", udp_port)
                print(f"[edge] sent {sent} UDP packets from 2 phones")
                _wait(lambda: app.state.hub.packets >= sent)

                ticks = 0
                saw_fall = False
                seen_inferences: set[str] = set()
                deadline = time.time() + 5
                while time.time() < deadline:
                    try:
                        msg = json.loads(ws.recv(timeout=0.25))
                    except TimeoutError:
                        if ticks > 0:
                            break
                        continue
                    if msg.get("k") != "tick":
                        continue
                    ticks += 1
                    latest = ((msg.get("inference") or {}).get("latest")) or {}
                    if ticks == 1 or ticks % 50 == 0:
                        print(
                            f"[edge] ws tick #{ticks} {msg.get('name')} "
                            f"mag={msg.get('mag')} db={msg.get('db')}"
                        )
                    inf_id = latest.get("inference_id")
                    if inf_id and inf_id not in seen_inferences:
                        seen_inferences.add(inf_id)
                        if latest.get("is_fall"):
                            saw_fall = True
                            print(
                                f"[edge] FALL DETECTED room={latest.get('room')} "
                                f"conf={latest.get('confidence')} "
                                f"node={latest.get('node_id')} "
                                f"id={inf_id}"
                            )
                        else:
                            print(
                                f"[edge] inference no-fall room={latest.get('room')} "
                                f"conf={latest.get('confidence')} id={inf_id}"
                            )

            assert ticks >= 1
            assert saw_fall

            store = InferenceLog(log_path)
            rows = store.tail(50)
            print("[edge] inference log:")
            for row in rows:
                line = row.model_dump_json()
                print(f"[edge]   {line}")
                if row.is_fall:
                    print(
                        f"[edge] FALL DETECTED (log) room={row.room} "
                        f"conf={row.confidence}"
                    )
            assert rows
            assert {row.node_id for row in rows} == {"Phone 1", "Phone 2"}
            assert {row.room for row in rows} == {1, 2}
            assert all(row.is_fall is True for row in rows)

            state = httpx.get(f"http://127.0.0.1:{http_port}/api/state", timeout=5).json()
            _print_server(http_port, udp_port, state)
            live = [s for s in state["slots"] if not s.get("empty")]
            assert {s["id"] for s in live} == {PHONE_A, PHONE_B}
            assert state["inference"]["latest"]["is_fall"] is True
        finally:
            server.should_exit = True
            thread.join(timeout=5)

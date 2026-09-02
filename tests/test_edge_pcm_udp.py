from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import uvicorn

from edge.app import DATA_DIR, create_app as create_edge
from edge.gate import RecordingCloudClient
from edge.infer import FakeCnn
from phone.fake_phone import make_packet, sample_stream, send_udp, send_udp_pcm
from shared.pcm import pack_frame

PHONE_A = "11111111-1111-1111-1111-111111111111"
N = 100
DT = 20
T0 = 1_000_000
HZ = 50.0


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
    raise AssertionError("timed out waiting")


def _boot():
    log_path = DATA_DIR / "inference.jsonl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    app = create_edge(
        enable_udp=True,
        udp_host="127.0.0.1",
        udp_port=0,
        udp_pcm_port=0,
        log_path=log_path,
        cloud_client=RecordingCloudClient(),
        classifier=FakeCnn(is_fall=True, confidence=0.95),
    )
    http_port = _free_tcp_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=http_port,
            log_level="warning",
            ws="websockets",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait(lambda: server.started)
    _wait(lambda: app.state.udp_port is not None)
    _wait(lambda: app.state.udp_pcm_port is not None)
    return app, server


def test_json_and_pcm_ports_are_distinct():
    app, server = _boot()
    try:
        assert app.state.udp_port != app.state.udp_pcm_port
        assert int(app.state.udp_port) > 0
        assert int(app.state.udp_pcm_port) > 0
    finally:
        server.should_exit = True


def test_udp_pcm_joins_json_window():
    app, server = _boot()
    try:
        json_port = int(app.state.udp_port)
        pcm_port = int(app.state.udp_pcm_port)
        packets = []
        frames = []
        pcm = np.full(320, 1000, dtype=np.int16).tobytes()
        for i in range(N):
            t = T0 + i * DT
            packets.append(make_packet(PHONE_A, t_ms=t, elapsed=i / HZ, kind="knee"))
            frames.append(pack_frame(PHONE_A, seq=i, t_ms=t, samples=pcm))
        send_udp_pcm(frames, "127.0.0.1", pcm_port)
        _wait(
            lambda: PHONE_A in app.state.hub.pcm_rings
            and app.state.hub.pcm_rings[PHONE_A].buffered_ms() >= 1500
        )
        send_udp(packets, "127.0.0.1", json_port)
        _wait(lambda: app.state.hub.packets >= N)
        _wait(lambda: PHONE_A in app.state.hub.last_pcm)
        got = app.state.hub.last_pcm[PHONE_A]
        assert got["coverage"] >= 0.9
    finally:
        server.should_exit = True


def test_pcm_port_ignores_json_json_port_still_counts():
    app, server = _boot()
    try:
        json_port = int(app.state.udp_port)
        pcm_port = int(app.state.udp_pcm_port)
        packets = sample_stream([PHONE_A], n=N, t0_ms=T0, kind="knee")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            blob = json.dumps(
                {k: v for k, v in packets[0].items() if k not in {"_nid", "from"}}
            ).encode("utf-8")
            sock.sendto(blob, ("127.0.0.1", pcm_port))
        finally:
            sock.close()
        time.sleep(0.05)
        assert app.state.hub.packets == 0
        sent = send_udp(packets, "127.0.0.1", json_port)
        _wait(lambda: app.state.hub.packets >= sent)
        assert app.state.hub.packets >= sent
    finally:
        server.should_exit = True

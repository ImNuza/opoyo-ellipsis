#!/usr/bin/env python3
"""Synthetic UDP phones for the edge (port 9000). Not a fall classifier.

--nodes is how many devices. --impact is the mag shape (quiet / knee / book).
Packets are SensorSample JSON; they do not carry a scenario label.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
import uuid

from shared.pcm import pack_frame

MODELS = [
    "iPhone 17 Pro Max",
    "iPhone 16",
    "iPhone 15 Pro",
    "iPhone 14",
    "iPhone SE",
]


def mag_for(kind: str, elapsed: float, nidx: int) -> float:
    """Return g at ``elapsed`` seconds. knee ~0.72, book ~1.2, else quiet + bump."""
    if kind == "knee":
        if elapsed < 0.4:
            return 0.01
        local = elapsed - 0.4
        if local < 0.06:
            return 0.01 + (0.72 - 0.01) * (local / 0.06)
        if local < 0.10:
            return 0.72
        if local < 0.30:
            return 0.72 - (0.72 - 0.01) * ((local - 0.10) / 0.20)
        return 0.01
    if kind == "book":
        if elapsed < 0.4:
            return 0.01
        local = elapsed - 0.4
        if local < 0.04:
            return 1.2
        return 0.01
    phase = elapsed * (1.4 + nidx * 0.35) + nidx
    bump = 3.2 + nidx * 1.1 < elapsed < 4.4 + nidx * 1.1
    if bump:
        return 0.02 + 0.55 * abs(math.sin(phase))
    return 0.012 + 0.01 * math.sin(phase * 7)


def make_packet(
    node_id: str,
    *,
    nidx: int = 0,
    t_ms: int,
    elapsed: float,
    kind: str = "",
) -> dict:
    """Build one v2 UDP/ingest packet, including `_nid` for Hub.ingest."""
    mag = mag_for(kind, elapsed, nidx)
    phase = elapsed * (1.4 + nidx * 0.35) + nidx
    ax = mag * (0.15 + 0.05 * nidx)
    ay = mag * 0.08 * math.sin(phase)
    az = mag
    db = -52 + 22 * mag + nidx * 2
    return {
        "v": 2,
        "id": node_id,
        "_nid": node_id,
        "model": MODELS[nidx % len(MODELS)],
        "t": t_ms,
        "ax": round(ax, 4),
        "ay": round(ay, 4),
        "az": round(az, 4),
        "mag": round(mag, 4),
        "db": round(db, 2),
        "from": "127.0.0.1:9",
    }


def sample_stream(
    node_ids: list[str],
    *,
    n: int,
    hz: float = 50.0,
    t0_ms: int = 1_735_689_600_000,
    kind: str = "",
) -> list[dict]:
    """N ticks of interleaved packets, same shape as the UDP CLI."""
    interval_ms = int(round(1000.0 / hz))
    packets: list[dict] = []
    for i in range(n):
        elapsed = i / hz
        t_ms = t0_ms + i * interval_ms
        for nidx, nid in enumerate(node_ids):
            packets.append(
                make_packet(nid, nidx=nidx, t_ms=t_ms, elapsed=elapsed, kind=kind)
            )
    return packets


def send_udp(packets: list[dict], host: str, port: int) -> int:
    """Send packets as real UDP datagrams, the way a phone would."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    try:
        for packet in packets:
            wire = {k: v for k, v in packet.items() if k not in {"_nid", "from"}}
            sock.sendto(json.dumps(wire).encode("utf-8"), (host, port))
            sent += 1
    finally:
        sock.close()
    return sent


def pcm_frame(node_id: str, *, seq: int, t_ms: int, elapsed: float, kind: str = "") -> bytes:
    """20 ms of 16 kHz s16le. Burst follows mag_for so knee/book are audible."""
    n = 320
    mag = mag_for(kind, elapsed, 0)
    amp = int(min(8000, 400 + 12000 * mag))
    samples = bytearray()
    for i in range(n):
        # cheap tone; amplitude tracks mag
        v = int(amp * math.sin(2 * math.pi * (i / n) * 4))
        v = max(-32767, min(32767, v))
        samples += int(v).to_bytes(2, "little", signed=True)
    return pack_frame(node_id, seq=seq, t_ms=t_ms, samples=bytes(samples))


def send_udp_pcm(frames: list[bytes], host: str, port: int) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    try:
        for frame in frames:
            sock.sendto(frame, (host, port))
            sent += 1
    finally:
        sock.close()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--pcm-port", type=int, default=0, help="PCM UDP port (default port+1)")
    parser.add_argument("--no-pcm", action="store_true", help="JSON only")
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--ids", default="", help="comma-separated stable ids")
    parser.add_argument("--impact", default="", choices=["", "knee", "book"])
    args = parser.parse_args()

    count = max(1, min(args.nodes, 5))
    if args.ids.strip():
        ids = [part.strip() for part in args.ids.split(",") if part.strip()][:count]
        while len(ids) < count:
            ids.append(str(uuid.uuid4()))
    else:
        ids = [str(uuid.uuid4()) for _ in range(count)]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pcm_port = args.pcm_port if args.pcm_port > 0 else args.port + 1
    interval = 1.0 / args.hz
    n = int(args.seconds * args.hz)
    t0 = time.time()
    sent = 0
    pcm_sent = 0
    kind = args.impact or ""
    seq = {nid: 0 for nid in ids}
    for _i in range(n):
        t = time.time()
        elapsed = t - t0
        t_ms = int(t * 1000)
        for nidx, nid in enumerate(ids):
            packet = make_packet(
                nid,
                nidx=nidx,
                t_ms=t_ms,
                elapsed=elapsed,
                kind=kind,
            )
            # Real phones do not send _nid; the edge UDP handler stamps it.
            wire = {k: v for k, v in packet.items() if k != "_nid"}
            sock.sendto(json.dumps(wire).encode("utf-8"), (args.host, args.port))
            sent += 1
            if not args.no_pcm:
                frame = pcm_frame(
                    nid, seq=seq[nid], t_ms=t_ms, elapsed=elapsed, kind=kind
                )
                sock.sendto(frame, (args.host, pcm_port))
                seq[nid] += 1
                pcm_sent += 1
        time.sleep(interval)
    sock.close()
    print(f"sent {sent} packets from {count} nodes to {args.host}:{args.port}")
    if pcm_sent:
        print(f"sent {pcm_sent} PCM frames to {args.host}:{pcm_port}")
    print("ids: " + ",".join(ids))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Send synthetic phone packets to the local receiver. Not a fall model."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
import uuid

MODELS = [
    "iPhone 17 Pro Max",
    "iPhone 16",
    "iPhone 15 Pro",
    "iPhone 14",
    "iPhone SE",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--ids", default="", help="comma-separated stable ids")
    args = parser.parse_args()

    count = max(1, min(args.nodes, 5))
    if args.ids.strip():
        ids = [part.strip() for part in args.ids.split(",") if part.strip()][:count]
        while len(ids) < count:
            ids.append(str(uuid.uuid4()))
    else:
        ids = [str(uuid.uuid4()) for _ in range(count)]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / args.hz
    n = int(args.seconds * args.hz)
    t0 = time.time()
    sent = 0
    for i in range(n):
        t = time.time()
        elapsed = t - t0
        for nidx, nid in enumerate(ids):
            phase = elapsed * (1.4 + nidx * 0.35) + nidx
            bump = 3.2 + nidx * 1.1 < elapsed < 4.4 + nidx * 1.1
            mag = 0.02 + 0.55 * abs(math.sin(phase)) if bump else 0.012 + 0.01 * math.sin(phase * 7)
            ax = mag * (0.15 + 0.05 * nidx)
            ay = mag * 0.08 * math.sin(phase)
            az = mag
            db = -52 + 22 * mag + nidx * 2
            packet = {
                "v": 2,
                "id": nid,
                "model": MODELS[nidx % len(MODELS)],
                "t": int(t * 1000),
                "ax": round(ax, 4),
                "ay": round(ay, 4),
                "az": round(az, 4),
                "mag": round(mag, 4),
                "db": round(db, 2),
            }
            sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))
            sent += 1
        time.sleep(interval)
    sock.close()
    print(f"sent {sent} packets from {count} nodes to {args.host}:{args.port}")
    print("ids: " + ",".join(ids))


if __name__ == "__main__":
    main()

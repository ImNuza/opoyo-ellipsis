#!/usr/bin/env python3
"""Send synthetic phone packets to the local receiver. Not a fall model."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / args.hz
    n = int(args.seconds * args.hz)
    t0 = time.time()
    for i in range(n):
        t = time.time()
        phase = (t - t0) * 2
        mag = 0.02 + 0.4 * abs(math.sin(phase)) if 3 < (t - t0) < 4.2 else 0.015 + 0.01 * math.sin(phase * 8)
        db = -48 + 18 * mag
        packet = {
            "v": 1,
            "t": int(t * 1000),
            "ax": mag * 0.2,
            "ay": 0.0,
            "az": mag,
            "mag": mag,
            "db": db,
        }
        sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))
        time.sleep(interval)
    sock.close()
    print(f"sent {n} packets to {args.host}:{args.port}")


if __name__ == "__main__":
    main()

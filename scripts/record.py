#!/usr/bin/env python3
"""Capture labelled OpoyoPhone UDP packets to data/<label>_<nn>.csv.

Listens on 0.0.0.0:9000 for the phone JSON: v,id,model,t,ax,ay,az,mag,db.
Cannot run at the same time as uvicorn — both bind UDP 9000. Stop the dashboard first.

From repo root:

  python scripts/record.py --label heeldrop --seconds 8
  python scripts/record.py --interactive
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
META_FIELDS = (
    "filename",
    "label",
    "floor",
    "room",
    "distance_m",
    "phone_model",
    "case",
    "footwear",
    "object",
    "notes",
)
HINT = (
    "no packets in 3s - campus Wi-Fi often blocks UDP. "
    "Use a phone hotspot (laptop IP is usually 172.20.10.11). "
    "OpoyoPhone host = this machine's LAN IP, port 9000."
)


def _num(pkt: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(pkt.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def next_csv(label: str) -> Path:
    n = 1
    while True:
        path = DATA / f"{label}_{n:02d}.csv"
        if not path.exists():
            return path
        n += 1


def append_meta(row: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "metadata.csv"
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in META_FIELDS})


def write_take(label: str, rows: list[dict], meta: dict) -> Path | None:
    if not rows:
        print("no packets - not writing")
        return None
    DATA.mkdir(parents=True, exist_ok=True)
    path = next_csv(label)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "ax", "ay", "az", "mag", "db"])
        writer.writeheader()
        writer.writerows(rows)
    append_meta({"filename": path.name, "label": label, **meta})
    return path


class Sink:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.armed = False
        self.n = 0
        self.last = 0.0
        self.t0 = 0.0
        self.model = ""
        self.recent: deque[float] = deque()

    def arm(self) -> None:
        with self.lock:
            self.rows = []
            self.n = 0
            self.last = 0.0
            self.t0 = time.time()
            self.model = ""
            self.recent.clear()
            self.armed = True

    def disarm(self) -> tuple[list[dict], int, str, float]:
        with self.lock:
            self.armed = False
            elapsed = time.time() - self.t0 if self.t0 else 0.0
            return list(self.rows), self.n, self.model, elapsed

    def push(self, pkt: dict) -> None:
        with self.lock:
            model = pkt.get("model")
            if isinstance(model, str) and model.strip():
                self.model = model.strip()[:48]
            if not self.armed:
                return
            t = pkt.get("t")
            try:
                t = int(t)
            except (TypeError, ValueError):
                t = int(time.time() * 1000)
            self.rows.append(
                {
                    "t": t,
                    "ax": _num(pkt, "ax"),
                    "ay": _num(pkt, "ay"),
                    "az": _num(pkt, "az"),
                    "mag": _num(pkt, "mag"),
                    "db": _num(pkt, "db", -120.0),
                }
            )
            now = time.time()
            self.n += 1
            self.last = now
            self.recent.append(now)
            while self.recent and now - self.recent[0] > 1.0:
                self.recent.popleft()


def recv_loop(sock: socket.socket, sink: Sink, stop: threading.Event) -> None:
    sock.settimeout(0.25)
    while not stop.is_set():
        try:
            data, _addr = sock.recvfrom(65535)
        except (TimeoutError, BlockingIOError):
            continue
        except OSError:
            break
        try:
            pkt = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(pkt, dict):
            sink.push(pkt)


def watch(sink: Sink, stop: threading.Event) -> None:
    hinted = False
    while not stop.wait(1.0):
        now = time.time()
        with sink.lock:
            n = sink.n
            last = sink.last
            t0 = sink.t0
            armed = sink.armed
            while sink.recent and now - sink.recent[0] > 1.0:
                sink.recent.popleft()
            hz = float(len(sink.recent))
        if not armed:
            continue
        elapsed = max(now - t0, 1e-6)
        print(f"\r  {hz:5.1f} pkt/s  {n} packets  {elapsed:.1f}s", end="", flush=True)
        gap = now - (last or t0)
        if not hinted and gap >= 3.0:
            print("\n" + HINT)
            hinted = True


def bind_udp(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        print(
            f"cannot bind {host}:{port} - uvicorn is probably running. "
            "Stop the dashboard first (UDP 9000 is exclusive).",
            file=sys.stderr,
        )
        sys.exit(1)
    return sock


def meta_from(args: argparse.Namespace, model: str) -> dict:
    return {
        "floor": args.floor,
        "room": args.room,
        "distance_m": args.distance,
        "phone_model": args.phone_model or model,
        "case": args.case,
        "footwear": args.footwear,
        "object": args.object,
        "notes": args.notes,
    }


def finish(label: str, rows: list[dict], n: int, model: str, elapsed: float, args: argparse.Namespace) -> None:
    print()
    if n == 0:
        print(HINT)
    path = write_take(label, rows, meta_from(args, model))
    if path is None:
        return
    if n >= 2:
        span = (int(rows[-1]["t"]) - int(rows[0]["t"])) / 1000.0
        hz = (n - 1) / span if span > 0 else 0.0
    else:
        hz = n / elapsed if elapsed > 0 else 0.0
    print(f"wrote {path.relative_to(ROOT).as_posix()}  ({n} rows, {hz:.1f} Hz)")


def record_seconds(sink: Sink, label: str, seconds: float, args: argparse.Namespace) -> None:
    print(f"recording {label} for {seconds:.0f}s - one event, 2s quiet before/after")
    sink.arm()
    stop = threading.Event()
    threading.Thread(target=watch, args=(sink, stop), daemon=True).start()
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        print("\nstopped")
    stop.set()
    rows, n, model, elapsed = sink.disarm()
    finish(label, rows, n, model, elapsed, args)


def record_interactive(sink: Sink, args: argparse.Namespace) -> None:
    default = args.label
    print("interactive: Ctrl+C to quit. One event per file.")
    while True:
        raw = input(f"label [{default or 'required'}]: ").strip()
        label = (raw or default).strip().lower().replace(" ", "")
        if not label:
            print("need a label")
            continue
        default = label
        input("Enter to start")
        print("recording - Enter to stop")
        sink.arm()
        stop = threading.Event()
        threading.Thread(target=watch, args=(sink, stop), daemon=True).start()
        try:
            input()
        except KeyboardInterrupt:
            print("\nstopped")
        stop.set()
        rows, n, model, elapsed = sink.disarm()
        finish(label, rows, n, model, elapsed, args)


def clean_label(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="", help="class name; writes data/<label>_<nn>.csv")
    p.add_argument("--interactive", action="store_true", help="prompt label, Enter to start/stop")
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--floor", default="tile")
    p.add_argument("--room", default="kitchen")
    p.add_argument("--distance", type=float, default=1.5)
    p.add_argument("--case", default="off")
    p.add_argument("--phone-model", default="", dest="phone_model")
    p.add_argument("--footwear", default="")
    p.add_argument("--object", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    label = clean_label(args.label)
    if not args.interactive and not label:
        p.error("need --label or --interactive")

    sock = bind_udp(args.host, args.port)
    sink = Sink()
    stop = threading.Event()
    threading.Thread(target=recv_loop, args=(sock, sink, stop), daemon=True).start()
    print(f"listening {args.host}:{args.port}")
    try:
        if args.interactive:
            record_interactive(sink, args)
        else:
            record_seconds(sink, label, args.seconds, args)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        stop.set()
        sock.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(path)


def chat_id() -> str | None:
    target = (os.getenv("TELEGRAM_TARGET") or "personal").strip().lower()
    if target == "family":
        raw = os.getenv("TELEGRAM_CHAT_ID_FAMILY") or ""
    else:
        raw = os.getenv("TELEGRAM_CHAT_ID") or ""
    cleaned = raw.strip()
    return cleaned or None


def token() -> str | None:
    raw = (os.getenv("TELEGRAM_BOT_API") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    return raw or None


def message_suspect(name: str) -> str:
    return f"OPOYO: possible fall. Node {name}. Cancel on the Mac."


def message_confirmed(name: str) -> str:
    return f"OPOYO: no recovery, still down. Node {name}."


def message_fall(room: str, timestamp_ms: int, confidence: float) -> str:
    ts = datetime.fromtimestamp(timestamp_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"OPOYO: fall. Room {room}. {ts}. confidence {confidence:.2f}."
    )


def send_sync(text: str) -> bool:
    tok = token()
    cid = chat_id()
    if not tok or not cid:
        return False
    import httpx

    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    try:
        response = httpx.post(url, json={"chat_id": cid, "text": text}, timeout=8.0)
        return response.status_code == 200
    except Exception:
        return False


async def send(text: str) -> bool:
    tok = token()
    cid = chat_id()
    if not tok or not cid:
        return False
    import httpx

    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(url, json={"chat_id": cid, "text": text})
        return response.status_code == 200
    except Exception:
        return False

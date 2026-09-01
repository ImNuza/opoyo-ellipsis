from __future__ import annotations

from datetime import datetime

from opoyo.config import secret
from opoyo.schemas import FallEvent


def chat_id() -> str | None:
    target = (secret("TELEGRAM_TARGET") or "personal").strip().lower()
    if target == "family":
        raw = secret("TELEGRAM_CHAT_ID_FAMILY")
    else:
        raw = secret("TELEGRAM_CHAT_ID")
    cleaned = raw.strip()
    return cleaned or None


def token() -> str | None:
    raw = (secret("TELEGRAM_BOT_TOKEN") or secret("TELEGRAM_BOT_API")).strip()
    return raw or None


def format_alert(ev: FallEvent, rung: int = 1) -> str:
    ts = datetime.fromtimestamp(ev.ts_unix).strftime("%H:%M:%S")
    if rung == 1:
        return (
            f"OPOYO alert · {ev.room}\n"
            f"{ts} · score {ev.fall_score:.2f} · peak {ev.peak_mag:.2f} m/s²\n"
            f"no recovery yet. Ack on the dashboard."
        )
    return (
        f"OPOYO still unacknowledged · {ev.room}\n"
        f"score {ev.fall_score:.2f} · {ts}"
    )


async def send_telegram(text: str, dry_run: bool = True) -> bool:
    if dry_run:
        print(f"[DRY RUN telegram] {text}")
        return True
    tok, cid = token(), chat_id()
    if not tok or not cid:
        print("[telegram] missing credentials; skipping")
        return False
    try:
        import httpx

        url = f"https://api.telegram.org/bot{tok}/sendMessage"
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(url, json={"chat_id": cid, "text": text})
        return response.status_code == 200
    except Exception as e:
        print(f"[telegram] failed: {e}")
        return False

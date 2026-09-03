# Server (cloud)

The cloud never sees raw vibration. It receives gated `FallEvent` JSON and drives the human ladder. Code: `src/server/`. Default: `http://127.0.0.1:8001`. No UDP. No dashboard.

## HTTP

| Method | Path | Notes |
|---|---|---|
| `POST` | `/events` | Open or reuse a case. **202** + `EscalationCase`. t+0 Telegram. |
| `POST` | `/cases/{case_id}/ack` | Apply an `AckEvent`. Path `case_id` wins if the body disagrees. |
| `GET` | `/cases/{case_id}` | Current case, or 404. |
| `GET` | `/health` | `{"ok": true}` |

There is no Telegram webhook. A 0.5 s `on_tick` loop (not a route) advances timers and polls `getUpdates`.

## Escalation ladder

One gated fall opens **one case**. Telegram goes out in rungs. Anyone answering can stop it.

**t+0 — two messages**

| Who | Env | Copy |
|---|---|---|
| Next of kin | `TELEGRAM_CHAT_ID_NEXT_OF_KIN` | `OPOYO: fall. Room N. <time>. Tap I'm on it … confidence 0.xx.` plus **I'm on it** |
| Senior | `TELEGRAM_CHAT_ID_SENIOR` | `OPOYO: possible fall. … Are you okay?` plus **I'm fine** / **I need help** |

State: `rung1_dispatched`.

```
t+0     family alert + senior check-in
          │
          ├─ senior I'm fine / yes     →  false_alarm_closed     (stop)
          ├─ family I'm on it / taken  →  family_handling        (stop, even at t+0)
          ├─ senior I need help / no
          │     → awaiting_family
          │         ├─ family taken within 60 s  →  family_handling
          │         └─ 60 s silence              →  Telegram secondary
          │                                           → secondary_alerted
          │               ├─ secondary taken     →  resolved
          │               └─ still open at t+180 →  careline_alerted (stub, no SMS)
          └─ senior silent 60 s  →  treated as no_answer, same family wait
```

Timers come from `config.yaml` (`senior_wait_s`, `family_wait_s`, `careline_at_s`). Twilio is **not** on this ladder; `server.adapters.twilio` is a future voice plug-in.

**Dedup.** Same `event_id` returns the existing case. A second fall from the same `node_id` inside `alert.cooldown_s` (3 s wall clock **or** window time) reuses the open case and does **not** send another ladder. Cooldown is stamped **before** `sendMessage` so a slow Telegram call cannot let a duplicate through.

## Acks

Buttons carry `ack:{case_id}:yes|not_fine|taken`. Typed replies:

- Senior chat: yes / ok / I'm fine → all-clear; no / help → not fine. Bind to the newest `rung1_dispatched` case if there is no `case_id`.
- Family chat: taken / I'm on it / got it / I'll handle. Bind to the newest case in `rung1_dispatched` or `awaiting_family`.

Family taking the case confirms: `Noted — you're handling this. Case closed.`

## Secrets

`.env`:

```
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID_NEXT_OF_KIN=
TELEGRAM_CHAT_ID_SECONDARY=
TELEGRAM_CHAT_ID_SENIOR=
```

Empty tokens: the process still runs; sends no-op.

## Run

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8001
```

Health: http://127.0.0.1:8001/health

Cases are **in memory**. Restarting the process drops open ladders. That is acceptable for the demo, not for production.

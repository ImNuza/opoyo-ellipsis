# OPOYO

Passive floor-vibration fall detector. Hackathon prototype.

The product node is a piezo puck bonded to the slab. This build uses iPhones on the floor as a proxy: accelerometer plus a sound-level meter, streamed to an **edge** process. A stub 1D CNN classifies each 2 s window. The **cloud** (`server/`) runs the escalation ladder. Telegram fires only after a gated fall. Raw vibration never leaves the edge.

```
phone (UDP JSON :9000)  →  edge (:8000 HTTP + :9000 UDP)  →  cloud (:8001 HTTP)
                                 │                              │
                                 │  FallEvent POST /events      │
                                 │  only if is_fall and         │  Telegram next of kin
                                 │  confidence ≥ 0.90           │  + Telegram senior (yes = ok)
                                 ▼                              ▼
                          edge/data/inference.jsonl        DecisionTree
```

Swap `StubCnn` for a trained 1D CNN with the same `infer(window) -> InferenceResult` signature.

---

## How they communicate

1. **Phone → edge (JSON).** One JSON object per sample, 50 Hz, UDP to `0.0.0.0:9000`. No HTTP. No TCP handshake. The packet is a `SensorSample` (`v, id, model, t, ax, ay, az, mag, db`). `id` is a UUID created on first launch (or by `fake_phone.py`).
2. **Phone → edge (PCM).** 16 kHz mono int16 frames on UDP **`:9001`** (JSON port + 1). Binary `OPYA` header + PCM — not a WAV file. Same `id` and phone unix `t_ms` as JSON. iOS resamples the mic tap to 16 kHz before send. `fake_phone.py` sends a synthetic 20 ms frame per mag tick (`--no-pcm` to skip).
3. **Edge internally.** JSON datagrams become `Node`s (max 5). Each node has a `WindowBuilder` (2 s window, 1 s hop, 50 Hz → 100 samples). PCM is stored in a 4 s `PcmRing` keyed by UUID. When a window closes, the hub slices PCM with `t_start_ms` / `t_end_ms` (fail-open if the clip is missing). `StubCnn.infer` still sees only the kinematic window. Combined dashboard traces are **max mag** and **max dB** among phones seen in the last 2.5 s. Ticks may include `pcm_coverage` / `pcm_buffered_ms`; the waveform is not sent to the browser or the cloud.
4. **Edge → cloud.** Only when `is_fall` is true **and** `confidence >= 0.90`. The edge POSTs a `FallEvent` to `{CLOUD_URL}/events` (`http://127.0.0.1:8001/events`). Axes, windows, and raw PCM stay on the edge.
5. **Cloud → humans.** `DecisionTree.ingest` opens a case: Telegram next of kin and Telegram to the senior at t+0. If the senior acks `yes`, the case closes as all-clear. Otherwise `POST /cases/{id}/ack` moves the ladder. A 0.5 s tick loop fires secondary (t+60) and CareLine stub (t+180). Twilio is not on this path; `server.adapters.twilio` is kept for a future voice plug-in.
6. **Dashboard → edge.** Browser uses `GET /api/state` and WebSocket `/ws`. It does not ingest samples.

If the edge log does not print `[edge] UDP listening on 0.0.0.0:9000` and `[edge] UDP PCM listening on 0.0.0.0:9001`, phones are sending into the void. Set `EDGE_ENABLE_UDP=1` in `.env` and restart.

---

## Shared models (`shared/schemas.py`)

Pydantic models used on the wire and in logs. Extra keys on `SensorSample` are ignored.

| Model | Who produces it | Who consumes it |
|---|---|---|
| `SensorSample` | Phone / `fake_phone.py` | Edge `Hub.ingest` |
| `SensorWindow` | Edge `WindowBuilder` | `Classifier.infer` |
| `InferenceResult` | Classifier | Gate + `inference.jsonl` + dashboard |
| `FallEvent` | Escalation gate | Cloud `POST /events` |
| `AckEvent` | Operator / tests | Cloud `POST /cases/{id}/ack` |
| `EscalationCase` | Decision tree | Cloud HTTP responses |

### `SensorSample` (UDP JSON)

```json
{"v":2,"id":"c0a1...","model":"iPhone 17 Pro Max","t":1735689600123,"ax":0.01,"ay":0.0,"az":-0.02,"mag":0.03,"db":-41.2}
```

| Field | Type | Notes |
|---|---|---|
| `v` | int | Packet version. Use `2`. |
| `id` | str | Stable phone UUID. Hub key. |
| `model` | str | Display only. Default `iPhone`. |
| `t` | int | Phone timestamp, ms. |
| `ax, ay, az` | float | User-acceleration, g. Default `0`. |
| `mag` | float | `sqrt(ax²+ay²+az²)`. Required. |
| `db` | float | Mic RMS dB. Default `-120`. |

UDP ingest also stamps `_nid` (from `id`) and `from` (`ip:port`). Those are not sent by the phone.

### PCM frame (UDP `:9001`, not WAV)

Little-endian. Magic `OPYA`. UUID is 16 raw bytes (same id as JSON). Typical payload: 20 ms = 320 samples = 640 B.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | `OPYA` |
| 4 | 1 | version `1` |
| 5 | 4 | seq uint32 |
| 9 | 8 | `t_ms` uint64 (unix ms of sample 0) |
| 17 | 16 | node UUID bytes |
| 33 | 2 | sample rate (16000) |
| 35 | 2 | n samples |
| 37 | 2n | PCM s16le mono |

Pack/unpack: `shared/pcm.py`. Join: `edge/pcm_ring.py` `slice_ms(t_start_ms, t_end_ms)`.

### `SensorWindow`

One 2 s slice for the CNN. `node_id` is `Phone 1` … `Phone 5`. `room` is the integer slot `1` … `5`. Lists `mag, ax, ay, az, db` are length 100 at 50 Hz. `t_start_ms` / `t_end_ms` come from the first and last sample `t`.

### `InferenceResult` (one JSONL line)

```json
{"inference_id":"a1b2c3d4e5f6","timestamp":1735689602123,"node_id":"Phone 1","room":1,"is_fall":false,"confidence":0.0}
```

`confidence` is in `[0, 1]`. Stub CNN always writes `is_fall: false`, `confidence: 0.0`.

### `FallEvent` (edge → cloud)

Same identity fields as inference, plus `threshold` (the gate, currently `0.90`). `is_fall` is always `true`. Built only after the gate.

### `AckEvent`

```json
{"case_id":"...","actor":"senior","outcome":"fine","timestamp":1}
```

`actor`: `senior` \| `family` \| `secondary` \| `careline`.  
`outcome`: `fine` \| `not_fine` \| `no_answer` \| `taken`.

---

## Phone

Two producers: the iOS app and `phone/fake_phone.py`. JSON `SensorSample` on `:9000`; PCM frames on `:9001`.

**Input:** none from the network. Sensors only (CoreMotion 50 Hz + mic).

**Output:** UDP JSON to `--host/--port` (default `127.0.0.1:9000`) and PCM to `port+1`. No HTTP API.

### iOS (`phone/OpoyoPhone`)

Needs Xcode, a paid team, and the phone on the same Wi-Fi or a personal hotspot.

```bash
cd phone
xcodegen generate
open OpoyoPhone.xcodeproj
```

On first launch allow Microphone and Local Network. Type the laptop LAN IP. Port `9000` (PCM uses `9001`). Start. Face-down on tile. Screen stays on while streaming (iOS will not stream CoreMotion/mic in the background).

If the phone is the hotspot, the Mac is almost always `172.20.10.11`. Campus Wi-Fi often blocks UDP; hotspot is the trusted demo network.

`DeviceIdentity.nodeId` is a UUID stored on the device. That is `SensorSample.id`.

### Fake sensors (`phone/fake_phone.py`)

Talks to the **edge UDP port**, not the cloud.

```bash
python phone/fake_phone.py --nodes 1 --impact knee --seconds 8
```

| Flag | Meaning |
|---|---|
| `--host` / `--port` | Edge JSON UDP target (default `127.0.0.1:9000`) |
| `--pcm-port` | PCM UDP port (default `port+1`) |
| `--no-pcm` | JSON only |
| `--hz` | Sample rate (default 50) |
| `--seconds` | Duration |
| `--nodes` | How many fake phones (max 5). Phone 1 / room 1, then Phone 2 / room 2, … |
| `--ids` | Comma-separated UUIDs so slots stay stable across runs |
| `--impact` | Scenario (see below) |

`--nodes` is device count, not the scenario. `--impact` is applied to every node in that run.

| `--impact` | Magnitude shape | Dashboard tell |
|---|---|---|
| *(omit)* | Quiet ~0.01 g, small bump after ~3 s | Flat combined mag |
| `knee` | After 0.4 s, ramp to **0.72 g** | Rounded spike |
| `book` | After 0.4 s, **1.2 g** for ~40 ms | Sharp spike, louder dB |

Packets do **not** carry a `knee`/`book` label. The waveform is how you tell scenarios apart. The stub CNN still returns no-fall, so fake impacts fill the log and do not open a cloud case.

Keep slots stable:

```bash
python phone/fake_phone.py --nodes 2 --ids 11111111-1111-1111-1111-111111111111,22222222-2222-2222-2222-222222222222 --impact book --seconds 8
```

---

## Edge (`edge/`)

FastAPI process. Default `http://0.0.0.0:8000` plus UDP `0.0.0.0:9000`.

**In:** UDP `SensorSample` on `:9000` and `OPYA` PCM on `:9001`. Optional dashboard HTTP/WS (no sample ingest over HTTP).

**Out:** dashboard JSON; `inference.jsonl`; gated `FallEvent` POST to the cloud.

### Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/` | — | Dashboard HTML |
| `GET` | `/static/*` | — | CSS, images |
| `GET` | `/api/state` | — | Snapshot: slots, combined mag/dB, histories, latest inference, last 50 log rows, read-only `cfg` |
| WebSocket | `/ws` | Client may send any text (ignored; keep-alive) | First message `{k: "state", ...}` (same as `/api/state`). Then `{k: "tick", ...}` per ingested packet |
| UDP | `:9000` | JSON `SensorSample` | — (side effect: ingest) |
| UDP | `:9001` | Binary `OPYA` PCM (16 kHz s16le) | — (side effect: ring + join on window) |

No `POST /api/edge/config`. Threshold, window, hop, and cloud URL are constants in `edge/app.py`. No phone rename route.

`GET /api/state` / WS `k: "state"` include five slots (`empty` or live phone), `combined` (`mag`, `db`, `live`, `hz`, `packets`), `dropped` (6th+ UUIDs), and `inference.latest` / `inference.log`.

WS `k: "tick"` is one sample plus `combined` and the current `inference` blob.

### Constants (`edge/app.py`)

```
ESCALATE_MIN_CONFIDENCE = 0.90
WINDOW_S = 2.0
HOP_S = 1.0
CLOUD_URL = http://127.0.0.1:8001
MAX_NODES = 5
```

Change them in code and restart. `EDGE_ENABLE_UDP` in `.env` can force the UDP socket off (`0` / `false` / `no`). Unset defaults to **on**.

### Internals

- **Node** — one phone the hub has seen. Keyed by UUID. Slot name `Phone N` is the inference `node_id`; slot number is `room`.
- **Hub** — ingest, windows, classify, gate, WS fan-out.
- **WindowBuilder** — 100-sample windows, hop 50 samples. First log line needs 100 packets per phone.
- **StubCnn** — always no-fall. **FakeCnn** — test double that can fire a fall.
- **EscalationGate** — log every inference; POST `/events` only if fall and confidence ≥ threshold. After a POST, the same `node_id` is quiet for 3 s so overlapping windows do not open two Telegram ladders.

---

## Cloud / server (`server/`)

FastAPI process. Default `http://127.0.0.1:8001`. No UDP. No dashboard.

**In:** `FallEvent` and `AckEvent` JSON. Senior **I'm fine** / **I need help** taps (and typed yes/no in the senior chat) are polled from Telegram on the 0.5 s tick — no webhook.

**Out:** `EscalationCase` JSON; Telegram `sendMessage` to family, senior, and secondary. Twilio adapter exists but is unused.

### Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `POST` | `/events` | `FallEvent` | `EscalationCase`, **202**. Dedupes on `event_id`. t+0: Telegram next of kin + Telegram senior check-in |
| `POST` | `/cases/{case_id}/ack` | `AckEvent` (`case_id` in the path wins if it disagrees with the body). Senior `yes` closes as all-clear. Optional; Telegram buttons do the same. | Updated case, or **404** |
| `GET` | `/cases/{case_id}` | — | Current case, or **404** |
| `GET` | `/health` | — | `{"ok": true}` |

Startup starts a 0.5 s `on_tick` loop (not an HTTP route): polls Telegram `getUpdates` (skips backlog on boot); senior silence for 60 s becomes `no_answer`; after `no_answer` / `not_fine`, wait 60 s then Telegram secondary; CareLine stub at t+180.

Family Telegram copy: `OPOYO: fall. Room {room}. {local time}. confidence {0.00}.`  
Senior Telegram copy asks them to tap **I'm fine** or **I need help** (or reply **yes** / **no**). Typed replies in the senior chat bind to the newest open check-in.

`.env` for a live tree: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID_NEXT_OF_KIN`, `TELEGRAM_CHAT_ID_SECONDARY`, `TELEGRAM_CHAT_ID_SENIOR`.

---

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Fill Telegram / senior phone for live alerts. Keep `EDGE_ENABLE_UDP=1`. Do not commit `.env`.

Two terminals:

```bash
# edge — dashboard + UDP
python -m uvicorn edge.app:app --host 0.0.0.0 --port 8000

# cloud
python -m uvicorn server.app:app --host 127.0.0.1 --port 8001
```

Or FastAPI CLI (`pip install "fastapi[standard]"`):

```bash
fastapi dev edge/app.py --host 0.0.0.0 --port 8000
fastapi dev server/app.py --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8000 and http://127.0.0.1:8001/health.

On boot the edge should print `[edge] UDP listening on 0.0.0.0:9000`. Then start a phone or `fake_phone.py`.

---

## Tests

```bash
pytest
```

`tests/test_edge.py` boots a live edge, sends fake UDP, reads `/ws`, and writes `edge/data/inference.jsonl` (gitignored via `data/`).

---

## Also on this tree (from `main`)

- `android/` collector APK (CSV share). Same floor job as iOS Collect.
- `brand/` lockup and square mark. Edge dashboard uses `edge/static/logo.png`.
- `opoyo/` and `scripts/record.py` are the older laptop kinematic pipeline. Live path is still edge → cloud.

---

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

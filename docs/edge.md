# Edge

The edge is the only process that sees raw vibration and audio. It lives in `src/edge/`. Default bind: HTTP `:8000`, UDP JSON `:9000`, UDP PCM `:9001`.

## What it does

1. Accept `SensorSample` JSON and `OPYA` PCM from phones.
2. Keep up to five nodes, keyed by UUID. Join order is Phone 1 / room 1, …
3. Build 2 s windows (1 s hop, 50 Hz → 100 samples).
4. Join time-aligned PCM from a 4 s ring (fail-open if the clip is missing).
5. Classify. Log every `InferenceResult` to `src/edge/data/inference.jsonl`.
6. POST a `FallEvent` to the cloud only if `is_fall` and confidence ≥ `escalate_min_confidence` (0.50 in `config.yaml`), after the 3 s cooldown.
7. Serve the operator dashboard and a WebSocket of per-packet ticks.

Axes, windows, and PCM never leave this process.

## HTTP

| Method | Path | Output |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/static/*` | CSS, images |
| `GET` | `/api/state` | Snapshot: slots, combined mag/dB, histories, latest inference, last 50 log rows, read-only `cfg` |
| WebSocket | `/ws` | First message `{k: "state"}`. Then `{k: "tick"}` per ingested packet |

There is no sample ingest over HTTP, no config POST, and no phone-rename route. Knobs are `config.yaml`.

If the log does not print `[edge] UDP listening on 0.0.0.0:9000` and `[edge] UDP PCM listening on 0.0.0.0:9001`, phones are sending into the void. `EDGE_ENABLE_UDP=0` in `.env` (or `enable_udp: false` in YAML) turns the sockets off.

## Internals

**Node.** One physical phone. UUID is the hub key. Slot name `Phone N` is the inference `node_id`; slot number is `room`. Silent for `live_s` (2.5 s) → not live on the dashboard.

**Hub.** Ingest, windows, classify, gate, WebSocket fan-out. An asyncio lock serializes ingest so two packets cannot race the gate.

**WindowBuilder.** Emits a `SensorWindow` every hop once 100 samples have arrived. First log line therefore needs ~2 s of packets per phone.

**PcmRing.** Time-indexed 16 kHz int16, default 4 s. `slice_ms(t_start_ms, t_end_ms)` uses phone timestamps, not arrival order. Coverage &lt; 1 is normal; the classifier then uses mag only.

**Classifier.** `load_runtime()` tries JointCnn (`joint_head.joblib`), then FusionCnn, then MagOnlyCnn. StubCnn is tests-only unless `OPOYO_ALLOW_STUB=1`. See [ml.md](ml.md).

**EscalationGate.** Always appends to the JSONL. POSTs only when `is_fall` and confidence ≥ threshold. Cooldown is per `node_id`: 3 s of **wall clock after the POST returns** and 3 s of **window timestamp**, so overlapping hops from one impact do not open two Telegram ladders. YAMNet + Telegram latency used to eat a wall-clock-only cooldown; both clocks are required.

**Dashboard.** Two columns. Left: live readings for Phone 1 (identity, mag/sound/rate/packets, mag/sound/frequency/ax/ay/az plots). Right: inference log, newest first, scroll for older rows. No QnA page. No night theme.

## Config that belongs here

From `config.yaml` → `edge` and `alert`:

- UDP/HTTP bind
- `max_nodes`, `max_history`, `live_s`, `rate_s`
- `window_s`, `hop_s`, `sensor_hz`
- `pcm_rate_hz`, `pcm_ring_s`
- `escalate_min_confidence`
- `cloud_url`, `cloud_post_timeout_s`
- `alert.cooldown_s`

`CLOUD_URL` in the environment overrides YAML (Compose sets `http://server:8001`).

## Run

```bash
python -m uvicorn edge.app:app --host 0.0.0.0 --port 8000
```

Or with the server, `docker compose up --build`. Dashboard: http://127.0.0.1:8000

# OPOYO

Passive floor-vibration fall detector. Ellipsis Tech Series 2026 prototype.

The product node is a piezo puck bonded to the slab. This repo uses a **phone on the tile** as a proxy: 50 Hz user-acceleration plus 16 kHz microphone PCM, streamed to a laptop **edge** process. The edge classifies each 2 s window. Only a gated fall leaves the machine, as a `FallEvent` posted to the **cloud**. The cloud runs a Telegram escalation ladder (family, senior, then secondary). Raw axes and PCM never go to the cloud.

Deeper dives: [phone](docs/phone.md) · [edge](docs/edge.md) · [server](docs/server.md) · [machine learning](docs/ml.md)

## Architecture

```
  iOS / Android / fake_phone
           │  UDP JSON :9000  (SensorSample, 50 Hz)
           │  UDP PCM  :9001  (OPYA 16 kHz int16)
           ▼
  edge  ── HTTP :8000 dashboard + /ws
           WindowBuilder 2 s / 1 s hop
           JointCnn or FusionCnn (mag + YAMNet)
           EscalationGate (threshold + 3 s cooldown)
           │
           │  POST /events   only if is_fall and confidence ≥ 0.50
           ▼
  server ─ HTTP :8001
           DecisionTree  →  Telegram family + senior at t+0
                            family "I'm on it" or senior "I'm fine" can stop
                            else secondary at t+60, CareLine stub at t+180
```

Shared Pydantic types live in `src/shared/schemas.py`: `SensorSample` → `SensorWindow` → `InferenceResult` → `FallEvent` → `EscalationCase`. Knobs are in `config.yaml`. Telegram tokens stay in `.env`.

| Piece | Role | Default |
|---|---|---|
| Phone | Sense; UDP only | JSON `:9000`, PCM `:9001` |
| Edge | Window, classify, gate, dashboard | HTTP `:8000` |
| Server | Cases and Telegram ladder | HTTP `:8001` |
| ML | Train heads offline; edge loads `src/ml/models/` | `python -m train.fit` |

## Repository layout

```
src/phone/     iOS app, Android collector, fake_phone.py
src/edge/      FastAPI hub, dashboard static, inference.jsonl
src/server/    FastAPI cloud + Telegram adapters
src/shared/    schemas, PCM codec, features, config loader
src/ml/        train/, models/, scripts/
src/tests/
docs/          this set of deep-dives, plus papers and original brief
config.yaml    runtime knobs (ports, window, threshold, ladder timers)
```

## Components

**Phone.** iOS (`src/phone/ios`) streams live UDP for the demo. Android (`src/phone/android`) records CSV takes for training. `python -m phone.fake_phone` synthesizes packets without a device. See [docs/phone.md](docs/phone.md).

**Edge.** One process owns UDP ingest (max five phones), 2 s windows, the classifier, the gate, and the operator dashboard. Inference is always logged; the cloud is contacted only after the gate. See [docs/edge.md](docs/edge.md).

**Server.** Stateless HTTP plus an in-memory decision tree. t+0 Telegram to next of kin (**I'm on it**) and the senior (**I'm fine** / **I need help**). Cooldown reuses the open case so overlapping hops do not open two ladders. See [docs/server.md](docs/server.md).

**Machine learning.** Offline training in `src/ml/train`. Live load order is JointCnn → FusionCnn → MagOnlyCnn. YAMNet is fail-open: missing PCM or a TF Hub error falls back to vibration only. See [docs/ml.md](docs/ml.md).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env             # Unix: cp .env.example .env
```

Fill Telegram chat ids in `.env`. Keep `EDGE_ENABLE_UDP=1`. Then two terminals:

```bash
python -m uvicorn edge.app:app --host 0.0.0.0 --port 8000
python -m uvicorn server.app:app --host 127.0.0.1 --port 8001
```

Dashboard: http://127.0.0.1:8000  
Health: http://127.0.0.1:8001/health  

The edge must print `[edge] UDP listening on 0.0.0.0:9000`. Then start a phone or:

```bash
python -m phone.fake_phone --nodes 1 --impact knee --seconds 8
```

## Docker

Needs Docker Desktop and a filled `.env`. Compose mounts `config.yaml` into both containers and sets `CLOUD_URL=http://server:8001`.

```bash
docker compose up --build
```

Same ports as local. `docker compose down` to stop.

## Config

Edit `config.yaml` and restart (or `docker compose restart`). Environment still wins for `CLOUD_URL` and `EDGE_ENABLE_UDP`. Telegram tokens stay in `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID_NEXT_OF_KIN`, `TELEGRAM_CHAT_ID_SECONDARY`, `TELEGRAM_CHAT_ID_SENIOR`.

## Tests

```bash
pytest
```

`src/tests/test_edge.py` boots a live edge, sends fake UDP, and writes `src/edge/data/inference.jsonl`.

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

# OPOYO

Passive floor-vibration fall detector. This repo is the hackathon prototype.

The production node is a piezo puck bonded to the slab. This build uses iPhones on the floor as a proxy: accelerometer plus a sound-level meter, streamed to an **edge** process. A stub 1D CNN classifies each 2 s window. The **cloud** process runs the escalation ladder. Telegram fires only after a gated fall.

## Live path

1. iOS app `phone/OpoyoPhone` reads user-acceleration at 50 Hz and microphone RMS in dB. Screen stays on while streaming.
2. Packets go over UDP JSON to the edge (`v, id, model, t, ax, ay, az, mag, db`).
3. Edge FastAPI listens on UDP 9000 and shows phones at http://127.0.0.1:8000
4. `edge/infer.py` `StubCnn.infer` returns `{timestamp, is_fall, confidence}`. Every result is appended to `edge/data/inference.jsonl`.
5. Edge POSTs a `FallEvent` to the cloud only when `is_fall` and `confidence >= EDGE_ESCALATE_MIN_CONFIDENCE` (default 0.90, also on the dashboard).
6. Cloud FastAPI on http://127.0.0.1:8001 runs Figure A5: Telegram family + stub senior call at t+0, family wait 60 s, secondary stub, CareLine stub at t+180.

Each phone carries a UUID created on first launch. Combined traces use max magnitude and max dB. Axes stay per phone. Raw vibration never leaves the edge process.

Swap `StubCnn` for a trained 1D CNN with the same `infer(window) -> InferenceResult` signature.

## Run both processes

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn edge.app:app --host 0.0.0.0 --port 8000
python3 -m uvicorn server.app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8000

Copy `.env.example` to `.env` and fill Telegram. Do not commit `.env`.

Without a phone, a synthetic stream:

```bash
python3 phone/fake_phone.py --nodes 1 --impact knee --seconds 4
```

The stub CNN always returns no-fall, so this will fill the inference log and will not create a cloud case.

## iPhone app

Needs Xcode 26, a paid Apple Developer team, and the physical iPhone on the same Wi-Fi or a personal hotspot.

```bash
cd phone
xcodegen generate
open OpoyoPhone.xcodeproj
```

On first launch allow Microphone and Local Network. Type the Mac LAN IP. Port `9000`. Start. Put the phone face-down on tile.

If the phone is the hotspot, the Mac address is almost always `172.20.10.11`. Campus Wi-Fi often blocks UDP. Hotspot is the trusted demo network.

## Packet

```json
{"v":2,"id":"c0a1...","model":"iPhone 17 Pro Max","t":1735689600123,"ax":0.01,"ay":0.0,"az":-0.02,"mag":0.03,"db":-41.2}
```

## Tests

```bash
pytest
```

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

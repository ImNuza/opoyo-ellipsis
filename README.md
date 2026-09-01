# OPOYO

Passive floor-vibration fall detector. This repo is the hackathon prototype.

The production node is a piezo puck bonded to the slab. This build uses iPhones on the floor as a proxy: accelerometer plus a sound-level meter, streamed to a Mac. A kinematic rule on the Mac decides suspect / recovered / confirmed. Telegram fires. No trained model yet.

## Live path

1. iOS app `phone/OpoyoPhone` reads user-acceleration at 50 Hz and microphone RMS in dB. Screen stays on while streaming.
2. Packets go over UDP JSON to the Mac (`v, id, model, t, ax, ay, az, mag, db`).
3. FastAPI on the Mac listens on UDP 9000 and shows up to five phones at http://127.0.0.1:8000
4. `server/detect.py` scores combined magnitude (peak, rise, decay). Suspect starts a 10 s demo clock. Cancel is on the dashboard. Extra motion recovers. Timeout confirms.
5. Telegram: t+0 on suspect, t+10 on confirmed. Target chat is `TELEGRAM_TARGET` in `.env`.

Each phone carries a UUID created on first launch. Combined traces use max magnitude and max dB. Axes stay per phone.

CS later: replace `Detector.tick` with a model that emits the same `FallEvent`. Disagree with the rule → no Telegram. QnA on the dashboard lists env knobs.

Not in this slice: YAMNet, 1D-CNN, piezo hardware, true background streaming.

## Mac receiver

```bash
cd "/Users/dewa/Documents/Projects/Ellipsis Tech Series/opoyo-ellipsis"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000

Copy `.env.example` to `.env` and fill Telegram. Do not commit `.env`.

Without a phone, a synthetic knee-drop:

```bash
python3 server/fake_phone.py --nodes 1 --impact knee --seconds 4
```

A book-shaped spike (should stay Quiet):

```bash
python3 server/fake_phone.py --nodes 1 --impact book --seconds 3
```

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

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

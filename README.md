# OPOYO

Passive floor-vibration fall detector. This repo is the hackathon prototype.

The production node is a piezo puck bonded to the slab. This build uses an iPhone on the floor as a proxy: accelerometer plus a sound-level meter, streamed to a Mac dashboard. No model yet.

## Today (signals only)

1. iOS app `phone/OpoyoPhone` reads device-motion user-acceleration at 50 Hz and microphone RMS in dB.
2. Packets go over UDP JSON to the Mac (`v, t, ax, ay, az, mag, db`).
3. FastAPI on the Mac listens on UDP 9000 and shows a live strip at http://127.0.0.1:8000

Not in this slice: fusion, YAMNet, Telegram, Twilio, piezo hardware.

## Mac receiver

```bash
cd "/Users/dewa/Documents/Projects/Ellipsis Tech Series/opoyo-ellipsis"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000

Without a phone:

```bash
python3 server/fake_phone.py
```

## iPhone app

Needs Xcode 26, a paid Apple Developer team, and the physical iPhone on the same Wi-Fi or a personal hotspot.

```bash
cd phone
xcodegen generate
open OpoyoPhone.xcodeproj
```

In Xcode: select the iPhone, Signing & Capabilities, team Dewa, Run.

On first launch allow Microphone and Local Network. Type the Mac LAN IP shown by `ipconfig getifaddr en0` (or the hotspot IP, usually `172.20.10.2` when the phone is the hotspot). Port `9000`. Start sensing. Put the phone face-down on tile.

If the phone is the hotspot, the Mac address is almost always `172.20.10.2`. If both are on campus Wi-Fi, use the Mac `en0` address.

## Packet

```json
{"v":1,"t":1735689600123,"ax":0.01,"ay":0.0,"az":-0.02,"mag":0.03,"db":-41.2}
```

`ax, ay, az` are user acceleration in g (gravity removed). `mag` is the vector length. `db` is 20*log10(rms).

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

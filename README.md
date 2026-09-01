# OPOYO

Passive floor-vibration fall detector. This repo is the hackathon prototype.

The production node is a piezo puck bonded to the slab. This build uses iPhones on the floor as a proxy: accelerometer plus a sound-level meter, streamed to a Mac dashboard. No model yet.

## Today (signals only)

1. iOS app `phone/OpoyoPhone` reads device-motion user-acceleration at 50 Hz and microphone RMS in dB.
2. Packets go over UDP JSON to the Mac (`v, id, model, t, ax, ay, az, mag, db`).
3. FastAPI on the Mac listens on UDP 9000 and shows up to five phones at http://127.0.0.1:8000

Each phone carries a UUID created on first launch. The dashboard names cards Phone 1, Phone 2 in join order. Rename on the Mac sticks to that UUID after quit. Combined traces use max magnitude and max dB across live phones. Axes stay per phone.

Not in this slice: fusion, YAMNet, Telegram, Twilio, piezo hardware, 1D-CNN.

## Mac receiver

```bash
cd "/Users/dewa/Documents/Projects/Ellipsis Tech Series/opoyo-ellipsis"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000

Without a phone, three synthetic nodes:

```bash
python3 server/fake_phone.py --nodes 3 --seconds 12
```

Reuse the same ids on the next run (rename test):

```bash
python3 server/fake_phone.py --nodes 3 --ids id-a,id-b,id-c --seconds 8
```

## iPhone app

Needs Xcode 26, a paid Apple Developer team, and the physical iPhone on the same Wi-Fi or a personal hotspot.

```bash
cd phone
xcodegen generate
open OpoyoPhone.xcodeproj
```

In Xcode: select the iPhone, Signing and Capabilities, team Dewa, Run.

On first launch allow Microphone and Local Network. Type the Mac LAN IP. Port `9000`. Start sensing. Put the phone face-down on tile.

If the phone is the hotspot, the Mac address is almost always `172.20.10.11`. If both are on campus Wi-Fi, use the Mac `en0` address. Campus Wi-Fi often blocks UDP. Hotspot is the trusted demo network.

Simulator: host `127.0.0.1`, port `9000`.

## Packet

```json
{"v":2,"id":"c0a1...","model":"iPhone 17 Pro Max","t":1735689600123,"ax":0.01,"ay":0.0,"az":-0.02,"mag":0.03,"db":-41.2}
```

`ax, ay, az` are user acceleration in g (gravity removed). `mag` is the vector length. `db` is 20*log10(rms). Combined floor is max mag and max dB across live phones.

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

# OPOYO

A senior falls, cannot get up, and cannot call. The harm is time on the floor (a long lie). Wall buttons, wearables, and cameras fail this segment: they need a press, a worn charged device, or a camera in the rooms people refuse.

OPOYO is a floor sensor. Production is a piezo and a microphone bonded to the slab; classification stays on the device and only event JSON leaves the flat. This repo is the hackathon prototype of that pipeline.

The pipeline is sequential: high-pass the floor vibration, trigger on a quiet-floor threshold, clip a short window, score seven features, optionally suppress with an audio check, then confirm (recovered / cancelled / alert). Only an alert sends Telegram. See `docs/OPOYO-briefing.md` for numbers and what is still a placeholder.

The demo sensor is an iPhone on the floor, not a slab transducer. It streams 50 Hz user-acceleration and a scalar dB over UDP. Nyquist is 25 Hz, so spectral features are weak; envelope features may still separate a bag from a book on hard tile. That is empirical and unmeasured until you have labelled CSVs. The phone does not measure the slab, and this build does not claim it does.

## Setup

Python 3.12. From `opoyo-ellipsis`:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**Mac**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill Telegram fields in `.env` when you want live messages. Leave `escalate.dry_run: true` in `config.yaml` until then. Do not commit `.env`.

## Run

From `opoyo-ellipsis`, with the venv active:

```bash
python -m uvicorn opoyo.server:app --host 127.0.0.1 --port 8000
```

Dashboard: http://127.0.0.1:8000

The server also binds UDP **9000** for phone packets.

## Phone

iOS app: `phone/OpoyoPhone`. Needs a Mac, Xcode, and a physical iPhone.

```bash
cd phone
xcodegen generate
open OpoyoPhone.xcodeproj
```

Same Wi-Fi as the laptop, or a phone hotspot. Campus Wi-Fi often blocks UDP; hotspot is the default. If the phone is the hotspot, the laptop address is often `172.20.10.11`. Port `9000`. Allow Microphone and Local Network. Put the phone face-down on hard floor (tile or vinyl), case off if you can.

Packets are JSON: `{v,id,model,t,ax,ay,az,mag,db}`.

## Record

Do not run the recorder at the same time as the server (both want UDP 9000).

```bash
python scripts/record.py --interactive
```

Labelling procedure: `docs/data-collection.md`. One event per file.

## Fake phone (no hardware)

Server must be running.

```bash
python scripts/fake_phone.py --impact heeldrop --seconds 8
```

## Tests

```bash
pytest -q
```

## Team

Dewa, Matteo, Gilchris, Shannon, Sonia. Ellipsis Tech Series 2026 Track 1.

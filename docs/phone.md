# Phone

The phone is a **sensor**, not a classifier. It never talks to the cloud. It samples the floor and sends UDP to the edge on the local network.

The product node is a piezo puck. This repo uses a phone face-down on tile as a proxy: gravity-removed acceleration (g) plus microphone.

## Producers

| Source | Path | Job |
|---|---|---|
| iOS | `src/phone/ios/OpoyoPhone` | Live UDP stream for the demo |
| Android | `src/phone/android` | Record CSV takes; share the file for training |
| Fake | `src/phone/fake_phone.py` | Synthetic UDP without a device |

## JSON channel (`:9000`)

One `SensorSample` object per tick, 50 Hz, UDP, no handshake.

```json
{"v":2,"id":"c0a1...","model":"iPhone 17 Pro Max","t":1735689600123,"ax":0.01,"ay":0.0,"az":-0.02,"mag":0.03,"db":-41.2}
```

| Field | Notes |
|---|---|
| `v` | Packet version. Use `2`. |
| `id` | Stable UUID created on first launch. Hub key. |
| `model` | Display only. |
| `t` | Phone unix time, milliseconds. |
| `ax, ay, az` | User-acceleration, **g**. |
| `mag` | `sqrt(ax² + ay² + az²)`. Required. |
| `db` | Mic RMS as dB. Default `-120` if unused. |

Extra keys are ignored. The edge stamps `_nid` (from `id`) and `from` (`ip:port`); the phone does not send those.

## PCM channel (`:9001`)

16 kHz mono int16, **not** a WAV file. Same `id` and unix `t_ms` as JSON so the edge can join the two streams. Typical frame: 20 ms = 320 samples.

Little-endian header (`src/shared/pcm.py`):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | Magic `OPYA` |
| 4 | 1 | Version `1` |
| 5 | 4 | seq uint32 |
| 9 | 8 | `t_ms` uint64 (unix ms of sample 0) |
| 17 | 16 | node UUID bytes |
| 33 | 2 | sample rate (16000) |
| 35 | 2 | n samples |
| 37 | 2n | PCM s16le mono |

iOS resamples the mic tap to 16 kHz before send.

## iOS live stream

Needs Xcode, a paid team, and the phone on the same Wi-Fi **or** a personal hotspot. Campus Wi-Fi often blocks phone-to-laptop UDP; hotspot is the trusted demo network. If the phone is the hotspot, the laptop is almost always `172.20.10.11`.

```bash
cd src/phone/ios
xcodegen generate
open OpoyoPhone.xcodeproj
```

On first launch allow **Microphone** and **Local Network**. Type the laptop LAN IP. Port `9000` (PCM uses `9001`). Start. Face-down on tile.

iOS will not stream CoreMotion or mic in the background. The screen stays on while streaming. `DeviceIdentity.nodeId` is the UUID stored on device; that is `SensorSample.id`.

The app can also capture a labelled take (heeldrop, bag, walk, …) as CSV for `data/takes/`.

## Android collector

Open `src/phone/android/` in Android Studio, run on a physical phone, allow microphone. Same CSV header as iOS: `t,ax,ay,az,mag,db`. Axes are linear acceleration ÷ 9.81 so they match iOS **g**. Share the file (Telegram is fine) and drop it under `data/takes/<label>/` on the laptop.

Android is for **dataset collection**, not the live UDP demo, unless you extend it.

## Fake phone

Talks to the **edge UDP port**, not the cloud.

```bash
python -m phone.fake_phone --nodes 1 --impact knee --seconds 8
```

| Flag | Meaning |
|---|---|
| `--host` / `--port` | JSON UDP target (default `127.0.0.1:9000`) |
| `--pcm-port` | PCM UDP (default `port+1`) |
| `--no-pcm` | JSON only |
| `--hz` | Sample rate (default 50) |
| `--seconds` | Duration |
| `--nodes` | How many fake devices (max 5 on the edge) |
| `--ids` | Comma-separated UUIDs so slots stay stable |
| `--impact` | Waveform: omit (quiet), `knee` (~0.72 g), `book` (~1.2 g) |

Packets do not carry a `knee`/`book` label. The live classifier **will** score real heads on this waveform; `FakeCnn` is only used in tests.

Keep slots stable across runs:

```bash
python -m phone.fake_phone --nodes 2 --ids 11111111-1111-1111-1111-111111111111,22222222-2222-2222-2222-222222222222 --impact book --seconds 8
```

## What the phone does not do

- No HTTP API.
- No Telegram.
- No on-device fall model in this prototype (production intent is an ESP32 on the puck; see the architecture discussion in the README).
- No cloud credentials.

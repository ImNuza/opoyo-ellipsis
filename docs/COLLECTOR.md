# Collector cheat sheet

One event per file. You are recording **floor vibration**, not a movie of a fall.

## Phone (what the team should use)

No laptop. Phone **face-down** on hard tile, case off. Allow microphone.

**iPhone:** rebuild `phone/OpoyoPhone` in Xcode.  
**Android:** open `android/` in Android Studio, Run on the device.

1. Open OPOYO. Pick a label (`heeldrop`, `bag`, `book`, …).
2. Tap **Record**. Stand still 2 s.
3. Do **exactly one** action.
4. Stand still 2 s. Tap **Stop take**.
5. Tap **Send …csv** → Telegram (or Files / WhatsApp) to Gilchris.

Filename is `heeldrop_20260901_211530.csv`. Header: `t,ax,ay,az,mag,db` (axes in g). Dump every CSV into `data/` on the laptop. Fill `data/metadata.csv` once when you copy them over (floor, room, distance).

## Laptop recorder (optional, live dashboard only)

Phone streams UDP to the Mac. Stop uvicorn first (both bind **9000**).

```powershell
python scripts/record.py --floor tile --room kitchen --distance 1.5 --case off --interactive
```

No phone? `python scripts/fake_phone.py --impact heeldrop --seconds 8`

## One take

1. Start the recorder (label = class).
2. Stand still **2 s**.
3. Do **exactly one** action.
4. Stand still **2 s**.
5. Stop. Next take = new file. Two drops in one file = delete and redo.

## Counts (minimum)

Positives — **not jumping** (a jump leaves the floor):

| label | do this | n |
|---|---|---|
| `heeldrop` | Socks or barefoot. Rise on toes, drop onto both heels, once. | 10 |
| `bag` | Backpack/tote **8–12 kg**, drop flat from ~waist / 1 m. | 10 |

Negatives:

| label | do this | n |
|---|---|---|
| `book` | Hardcover onto the same floor. | 5 |
| `pan` | Metal saucepan or similar. | 5 |
| `door` | Slam a door in the same room. Phone stays on the floor. | 8 |
| `walk` | Walk past the phone, five paces, no stomp. | 8 |
| `quiet` | 15 s, nobody moving. | 2 |

Optional: `tv` — speaker on, no impact, 10 s × 2.

## Metadata (required)

Recorder appends one row per file to `data/metadata.csv`:

```text
filename,label,floor,room,distance_m,phone_model,case,footwear,object,notes
```

Session flags: `--floor tile --room kitchen --distance 1.5 --case off`. Also `--phone-model`, `--footwear`, `--object`, `--notes`.

- `floor`: `tile` | `vinyl` | `laminate` | `carpet` | `concrete`
- `case`: `on` | `off`
- `object`: what you dropped (e.g. `backpack 10kg`, `hardcover`)
- New room or floor = new session, say so in `notes`.

Send `data/` (CSVs + `metadata.csv`). Procedure detail: `docs/data-collection.md`.

# Labelled data collection (for the person recording)

You are recording **floor vibration from a phone**, not a movie of a person falling. Each file is one event. The filename is the label.

## Setup

- Hard floor (tile or vinyl). Not a thick rug.
- iPhone with OpoyoPhone, face-down on the floor, **case off** if you can.
- Same Wi-Fi as the laptop, or a phone hotspot. Campus Wi-Fi often blocks this; hotspot is the default.
- Laptop: recorder running (when it exists: `python scripts/record.py --label LABEL`). It writes one CSV per take: `t,ax,ay,az,mag,db`.
- Phone ↔ impact distance: **1.5–2 m**. Keep it fixed for the whole session. Write it in metadata.
- Quiet room. TV off unless the take is the `tv` class.

Do **not** use jumping as a “fall.” A jump is a different signal (you leave the floor and come back). Positives are **heel-drop** and **bag-drop** only.

## One take

1. Start the recorder with the label for this take.
2. Stand still 2 s (quiet lead-in).
3. Do **exactly one** action (below).
4. Stand still 2 s (quiet tail).
5. Stop. Next take = new file. Do not put two drops in one file.

If you talked, walked extra, or the drop was messy: delete the file and redo. Do not keep bad takes.

## Classes and counts (minimum)

| label | what to do | count |
|---|---|---|
| `heeldrop` | Barefoot or socks. Rise on toes, drop onto both heels, once. Stay still after. | 10 |
| `bag` | Backpack or tote 8–12 kg, drop flat from about waist / 1 m. Once. | 10 |
| `book` | Hardcover book, drop onto the same floor. | 5 |
| `pan` | Metal saucepan or similar, drop. | 5 |
| `door` | Slam a door in the same room. Phone still on the floor. | 8 |
| `walk` | Walk past the phone, five paces, normal. Do not stomp. | 8 |
| `quiet` | 15 s, nobody moving. | 2 |
| `tv` (optional) | TV or speaker on, no impact. 10 s. | 2 |

`heeldrop` and `bag` are **positives**. Everything else is **negative**.

## Metadata (required)

Fill one row per file in `data/metadata.csv`:

```text
filename,label,floor,room,distance_m,phone_model,case,footwear,object,notes
heeldrop_01.csv,heeldrop,tile,kitchen,1.5,iPhone 15,off,socks,,first session
bag_01.csv,bag,tile,kitchen,1.5,iPhone 15,off,,backpack 10kg,
book_01.csv,book,tile,kitchen,1.5,iPhone 15,off,,hardcover,
```

- `floor`: `tile` | `vinyl` | `laminate` | `carpet` | `concrete`
- `distance_m`: phone to where the impact happened
- `case`: `on` | `off`
- `object`: what you dropped (or empty)
- `notes`: TV on, other people, failed then redone, etc.

Floor material matters. Decay and amplitude change with it. If you change room or floor, start a new session and say so in `notes`.

## Video

**Not required.** Optional phone video of the room is only for arguing later about a mislabel. Do not try to sync video frames to the CSV. If you film, name the video the same as the CSV (`heeldrop_01.mp4`) and leave it in `data/video/`. Training uses the CSV only.

## What “good” looks like

- ≥ 48 files, names matching the table.
- Every CSV has a header and more than ~200 rows.
- One action per file.
- `metadata.csv` complete.
- Same phone, same distance, same floor for the main set.

Send the `data/` folder (CSVs + `metadata.csv`). That is the dataset.

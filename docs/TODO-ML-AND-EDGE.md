# OPOYO split: Gilchris (ML) vs friend (edge / phone / cloud)

Repo of record for plumbing: `feat/refactor` on `opoyo-ellipsis`.  
Collect APK (csv + wav): `opoyo-pipeline-clean/android/OPOYO.apk` (also copied to Downloads).  
Do not mix the old csv-only APK.

Live detection and labelled collect are two different paths. Do not treat Record as UDP.

---

## 0. Shared, tonight (both)

- [ ] Uninstall old OPOYO. Install **new** APK: `C:\Users\Gilchris\Downloads\OPOYO.apk`.
- [ ] iPhone: Dewa rebuilds `opoyo-pipeline-clean/phone/` in Xcode (no APK).
- [ ] Record at **venue carpet** and at **house**, same phone, same spot, same distance you will demo.
- [ ] One action per take. 2 s still, action, 2 s still, Stop.
- [ ] Send **both** files: `stem.csv` and `stem.wav` (same stem).
- [ ] Drop into `data/takes/<label>/`. **Folder name is the label.** Ignore that the app may still prefix `heeldrop_` on the filename.
- [ ] Labels for this round:
  - Positive: `heeldrop` only (many takes, not 5).
  - Negative: whatever you will actually drop on stage (`key`, `bottle`, `bag`, `book`, …) plus `walk` / `quiet` if people will walk.
- [ ] Do not jump unless you want a separate `jump` negative folder.
- [ ] Hard tile: extra set if you can, **same labels**, different folder tree e.g. `data/takes-tile/<label>/`. Does not replace venue carpet.

---

## 1. Gilchris — ML

Goal: `infer(window) -> InferenceResult` that is better than `StubCnn`, trained on the new csv+wav.

### Data

- [ ] Keep label = parent folder. Never parse class from filename.
- [ ] Pair wav to csv by stem. Skip csv-only files for any **sound** model. Vibration-only may still use csv-only.
- [ ] Peak-normalize vibration clips (carpet / close range is not a constant; still divide out peak so distance does not dominate).
- [ ] Do not train 10-way. Binary: `heeldrop` vs rest.
- [ ] Eval: **stratified 5-fold** (honest). 60/20/20 is ~3/1/1 positives until you have many heel-drops; print it but do not quote it.
- [ ] Augment **train folds only**: time shift, noise, stretch, polarity, mix-quiet. Never augment the test fold.
- [ ] Report precision/recall/F1 and confusion. Do not invent field false-alarm rates.

### Vibration model

- [ ] Input: 2 s mag (and optional `spec_mag`). 50 Hz → ~100 samples. Nyquist 25 Hz.
- [ ] Carpet set: mag traces of heel-drop vs key/bag **look similar**. Do not expect mag-only to carry the demo.
- [ ] Plug class in `edge/infer.py` next to `StubCnn`:
  - `infer(self, window: SensorWindow) -> InferenceResult`
  - set `is_fall`, `confidence` in `[0, 1]`
  - copy `node_id`, `room`, `timestamp=window.t_end_ms`
- [ ] Friend’s live windows are **2 s length, 1 s hop**, not bang-then-cut. Train on files first; if live is noisy, ask friend to switch to trigger later.

### Sound model (the one that can still work on carpet)

- [ ] Read the 16 kHz wav. FFT (and/or a small 1-D CNN) on a window around the impact.
- [ ] Keys/bottle = high-pitch; heel-drop/jump = thud. Jump may need a **longer** air window (hop then hit).
- [ ] `spec_air` on `SensorWindow` is the slot. Today it is `|rfft(db)|` dummy. Fill from wav in **train**. Live fill needs friend (below).
- [ ] Fuse: vibration score + air score → one `is_fall` / `confidence`. Write down the rule (e.g. air veto if high-pitch, or max, or logistic on two scores).

### Hand-off to friend

- [ ] Weights on disk: e.g. `models/probe.joblib` and/or `models/backbone.npz` + `models/head.npz`.
- [ ] One class friend can construct with no args (or a path).
- [ ] `python -m train.fit` (or equivalent) documented.
- [ ] Tests: dummy window in, `InferenceResult` out, score in `[0, 1]`, no throw on zeros.
- [ ] Do not change UDP, dashboard, Telegram, or `DecisionTree`.

---

## 2. Friend — everything else

Do not retrain. Do not relabel CSVs.

### Phone (live)

- [ ] Keep UDP JSON as today: `{v,id,model,t,ax,ay,az,mag,db}` port 9000.
- [ ] Collect path: ship/rebuild the **csv+wav** app (copy from `opoyo-pipeline-clean` if `feat/refactor` still csv-only).
- [ ] Optional for **live sound CNN**: send mic PCM (or a compact spectrum) with the stream. Not required for mag-only. Without this, live `spec_air` cannot be a real FFT.
- [ ] Do not send 16 kHz over UDP unless you have a plan (bandwidth). Spectrum bins or a side channel is enough.

### Edge

- [ ] Leave `WindowBuilder`, hub, dashboard, `EscalationGate` as-is unless Gilchris asks for trigger windows.
- [ ] Wire `create_app(classifier=...)` to Gilchris’s class when weights exist. Default stays `StubCnn` so tests stay green.
- [ ] If live sound exists: fill `SensorWindow.spec_air` / `spec_hz` from that, not from `db`.
- [ ] Gate is still `is_fall and confidence >= 0.90`. If the probe uses 0.50, **agree the cut** (change gate or scale confidence). Do not silently mix 0.50 and 0.90.
- [ ] Only gated `FallEvent` POSTs to cloud. Raw mag/wav stay on the edge.

### Cloud

- [ ] Telegram ladder unchanged (t+0 family + senior, t+60 secondary, t+180 CareLine stub).
- [ ] Twilio stays unused unless you add it.

### Tests / demo laptop

- [ ] `pytest` on `feat/refactor` still passes with `StubCnn`.
- [ ] One live dry run: fake_phone or real phone → dashboard → (if FakeCnn) Telegram dry path.
- [ ] Venue: same Wi-Fi/hotspot, UDP 9000 free, edge `:8000`, cloud `:8001`.

---

## 3. Demo day (both)

- [ ] Recalibrate on **that carpet, that phone, that spot** if you have time (more heel-drops + the exact objects).
- [ ] Script: heel-drop should alert; key/bottle should not. Walk past once.
- [ ] If mag-only model is weak, lean on **sound** for the object vs thud distinction. Say so if a judge asks.
- [ ] Do not quote the old 50-csv 5-fold F1 (0.20) as the demo metric.

---

## 4. Not this week

- Piezo puck, on-device MCU, Twilio voice, CareLine real API, YAMNet unless wav + 16 kHz live exist, gait.

---

## Quick map

| Piece | Who | Now |
|---|---|---|
| csv+wav Record APK | Friend ships; Gilchris already built in pipeline-clean | Ready in pipeline-clean; not in feat/refactor until copied |
| UDP `mag`+`db` | Friend | Ready |
| Live wav/spectrum on UDP | Friend, if live sound CNN | Not done |
| `StubCnn` slot | Friend wires; Gilchris fills | Stub always no-fall |
| Train on `data/takes/<label>/` | Gilchris | Loader exists; need new wavs |
| `spec_air` from real wav | Gilchris train; Friend live | Dummy on `db` |
| Telegram / dashboard / gate | Friend | Ready |

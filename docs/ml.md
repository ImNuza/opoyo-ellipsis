# Machine learning

Training is offline. The edge loads frozen heads from `src/ml/models/` at process start. Nothing in `src/ml/train` runs on the live UDP path.

## Live load order

`edge.infer.load_runtime()`:

1. **JointCnn** if `joint_head.joblib` and `mag_head.joblib` exist — one logistic over `[6 hand features | 1024-d YAMNet embedding]`.
2. Else **FusionCnn** if mag + yamnet + fuse heads exist — two logistics fused on `[p_mag, p_yam]`.
3. Else **MagOnlyCnn** — vibration features only.
4. Else hard fail. `StubCnn` is tests-only unless `OPOYO_ALLOW_STUB=1` (a stub never fires a fall, so a missing model would look like a quiet flat).

YAMNet is **fail-open**. If the window has no PCM (`pcm` shorter than 16 samples) or TF Hub throws, the vibration head stands. The committed copy under `src/ml/models/tfhub/` is pointed at via `TFHUB_CACHE_DIR` so the first inference does not download on venue Wi-Fi.

`is_fall` on the model is `confidence >= 0.5`. The **edge gate** then requires `confidence >= escalate_min_confidence` (0.50 in `config.yaml`) before POST `/events`.

## Features

`src/shared/features.py` is the single vector used at train and serve time (six floats: `rms`, `crest`, `decay_ms`, `db_med`, `low`, `cent`). The window is peak-normalised before shape features so carpet and range drop out. Crest is measured on the **raw** clip because it is a dimensionless peak-to-RMS ratio.

## Training

```bash
python -m train.fit data/takes
```

Folder name = label. `heeldrop` and `jump` are treated as positives. Writes:

| File | Head |
|---|---|
| `src/ml/models/mag_head.joblib` | Vibration logistic + scaler |
| `src/ml/models/yamnet_head.joblib` | Frozen YAMNet embed → logistic |
| `src/ml/models/fuse_head.joblib` | `[p_mag, p_yam]` logistic |
| `src/ml/models/joint_head.joblib` | Concatenated mag features + YAMNet embed |

Mag is in **g**. Wav is in **[-1, 1]**. Each clip is peak-normalised on its own. Each head has its own `StandardScaler` from training.

Takes live in `data/takes/<label>/` as `*.csv` (and optional `*.wav`). Collect with the iOS capture UI or the Android APK; see [phone.md](phone.md).

## Other scripts

`src/ml/train/` also has probes and borrowed-backbone experiments (LIMU-BERT, PANNs, ssl-wearables, Tarkett). Those are research; they are not loaded by the edge. `src/ml/scripts/preflight.py` checks models, TF Hub cache, `.env`, and UDP ports before a demo:

```bash
python -m scripts.preflight
```

## Production note

JointCnn / YAMNet will not run on an ESP32. Distill a tiny mag (or mag+tiny audio) TFLite head from the same labelled takes for on-puck inference. Keep this stack as the laptop lab and, if needed, a cloud second look on rare clips.

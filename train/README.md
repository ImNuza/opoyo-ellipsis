# train/

Offline only. Runtime loads heads from `models/` in `edge/infer.py` (`FusionCnn`).

```
python -m train.fit "C:\path\to\dataset"
```

Folder name = label. `heeldrop` = positive. Writes `models/mag_head.joblib`, `yamnet_head.joblib`, `fuse_head.joblib`.

**Scale:** mag is in g, wav is in [-1, 1]. Each clip is peak-normalized on its own (carpet/distance). Each head has its own `StandardScaler` from training. Fusion sees two probabilities already in [0, 1], then scales those.

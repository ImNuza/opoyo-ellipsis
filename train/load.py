"""Load takes. Label = parent folder. Filenames are ignored (this batch is all heeldrop_*)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAKES = ROOT / "data" / "takes"


def label_of(path: Path) -> str:
    return path.parent.name.lower()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    t, ax, ay, az, mag, db = [], [], [], [], [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            ax.append(float(row.get("ax") or 0))
            ay.append(float(row.get("ay") or 0))
            az.append(float(row.get("az") or 0))
            mag.append(float(row["mag"]))
            db.append(float(row.get("db") or -120))
    t_a = np.asarray(t, dtype=np.float64)
    dt = np.diff(t_a) / 1000.0 if t_a.size > 1 else np.array([0.02])
    fs = float(1.0 / np.median(dt)) if dt.size else 50.0
    return {
        "t": t_a,
        "ax": np.asarray(ax, dtype=np.float64),
        "ay": np.asarray(ay, dtype=np.float64),
        "az": np.asarray(az, dtype=np.float64),
        "mag": np.asarray(mag, dtype=np.float64),
        "db": np.asarray(db, dtype=np.float64),
        "fs": np.array(fs),
    }


def iter_takes(root: Path | None = None):
    """Yield (label, path, arrays). Skips unknown nested junk."""
    base = root or DEFAULT_TAKES
    if not base.is_dir():
        return
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        for csv_path in sorted(folder.glob("*.csv")):
            yield label_of(csv_path), csv_path, load_csv(csv_path)

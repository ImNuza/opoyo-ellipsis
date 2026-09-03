#!/usr/bin/env python3
"""Print per-class medians. Label = directory. Do not trust the filename."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.features import vector  # noqa: E402
from train.load import iter_takes  # noqa: E402

POS = {"heeldrop"}


def main() -> None:
    by: dict[str, list[np.ndarray]] = defaultdict(list)
    n = 0
    for label, path, arr in iter_takes():
        fs = float(arr["fs"])
        by[label].append(vector(arr["mag"], arr["db"], fs))
        n += 1
        print(f"{label:10s} {path.name}  peak={by[label][-1][0]:.4f}g  decay={by[label][-1][3]:.0f}ms")
    if not n:
        print("no data/takes/<label>/*.csv")
        sys.exit(1)
    print()
    print(f"{'label':10s} n  peak_g  crest  decay_ms  db  low8  cent_Hz")
    for label, vecs in sorted(by.items()):
        m = np.median(np.stack(vecs), axis=0)
        print(
            f"{label:10s} {len(vecs):2d} {m[0]:7.4f} {m[2]:6.1f} {m[3]:8.0f} {m[4]:6.1f} {m[5]:5.2f} {m[6]:7.1f}"
        )
    print()
    print("filename is not the label. mag-FFT (low8, cent) is a 0–25 Hz placeholder, not a thud spectrum.")
    print("spec_air in SensorWindow is |rfft(db)| until WAV exists.")


if __name__ == "__main__":
    main()

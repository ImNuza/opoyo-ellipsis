from pathlib import Path

import numpy as np

from train.features import peak_normalize
from train.labels import y_of
from train.load import iter_takes, label_of, load_csv


def test_label_is_parent_dir_not_stem(tmp_path: Path):
    d = tmp_path / "bag"
    d.mkdir()
    p = d / "heeldrop_20260902_000000.csv"
    p.write_text("t,ax,ay,az,mag,db\n1,0,0,0,0.01,-40\n2,0,0,0,0.02,-41\n", encoding="utf-8")
    assert label_of(p) == "bag"
    got = list(iter_takes(tmp_path))
    assert len(got) == 1
    assert got[0][0] == "bag"
    arr = load_csv(p)
    assert arr["mag"].tolist() == [0.01, 0.02]
    assert y_of("heeldrop") == 1
    assert y_of("bag") == 0
    assert y_of("jump") == 1  # jump landing is a body-weight impact, same class as heeldrop
    n = peak_normalize(np.array([0.0, 0.02, -0.01]))
    assert abs(float(np.max(np.abs(n))) - 1.0) < 1e-9


def test_augment_stays_normalized_and_same_length():
    from train.augment import expand

    rng = np.random.default_rng(0)
    w = peak_normalize(np.linspace(-0.2, 1.0, 80))
    quiet = peak_normalize(np.ones(80) * 0.01)
    got = expand(w, 1, rng, quiet, copies=5)
    assert len(got) == 6
    for g in got:
        assert g.shape == w.shape
        assert abs(float(np.max(np.abs(g))) - 1.0) < 1e-6 or float(np.max(np.abs(g))) == 0.0

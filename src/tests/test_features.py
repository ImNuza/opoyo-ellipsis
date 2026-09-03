import numpy as np

from opoyo.features import FEATURE_NAMES, extract
from opoyo.synth import decaying


def test_all_features_present_and_finite():
    f = extract(decaying(100, 0.3, 12), 100)
    assert set(f) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())


def test_body_decays_slower_than_rigid_object():
    # abs() envelope zeros every half-cycle; 8 Hz vs 40 Hz keeps tau visible
    soft = extract(decaying(100, 0.45, 8), 100)
    hard = extract(decaying(100, 0.04, 40), 100)
    assert soft["decay_ms"] > hard["decay_ms"] * 2


def test_body_has_more_low_frequency_energy():
    soft = extract(decaying(100, 0.45, 8), 100)
    hard = extract(decaying(100, 0.04, 40), 100)
    assert soft["low_ratio"] > hard["low_ratio"]


def test_short_window_returns_zeros_not_error():
    assert extract(np.array([1.0, 2.0]), 100)["peak"] == 0.0
    assert extract(np.array([1.0, 2.0]), 100) == {k: 0.0 for k in FEATURE_NAMES}

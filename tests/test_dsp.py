import numpy as np

from opoyo.dsp import Trigger, mad, make_highpass, rolling_rms, StreamingFilter
from opoyo.ring import RingBuffer


def test_ring_wraps_and_preserves_order():
    r = RingBuffer(5)
    r.push([1, 2, 3])
    assert list(r.last(3)) == [1, 2, 3]
    r.push([4, 5, 6, 7])
    assert list(r.last(5)) == [3, 4, 5, 6, 7]
    assert r.total == 7


def test_ring_oversized_push_keeps_tail():
    r = RingBuffer(4)
    r.push(np.arange(10))
    assert list(r.last(4)) == [6, 7, 8, 9]


def test_ring_pads_when_empty():
    r = RingBuffer(8)
    assert r.last(4).tolist() == [0, 0, 0, 0]


def test_highpass_removes_gravity():
    fs = 100
    sig = np.full(fs * 5, 9.81, dtype=np.float32)
    f = StreamingFilter(make_highpass(2.0, fs))
    out = f(sig)
    assert abs(out[-fs:].mean()) < 0.05


def test_streaming_filter_matches_chunked():
    fs, rng = 100, np.random.default_rng(0)
    sig = rng.normal(0, 1, fs * 3).astype(np.float32)
    whole = StreamingFilter(make_highpass(2.0, fs))(sig)
    f = StreamingFilter(make_highpass(2.0, fs))
    chunked = np.concatenate([f(sig[i : i + 37]) for i in range(0, sig.size, 37)])
    assert np.allclose(whole, chunked, atol=1e-4)


def test_rolling_rms_length_and_value():
    x = np.ones(100)
    r = rolling_rms(x, 10)
    assert r.size == 100 and abs(r[-1] - 1.0) < 1e-6


def test_mad_of_constant_is_zero():
    assert mad(np.full(50, 3.0)) == 0.0


def test_trigger_ignores_quiet_and_fires_on_impact():
    fs = 100
    t = Trigger(fs=fs, k_mad=6.0, min_peak_mag=1.0, refractory_s=1.0)
    rng = np.random.default_rng(1)
    for _ in range(40):
        fired, _ = t.process(rng.normal(0, 0.05, 10))
        assert not fired
    impact = np.zeros(10)
    impact[3] = 12.0
    fired, peak = t.process(impact)
    assert fired and peak > 1.0


def test_trigger_refractory_blocks_double_fire():
    fs = 100
    t = Trigger(fs=fs, k_mad=6.0, min_peak_mag=1.0, refractory_s=2.0)
    rng = np.random.default_rng(2)
    for _ in range(40):
        t.process(rng.normal(0, 0.05, 10))
    spike = np.zeros(10)
    spike[3] = 12.0
    assert t.process(spike)[0] is True
    assert t.process(spike)[0] is False


def test_threshold_is_median_plus_k_mad():
    fs = 100
    t = Trigger(fs=fs, k_mad=6.0, min_peak_mag=100.0, refractory_s=0.0)
    for _ in range(8):
        t.process(np.full(10, 0.2))
    h = np.asarray(t._hist)
    expected = float(np.median(h) + 6.0 * max(mad(h), 1e-3))
    assert abs(t.threshold() - expected) < 1e-9


def test_trigger_baseline_not_poisoned_by_impacts():
    fs = 100
    t = Trigger(fs=fs, k_mad=6.0, min_peak_mag=1.0, refractory_s=0.0)
    rng = np.random.default_rng(3)
    for _ in range(40):
        t.process(rng.normal(0, 0.05, 10))
    before = t.threshold()
    for _ in range(10):
        spike = np.zeros(10)
        spike[3] = 15.0
        t.process(spike)
    assert t.threshold() < before * 1.5

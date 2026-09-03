from __future__ import annotations

import numpy as np
import pytest

from edge.pcm_ring import PcmRing

RATE = 16000
FRAME_N = 320  # 20 ms
FRAME_MS = 20


def _s16(n: int, value: int = 1000) -> bytes:
    return np.full(n, value, dtype=np.int16).tobytes()


def _append_span(ring: PcmRing, t0: int, duration_ms: int, value: int = 1000, seq0: int = 0) -> None:
    seq = seq0
    t = t0
    end = t0 + duration_ms
    while t < end:
        ring.append(t, _s16(FRAME_N, value), seq=seq)
        seq += 1
        t += FRAME_MS


def test_full_two_second_slice_has_32000_samples():
    ring = PcmRing(rate=RATE, seconds=4.0)
    _append_span(ring, 1000, 2000)
    clip, coverage = ring.slice_ms(1000, 3000)
    assert clip.dtype == np.float32
    assert clip.size == 32000
    assert float(np.max(np.abs(clip))) <= 1.0
    assert coverage == 1.0


def test_int16_scale_to_unit_float():
    ring = PcmRing(rate=RATE, seconds=4.0)
    peak = np.array([32767, 0], dtype=np.int16).tobytes()
    ring.append(0, peak, seq=0)
    clip, coverage = ring.slice_ms(0, 1)  # 16 samples at 16 kHz; first two set
    assert coverage > 0
    assert clip[0] == pytest.approx(32767 / 32768.0, rel=1e-5)
    assert clip[1] == 0.0


def test_empty_ring_slice_is_empty():
    ring = PcmRing(rate=RATE, seconds=4.0)
    clip, coverage = ring.slice_ms(0, 2000)
    assert clip.size == 0
    assert coverage == 0.0


def test_partial_overlap_keeps_span_zeros_holes():
    ring = PcmRing(rate=RATE, seconds=4.0)
    # Only the second half of [1000, 3000): [2000, 3000)
    _append_span(ring, 2000, 1000, value=8000)
    clip, coverage = ring.slice_ms(1000, 3000)
    assert clip.size == 32000
    assert abs(coverage - 0.5) <= 0.02
    first_half = clip[:16000]
    second_half = clip[16000:]
    assert float(np.max(np.abs(first_half))) == 0.0
    assert float(np.max(np.abs(second_half))) > 0.0


def test_gap_zero_filled_does_not_slide_later_samples():
    ring = PcmRing(rate=RATE, seconds=4.0)
    ring.append(1000, _s16(FRAME_N, 5000), seq=0)
    # skip 20 ms (seq 1), then continue — 20 ms < 200 ms gap budget
    ring.append(1040, _s16(FRAME_N, 9000), seq=2)
    clip, coverage = ring.slice_ms(1000, 1060)
    assert clip.size == 960  # 60 ms * 16
    assert coverage < 1.0
    # first 20 ms present, next 20 ms zeros, last 20 ms present
    assert float(np.max(np.abs(clip[:320]))) > 0.0
    assert float(np.max(np.abs(clip[320:640]))) == 0.0
    assert float(np.max(np.abs(clip[640:960]))) > 0.0


def test_capacity_drops_oldest_keeps_newest():
    ring = PcmRing(rate=RATE, seconds=4.0)
    t0 = 1000
    _append_span(ring, t0, 5000, value=2000)
    old, old_cov = ring.slice_ms(t0, t0 + 1000)
    assert old.size == 0 or old_cov == 0.0
    newest, new_cov = ring.slice_ms(t0 + 3000, t0 + 5000)
    assert newest.size == 32000
    assert new_cov == 1.0

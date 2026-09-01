from __future__ import annotations

from server.detect import Detector, DetectorConfig


DT = 0.02  # 50 Hz


def _cfg(**kwargs) -> DetectorConfig:
    base = dict(
        peak_g=0.4,
        quiet_g=0.05,
        min_peak_samples=2,
        rise_ms=80.0,
        decay_min_ms=150.0,
        decay_max_ms=400.0,
        decay_hold_ms=80.0,
        recover_g=0.15,
        recover_after_ms=500.0,
        escalate_s=10.0,
        cooldown_s=0.4,
    )
    base.update(kwargs)
    return DetectorConfig(**base)


def _feed(det: Detector, mags: list[float], t0: float = 0.0, node: str = "n1") -> list[dict]:
    changes: list[dict] = []
    t = t0
    for mag in mags:
        changes.extend(det.tick(t, mag, node))
        t += DT
    return changes


def _knee(quiet_before: int = 25) -> list[float]:
    rise = [0.01 + (0.72 - 0.01) * (i + 1) / 3 for i in range(3)]  # 60 ms
    peak = [0.72, 0.72]
    decay = [0.72 - (0.72 - 0.01) * (i + 1) / 10 for i in range(10)]  # 200 ms
    hold = [0.01] * 8
    return [0.01] * quiet_before + rise + peak + decay + hold


def test_quiet_floor_stays_quiet():
    det = Detector(_cfg())
    changes = _feed(det, [0.01] * 80)
    assert changes == []
    assert det.public(1.6)["state"] == "quiet"


def test_knee_shaped_envelope_goes_suspect():
    det = Detector(_cfg())
    changes = _feed(det, _knee())
    states = [c["state"] for c in changes]
    assert "suspect" in states
    pub = det.public(len(_knee()) * DT)
    assert pub["state"] == "suspect"
    assert pub["banner"].startswith("Suspect")
    assert pub["event"]["peak_g"] >= 0.4
    assert pub["event"]["node_id"] == "n1"


def test_slow_rise_is_not_a_fall():
    det = Detector(_cfg())
    rise = [0.01 + 0.55 * (i + 1) / 25 for i in range(25)]  # 500 ms
    peak = [0.56, 0.56]
    decay = [0.56 - 0.55 * (i + 1) / 10 for i in range(10)]
    hold = [0.01] * 8
    changes = _feed(det, [0.01] * 20 + rise + peak + decay + hold)
    assert all(c["state"] != "suspect" for c in changes)


def test_short_decay_book_is_not_a_fall():
    det = Detector(_cfg())
    rise = [0.2, 1.1]
    peak = [1.2, 1.2]
    decay = [0.4, 0.01]  # 40 ms
    hold = [0.01] * 10
    changes = _feed(det, [0.01] * 20 + rise + peak + decay + hold)
    assert all(c["state"] != "suspect" for c in changes)


def test_periodic_hops_fail_decay_hold():
    det = Detector(_cfg())
    mags = [0.01] * 15
    for _ in range(6):
        mags.extend([0.7, 0.7, 0.2, 0.01, 0.01])
    changes = _feed(det, mags)
    assert all(c["state"] != "suspect" for c in changes)


def test_cancel_from_suspect():
    det = Detector(_cfg())
    _feed(det, _knee())
    t = len(_knee()) * DT
    ev = det.cancel(t)
    assert ev is not None
    assert ev.state == "cancelled"
    assert det.public(t)["state"] == "cancelled"


def test_recovery_motion_after_impact():
    det = Detector(_cfg())
    mags = _knee()
    _feed(det, mags)
    t = len(mags) * DT
    assert det.public(t)["state"] == "suspect"
    # 500 ms after peak: still in hold, then a stand-up bump
    rest = [0.01] * 20 + [0.22, 0.28, 0.18] + [0.04] * 5
    changes = _feed(det, rest, t0=t)
    assert any(c["state"] == "recovered" for c in changes)
    assert det.public(t + len(rest) * DT)["state"] == "recovered"


def test_timeout_confirms_when_still():
    det = Detector(_cfg(escalate_s=0.5))
    mags = _knee()
    first = _feed(det, mags)
    t = len(mags) * DT
    assert any(c["state"] == "suspect" for c in first)
    rest = [0.01] * 20  # 0.4 s more still, total > 0.5 s from peak
    changes = _feed(det, rest, t0=t)
    assert any(c["state"] == "confirmed" for c in changes)


def test_update_peak_g():
    det = Detector(_cfg())
    det.update_cfg(peak_g=0.2)
    assert det.cfg.peak_g == 0.2


def test_short_decay_records_reject():
    det = Detector(_cfg())
    rise = [0.2, 1.1]
    peak = [1.2, 1.2]
    decay = [0.4, 0.01]
    hold = [0.01] * 10
    _feed(det, [0.01] * 20 + rise + peak + decay + hold)
    assert det.last_reject.startswith("decay")


def test_simple_smash_triggers_without_shape():
    det = Detector(_cfg(simple=True, peak_g=0.2, min_peak_samples=1))
    mags = [0.01] * 20 + [0.05, 0.9, 0.4, 0.01] + [0.01] * 8
    changes = _feed(det, mags)
    assert any(c["state"] == "suspect" for c in changes)


def test_cancel_does_nothing_when_quiet():
    det = Detector(_cfg())
    _feed(det, [0.01] * 10)
    assert det.cancel(0.2) is None

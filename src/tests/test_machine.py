from types import SimpleNamespace

import numpy as np

from opoyo.machine import EventMachine


class _Clf:
    def __init__(self, p):
        self.p = p

    def score(self, feats):
        return self.p


class _Audio:
    def __init__(self, blocked=False, label="", score=0.0):
        self._blocked = blocked
        self._label = label
        self._score = score

    def check(self, audio):
        return self._blocked, self._label, self._score


def _cfg(window_s=2.0, threshold=0.60, recovery_k=3.0, room="Living room"):
    return SimpleNamespace(
        room=room,
        sensor=SimpleNamespace(accel_rate_hz=50),
        classify=SimpleNamespace(fall_threshold=threshold),
        confirm=SimpleNamespace(window_s=window_s, recovery_k=recovery_k),
    )


def _machine(score, blocked=False, window_s=2.0):
    t = [0.0]
    events = []
    states = []

    def on_event(e):
        events.append(e)
        states.append(e.state)  # snapshot; FallEvent is mutated in place

    m = EventMachine(
        _Clf(score),
        _Audio(blocked=blocked, label="tv" if blocked else "", score=0.9 if blocked else 0.0),
        _cfg(window_s=window_s),
        now_fn=lambda: t[0],
        on_event=on_event,
        fs=50.0,
    )
    return m, t, events, states


def _window():
    return np.ones(50, dtype=np.float32)


def test_low_score_is_candidate_no_alert():
    m, t, events, states = _machine(0.2)
    ev = m.on_trigger(_window(), np.array([]), quiet_rms=0.05)
    assert ev is not None
    assert ev.state == "candidate"
    assert m.phase == "idle"
    assert "alert" not in states


def test_high_score_audio_block_no_alert():
    m, t, events, states = _machine(0.9, blocked=True)
    ev = m.on_trigger(_window(), np.array([]), quiet_rms=0.05)
    assert ev is not None
    assert ev.state == "blocked"
    assert m.phase == "idle"
    assert "alert" not in states


def test_high_score_rms_rise_recovers():
    m, t, events, states = _machine(0.9, blocked=False)
    ev = m.on_trigger(_window(), np.array([]), quiet_rms=0.05)
    assert ev.state == "candidate"
    assert m.phase == "confirming"
    t[0] = 0.5
    got = m.tick(1.0)
    assert got is not None
    assert got.state == "recovered"
    assert m.phase == "idle"
    assert "alert" not in states
    assert "recovered" in states


def test_high_score_still_alerts_exactly_once():
    m, t, events, states = _machine(0.9, blocked=False, window_s=2.0)
    m.on_trigger(_window(), np.array([]), quiet_rms=0.05)
    assert m.phase == "confirming"
    t[0] = 2.0
    first = m.tick(0.0)
    second = m.tick(0.0)
    t[0] = 4.0
    third = m.tick(0.0)
    assert first is not None and first.state == "alert"
    assert second is None and third is None
    assert states.count("alert") == 1


def test_cancel_from_confirming():
    m, t, events, states = _machine(0.9, blocked=False)
    m.on_trigger(_window(), np.array([]), quiet_rms=0.05)
    assert m.phase == "confirming"
    got = m.cancel()
    assert got is not None
    assert got.state == "cancelled"
    assert m.phase == "idle"
    assert "cancelled" in states
    assert m.cancel() is None

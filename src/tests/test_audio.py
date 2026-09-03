import numpy as np

from opoyo.audio_check import AudioCheck


def test_disabled_is_fail_open():
    chk = AudioCheck(classes=["Television"], enabled=False)
    assert chk.check(np.full(32, -18.0)) == (False, "", 0.0)


def test_empty_db_is_fail_open():
    chk = AudioCheck(classes=["Television"], enabled=True)
    chk.ok = False
    assert chk.check(np.array([])) == (False, "", 0.0)


def test_broken_check_is_fail_open():
    chk = AudioCheck(classes=["Television"], enabled=True)
    chk.ok = False

    def boom(_w):
        raise RuntimeError("boom")

    chk._envelope = boom  # type: ignore[method-assign]
    blocked, label, score = chk.check(np.full(32, -18.0))
    assert blocked is False
    assert label == ""
    assert score == 0.0


def test_sustained_loudness_may_suppress():
    chk = AudioCheck(classes=["Television"], enabled=True)
    chk.ok = False
    db = np.full(32, -20.0)
    blocked, label, score = chk.check(db)
    assert blocked is True
    assert label == "sustained_loudness"
    assert score > 0.0

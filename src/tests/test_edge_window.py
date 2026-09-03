from __future__ import annotations

from shared.schemas import SensorSample
from edge.window import WindowBuilder


def _samples(n: int, hz: float = 50.0, t0: int = 1_000_000, node: str = "n1") -> list[SensorSample]:
    dt = int(round(1000.0 / hz))
    out: list[SensorSample] = []
    for i in range(n):
        out.append(
            SensorSample(
                v=2,
                id=node,
                model="iPhone",
                t=t0 + i * dt,
                ax=0.0,
                ay=0.0,
                az=0.0,
                mag=0.01,
                db=-40.0,
            )
        )
    return out


def test_two_second_window_has_100_samples():
    builder = WindowBuilder(
        window_s=2.0, hop_s=1.0, hz=50.0, node_id="Phone 1", room=1
    )
    window = None
    for sample in _samples(100, node="abc"):
        window = builder.push(sample) or window
    assert window is not None
    assert len(window.mag) == 100
    assert abs((window.t_end_ms - window.t_start_ms) - 2000) < 50
    assert window.node_id == "Phone 1"
    assert window.room == 1


def test_incomplete_buffer_returns_none():
    builder = WindowBuilder()
    got = None
    for sample in _samples(50):
        got = builder.push(sample)
        assert got is None
    assert got is None


def test_hop_one_second_yields_two_windows_from_three_seconds():
    builder = WindowBuilder(
        window_s=2.0, hop_s=1.0, hz=50.0, node_id="Phone 2", room=2
    )
    windows = []
    for sample in _samples(150, node="n9"):
        window = builder.push(sample)
        if window is not None:
            windows.append(window)
    assert len(windows) == 2
    assert windows[0].node_id == "Phone 2"
    assert windows[0].room == 2
    assert windows[1].room == 2

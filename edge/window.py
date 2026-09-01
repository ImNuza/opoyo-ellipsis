from __future__ import annotations

from shared.schemas import SensorSample, SensorWindow


class WindowBuilder:
    def __init__(
        self,
        window_s: float = 2.0,
        hop_s: float = 1.0,
        hz: float = 50.0,
        room: str = "",
    ) -> None:
        self.window_s = window_s
        self.hop_s = hop_s
        self.hz = hz
        self.room = room
        self._buf: list[SensorSample] = []
        self._start = 0

    @property
    def window_n(self) -> int:
        return max(1, int(round(self.window_s * self.hz)))

    @property
    def hop_n(self) -> int:
        return max(1, int(round(self.hop_s * self.hz)))

    def push(self, sample: SensorSample, room: str | None = None) -> SensorWindow | None:
        if room is not None:
            self.room = room
        self._buf.append(sample)
        n = self.window_n
        if len(self._buf) < self._start + n:
            return None
        chunk = self._buf[self._start : self._start + n]
        self._start += self.hop_n
        cutoff = max(0, self._start - n)
        if cutoff:
            self._buf = self._buf[cutoff:]
            self._start -= cutoff
        return SensorWindow(
            node_id=chunk[-1].id,
            room=self.room,
            t_start_ms=chunk[0].t,
            t_end_ms=chunk[-1].t,
            hz=self.hz,
            mag=[s.mag for s in chunk],
            ax=[s.ax for s in chunk],
            ay=[s.ay for s in chunk],
            az=[s.az for s in chunk],
            db=[s.db for s in chunk],
        )

"""Sliding windows over a single phone's 50 Hz stream.

Defaults are 2 s windows and a 1 s hop (100 samples, hop 50).
"""

from __future__ import annotations

from shared.schemas import SensorSample, SensorWindow


class WindowBuilder:
    """Return a SensorWindow every hop once a full window of samples has arrived."""
    def __init__(
        self,
        window_s: float = 2.0,
        hop_s: float = 1.0,
        hz: float = 50.0,
        node_id: str = "",
        room: int = 1,
    ) -> None:
        self.window_s = window_s
        self.hop_s = hop_s
        self.hz = hz
        self.node_id = node_id
        self.room = room
        self._buf: list[SensorSample] = []
        self._start = 0

    @property
    def window_n(self) -> int:
        return max(1, int(round(self.window_s * self.hz)))

    @property
    def hop_n(self) -> int:
        return max(1, int(round(self.hop_s * self.hz)))

    def push(
        self,
        sample: SensorSample,
        node_id: str | None = None,
        room: int | None = None,
    ) -> SensorWindow | None:
        """Append one sample and emit a window when the hop is due.

        Args:
            sample: Latest 50 Hz packet.
            node_id: Optional dashboard name (Phone N).
            room: Optional slot number.

        Returns:
            SensorWindow, or None until ``window_n`` samples sit in the buffer.
        """
        if node_id is not None:
            self.node_id = node_id
        if room is not None:
            self.room = room
        self._buf.append(sample)
        n = self.window_n
        if len(self._buf) < self._start + n:
            return None
        chunk = self._buf[self._start : self._start + n]
        self._start += self.hop_n
        # Drop samples that can no longer sit in a future window.
        cutoff = max(0, self._start - n)
        if cutoff:
            self._buf = self._buf[cutoff:]
            self._start -= cutoff
        return SensorWindow(
            node_id=self.node_id or chunk[-1].id,
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

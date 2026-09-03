"""Time-indexed PCM ring.

Slices by phone unix milliseconds, not datagram arrival order, so a late UDP
frame still lands at the correct offset in the window.
"""

from __future__ import annotations

import numpy as np


class PcmRing:
    """Fixed-capacity int16 ring addressed by phone timestamps."""

    def __init__(self, rate: float = 16000.0, seconds: float = 4.0) -> None:
        self.rate = float(rate)
        self.cap = max(1, int(round(self.rate * seconds)))
        self.buf = np.zeros(self.cap, dtype=np.int16)
        self.ok = np.zeros(self.cap, dtype=np.uint8)
        self.t0_ms: float | None = None
        self.n = 0

    def _index(self, t_ms: float) -> int:
        assert self.t0_ms is not None
        return int(round((t_ms - self.t0_ms) * self.rate / 1000.0))

    def _drop_left(self, drop: int) -> None:
        if drop <= 0 or self.n <= 0:
            return
        drop = min(drop, self.n)
        keep = self.n - drop
        if keep:
            self.buf[:keep] = self.buf[drop : self.n]
            self.ok[:keep] = self.ok[drop : self.n]
        self.buf[keep : self.n] = 0
        self.ok[keep : self.n] = 0
        if self.t0_ms is not None:
            self.t0_ms += drop * 1000.0 / self.rate
        self.n = keep

    def append(self, t_ms: int, pcm_s16: bytes, seq: int = 0) -> None:
        """Write a frame at ``t_ms``. Gaps are zero-filled; overflow drops the left."""
        del seq  # Sequence is for debugging on the phone; the ring is time-based.
        pcm = np.frombuffer(pcm_s16, dtype=np.int16).copy()
        if pcm.size == 0:
            return
        if self.t0_ms is None:
            self.t0_ms = float(t_ms)
            take = min(pcm.size, self.cap)
            self.buf[:take] = pcm[-take:]
            self.ok[:take] = 1
            self.n = take
            return

        i = self._index(float(t_ms))
        if i + pcm.size <= 0:
            return
        if i < 0:
            pcm = pcm[-i:]
            i = 0
        end = i + pcm.size
        if end > self.cap:
            self._drop_left(end - self.cap)
            i = self._index(float(t_ms))
            if i < 0:
                pcm = pcm[-i:]
                i = 0
            end = i + pcm.size
            if end > self.cap:
                pcm = pcm[: max(0, self.cap - i)]
                end = i + pcm.size
        if pcm.size == 0 or i >= self.cap:
            return
        if i > self.n:
            self.buf[self.n : i] = 0
            self.ok[self.n : i] = 0
        self.buf[i:end] = pcm
        self.ok[i:end] = 1
        self.n = max(self.n, end)

    def buffered_ms(self) -> float:
        """Return how many milliseconds of audio are currently held."""
        if self.n <= 0:
            return 0.0
        return self.n * 1000.0 / self.rate

    def slice_ms(self, t_start_ms: int, t_end_ms: int) -> tuple[np.ndarray, float]:
        """Return a float32 clip in [-1, 1] and the fraction of samples present.

        Missing samples stay zero. Coverage is 0.0 when the ring has nothing
        overlapping the request, which is the fail-open path for the classifier.
        """
        empty = np.zeros(0, dtype=np.float32)
        if self.t0_ms is None or self.n <= 0:
            return empty, 0.0
        expected = int(round((t_end_ms - t_start_ms) * self.rate / 1000.0))
        if expected <= 0:
            return empty, 0.0
        i0 = self._index(float(t_start_ms))
        i1 = i0 + expected
        if i1 <= 0 or i0 >= self.n:
            return empty, 0.0
        out = np.zeros(expected, dtype=np.int16)
        ok = np.zeros(expected, dtype=np.uint8)
        src0 = max(0, i0)
        src1 = min(self.n, i1)
        dst0 = src0 - i0
        dst1 = dst0 + (src1 - src0)
        out[dst0:dst1] = self.buf[src0:src1]
        ok[dst0:dst1] = self.ok[src0:src1]
        coverage = float(ok.sum() / expected)
        return out.astype(np.float32) / 32768.0, coverage

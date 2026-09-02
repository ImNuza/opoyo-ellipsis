import threading

import numpy as np


class RingBuffer:
    """Fixed-size circular float buffer. Thread-safe for one writer, many readers."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._n = 0
        self._lock = threading.Lock()

    def push(self, samples) -> None:
        s = np.asarray(samples, dtype=np.float32).ravel()
        if s.size == 0:
            return
        if s.size >= self.capacity:
            s = s[-self.capacity:]
        with self._lock:
            start = self._n % self.capacity
            end = start + s.size
            if end <= self.capacity:
                self._buf[start:end] = s
            else:
                cut = self.capacity - start
                self._buf[start:] = s[:cut]
                self._buf[: end - self.capacity] = s[cut:]
            self._n += s.size

    def last(self, k: int) -> np.ndarray:
        """Most recent k samples, oldest first. Zero-padded if not yet filled."""
        k = min(int(k), self.capacity)
        with self._lock:
            if self._n == 0:
                return np.zeros(k, dtype=np.float32)
            end = self._n % self.capacity
            out = np.concatenate([self._buf[end:], self._buf[:end]])
            return out[-k:].copy()

    @property
    def total(self) -> int:
        return self._n

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields
from uuid import uuid4


@dataclass(frozen=True)
class DetectorConfig:
    peak_g: float = 0.4
    quiet_g: float = 0.05
    min_peak_samples: int = 2
    rise_ms: float = 80.0
    decay_min_ms: float = 150.0
    decay_max_ms: float = 400.0
    decay_hold_ms: float = 80.0
    recover_g: float = 0.15
    recover_after_ms: float = 500.0
    escalate_s: float = 10.0
    cooldown_s: float = 2.0
    buffer_s: float = 1.2
    simple: bool = False


@dataclass
class FallEvent:
    id: str
    t0: float
    peak_g: float
    rise_ms: float
    decay_ms: float
    node_id: str
    state: str
    remaining_s: float = 0.0

    def public(self) -> dict:
        return {
            "id": self.id,
            "t0": self.t0,
            "peak_g": round(self.peak_g, 4),
            "rise_ms": round(self.rise_ms, 1),
            "decay_ms": round(self.decay_ms, 1),
            "node_id": self.node_id,
            "state": self.state,
            "remaining_s": round(self.remaining_s, 1),
        }


def _banner(state: str, remaining_s: float) -> str:
    if state == "suspect":
        left = max(0, int(round(remaining_s)))
        return f"Suspect · {left} s to confirm"
    if state == "recovered":
        return "Recovered"
    if state == "cancelled":
        return "Cancelled"
    if state == "confirmed":
        return "Confirmed"
    return "Quiet"


@dataclass
class Detector:
    cfg: DetectorConfig = field(default_factory=DetectorConfig)
    buf: deque = field(default_factory=deque)
    event: FallEvent | None = None
    last_mag: float = 0.0
    last_node: str = ""
    last_hit_g: float = 0.0
    last_reject: str = ""
    _terminal_at: float | None = None
    _armed_peak_t: float | None = None

    def update_cfg(self, **kwargs) -> DetectorConfig:
        allowed = {item.name for item in fields(DetectorConfig)}
        current = {item.name: getattr(self.cfg, item.name) for item in fields(DetectorConfig)}
        for key, value in kwargs.items():
            if key not in allowed or value is None:
                continue
            current[key] = value
        self.cfg = DetectorConfig(**current)
        self._armed_peak_t = None
        return self.cfg

    def tick(self, now: float, mag: float | None = None, node_id: str = "") -> list[dict]:
        if mag is not None:
            self.last_mag = mag
            if node_id:
                self.last_node = node_id
            self.buf.append((now, mag, node_id or self.last_node))
            cutoff = now - self.cfg.buffer_s
            while self.buf and self.buf[0][0] < cutoff:
                self.buf.popleft()
            self.last_hit_g = max((sample[1] for sample in self.buf), default=0.0)
        else:
            mag = self.last_mag
            node_id = node_id or self.last_node

        changes: list[dict] = []
        if self.event and self.event.state == "suspect":
            elapsed = now - self.event.t0
            self.event.remaining_s = max(0.0, self.cfg.escalate_s - elapsed)
            if elapsed >= self.cfg.recover_after_ms / 1000.0 and mag >= self.cfg.recover_g:
                self.event.state = "recovered"
                self.event.remaining_s = 0.0
                self._terminal_at = now
                changes.append(self._change("recovered"))
            elif elapsed >= self.cfg.escalate_s:
                self.event.state = "confirmed"
                self.event.remaining_s = 0.0
                self._terminal_at = now
                changes.append(self._change("confirmed"))
            return changes

        if self.event and self.event.state in {"recovered", "cancelled"}:
            if self._terminal_at is not None and now - self._terminal_at >= self.cfg.cooldown_s:
                self.event = None
            return changes

        if self.event and self.event.state == "confirmed":
            return changes

        found = self._try_trigger()
        if found is not None:
            self.event = found
            self.last_reject = ""
            changes.append(self._change("suspect"))
        return changes

    def cancel(self, now: float) -> FallEvent | None:
        if self.event is None:
            return None
        if self.event.state == "suspect":
            self.event.state = "cancelled"
            self.event.remaining_s = 0.0
            self._terminal_at = now
            return self.event
        if self.event.state in {"confirmed", "recovered", "cancelled"}:
            self.event = None
            self._terminal_at = None
        return None

    def public(self, now: float | None = None) -> dict:
        ev = self.event
        state = ev.state if ev else "quiet"
        remaining = ev.remaining_s if ev else 0.0
        if ev and ev.state == "suspect" and now is not None:
            remaining = max(0.0, self.cfg.escalate_s - (now - ev.t0))
            ev.remaining_s = remaining
        return {
            "state": state,
            "banner": _banner(state, remaining),
            "remaining_s": round(remaining, 1),
            "event": ev.public() if ev else None,
            "last_hit_g": round(self.last_hit_g, 4),
            "last_reject": self.last_reject,
            "cfg": {
                "peak_g": self.cfg.peak_g,
                "quiet_g": self.cfg.quiet_g,
                "min_peak_samples": self.cfg.min_peak_samples,
                "rise_ms": self.cfg.rise_ms,
                "decay_min_ms": self.cfg.decay_min_ms,
                "decay_max_ms": self.cfg.decay_max_ms,
                "decay_hold_ms": self.cfg.decay_hold_ms,
                "recover_g": self.cfg.recover_g,
                "recover_after_ms": self.cfg.recover_after_ms,
                "escalate_s": self.cfg.escalate_s,
                "simple": self.cfg.simple,
            },
        }

    def _change(self, state: str) -> dict:
        assert self.event is not None
        payload = self.event.public()
        payload["state"] = state
        payload["banner"] = _banner(state, self.event.remaining_s)
        payload["k"] = "detect"
        return payload

    def _try_trigger(self) -> FallEvent | None:
        samples = list(self.buf)
        if len(samples) < 8:
            return None
        t_end = samples[-1][0]
        peak_i = -1
        peak_mag = -1.0
        for i in range(len(samples) - 1, -1, -1):
            t, mag, _nid = samples[i]
            if t_end - t > 0.7:
                break
            if mag >= self.cfg.peak_g and mag >= peak_mag:
                peak_mag = mag
                peak_i = i
        if peak_i < 0:
            hit = max((sample[1] for sample in samples), default=0.0)
            if hit >= self.cfg.quiet_g:
                self.last_reject = (
                    f"last hit {hit:.2f} g, need {self.cfg.peak_g:.2f} g"
                )
            return None
        peak_t, peak_mag, peak_nid = samples[peak_i]
        if self._armed_peak_t is not None and abs(peak_t - self._armed_peak_t) < 0.05:
            return None

        run = 1
        j = peak_i - 1
        while j >= 0 and samples[j][1] >= self.cfg.peak_g:
            run += 1
            j -= 1
        j = peak_i + 1
        while j < len(samples) and samples[j][1] >= self.cfg.peak_g:
            run += 1
            j += 1
        if run < self.cfg.min_peak_samples:
            self.last_reject = (
                f"peak {peak_mag:.2f} g lasted {run} sample, need {self.cfg.min_peak_samples}"
            )
            return None

        if self.cfg.simple:
            self._armed_peak_t = peak_t
            return FallEvent(
                id=str(uuid4())[:8],
                t0=peak_t,
                peak_g=peak_mag,
                rise_ms=0.0,
                decay_ms=0.0,
                node_id=peak_nid or self.last_node,
                state="suspect",
                remaining_s=self.cfg.escalate_s,
            )

        k = peak_i
        while k > 0 and samples[k][1] >= self.cfg.quiet_g:
            k -= 1
        if samples[k][1] >= self.cfg.quiet_g:
            self.last_reject = "no quiet before peak"
            return None
        rise_ms = (peak_t - samples[k][0]) * 1000.0
        if not (0.0 < rise_ms <= self.cfg.rise_ms):
            self.last_reject = (
                f"rise {rise_ms:.0f} ms, need ≤ {self.cfg.rise_ms:.0f} ms"
            )
            return None

        k = peak_i
        while k < len(samples) - 1 and samples[k][1] >= self.cfg.quiet_g:
            k += 1
        if samples[k][1] >= self.cfg.quiet_g:
            return None
        decay_ms = (samples[k][0] - peak_t) * 1000.0
        if not (self.cfg.decay_min_ms <= decay_ms <= self.cfg.decay_max_ms):
            self.last_reject = (
                f"decay {decay_ms:.0f} ms, need {self.cfg.decay_min_ms:.0f} to {self.cfg.decay_max_ms:.0f} ms"
            )
            return None

        hold_end = samples[k][0] + self.cfg.decay_hold_ms / 1000.0
        if samples[-1][0] < hold_end:
            return None
        for t, mag, _nid in samples[k:]:
            if t > hold_end + 1e-9:
                break
            if mag >= self.cfg.quiet_g:
                self.last_reject = "ringing after decay"
                return None

        self._armed_peak_t = peak_t
        remaining = self.cfg.escalate_s
        return FallEvent(
            id=str(uuid4())[:8],
            t0=peak_t,
            peak_g=peak_mag,
            rise_ms=rise_ms,
            decay_ms=decay_ms,
            node_id=peak_nid or self.last_node,
            state="suspect",
            remaining_s=remaining,
        )

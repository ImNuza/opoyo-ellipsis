from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from opoyo.features import extract
from opoyo.schemas import FallEvent


class EventMachine:
    """idle → (trigger) score+audio → confirming → recovered | alert | cancelled.

    Public event states: candidate, blocked, recovered, alert, cancelled.
    Clock is injected so tests never sleep.
    """

    def __init__(
        self,
        classifier,
        audio_check,
        cfg,
        now_fn: Callable[[], float] | None = None,
        on_event: Callable[[FallEvent], None] | None = None,
        fs: float | None = None,
    ) -> None:
        self.classifier = classifier
        self.audio_check = audio_check
        self.cfg = cfg
        self.now = now_fn or time.time
        self.on_event = on_event or (lambda _e: None)
        self.fs = fs or float(cfg.sensor.accel_rate_hz)
        self.phase = "idle"
        self.event: FallEvent | None = None
        self._t0 = 0.0
        self._recover_rms = 0.0
        self._alerted = False

    def on_trigger(
        self,
        window,
        audio,
        room: str = "",
        node_id: str = "",
        quiet_rms: float = 0.0,
    ) -> FallEvent | None:
        if self.phase == "confirming":
            return None
        fs = self.fs
        feats = extract(window, fs)
        p = float(self.classifier.score(feats))
        blocked, label, ascore = self.audio_check.check(audio)
        room = room or str(self.cfg.room)
        ev = FallEvent.new(
            room,
            ts_unix=self.now(),
            fall_score=p,
            veto_label=label or None,
            veto_score=ascore,
            peak_mag=feats["peak"],
            decay_ms=feats["decay_ms"],
            rise_ms=feats["rise_ms"],
            node_id=node_id,
            remaining_s=float(self.cfg.confirm.window_s),
        )
        threshold = float(self.cfg.classify.fall_threshold)
        if p < threshold:
            ev.state = "candidate"
            self.on_event(ev)
            return ev
        if blocked:
            ev.state = "blocked"
            self.on_event(ev)
            return ev
        self.phase = "confirming"
        self.event = ev
        self._t0 = self.now()
        self._alerted = False
        k = float(self.cfg.confirm.recovery_k)
        self._recover_rms = max(quiet_rms * k, 0.15)
        ev.state = "candidate"
        ev.remaining_s = float(self.cfg.confirm.window_s)
        self.on_event(ev)
        return ev

    def tick(self, rms_now: float) -> FallEvent | None:
        if self.phase != "confirming" or self.event is None:
            return None
        now = self.now()
        elapsed = now - self._t0
        window_s = float(self.cfg.confirm.window_s)
        self.event.remaining_s = max(0.0, window_s - elapsed)
        recover_after = 0.4
        if elapsed >= recover_after and rms_now >= self._recover_rms:
            return self._finish("recovered")
        if elapsed >= window_s and not self._alerted:
            return self._finish("alert")
        return None

    def cancel(self) -> FallEvent | None:
        if self.phase != "confirming" or self.event is None:
            return None
        return self._finish("cancelled")

    def acknowledge(self, event_id: str) -> bool:
        if self.event and self.event.event_id == event_id:
            if self.phase == "confirming":
                self._finish("cancelled")
            return True
        return False

    def public(self) -> dict:
        ev = self.event
        if self.phase == "confirming" and ev is not None:
            remaining = max(0.0, float(self.cfg.confirm.window_s) - (self.now() - self._t0))
            ev.remaining_s = remaining
            return {
                "phase": self.phase,
                "state": "confirming",
                "banner": f"Confirming · {int(round(remaining))} s",
                "remaining_s": round(remaining, 1),
                "event": ev.public(),
            }
        state = ev.state if ev else "idle"
        return {
            "phase": self.phase,
            "state": state,
            "banner": state.capitalize() if ev else "Idle",
            "remaining_s": 0.0,
            "event": ev.public() if ev else None,
        }

    def _finish(self, state: str) -> FallEvent:
        assert self.event is not None
        self.event.state = state
        self.event.remaining_s = 0.0
        if state == "alert":
            self._alerted = True
        ev = self.event
        self.phase = "idle"
        self.on_event(ev)
        return ev

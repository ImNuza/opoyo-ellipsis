from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from opoyo.audio_check import AudioCheck
from opoyo.classify import load_classifier
from opoyo.dsp import StreamingFilter, Trigger, magnitude, make_highpass
from opoyo.machine import EventMachine
from opoyo.ring import RingBuffer
from opoyo.schemas import FallEvent


class LivePipeline:
    def __init__(self, cfg, now_fn=None, on_event: Callable[[FallEvent], None] | None = None):
        self.cfg = cfg
        self.now = now_fn or time.time
        self.fs = float(cfg.sensor.accel_rate_hz)
        self.g_to_ms2 = float(cfg.sensor.g_to_ms2)
        cap = int(self.fs * float(cfg.sensor.buffer_seconds))
        self.mag = RingBuffer(cap)
        self.db = RingBuffer(cap)
        self.hp = StreamingFilter(make_highpass(float(cfg.trigger.highpass_hz), self.fs))
        self.trigger = Trigger(
            fs=self.fs,
            rms_window_ms=float(cfg.trigger.rms_window_ms),
            baseline_seconds=float(cfg.trigger.baseline_seconds),
            k_mad=float(cfg.trigger.k_mad),
            refractory_s=float(cfg.trigger.refractory_s),
            min_peak_mag=float(cfg.trigger.min_peak_mag),
        )
        self.classifier = load_classifier(cfg)
        self.audio = AudioCheck(
            classes=list(cfg.veto.classes),
            threshold=float(cfg.veto.threshold),
            enabled=bool(cfg.veto.enabled),
        )
        self.machine = EventMachine(
            self.classifier,
            self.audio,
            cfg,
            now_fn=self.now,
            on_event=on_event,
            fs=self.fs,
        )
        self.pre_n = int(self.fs * float(cfg.window.pre_s))
        self.post_n = int(self.fs * float(cfg.window.post_s))
        self._pending_t: float | None = None
        self._chunk: list[float] = []
        self.last_rms = 0.0
        self.last_fired = False
        self.packets = 0

    def feed(
        self,
        ax: float,
        ay: float,
        az: float,
        db: float = -120.0,
        mag_g: float | None = None,
        node_id: str = "",
        room: str = "",
    ) -> list[FallEvent]:
        if mag_g is None:
            mag_g = float(magnitude([ax], [ay], [az])[0])
        mag_ms2 = mag_g * self.g_to_ms2
        hp = self.hp(np.array([mag_ms2], dtype=np.float64))
        self.mag.push(hp)
        self.db.push([db])
        self.packets += 1
        self._chunk.extend(hp.tolist())
        events: list[FallEvent] = []
        now = self.now()

        win = max(1, self.trigger._win)
        if len(self._chunk) >= win:
            fired, peak = self.trigger.process(np.asarray(self._chunk, dtype=np.float64))
            self.last_rms = peak
            self._chunk = []
            if fired and self._pending_t is None:
                self._pending_t = now
                self.last_fired = True

        ev = self.machine.tick(self.last_rms)
        if ev is not None:
            events.append(ev)

        if self._pending_t is not None and now - self._pending_t >= float(self.cfg.window.post_s):
            n = self.pre_n + self.post_n
            window = self.mag.last(n)
            audio = self.db.last(n)
            quiet = float(self.trigger.threshold())
            if not np.isfinite(quiet):
                quiet = 0.05
            got = self.machine.on_trigger(
                window, audio, room=room, node_id=node_id, quiet_rms=quiet
            )
            self._pending_t = None
            if got is not None:
                events.append(got)
        return events

    def cancel(self) -> FallEvent | None:
        return self.machine.cancel()

    def public(self) -> dict:
        d = self.machine.public()
        thr = self.trigger.threshold()
        d["rms"] = round(self.last_rms, 4)
        d["threshold"] = None if not np.isfinite(thr) else round(thr, 4)
        d["packets"] = self.packets
        d["warmup"] = not np.isfinite(thr)
        return d

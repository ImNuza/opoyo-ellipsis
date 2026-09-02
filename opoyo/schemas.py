from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
import uuid
from typing import Optional

STATES = ("candidate", "vetoed", "recovered", "alert", "cancelled")


@dataclass
class FallEvent:
    event_id: str
    ts_unix: float
    room: str
    state: str
    fall_score: float
    veto_label: Optional[str]
    veto_score: float
    peak_mag: float
    decay_ms: float
    rise_ms: float = 0.0
    node_id: str = ""
    remaining_s: float = 0.0

    @staticmethod
    def new(room: str, **kw) -> "FallEvent":
        return FallEvent(
            event_id=uuid.uuid4().hex[:12],
            ts_unix=kw.pop("ts_unix", time.time()),
            room=room,
            state=kw.pop("state", "candidate"),
            fall_score=kw.pop("fall_score", 0.0),
            veto_label=kw.pop("veto_label", None),
            veto_score=kw.pop("veto_score", 0.0),
            peak_mag=kw.pop("peak_mag", 0.0),
            decay_ms=kw.pop("decay_ms", 0.0),
            rise_ms=kw.pop("rise_ms", 0.0),
            node_id=kw.pop("node_id", ""),
            remaining_s=kw.pop("remaining_s", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    def public(self) -> dict:
        payload = asdict(self)
        payload["fall_score"] = round(self.fall_score, 3)
        payload["veto_score"] = round(self.veto_score, 3)
        payload["peak_mag"] = round(self.peak_mag, 4)
        payload["decay_ms"] = round(self.decay_ms, 1)
        payload["rise_ms"] = round(self.rise_ms, 1)
        payload["remaining_s"] = round(self.remaining_s, 1)
        return payload

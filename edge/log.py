"""JSONL store for every InferenceResult. Default path: edge/data/inference.jsonl."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from shared.schemas import InferenceResult


class InferenceLog:
    """Append-only file plus a 500-row ring for the dashboard tail."""
    def __init__(self, path: Path, maxlen: int = 500) -> None:
        self.path = path
        self._ring: deque[InferenceResult] = deque(maxlen=maxlen)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                self._ring.append(InferenceResult.model_validate_json(stripped))

    def append(self, result: InferenceResult) -> None:
        self._ring.append(result)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")

    def tail(self, n: int) -> list[InferenceResult]:
        items = list(self._ring)
        if n >= len(items):
            return items
        return items[-n:]

from __future__ import annotations

from pathlib import Path

from shared.schemas import InferenceResult
from edge.log import InferenceLog


def _row(i: int, is_fall: bool = False) -> InferenceResult:
    return InferenceResult(
        inference_id=f"id{i}",
        timestamp=1000 + i,
        node_id="n1",
        room="Phone 1",
        is_fall=is_fall,
        confidence=0.1 if not is_fall else 0.93,
    )


def test_appends_fall_and_no_fall(tmp_path: Path):
    store = InferenceLog(tmp_path / "inference.jsonl")
    store.append(_row(1, False))
    store.append(_row(2, True))
    tail = store.tail(10)
    assert [r.is_fall for r in tail] == [False, True]
    text = (tmp_path / "inference.jsonl").read_text(encoding="utf-8")
    assert text.count("\n") == 2


def test_tail_returns_last_n(tmp_path: Path):
    store = InferenceLog(tmp_path / "inference.jsonl")
    for i in range(5):
        store.append(_row(i))
    ids = [r.inference_id for r in store.tail(2)]
    assert ids == ["id3", "id4"]


def test_reopen_reads_existing_jsonl(tmp_path: Path):
    path = tmp_path / "inference.jsonl"
    first = InferenceLog(path)
    first.append(_row(0))
    first.append(_row(1, True))
    second = InferenceLog(path)
    assert [r.inference_id for r in second.tail(10)] == ["id0", "id1"]
    assert second.tail(10)[1].is_fall is True

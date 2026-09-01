from __future__ import annotations

from shared.schemas import AckEvent, FallEvent
from server.adapters import RecordingAdapter
from server.tree import EscalationTree, FakeClock


def _event() -> FallEvent:
    return FallEvent(
        event_id="evt1",
        inference_id="evt1",
        timestamp=1735689602123,
        node_id="n1",
        room="Bathroom",
        is_fall=True,
        confidence=0.94,
        threshold=0.9,
    )


def _tree() -> tuple[EscalationTree, FakeClock, dict[str, RecordingAdapter]]:
    clock = FakeClock(0.0)
    adapters = {
        "telegram": RecordingAdapter(),
        "twilio": RecordingAdapter(),
        "secondary": RecordingAdapter(),
        "careline": RecordingAdapter(),
    }
    tree = EscalationTree(
        clock=clock,
        telegram=adapters["telegram"],
        twilio=adapters["twilio"],
        secondary=adapters["secondary"],
        careline=adapters["careline"],
    )
    return tree, clock, adapters


def test_ingest_dispatches_rung1():
    tree, _clock, adapters = _tree()
    case = tree.ingest(_event())
    assert case.state == "rung1_dispatched"
    assert len(adapters["telegram"].commands) == 1
    assert len(adapters["twilio"].commands) == 1
    cmd = adapters["telegram"].commands[0]
    assert cmd.at_s == 0
    assert cmd.rung == "family_telegram"
    assert cmd.event.room == "Bathroom"
    assert cmd.event.timestamp == 1735689602123
    assert cmd.event.confidence == 0.94
    assert adapters["twilio"].commands[0].rung == "senior_call"


def test_senior_fine_closes_and_blocks_later_rungs():
    tree, clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="fine", timestamp=1)
    )
    assert tree.cases[case.case_id].state == "false_alarm_closed"
    clock.advance(60)
    tree.on_tick()
    clock.advance(120)
    tree.on_tick()
    assert adapters["secondary"].commands == []
    assert adapters["careline"].commands == []


def test_senior_no_answer_waits_for_family():
    tree, _clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1)
    )
    assert tree.cases[case.case_id].state == "awaiting_family"
    assert len(adapters["telegram"].commands) == 1
    assert len(adapters["twilio"].commands) == 1
    assert adapters["secondary"].commands == []


def test_family_taken_at_t30_stops_ladder():
    tree, clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1)
    )
    clock.advance(30)
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="family", outcome="taken", timestamp=30)
    )
    assert tree.cases[case.case_id].state == "family_handling"
    clock.advance(30)
    tree.on_tick()
    clock.advance(120)
    tree.on_tick()
    assert adapters["secondary"].commands == []
    assert adapters["careline"].commands == []


def test_no_family_at_t60_alerts_secondary():
    tree, clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="not_fine", timestamp=1)
    )
    clock.advance(60)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "secondary_alerted"
    assert len(adapters["secondary"].commands) == 1


def test_secondary_taken_at_t90_skips_careline():
    tree, clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1)
    )
    clock.advance(60)
    tree.on_tick()
    clock.advance(30)
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="secondary", outcome="taken", timestamp=90)
    )
    assert tree.cases[case.case_id].state == "resolved"
    clock.advance(90)
    tree.on_tick()
    assert adapters["careline"].commands == []


def test_nobody_at_t180_alerts_careline():
    tree, clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1)
    )
    clock.advance(60)
    tree.on_tick()
    clock.advance(120)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "careline_alerted"
    assert len(adapters["careline"].commands) == 1
    assert adapters["careline"].commands[0].at_s == 180


def test_ack_on_terminal_is_noop():
    tree, _clock, adapters = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="fine", timestamp=1)
    )
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="family", outcome="taken", timestamp=2)
    )
    assert tree.cases[case.case_id].state == "false_alarm_closed"
    assert len(adapters["telegram"].commands) == 1


def test_duplicate_event_id_does_not_start_second_case():
    tree, _clock, adapters = _tree()
    first = tree.ingest(_event())
    second = tree.ingest(_event())
    assert first.case_id == second.case_id
    assert len(tree.cases) == 1
    assert len(adapters["telegram"].commands) == 1

from __future__ import annotations

from server.decision_tree import (
    SENIOR_WAIT_S,
    DecisionTree,
    fall_message,
    senior_check_message,
)
from shared.schemas import AckEvent, FallEvent
from tests.fakes import FakeClock, FakeSender


KIN = "kin-chat"
SECONDARY = "sec-chat"
SENIOR = "senior-chat"


def _event() -> FallEvent:
    return FallEvent(
        event_id="evt1",
        inference_id="evt1",
        timestamp=1735689602123,
        node_id="Phone 1",
        room=1,
        is_fall=True,
        confidence=0.94,
        threshold=0.9,
    )


def _tree() -> tuple[DecisionTree, FakeClock, FakeSender]:
    clock = FakeClock(0.0)
    telegram = FakeSender()
    tree = DecisionTree(
        clock=clock,
        telegram=telegram,
        next_of_kin_chat_id=KIN,
        secondary_chat_id=SECONDARY,
        senior_chat_id=SENIOR,
    )
    return tree, clock, telegram


def test_fall_message_includes_room_and_confidence():
    text = fall_message(_event())
    assert "Room 1" in text
    assert "0.94" in text


def test_ingest_dispatches_rung1():
    tree, _clock, telegram = _tree()
    case = tree.ingest(_event())
    assert case.state == "rung1_dispatched"
    assert [dest for dest, _text in telegram.sent] == [KIN, SENIOR]
    assert "Room 1" in telegram.sent[0][1]
    assert "0.94" in telegram.sent[0][1]
    assert "I'm fine" in telegram.sent[1][1]
    assert "reply yes" in telegram.sent[1][1]
    assert telegram.sent[1][1] == senior_check_message(_event())
    assert telegram.markups[0] is None
    senior_markup = telegram.markups[1]
    assert senior_markup is not None
    data = {
        btn["callback_data"]
        for row in senior_markup["inline_keyboard"]
        for btn in row
    }
    assert f"ack:{case.case_id}:yes" in data
    assert f"ack:{case.case_id}:not_fine" in data


def test_senior_yes_closes_and_blocks_later_rungs():
    tree, clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="yes", timestamp=1)
    )
    assert tree.cases[case.case_id].state == "false_alarm_closed"
    clock.advance(60)
    tree.on_tick()
    clock.advance(120)
    tree.on_tick()
    assert len(telegram.sent) == 2


def test_senior_no_answer_waits_for_family():
    tree, _clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1
        )
    )
    assert tree.cases[case.case_id].state == "awaiting_family"
    assert len(telegram.sent) == 2


def test_family_taken_at_t30_stops_ladder():
    tree, clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1
        )
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
    assert len(telegram.sent) == 2


def test_no_family_at_t60_alerts_secondary():
    tree, clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="not_fine", timestamp=1
        )
    )
    clock.advance(60)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "secondary_alerted"
    assert len(telegram.sent) == 3
    assert telegram.sent[2][0] == SECONDARY
    assert "Room 1" in telegram.sent[2][1]


def test_secondary_taken_at_t90_skips_careline():
    tree, clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1
        )
    )
    clock.advance(60)
    tree.on_tick()
    clock.advance(30)
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="secondary", outcome="taken", timestamp=90
        )
    )
    assert tree.cases[case.case_id].state == "resolved"
    clock.advance(90)
    tree.on_tick()
    assert len(telegram.sent) == 3
    assert all(c.rung != "careline" for c in tree.cases[case.case_id].commands)


def test_nobody_at_t180_alerts_careline():
    tree, clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1
        )
    )
    clock.advance(60)
    tree.on_tick()
    clock.advance(120)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "careline_alerted"
    careline = [c for c in tree.cases[case.case_id].commands if c.rung == "careline"]
    assert len(careline) == 1
    assert careline[0].at_s == 180
    assert len(telegram.sent) == 3


def test_ack_on_terminal_is_noop():
    tree, _clock, telegram = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="yes", timestamp=1)
    )
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="family", outcome="taken", timestamp=2)
    )
    assert tree.cases[case.case_id].state == "false_alarm_closed"
    assert len(telegram.sent) == 2


def test_senior_silence_becomes_no_answer():
    tree, clock, telegram = _tree()
    case = tree.ingest(_event())
    clock.advance(SENIOR_WAIT_S - 1)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "rung1_dispatched"
    clock.advance(1)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "awaiting_family"
    clock.advance(60)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "secondary_alerted"
    assert len(telegram.sent) == 3


def test_duplicate_event_id_does_not_start_second_case():
    tree, _clock, telegram = _tree()
    first = tree.ingest(_event())
    second = tree.ingest(_event())
    assert first.case_id == second.case_id
    assert len(tree.cases) == 1
    assert len(telegram.sent) == 2

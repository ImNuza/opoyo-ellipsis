from __future__ import annotations

from server.decision_tree import DecisionTree, fall_message
from shared.schemas import AckEvent, FallEvent
from tests.fakes import FakeClock, FakeSender


KIN = "kin-chat"
SECONDARY = "sec-chat"
PHONE = "+6500000000"


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


def _tree() -> tuple[DecisionTree, FakeClock, FakeSender, FakeSender]:
    clock = FakeClock(0.0)
    telegram = FakeSender()
    twilio = FakeSender()
    tree = DecisionTree(
        clock=clock,
        telegram=telegram,
        twilio=twilio,
        next_of_kin_chat_id=KIN,
        secondary_chat_id=SECONDARY,
        senior_phone=PHONE,
    )
    return tree, clock, telegram, twilio


def test_fall_message_includes_room_and_confidence():
    text = fall_message(_event())
    assert "Room 1" in text
    assert "0.94" in text


def test_ingest_dispatches_rung1():
    tree, _clock, telegram, twilio = _tree()
    case = tree.ingest(_event())
    assert case.state == "rung1_dispatched"
    assert len(telegram.sent) == 1
    assert telegram.sent[0][0] == KIN
    assert "Room 1" in telegram.sent[0][1]
    assert "0.94" in telegram.sent[0][1]
    assert twilio.sent == [(PHONE, telegram.sent[0][1])]


def test_senior_fine_closes_and_blocks_later_rungs():
    tree, clock, telegram, twilio = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="fine", timestamp=1)
    )
    assert tree.cases[case.case_id].state == "false_alarm_closed"
    clock.advance(60)
    tree.on_tick()
    clock.advance(120)
    tree.on_tick()
    assert len(telegram.sent) == 1
    assert len(twilio.sent) == 1


def test_senior_no_answer_waits_for_family():
    tree, _clock, telegram, twilio = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="no_answer", timestamp=1
        )
    )
    assert tree.cases[case.case_id].state == "awaiting_family"
    assert len(telegram.sent) == 1
    assert len(twilio.sent) == 1


def test_family_taken_at_t30_stops_ladder():
    tree, clock, telegram, twilio = _tree()
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
    assert len(telegram.sent) == 1
    assert len(twilio.sent) == 1


def test_no_family_at_t60_alerts_secondary():
    tree, clock, telegram, _twilio = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(
            case_id=case.case_id, actor="senior", outcome="not_fine", timestamp=1
        )
    )
    clock.advance(60)
    tree.on_tick()
    assert tree.cases[case.case_id].state == "secondary_alerted"
    assert len(telegram.sent) == 2
    assert telegram.sent[1][0] == SECONDARY
    assert "Room 1" in telegram.sent[1][1]


def test_secondary_taken_at_t90_skips_careline():
    tree, clock, telegram, twilio = _tree()
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
    assert len(telegram.sent) == 2
    assert len(twilio.sent) == 1
    assert all(c.rung != "careline" for c in tree.cases[case.case_id].commands)


def test_nobody_at_t180_alerts_careline():
    tree, clock, telegram, twilio = _tree()
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
    assert len(telegram.sent) == 2
    assert len(twilio.sent) == 1


def test_ack_on_terminal_is_noop():
    tree, _clock, telegram, _twilio = _tree()
    case = tree.ingest(_event())
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="senior", outcome="fine", timestamp=1)
    )
    tree.on_ack(
        AckEvent(case_id=case.case_id, actor="family", outcome="taken", timestamp=2)
    )
    assert tree.cases[case.case_id].state == "false_alarm_closed"
    assert len(telegram.sent) == 1


def test_duplicate_event_id_does_not_start_second_case():
    tree, _clock, telegram, twilio = _tree()
    first = tree.ingest(_event())
    second = tree.ingest(_event())
    assert first.case_id == second.case_id
    assert len(tree.cases) == 1
    assert len(telegram.sent) == 1
    assert len(twilio.sent) == 1

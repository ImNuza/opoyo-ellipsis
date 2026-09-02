from __future__ import annotations

from server.adapters.twilio import Twilio


def test_twilio_sender_signature_is_pluggable():
    """Live ladder does not call Twilio; the adapter must stay drop-in for later."""
    client = Twilio()
    assert client.send("+6500000000", "OPOYO: fall") is True

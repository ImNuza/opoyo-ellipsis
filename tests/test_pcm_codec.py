from __future__ import annotations

import struct
import uuid

import numpy as np

from shared.pcm import HEADER_SIZE, PCM_MAGIC, PCM_RATE, pack_frame, unpack_frame

NODE = uuid.UUID("11111111-1111-1111-1111-111111111111")
T_MS = 1_735_689_600_000
N = 320


def _ramp() -> bytes:
    samples = np.arange(N, dtype=np.int16)
    return samples.tobytes()


def test_roundtrip_restores_fields_and_pcm():
    pcm = _ramp()
    packed = pack_frame(NODE, seq=7, t_ms=T_MS, samples=pcm)
    frame = unpack_frame(packed)
    assert frame is not None
    assert frame.node_id == str(NODE)
    assert frame.seq == 7
    assert frame.t_ms == T_MS
    assert frame.rate == PCM_RATE
    assert frame.pcm == pcm


def test_packed_length_is_header_plus_s16():
    packed = pack_frame(NODE, seq=0, t_ms=T_MS, samples=_ramp())
    assert len(packed) == HEADER_SIZE + 2 * N


def test_bad_magic_returns_none():
    packed = bytearray(pack_frame(NODE, seq=1, t_ms=T_MS, samples=_ramp()))
    packed[0:4] = b"XXXX"
    assert unpack_frame(bytes(packed)) is None


def test_truncated_header_returns_none():
    packed = pack_frame(NODE, seq=1, t_ms=T_MS, samples=_ramp())
    assert unpack_frame(packed[: HEADER_SIZE - 1]) is None


def test_truncated_payload_returns_none():
    packed = pack_frame(NODE, seq=1, t_ms=T_MS, samples=_ramp())
    assert unpack_frame(packed[:-2]) is None


def test_n_samples_larger_than_payload_returns_none():
    packed = bytearray(pack_frame(NODE, seq=1, t_ms=T_MS, samples=_ramp()))
    struct.pack_into("<H", packed, 35, N + 50)
    assert unpack_frame(bytes(packed)) is None


def test_wrong_version_returns_none():
    packed = bytearray(pack_frame(NODE, seq=1, t_ms=T_MS, samples=_ramp()))
    packed[4] = 2
    assert unpack_frame(bytes(packed)) is None


def test_pack_never_prepends_riff():
    packed = pack_frame(NODE, seq=1, t_ms=T_MS, samples=_ramp())
    assert packed[:4] == PCM_MAGIC
    assert packed[:4] != b"RIFF"
    assert b"WAVE" not in packed[: HEADER_SIZE]

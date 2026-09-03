"""Binary PCM frames for phone to edge UDP :9001.

This is not a WAV or RIFF container. The header is a fixed 37-byte ``OPYA``
layout so a datagram can carry 16 kHz int16 without a file wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
import struct

PCM_MAGIC = b"OPYA"
PCM_VERSION = 1
PCM_RATE = 16000
HEADER_SIZE = 37
# Little-endian: magic, version, seq, t_ms, uuid, rate, sample count.
_HEADER = struct.Struct("<4sBIQ16sHH")


@dataclass(frozen=True)
class PcmFrame:
    node_id: str
    seq: int
    t_ms: int
    rate: int
    pcm: bytes


def _uuid_bytes(node_id: str | UUID) -> bytes:
    if isinstance(node_id, UUID):
        return node_id.bytes
    return UUID(str(node_id)).bytes


def pack_frame(
    node_id: str | UUID,
    seq: int,
    t_ms: int,
    samples: bytes,
    rate: int = PCM_RATE,
) -> bytes:
    """Pack a PCM datagram.

    Args:
        node_id: Phone UUID, same as SensorSample.id.
        seq: Monotonic frame counter from the phone.
        t_ms: Phone unix time in milliseconds for the first sample.
        samples: Little-endian int16 bytes.
        rate: Sample rate in Hz. YAMNet requires 16000.

    Returns:
        Header plus sample payload.
    """
    n = len(samples) // 2
    header = _HEADER.pack(
        PCM_MAGIC,
        PCM_VERSION,
        int(seq) & 0xFFFFFFFF,
        int(t_ms) & 0xFFFFFFFFFFFFFFFF,
        _uuid_bytes(node_id),
        int(rate) & 0xFFFF,
        n & 0xFFFF,
    )
    return header + samples


def unpack_frame(data: bytes) -> PcmFrame | None:
    """Parse one datagram. Returns None if the header is truncated or invalid.

    Args:
        data: UDP payload.

    Returns:
        PcmFrame, or None so the edge can fail open on a bad packet.
    """
    if len(data) < HEADER_SIZE:
        return None
    try:
        magic, version, seq, t_ms, raw_id, rate, n = _HEADER.unpack_from(data, 0)
    except struct.error:
        return None
    if magic != PCM_MAGIC or version != PCM_VERSION:
        return None
    need = HEADER_SIZE + 2 * n
    if n < 0 or len(data) < need:
        return None
    try:
        node_id = str(UUID(bytes=raw_id))
    except ValueError:
        return None
    return PcmFrame(
        node_id=node_id,
        seq=seq,
        t_ms=t_ms,
        rate=rate,
        pcm=data[HEADER_SIZE:need],
    )

"""Tiny dependency-free Standard MIDI File (format 1) writer and reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PPQ = 480  # ticks per quarter note


@dataclass
class Note:
    start: float  # in beats (quarter notes)
    duration: float  # in beats
    pitch: int  # 0..127
    velocity: int = 90


@dataclass
class Track:
    name: str
    notes: list[Note] = field(default_factory=list)
    channel: int = 0
    program: int = 0
    is_drums: bool = False


def _vlq(value: int) -> bytes:
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _meta(delta: int, kind: int, data: bytes) -> bytes:
    return _vlq(delta) + bytes([0xFF, kind]) + _vlq(len(data)) + data


def write_midi(path: str | Path, tracks: list[Track], bpm: float, time_sig: tuple[int, int] = (4, 4)) -> Path:
    """Write tracks to a .mid file (format 1: tempo track + one track per part)."""
    path = Path(path)
    chunks: list[bytes] = []

    # tempo / meta track
    tempo = int(60_000_000 / bpm)
    num, den = time_sig
    meta = bytearray()
    meta += _meta(0, 0x03, b"Tempo")
    meta += _meta(0, 0x51, tempo.to_bytes(3, "big"))
    meta += _meta(0, 0x58, bytes([num, den.bit_length() - 1, 24, 8]))
    meta += _meta(0, 0x2F, b"")
    chunks.append(b"MTrk" + len(meta).to_bytes(4, "big") + bytes(meta))

    for tr in tracks:
        ch = 9 if tr.is_drums else (tr.channel & 0x0F)
        events: list[tuple[int, int, bytes]] = []  # (tick, order, data)
        for n in tr.notes:
            on = int(round(n.start * PPQ))
            off = int(round((n.start + max(n.duration, 0.01)) * PPQ))
            pitch = max(0, min(127, int(n.pitch)))
            vel = max(1, min(127, int(n.velocity)))
            events.append((on, 1, bytes([0x90 | ch, pitch, vel])))
            events.append((off, 0, bytes([0x80 | ch, pitch, 0])))
        events.sort(key=lambda e: (e[0], e[1]))

        body = bytearray()
        body += _meta(0, 0x03, tr.name.encode("utf-8", "replace"))
        if not tr.is_drums:
            body += _vlq(0) + bytes([0xC0 | ch, tr.program & 0x7F])
        last = 0
        for tick, _, data in events:
            body += _vlq(tick - last) + data
            last = tick
        body += _meta(0, 0x2F, b"")
        chunks.append(b"MTrk" + len(body).to_bytes(4, "big") + bytes(body))

    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(chunks).to_bytes(2, "big") + PPQ.to_bytes(2, "big")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(chunks))
    return path


def read_midi(path: str | Path) -> tuple[list[Track], float]:
    """Minimal reader for files written by write_midi (and most simple SMFs)."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("Not a MIDI file")
    ntrks = int.from_bytes(data[10:12], "big")
    ppq = int.from_bytes(data[12:14], "big")
    pos = 14
    bpm = 120.0
    tracks: list[Track] = []
    for _ in range(ntrks):
        assert data[pos:pos + 4] == b"MTrk"
        length = int.from_bytes(data[pos + 4:pos + 8], "big")
        end = pos + 8 + length
        p = pos + 8
        tick = 0
        status = 0
        name = ""
        notes: list[Note] = []
        open_notes: dict[tuple[int, int], tuple[int, int]] = {}
        is_drums = False
        while p < end:
            delta = 0
            while True:
                b = data[p]; p += 1
                delta = (delta << 7) | (b & 0x7F)
                if not b & 0x80:
                    break
            tick += delta
            b = data[p]
            if b == 0xFF:
                kind = data[p + 1]; p += 2
                ln = 0
                while True:
                    c = data[p]; p += 1
                    ln = (ln << 7) | (c & 0x7F)
                    if not c & 0x80:
                        break
                payload = data[p:p + ln]; p += ln
                if kind == 0x03:
                    name = payload.decode("utf-8", "replace")
                elif kind == 0x51 and ln == 3:
                    bpm = 60_000_000 / int.from_bytes(payload, "big")
                continue
            if b & 0x80:
                status = b; p += 1
            kind = status & 0xF0
            ch = status & 0x0F
            if kind in (0xC0, 0xD0):
                p += 1
            else:
                d1, d2 = data[p], data[p + 1]; p += 2
                if kind == 0x90 and d2 > 0:
                    open_notes[(ch, d1)] = (tick, d2)
                    is_drums = is_drums or ch == 9
                elif kind == 0x80 or (kind == 0x90 and d2 == 0):
                    st = open_notes.pop((ch, d1), None)
                    if st:
                        notes.append(Note(st[0] / ppq, (tick - st[0]) / ppq, d1, st[1]))
        tracks.append(Track(name, notes, is_drums=is_drums))
        pos = end
    return tracks, bpm

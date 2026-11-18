"""Builds a tiny but spec-conformant SoundFont 2 file for tests (looped sine 'instrument' + one-shot kick 'kit')."""

import struct

import numpy as np


def _chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\0" if len(payload) & 1 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _list(kind: bytes, chunks: list[bytes]) -> bytes:
    return _chunk(b"LIST", kind + b"".join(chunks))


def _name(s: str) -> bytes:
    return s.encode().ljust(20, b"\0")[:20]


def build_sf2(path, root_key: int = 60):
    # sample 0: sine with a 100-sample period at 26163 Hz  -> exactly 261.63 Hz = MIDI 60, looped
    rate0 = 26163
    s0 = (np.sin(2 * np.pi * np.arange(3000) / 100.0) * 20000).astype("<i2")
    # sample 1: decaying 55 Hz thump, one shot at 44100 Hz
    t = np.arange(4000) / 44100.0
    s1 = (np.sin(2 * np.pi * 55 * t) * np.exp(-t * 18) * 28000).astype("<i2")
    smpl = s0.tobytes() + b"\0" * 92 + s1.tobytes() + b"\0" * 92        # 46 zero samples of padding after each
    off1 = len(s0) + 46
    shdr = (struct.pack("<20sIIIIIBbHH", _name("Sine"), 0, 3000, 100, 2900, rate0, root_key, 0, 0, 1)
            + struct.pack("<20sIIIIIBbHH", _name("Kick"), off1, off1 + 4000, off1 + 100, off1 + 200, 44100, 36, 0, 0, 1)
            + struct.pack("<20sIIIIIBbHH", _name("EOS"), 0, 0, 0, 0, 0, 0, 0, 0, 0))
    gen = lambda op, lo, hi=0: struct.pack("<HBB", op, lo, hi)  # noqa: E731
    # instruments
    inst = struct.pack("<20sH", _name("SineInst"), 0) + struct.pack("<20sH", _name("KitInst"), 1) + struct.pack("<20sH", _name("EOI"), 2)
    ibag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 3, 0) + struct.pack("<HH", 6, 0)
    igen = (gen(43, 0, 127) + gen(54, 1, 0) + gen(53, 0, 0)            # key 0-127, loop on, sample 0
            + gen(43, 36, 36) + gen(34, 0x00, 0xE0) + gen(53, 1, 0)    # key 36, (attack = -8192 tc), sample 1
            + gen(0, 0, 0))
    # presets: (name, program, bank, bag index)
    phdr = (struct.pack("<20sHHHIII", _name("Sine"), 0, 0, 0, 0, 0, 0)
            + struct.pack("<20sHHHIII", _name("Kit"), 0, 128, 1, 0, 0, 0)
            + struct.pack("<20sHHHIII", _name("EOP"), 0, 0, 2, 0, 0, 0))
    pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0) + struct.pack("<HH", 2, 0)
    pgen = gen(41, 0, 0) + gen(41, 1, 0) + gen(0, 0, 0)
    pdta = _list(b"pdta", [
        _chunk(b"phdr", phdr), _chunk(b"pbag", pbag), _chunk(b"pmod", b"\0" * 10), _chunk(b"pgen", pgen),
        _chunk(b"inst", inst), _chunk(b"ibag", ibag), _chunk(b"imod", b"\0" * 10), _chunk(b"igen", igen),
        _chunk(b"shdr", shdr)])
    body = b"sfbk" + _list(b"INFO", [_chunk(b"ifil", struct.pack("<HH", 2, 1))]) + _list(b"sdta", [_chunk(b"smpl", smpl)]) + pdta
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path

"""Minimal SoundFont 2 (.sf2) reader + sample player (numpy only).

Supports what real General-MIDI fonts need for plain playback: presets / instruments / zones with key and
velocity ranges, global zones, sample pitch + tuning, loops, volume ADSR, attenuation and drum kits (bank 128).
Not supported: modulators, filters, LFOs, 24-bit extension, compressed .sf3 (these are ignored / rejected).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# generator operators we use
G_PAN, G_ATTACK, G_HOLD, G_DECAY, G_SUSTAIN, G_RELEASE = 17, 34, 35, 36, 37, 38
G_INSTRUMENT, G_KEYRANGE, G_VELRANGE, G_ATTEN = 41, 43, 44, 48
G_COARSE, G_FINE, G_SAMPLEID, G_MODES, G_SCALE, G_ROOT = 51, 52, 53, 54, 56, 58
_RANGE_OPS = (G_KEYRANGE, G_VELRANGE)
_ADDITIVE = (G_COARSE, G_FINE, G_ATTEN, G_ATTACK, G_HOLD, G_DECAY, G_SUSTAIN, G_RELEASE)


@dataclass
class _Sample:
    start: int
    end: int
    loop_start: int
    loop_end: int
    rate: int
    root: int
    correction: int
    kind: int          # 1 mono, 2 right, 4 left


def _chunks(data: bytes, pos: int, end: int):
    while pos + 8 <= end:
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        yield cid, pos + 8, size
        pos += 8 + size + (size & 1)


def _records(data: bytes, off: int, size: int, fmt: str):
    n = struct.calcsize(fmt)
    return [struct.unpack_from(fmt, data, off + i * n) for i in range(size // n)]


def _gens(raw) -> list[tuple[int, object]]:
    out = []
    for oper, lo, hi in raw:
        if oper in _RANGE_OPS:
            out.append((oper, (lo, hi)))
        else:
            out.append((oper, struct.unpack("<h", bytes([lo, hi]))[0]))
    return out


def _zones(bags, gens, first: int, last: int, link_op: int):
    """Split bag range into (global_gens, [zone_gens...]) where a zone has a `link_op` generator."""
    glob: dict = {}
    zones: list[dict] = []
    for j in range(first, last):
        g0, g1 = bags[j][0], bags[j + 1][0]
        d = dict(gens[g0:g1])
        if link_op in d:
            zones.append({**glob, **d})
        elif j == first:
            glob = d
    return zones


class SoundFont:
    def __init__(self, path: str | Path):
        data = Path(path).read_bytes()
        if data[:4] != b"RIFF" or data[8:12] != b"sfbk":
            raise ValueError("Not a SoundFont 2 file")
        self.name = Path(path).name
        smpl = None
        pdta = {}
        for cid, off, size in _chunks(data, 12, len(data)):
            if cid != b"LIST":
                continue
            kind = data[off:off + 4]
            for sid, so, ss in _chunks(data, off + 4, off + size):
                if kind == b"sdta" and sid == b"smpl":
                    smpl = np.frombuffer(data, dtype="<i2", count=ss // 2, offset=so)
                elif kind == b"sdta" and sid == b"sm24":
                    pass
                elif kind == b"pdta":
                    pdta[sid] = (so, ss)
        if smpl is None:
            raise ValueError("SoundFont has no sample data (compressed .sf3 is not supported)")
        self.pcm = smpl.astype(np.float32) / 32768.0
        rec = lambda k, fmt: _records(data, pdta[k][0], pdta[k][1], fmt)  # noqa: E731
        phdr = rec(b"phdr", "<20sHHHIII")
        pbag = rec(b"pbag", "<HH")
        pgen = _gens(rec(b"pgen", "<HBB"))
        inst = rec(b"inst", "<20sH")
        ibag = rec(b"ibag", "<HH")
        igen = _gens(rec(b"igen", "<HBB"))
        shdr = rec(b"shdr", "<20sIIIIIBbHH")
        self.samples = [_Sample(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[9]) for s in shdr]
        self._inst_zones = [_zones(ibag, igen, inst[i][1], inst[i + 1][1], G_SAMPLEID) for i in range(len(inst) - 1)]
        self.presets: dict[tuple[int, int], list[dict]] = {}
        self.preset_names: dict[tuple[int, int], str] = {}
        for i in range(len(phdr) - 1):
            name, prog, bank, bag0 = phdr[i][:4]
            zones = _zones(pbag, pgen, bag0, phdr[i + 1][3], G_INSTRUMENT)
            self.presets[(bank, prog)] = zones
            self.preset_names[(bank, prog)] = name.split(b"\0")[0].decode("latin-1")
        self._cache: dict[tuple, np.ndarray] = {}

    # ---- lookup
    def has(self, program: int, drums: bool = False) -> bool:
        return (128, program) in self.presets or (128, 0) in self.presets if drums else (0, program) in self.presets

    def _find(self, bank: int, program: int, key: int, vel: int):
        zones = self.presets.get((bank, program))
        if zones is None and bank == 128:
            zones = self.presets.get((128, 0)) or next((z for (b, _), z in self.presets.items() if b == 128), None)
        out = []
        for pz in zones or []:
            kr, vr = pz.get(G_KEYRANGE, (0, 127)), pz.get(G_VELRANGE, (0, 127))
            if not (kr[0] <= key <= kr[1] and vr[0] <= vel <= vr[1]):
                continue
            for iz in self._inst_zones[pz[G_INSTRUMENT]]:
                ikr, ivr = iz.get(G_KEYRANGE, (0, 127)), iz.get(G_VELRANGE, (0, 127))
                if ikr[0] <= key <= ikr[1] and ivr[0] <= vel <= ivr[1]:
                    merged = dict(iz)
                    for op in _ADDITIVE:
                        if op in pz:
                            merged[op] = merged.get(op, 0) + pz[op]
                    out.append(merged)
        return out

    # ---- playback
    def render(self, program: int, pitch: int, dur_s: float, vel: int, sr: int = 44100, drums: bool = False) -> np.ndarray | None:
        """Mono float32 audio for one note, or None if the font has nothing for it."""
        key = (program, pitch, round(dur_s, 2), vel // 4, drums)
        if key in self._cache:
            return self._cache[key]
        zones = self._find(128 if drums else 0, program, pitch, vel)
        if not zones:
            self._cache[key] = None
            return None
        mix = np.zeros(int((dur_s + 2.5) * sr), dtype=np.float32)
        used = 0
        for z in zones:
            snd = self._play_zone(z, pitch, dur_s, vel, sr, drums)
            if snd is not None:
                n = min(len(mix), len(snd))
                mix[:n] += snd[:n]
                used += 1
        out = None
        if used:
            nz = np.nonzero(np.abs(mix) > 1e-5)[0]
            out = mix[: (nz[-1] + 1 if len(nz) else 1)].copy()
        self._cache[key] = out
        return out

    def _play_zone(self, z: dict, pitch: int, dur_s: float, vel: int, sr: int, drums: bool):
        sid = z.get(G_SAMPLEID)
        if sid is None or sid >= len(self.samples):
            return None
        s = self.samples[sid]
        if s.end - s.start < 2:
            return None
        src = self.pcm[s.start:s.end]
        root = z.get(G_ROOT, -1)
        root = s.root if not (0 <= root <= 127) else root
        if root > 127:
            root = 60
        cents = (pitch - root) * z.get(G_SCALE, 100) + z.get(G_COARSE, 0) * 100 + z.get(G_FINE, 0) + s.correction
        ratio = 2.0 ** (cents / 1200.0) * s.rate / sr
        tc = lambda op, default: 2.0 ** (z.get(op, default) / 1200.0)  # noqa: E731
        att, hold, dec, rel = tc(G_ATTACK, -12000), tc(G_HOLD, -12000), tc(G_DECAY, -12000), max(tc(G_RELEASE, -12000), 0.008)
        sus = 10 ** (-np.clip(z.get(G_SUSTAIN, 0), 0, 1440) / 200.0)
        loop = (z.get(G_MODES, 0) & 1) == 1 and s.loop_end - s.loop_start > 8
        play = dur_s if not drums else min(len(src) / (s.rate * max(ratio * sr / s.rate, 1e-6)), 6.0)
        n_out = int((play + min(rel, 2.0) + 0.02) * sr)
        pos = np.arange(n_out) * ratio
        if loop:
            ls, le = s.loop_start - s.start, s.loop_end - s.start
            over = pos >= le
            pos = np.where(over, ls + (pos - ls) % (le - ls), pos)
        valid = pos < len(src) - 1
        y = np.interp(np.minimum(pos, len(src) - 1), np.arange(len(src)), src) * valid
        t = np.arange(n_out) / sr
        env = np.minimum(1.0, t / max(att, 1e-4))
        td = t - att - hold
        env = np.where(td > 0, sus + (1 - sus) * np.exp(-td / max(dec / 3.0, 1e-3)), env)
        if not drums:
            tr = t - play
            env = env * np.where(tr > 0, np.clip(1 - tr / rel, 0.0, 1.0), 1.0)
        gain = (vel / 127.0) ** 1.6 * 10 ** (-z.get(G_ATTEN, 0) / 200.0)
        if s.kind in (2, 4):
            gain *= 0.5            # one half of a stereo pair: both halves are summed to mono
        return (y * env * gain).astype(np.float32)


def find_soundfonts(extra_dirs: list[Path] | None = None) -> list[Path]:
    """.sf2 files in ./soundfonts next to the program (or the project / given folders)."""
    import sys
    dirs = list(extra_dirs or [])
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).parent / "soundfonts")
    dirs.append(Path.cwd() / "soundfonts")
    dirs.append(Path(__file__).resolve().parents[2] / "soundfonts")
    found: list[Path] = []
    for d in dirs:
        if d.is_dir():
            found += sorted(d.glob("*.sf2"))
    return found

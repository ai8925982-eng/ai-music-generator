"""Rule-based + stochastic composer: turns a genre profile into MIDI-like tracks."""

from __future__ import annotations

import random

import numpy as np
from dataclasses import dataclass, field

from .genres import Genre, get_genre
from .midi_io import Note, Track
from .theory import (
    CHORD_TYPES, NOTE_NAMES, SCALES, chord_midi, degree_to_chord, pent_for, scale_notes,
)

BEATS_PER_BAR = 4
KICK, SNARE, CLAP, HAT, OPEN_HAT, RIDE = 36, 38, 39, 42, 46, 51
CONGA_HI, CONGA_LO, SHAKER, CHING = 63, 64, 70, 81

# (section name, bars, intensity 0..1)
SECTION_TEMPLATE = [
    ("intro", 4, 0.35),
    ("verse", 8, 0.6),
    ("chorus", 8, 1.0),
    ("verse", 8, 0.65),
    ("chorus", 8, 1.0),
    ("outro", 4, 0.3),
]


@dataclass
class Song:
    genre: str
    bpm: float
    root_pc: int
    scale: str
    bars: int
    seed: int
    tracks: list[Track] = field(default_factory=list)
    sections: list[tuple[str, int, int, float]] = field(default_factory=list)  # name, start_bar, bars, intensity
    offset_s: float = 0.0  # music starts this many seconds in (aligned to the vocal's bar line)

    @property
    def key_name(self) -> str:
        return f"{NOTE_NAMES[self.root_pc]} {self.scale}"

    @property
    def seconds(self) -> float:
        return self.offset_s + self.bars * BEATS_PER_BAR * 60.0 / self.bpm


def build_sections(total_bars: int) -> list[tuple[str, int, int, float]]:
    """Scale the section template so the song is ~total_bars long."""
    base = sum(b for _, b, _ in SECTION_TEMPLATE)
    plan = []
    bar = 0
    for i, (name, bars, inten) in enumerate(SECTION_TEMPLATE):
        n = max(2, round(bars * total_bars / base / 2) * 2)
        plan.append((name, bar, n, inten))
        bar += n
    # trim / pad the last section so we hit total_bars exactly
    name, start, n, inten = plan[-1]
    plan[-1] = (name, start, max(1, total_bars - start), inten)
    plan = [p for p in plan if p[1] < total_bars]
    return plan


def _swing_offset(step: int, swing: float) -> float:
    """Beat offset for a 16th step (odd 16ths pushed late by swing)."""
    return 0.25 * swing if step % 2 == 1 else 0.0


def _drums(rng: random.Random, g: Genre, sections, bars: int, tight: bool = False) -> Track:
    tr = Track("Drums", is_drums=True)
    if g.extra.get("no_drums"):
        return tr
    for name, start, n, inten in sections:
        if name == "intro" and inten < 0.5:
            use_hat_only = True
        else:
            use_hat_only = False
        for b in range(start, start + n):
            bar_beat = b * BEATS_PER_BAR
            last_bar_of_section = b == start + n - 1
            for step in range(16):
                t = bar_beat + step * 0.25 + _swing_offset(step, g.swing)
                k, s, h = g.kick[step], g.snare[step], g.hat[step]
                human = rng.randint(-4, 4)
                if h != ".":
                    vel = (62 if step % 4 else 78) + human
                    if not tight and rng.random() < 0.08 and step % 4 == 2:
                        tr.notes.append(Note(t, 0.2, OPEN_HAT, 70))
                    else:
                        tr.notes.append(Note(t, 0.1, HAT, int(vel * (0.7 + 0.3 * inten))))
                if use_hat_only:
                    continue
                if k != ".":
                    tr.notes.append(Note(t, 0.15, KICK, 108 + human))
                elif not tight and inten > 0.7 and rng.random() < 0.06 and step % 2 == 0:
                    tr.notes.append(Note(t, 0.15, KICK, 80))  # ghost kick
                if s != ".":
                    tr.notes.append(Note(t, 0.15, SNARE, 104 + human))
                    if inten >= 1.0 and g.name in ("pop", "edm", "house"):
                        tr.notes.append(Note(t, 0.15, CLAP, 70))
                elif not tight and rng.random() < 0.05 * inten and step % 2 == 1:
                    tr.notes.append(Note(t, 0.1, SNARE, 45))  # ghost snare
            if last_bar_of_section and n >= 4 and inten > 0.5:
                for i in range(4):  # small fill on last beat
                    tr.notes.append(Note(bar_beat + 3 + i * 0.25, 0.12, SNARE, 70 + i * 10))
    return tr


def _bass(rng: random.Random, g: Genre, chords, sections, bars: int, guide=None) -> Track:
    tr = Track("Bass", program=38, channel=1)
    for name, start, n, inten in sections:
        if name == "intro" and g.bass_style == "sub" and inten < 0.4:
            pass
        for b in range(start, start + n):
            root_pc, kind = chords[b % len(chords)]
            root = 28 + ((root_pc - 28) % 12)  # E1..D#2 region
            tt = b * BEATS_PER_BAR
            tones = chord_midi(root_pc, kind, root)
            if g.bass_style == "root":
                tr.notes.append(Note(tt, 3.8, root, 96))
            elif g.bass_style == "octave":
                for i in range(8):
                    p = root + (12 if i % 2 else 0)
                    tr.notes.append(Note(tt + i * 0.5, 0.45, p, 92 if i % 2 == 0 else 78))
            elif g.bass_style == "walking":
                seq = [tones[0], tones[1 % len(tones)], tones[2 % len(tones)], tones[1 % len(tones)]]
                for i, p in enumerate(seq):
                    tr.notes.append(Note(tt + i, 0.92, p, 88))
            elif g.bass_style == "sub":
                tr.notes.append(Note(tt, 1.9, root, 100))
                tr.notes.append(Note(tt + 2, 1.9 if rng.random() < 0.6 else 0.9, root, 92))
            elif g.bass_style == "syncopated":
                sung = guide is not None and sum(0 if _free_fn(guide)(tt + k * 0.5) else 1 for k in range(8)) >= 4
                pattern = [0, 2.0, 3.5] if sung else [0, 0.75, 1.5, 2.5, 3.0, 3.5]
                for off in pattern:
                    if rng.random() < 0.9:
                        p = root if rng.random() < 0.7 else tones[2 % len(tones)]
                        tr.notes.append(Note(tt + off, 0.4, p, 94))
            else:  # arp
                for i in range(8):
                    tr.notes.append(Note(tt + i * 0.5, 0.45, tones[i % len(tones)], 86))
    return tr


def _chords(rng: random.Random, g: Genre, chords, sections, bars: int) -> Track:
    tr = Track("Chords", program=89 if g.chord_timbre == "pad" else 19, channel=2)
    for name, start, n, inten in sections:
        for b in range(start, start + n):
            root_pc, kind = chords[b % len(chords)]
            tones = chord_midi(root_pc, kind, 55)
            tt = b * BEATS_PER_BAR
            vel = int(60 + 30 * inten)
            style = g.chord_style
            if style == "pad":
                for p in tones:
                    tr.notes.append(Note(tt, 3.95, p, vel - 10))
            elif style == "stabs":
                for off in (0, 1.5, 2.5):
                    for p in tones:
                        tr.notes.append(Note(tt + off, 0.6, p, vel))
            elif style == "arp":
                order = tones + [tones[0] + 12, tones[1] + 12]
                for i in range(8):
                    tr.notes.append(Note(tt + i * 0.5, 0.5, order[i % len(order)], vel - 4))
            elif style == "strum":
                for beat in (0, 1, 2, 3):
                    for j, p in enumerate(tones):
                        tr.notes.append(Note(tt + beat + j * 0.02, 0.9, p, vel - (0 if beat % 2 == 0 else 10)))
            elif style == "offbeat":
                for beat in (0.5, 1.5, 2.5, 3.5):
                    for p in tones:
                        tr.notes.append(Note(tt + beat, 0.4, p, vel))
    return tr


def _melody(rng: random.Random, g: Genre, root_pc: int, scale: str, chords, sections, bars: int) -> Track:
    tr = Track("Lead", program=80 if g.lead_timbre in ("saw", "square") else 73, channel=3)
    lo, hi = g.melody_range
    pool = scale_notes(root_pc, scale, lo, hi)
    if not pool:
        return tr
    idx = len(pool) // 2
    motif: list[tuple[float, float, int]] = []  # (offset, dur, step) reused across bars
    for name, start, n, inten in sections:
        if name in ("intro", "outro") and inten < 0.4:
            density = g.melody_density * 0.4
        else:
            density = g.melody_density * (0.7 + 0.5 * inten)
        for b in range(start, start + n):
            tt = b * BEATS_PER_BAR
            root_pc_c, kind = chords[b % len(chords)]
            chord_pcs = {(root_pc_c + i) % 12 for i in (0, 4 if "maj" in kind or kind == "dom7" else 3, 7)}
            if (b - start) % 4 == 0 or not motif:
                motif = []
                pos = 0.0
                while pos < 4.0:
                    dur = rng.choice([0.5, 0.5, 1.0, 0.25, 0.75, 1.5])
                    dur = min(dur, 4.0 - pos)
                    if rng.random() < density:
                        motif.append((pos, dur, rng.choice([-2, -1, -1, 0, 1, 1, 2])))
                    pos += dur
            for off, dur, step in motif:
                idx = max(0, min(len(pool) - 1, idx + step))
                if off in (0.0, 2.0):  # strong beats lean on chord tones
                    for _ in range(3):
                        if pool[idx] % 12 in chord_pcs:
                            break
                        idx = max(0, min(len(pool) - 1, idx + rng.choice([-1, 1])))
                tr.notes.append(Note(tt + off, dur * 0.95, pool[idx], int(70 + 30 * inten + rng.randint(-5, 5))))
        # resolve to the tonic at the end of each section
        end = (start + n) * BEATS_PER_BAR - 2
        tonic = min(pool, key=lambda p: (abs(p - pool[idx]) if (p - root_pc) % 12 == 0 else 99))
        tr.notes.append(Note(end, 2.0, tonic, 80))
    return tr



# ---------------------------------------------------------------------------
# Vocal-aware helpers + Isan-indie arrangement
# ---------------------------------------------------------------------------

def _chord_pcs(chord: tuple[int, str]) -> set[int]:
    root, kind = chord
    return {(root + i) % 12 for i in CHORD_TYPES[kind]}


def _free_fn(guide):
    """free(t_beats) -> True when the vocal is silent at that moment (always True with no vocal)."""
    if guide is None:
        return lambda t: True
    act = guide.activity

    def free(t: float) -> bool:
        i = int(t * 2)
        return not (0 <= i < len(act) and bool(act[i]))
    return free


def _vocal_chords(root_pc, scale, guide, prog, bars, seventh):
    """Choose one chord per bar by Viterbi search: fit the sung notes, stay close to the genre's
    progression, avoid pointless changes and favour natural moves (4th/5th)."""
    cands = []
    for d in range(1, 8):
        c = degree_to_chord(root_pc, scale, d, seventh)
        if c[1] != "dim" and c not in cands:
            cands.append(c)
    base = [degree_to_chord(root_pc, scale, d, seventh) for d in prog]
    K = len(cands)
    emit = np.zeros((bars, K))
    for b in range(bars):
        want = base[b % len(base)]
        chroma = guide.bar_chroma[b] if b < len(guide.bar_chroma) else None
        if chroma is not None and chroma.sum() > 0:      # smooth over neighbouring bars: harmony changes slowly
            nb = [guide.bar_chroma[k] for k in (b - 1, b + 1) if 0 <= k < len(guide.bar_chroma)]
            chroma = 0.6 * chroma + 0.2 * sum(nb) / max(len(nb), 1) * 2 / 2
            chroma = chroma / max(chroma.sum(), 1e-9)
        sung = chroma is not None and chroma.sum() > 0
        for k, c in enumerate(cands):
            r, kind = c
            pcs = [(r + i) % 12 for i in CHORD_TYPES[kind]]
            if sung:
                sc = chroma[pcs[0]] * 1.0 + chroma[pcs[1]] * 0.85 + chroma[pcs[2]] * 0.85
                sc -= 0.45 * (1.0 - sum(chroma[p] for p in pcs))
                sc += 0.10 * (c == want)
            else:
                sc = 0.35 * (c == want)
            emit[b, k] = sc
    trans = np.zeros((K, K))
    for i, (r1, _) in enumerate(cands):
        for j, (r2, _) in enumerate(cands):
            iv = (r2 - r1) % 12
            trans[i, j] = 0.32 if i == j else (0.06 if iv in (5, 7) else (0.02 if iv in (2, 10, 3, 8) else 0.0))
    score = emit[0].copy()
    back = np.zeros((bars, K), dtype=int)
    for b in range(1, bars):
        cand = score[:, None] + trans
        back[b] = np.argmax(cand, axis=0)
        score = cand[back[b], np.arange(K)] + emit[b]
    path = [int(np.argmax(score))]
    for b in range(bars - 1, 0, -1):
        path.append(int(back[b][path[-1]]))
    return [cands[k] for k in reversed(path)]


def _bar_info(sections, bars, guide=None):
    """(section name, intensity) per bar. With a vocal, intensity follows how hard the singer sings."""
    info = [("verse", 0.6)] * bars
    for name, start, n, inten in sections:
        for b in range(start, min(bars, start + n)):
            info[b] = (name, inten)
    if guide is None:
        return info
    e = np.asarray(guide.energy[:bars], dtype=float)
    k = np.array([0.25, 0.5, 0.25])
    sm = np.convolve(np.pad(e, 1, mode="edge"), k, mode="valid")
    out = []
    for b, (name, inten) in enumerate(info):
        if b < len(sm) and e[b] > 0.05:
            mixed = 0.5 * inten + 0.5 * (0.3 + 0.7 * sm[b])
            out.append((name, float(np.clip(mixed, 0.25, 1.0))))
        else:
            out.append((name, inten * 0.9))   # instrumental bar: follow the section plan
    return out


def _snap(pool, midi):
    """Nearest note of `pool` to `midi` (after folding by octaves into the pool range)."""
    lo, hi = pool[0], pool[-1]
    while midi < lo:
        midi += 12
    while midi > hi:
        midi -= 12
    return min(pool, key=lambda p: abs(p - midi))


def _echo(guide, slot, pool, free, max_notes=4, up=0):
    """Answer the phrase the singer just finished: same contour (snapped to the pentatonic pool), in the gap.
    Returns [(t_beats, dur_beats, midi)]."""
    if guide is None:
        return []
    ph = guide.recent_phrase(slot, 8)
    if len(ph) < 2:
        return []
    ph = ph[-max_notes:]
    base = ph[0][0]
    out = []
    for rel, ln, m in ph:
        t = slot * 0.5 + (rel - base) * 0.5
        if not free(t):
            break
        out.append((t, max(0.4, ln * 0.5 * 0.95), _snap(pool, m + up)))
    return out


# ---------------------------------------------------------------------------
# Isan-indie arrangement: written parts (riffs / hooks / patterns), not random notes.
# Every part repeats in 4-bar phrases, changes on the 8th bar, and steps aside for the singer.
# ---------------------------------------------------------------------------

# (step in 16ths, pentatonic-degree offset from the chord root, length in 16ths, accent 0..1)
PHIN_RIFFS = {
    "intro":  [(0, 0, 4, 1.0), (8, 2, 4, .8)],
    "verse":  [(0, 0, 3, 1.0), (4, 2, 1, .6), (5, 3, 1, .7), (6, 2, 2, .6),
               (8, 0, 3, .9), (12, -1, 1, .5), (13, 0, 1, .6), (14, 2, 2, .7)],
    "chorus": [(0, 0, 2, 1.0), (2, 0, 1, .5), (3, 2, 1, .6), (4, 3, 2, .9), (6, 2, 1, .6), (7, 0, 1, .5),
               (8, 0, 2, 1.0), (10, 0, 1, .5), (11, 2, 1, .6), (12, 4, 2, .9), (14, 3, 1, .6), (15, 2, 1, .6)],
    "turn":   [(0, 0, 2, 1.0), (2, 2, 1, .7), (4, 3, 1, .7), (6, 4, 1, .8),
               (8, 3, 1, .7), (10, 2, 1, .7), (12, 0, 3, .8)],
}
WOT_HOOK = [(0, 4, 4, .9), (4, 5, 2, .8), (6, 4, 2, .8), (8, 2, 6, .9)]
PONG_PAT = (0, 2, 1, 3, 0, 2, 3, 4)


def _pent_pitch(root_pc: int, pent: str, deg: int, base: int) -> int:
    """MIDI pitch of pentatonic degree `deg` (0 = tonic, may be negative / > 4) with the tonic placed at/above `base`."""
    steps = SCALES[pent]
    octv, i = divmod(deg, len(steps))
    return base + ((root_pc - base) % 12) + steps[i] + 12 * octv


def _chord_deg(root_pc: int, chord_root: int, pent: str) -> int:
    """Pentatonic degree (kept low, -5..4) that is closest to the chord root."""
    steps = SCALES[pent]
    iv = (chord_root - root_pc) % 12
    d = min(range(len(steps)), key=lambda i: min((iv - steps[i]) % 12, (steps[i] - iv) % 12))
    return d - len(steps) if steps[d] > 7 else d


def _bass_isan(g: Genre, chords, info, bars: int, guide) -> Track:
    """Bass locked to the kick drum, with a walking-up approach note before each chord change."""
    free = _free_fn(guide)
    tr = Track("Bass", program=33, channel=1)
    kicks = [i for i, c in enumerate(g.kick) if c != "."] or [0, 8]
    root_of = lambda r: 28 + ((r - 28) % 12)  # noqa: E731
    for b in range(bars):
        name, inten = info[b]
        r, kind = chords[b % len(chords)]
        root = root_of(r)
        tt = b * 4
        sung = sum(0 if free(tt + k * 0.5) else 1 for k in range(8)) >= 4
        steps = [0, 8] if sung else kicks
        for i, s in enumerate(steps):
            end = steps[i + 1] if i + 1 < len(steps) else 16
            pitch = root + (12 if (i == len(steps) - 1 and not sung and b % 2 == 1 and len(steps) > 2) else 0)
            tr.notes.append(Note(tt + s * 0.25, (end - s) * 0.25 * 0.88, pitch, 96 if s == 0 else 82))
        nxt = chords[(b + 1) % len(chords)]
        if b + 1 < bars and nxt[0] != r and not sung:
            tr.notes.append(Note(tt + 3.5, 0.4, root_of(nxt[0]) - 2, 72))     # approach from a whole step below
    return tr


def _isan_arrange(rng, g, root_pc, scale, chords, sections, bars, guide) -> list[Track]:
    free = _free_fn(guide)
    info = _bar_info(sections, bars, guide)
    pent = pent_for(scale)
    ch = lambda b: chords[b % len(chords)]  # noqa: E731
    pos = {}
    for name, start, n, inten in sections:
        for b in range(start, min(bars, start + n)):
            pos[b] = (name, b - start)

    def vshare(b):
        return sum(0 if free(b * 4 + k * 0.5) else 1 for k in range(8)) / 8.0

    reg = guide.register if guide is not None else None

    def under(ps, share):
        ps = list(ps)
        if reg is None or share < 0.35:
            return ps
        while max(ps) > reg - 2 and min(ps) - 12 >= 36:
            ps = [p - 12 for p in ps]
        return ps

    def run_len(t, max_steps):
        """How many 16ths from t stay free of vocal (so a note never rings into the singer)."""
        k = 0
        while k < max_steps and free(t + k * 0.25):
            k += 1
        return k

    phin = Track("Phin", program=105, channel=4)
    khaen = Track("Khaen", program=21, channel=5)
    wot = Track("Wot", program=75, channel=6)
    gtr = Track("Guitar", program=27, channel=7)
    pno = Track("Piano", program=0, channel=8)
    syn = Track("Synth", program=89, channel=10)
    pong = Track("Pong", program=13, channel=11)
    perc = Track("Perc", is_drums=True)
    wot_pool = scale_notes(root_pc, pent, 67, 91)

    for b in range(bars):
        name, inten = info[b]
        sname, p = pos.get(b, ("verse", b))
        r, kind = ch(b)
        tt = b * 4
        share = vshare(b)
        dc = _chord_deg(root_pc, r, pent)
        soft = 0.75 if share > 0.5 else 1.0

        # ---- Phin: the written riff; soft pedal dyad when the singer is busy
        if share > 0.5:
            for s0, off in ((0, 0), (8, 2 if inten > 0.6 else 0)):
                phin.notes.append(Note(tt + s0 * 0.25, 1.4, _pent_pitch(root_pc, pent, dc + off, 55), 50))
        else:
            riff = ("intro" if sname == "intro" else "turn" if (p % 8 == 7 and sname in ("verse", "chorus"))
                    else "chorus" if (sname == "chorus" or inten >= 0.85) else "verse")
            for s0, off, ln, acc in PHIN_RIFFS[riff]:
                t = tt + s0 * 0.25
                if not free(t):
                    continue
                ln = max(1, min(ln, run_len(t, ln)))
                pitch = _pent_pitch(root_pc, pent, dc + off, 55)
                if acc >= 0.9 and s0 > 0 and free(t - 0.125):          # grace-note slide into accents
                    phin.notes.append(Note(t - 0.125, 0.1, _pent_pitch(root_pc, pent, dc + off - 1, 55), 48))
                phin.notes.append(Note(t, ln * 0.25 * 0.95, pitch, int(56 + 34 * acc * (0.5 + 0.5 * inten))))

        # ---- Khaen: drone (root + fifth) + chorus chops
        dr = 40 + ((r - 40) % 12)
        kv = int((42 + 25 * inten) * soft)
        for q in (dr, dr + 7, dr + 12):
            khaen.notes.append(Note(tt, 3.9, q, kv))
        if inten >= 0.9:
            tones = [q for q in scale_notes(root_pc, pent, 60, 76) if q % 12 in _chord_pcs(ch(b))][:3]
            for off in (1.5, 3.5):
                if free(tt + off):
                    for q in tones:
                        khaen.notes.append(Note(tt + off, 0.4, q, 62))

        # ---- Wot: answers the singer's phrase; plays a written hook in instrumental bars
        placed = False
        if guide is not None and wot_pool and sname != "outro":
            for s8 in range(8):
                gslot, t = b * 8 + s8, tt + s8 * 0.5
                if gslot > 0 and free(t) and not free(t - 0.5):
                    ans = _echo(guide, gslot, wot_pool, free, max_notes=4, up=12)
                    if len(ans) >= 2:
                        for t0, d0, m0 in ans:
                            wot.notes.append(Note(t0, d0, m0, int(64 + 20 * inten)))
                        placed = True
        if not placed and share == 0 and sname in ("verse", "chorus", "intro") and p % 4 in (1, 3):
            hook_ok = all(free(tt + s0 * 0.25) for s0, _, _, _ in WOT_HOOK)
            if hook_ok:
                for s0, off, ln, acc in WOT_HOOK:
                    wot.notes.append(Note(tt + s0 * 0.25, ln * 0.25 * 0.95, _pent_pitch(root_pc, pent, dc + off, 67),
                                          int(60 + 22 * acc * inten + 8)))

        # ---- Guitar: arpeggio (verse) / strum (chorus), voiced under the singer
        tones = under(chord_midi(r, kind, 52), share)
        gv = int((48 + 28 * inten) * soft)
        arp = tones + [tones[0] + 12]
        if inten >= 0.9:
            for beat in (0, 1, 1.5, 2.5, 3):
                for j, q in enumerate(tones):
                    gtr.notes.append(Note(tt + beat + j * 0.015, 0.9, q, gv))
        else:
            for i, k in enumerate((0, 2, 1, 2, 0, 2, 1, 2)):
                gtr.notes.append(Note(tt + i * 0.5, 0.5, arp[k % len(arp)], gv - (6 if i % 2 else 0)))

        # ---- Piano: open chords on 1 and 3; a short written lick at the end of every 4th bar
        pv = int((46 + 22 * inten) * soft)
        t3 = chord_midi(r, kind, 48)
        voicing = under([t3[0], t3[2], t3[1] + 12], share)
        for beat in (0, 2):
            for q in voicing:
                pno.notes.append(Note(tt + beat, 1.9, q, pv))
        if sname != "intro" and p % 4 == 3:
            for s8, off in ((6, 4), (7, 3)):
                if free(tt + s8 * 0.5):
                    pno.notes.append(Note(tt + s8 * 0.5, 0.5, _pent_pitch(root_pc, pent, dc + off, 72), 60))

        # ---- Synth pad
        for q in under(chord_midi(r, kind, 60), share):
            syn.notes.append(Note(tt, 3.95, q, int(34 + 20 * inten)))

        # ---- Pong lang: fixed 8th-note pattern in verse/chorus gaps
        if inten >= 0.55 and sname not in ("intro", "outro"):
            for s8 in range(8):
                t = tt + s8 * 0.5
                if free(t):
                    pong.notes.append(Note(t, 0.3, _pent_pitch(root_pc, pent, dc + PONG_PAT[s8], 64), int(56 + 16 * inten)))

        # ---- Percussion: ching on the off-beats, shaker, congas
        sv = int(62 * soft)
        for off in (0.5, 1.5, 2.5, 3.5):
            perc.notes.append(Note(tt + off, 0.1, CHING, sv))
        for s0 in range(0, 16, 2):
            perc.notes.append(Note(tt + s0 * 0.25, 0.05, SHAKER, int((44 + (8 if s0 % 4 == 0 else 0)) * soft)))
        if inten >= 0.5:
            for s0 in (2, 5, 10, 13):
                perc.notes.append(Note(tt + s0 * 0.25, 0.15, CONGA_HI, int(70 * soft)))
            for s0 in (7, 15):
                perc.notes.append(Note(tt + s0 * 0.25, 0.15, CONGA_LO, int(74 * soft)))

    return [phin, khaen, wot, gtr, pno, syn, pong, perc]



def compose(
    genre: str = "all",
    bpm: float | None = None,
    key: int | None = None,
    scale: str | None = None,
    bars: int = 32,
    seed: int | None = None,
    lead: bool = True,
    guide=None,
) -> Song:
    """Compose a song. All arguments optional; `genre='all'` picks a random genre."""
    if seed is None:
        seed = random.randrange(1, 2**31)
    rng = random.Random(seed)
    g = get_genre(genre, rng)
    bpm = float(bpm) if bpm else float(rng.randint(*g.bpm))
    scale = scale if scale in SCALES else rng.choice(g.scales)
    root_pc = key if key is not None else rng.randrange(12)
    prog = rng.choice(g.progressions)
    if guide is not None:
        chords = _vocal_chords(root_pc, scale, guide, prog, bars, g.seventh)
    else:
        chords = [degree_to_chord(root_pc, scale, d, g.seventh) for d in prog]

    sections = build_sections(bars)
    song = Song(g.name, bpm, root_pc, scale, bars, seed, sections=sections)
    song.tracks.append(_drums(rng, g, sections, bars, tight=bool(g.extra.get("isan"))))
    if g.extra.get("isan"):
        song.tracks.append(_bass_isan(g, chords, _bar_info(sections, bars, guide), bars, guide))
    else:
        song.tracks.append(_bass(rng, g, chords, sections, bars, guide))
    if g.extra.get("isan"):
        song.tracks.extend(_isan_arrange(rng, g, root_pc, scale, chords, sections, bars, guide))
    else:
        song.tracks.append(_chords(rng, g, chords, sections, bars))
        if lead:
            song.tracks.append(_melody(rng, g, root_pc, scale, chords, sections, bars))
    song.tracks = [t for t in song.tracks if t.notes]
    return song

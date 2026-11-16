"""Rule-based + stochastic composer: turns a genre profile into MIDI-like tracks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .genres import Genre, get_genre
from .midi_io import Note, Track
from .theory import (
    NOTE_NAMES, SCALES, chord_midi, degree_to_chord, scale_notes,
)

BEATS_PER_BAR = 4
KICK, SNARE, CLAP, HAT, OPEN_HAT, RIDE = 36, 38, 39, 42, 46, 51

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

    @property
    def key_name(self) -> str:
        return f"{NOTE_NAMES[self.root_pc]} {self.scale}"

    @property
    def seconds(self) -> float:
        return self.bars * BEATS_PER_BAR * 60.0 / self.bpm


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


def _drums(rng: random.Random, g: Genre, sections, bars: int) -> Track:
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
                    if rng.random() < 0.08 and step % 4 == 2:
                        tr.notes.append(Note(t, 0.2, OPEN_HAT, 70))
                    else:
                        tr.notes.append(Note(t, 0.1, HAT, int(vel * (0.7 + 0.3 * inten))))
                if use_hat_only:
                    continue
                if k != ".":
                    tr.notes.append(Note(t, 0.15, KICK, 108 + human))
                elif inten > 0.7 and rng.random() < 0.06 and step % 2 == 0:
                    tr.notes.append(Note(t, 0.15, KICK, 80))  # ghost kick
                if s != ".":
                    tr.notes.append(Note(t, 0.15, SNARE, 104 + human))
                    if inten >= 1.0 and g.name in ("pop", "edm", "house"):
                        tr.notes.append(Note(t, 0.15, CLAP, 70))
                elif rng.random() < 0.05 * inten and step % 2 == 1:
                    tr.notes.append(Note(t, 0.1, SNARE, 45))  # ghost snare
            if last_bar_of_section and n >= 4 and inten > 0.5:
                for i in range(4):  # small fill on last beat
                    tr.notes.append(Note(bar_beat + 3 + i * 0.25, 0.12, SNARE, 70 + i * 10))
    return tr


def _bass(rng: random.Random, g: Genre, chords, sections, bars: int) -> Track:
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
                pattern = [0, 0.75, 1.5, 2.5, 3.0, 3.5]
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


def compose(
    genre: str = "all",
    bpm: float | None = None,
    key: int | None = None,
    scale: str | None = None,
    bars: int = 32,
    seed: int | None = None,
    lead: bool = True,
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
    chords = [degree_to_chord(root_pc, scale, d, g.seventh) for d in prog]

    sections = build_sections(bars)
    song = Song(g.name, bpm, root_pc, scale, bars, seed, sections=sections)
    song.tracks.append(_drums(rng, g, sections, bars))
    song.tracks.append(_bass(rng, g, chords, sections, bars))
    song.tracks.append(_chords(rng, g, chords, sections, bars))
    if lead:
        song.tracks.append(_melody(rng, g, root_pc, scale, chords, sections, bars))
    song.tracks = [t for t in song.tracks if t.notes]
    return song

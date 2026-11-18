"""Genre profiles. The program supports 'all' genres through this table.

Each profile drives the composer (tempo, scale, progressions, drum grid, bass /
chord / melody behaviour) and the synth (timbres).
Drum grids are 16 steps per bar (1 = hit, 0 = rest, 2 = ghost/soft hit).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Genre:
    name: str
    bpm: tuple[int, int]
    scales: tuple[str, ...]
    progressions: tuple[tuple[int, ...], ...]  # scale degrees, 1 chord per bar
    kick: str
    snare: str
    hat: str
    swing: float = 0.0  # 0..0.35 shifts off-beat 16ths
    bass_style: str = "root"  # root | walking | octave | syncopated | sub | arp
    chord_style: str = "pad"  # pad | stabs | arp | strum | offbeat
    melody_density: float = 0.6  # probability a 8th/16th slot has a note
    melody_range: tuple[int, int] = (60, 84)
    seventh: bool = False
    lead_timbre: str = "saw"  # sine | saw | square | pluck | bell | organ
    chord_timbre: str = "pad"
    bass_timbre: str = "bass"
    extra: dict = field(default_factory=dict)


def _g(s: str) -> str:
    """Pattern string like 'x...x...x...x...' -> compact 16-char string."""
    s = s.replace(" ", "")
    assert len(s) == 16, s
    return s


GENRES: dict[str, Genre] = {
    "pop": Genre(
        "pop", (96, 124), ("major", "minor"),
        ((1, 5, 6, 4), (6, 4, 1, 5), (1, 6, 4, 5), (4, 1, 5, 6)),
        _g("x...x...x...x..."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.0, "root", "arp", 0.65, (62, 84), False, "pluck", "pad",
    ),
    "rock": Genre(
        "rock", (110, 150), ("minor", "mixolydian", "pentatonic_minor"),
        ((1, 4, 5, 5), (1, 7, 4, 1), (6, 7, 1, 1), (1, 5, 6, 4)),
        _g("x.....x.x.x....."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.0, "octave", "strum", 0.55, (57, 79), False, "saw", "organ",
    ),
    "hiphop": Genre(
        "hiphop", (78, 98), ("minor", "pentatonic_minor", "dorian"),
        ((1, 1, 4, 5), (1, 6, 7, 1), (6, 5, 4, 5)),
        _g("x.....x...x....."), _g("....x.......x..."), _g("x.xxx.x.x.xxx.x."),
        0.12, "sub", "stabs", 0.35, (60, 79), True, "bell", "pad",
    ),
    "edm": Genre(
        "edm", (124, 134), ("minor", "major"),
        ((6, 4, 1, 5), (1, 5, 6, 4), (1, 6, 3, 7)),
        _g("x...x...x...x..."), _g("....x.......x..."), _g("..x...x...x...x."),
        0.0, "syncopated", "offbeat", 0.7, (64, 88), False, "saw", "pad",
    ),
    "house": Genre(
        "house", (120, 128), ("minor", "dorian"),
        ((1, 1, 4, 5), (6, 7, 1, 1), (1, 4, 6, 5)),
        _g("x...x...x...x..."), _g("....x.......x..."), _g("..x...x...x...x."),
        0.05, "syncopated", "offbeat", 0.45, (60, 82), True, "square", "organ",
    ),
    "lofi": Genre(
        "lofi", (68, 88), ("dorian", "minor", "major"),
        ((2, 5, 1, 6), (1, 6, 2, 5), (4, 3, 2, 1)),
        _g("x.....x...x....."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.2, "walking", "stabs", 0.3, (60, 79), True, "bell", "pad",
    ),
    "jazz": Genre(
        "jazz", (84, 140), ("dorian", "major", "mixolydian"),
        ((2, 5, 1, 1), (1, 6, 2, 5), (3, 6, 2, 5)),
        _g("x.......x......."), _g("....x.......x..."), _g("x..xx..xx..xx..x"),
        0.3, "walking", "stabs", 0.5, (58, 82), True, "sine", "organ",
    ),
    "blues": Genre(
        "blues", (72, 112), ("blues", "pentatonic_minor", "mixolydian"),
        ((1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5),),
        _g("x.....x...x....."), _g("....x.......x..."), _g("x..xx..xx..xx..x"),
        0.3, "walking", "strum", 0.45, (55, 79), True, "saw", "organ",
    ),
    "rnb": Genre(
        "rnb", (66, 96), ("minor", "dorian", "major"),
        ((2, 5, 1, 6), (6, 5, 4, 5), (1, 3, 6, 4)),
        _g("x.....x...x....."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.15, "syncopated", "pad", 0.5, (60, 82), True, "sine", "pad",
    ),
    "country": Genre(
        "country", (88, 130), ("major", "mixolydian", "pentatonic_major"),
        ((1, 4, 5, 1), (1, 5, 6, 4), (1, 1, 4, 5)),
        _g("x...x...x...x..."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.05, "walking", "strum", 0.55, (57, 81), False, "pluck", "organ",
    ),
    "reggae": Genre(
        "reggae", (66, 84), ("major", "minor"),
        ((1, 4, 1, 5), (6, 4, 1, 5)),
        _g("........x......."), _g("........x......."), _g("x.x.x.x.x.x.x.x."),
        0.08, "sub", "offbeat", 0.4, (60, 79), False, "organ", "organ",
    ),
    "latin": Genre(
        "latin", (92, 124), ("minor", "phrygian", "harmonic_minor"),
        ((1, 4, 5, 1), (1, 7, 6, 5), (6, 7, 1, 1)),
        _g("x..x..x...x..x.."), _g("....x..x....x..."), _g("x.x.x.x.x.x.x.x."),
        0.05, "syncopated", "strum", 0.6, (60, 84), False, "pluck", "organ",
    ),
    "classical": Genre(
        "classical", (60, 112), ("major", "minor", "harmonic_minor"),
        ((1, 4, 5, 1), (1, 6, 4, 5), (6, 4, 1, 5)),
        _g("................"), _g("................"), _g("................"),
        0.0, "root", "arp", 0.55, (60, 88), False, "sine", "pad",
        extra={"no_drums": True},
    ),
    "ambient": Genre(
        "ambient", (56, 78), ("major", "dorian", "pentatonic_major"),
        ((1, 5, 6, 4), (4, 1, 5, 1)),
        _g("................"), _g("................"), _g("................"),
        0.0, "sub", "pad", 0.2, (64, 91), True, "bell", "pad",
        extra={"no_drums": True},
    ),
    # Isan indie: phin, khaen, wot, pong lang + guitar, piano, synth, drums, bass, percussion.
    # Layers are built in composer._isan_layers (extra["isan"]).
    "isan": Genre(
        "isan", (92, 118), ("pentatonic_minor", "isan_yai", "isan_noi", "pentatonic_major"),
        ((1, 7, 6, 7), (1, 4, 1, 5), (6, 7, 1, 1), (1, 3, 7, 4)),
        _g("x.....x...x....."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.06, "syncopated", "pad", 0.5, (55, 84), False, "phin", "khaen", "bass",
        extra={"isan": True},
    ),
    "metal": Genre(
        "metal", (130, 180), ("phrygian", "minor", "harmonic_minor"),
        ((1, 2, 1, 7), (1, 6, 7, 1), (1, 1, 4, 5)),
        _g("x.x.x.x.x.x.x.x."), _g("....x.......x..."), _g("x.x.x.x.x.x.x.x."),
        0.0, "octave", "strum", 0.65, (57, 84), False, "saw", "organ",
    ),
}

GENRE_NAMES = sorted(GENRES)


def get_genre(name: str, rng=None) -> Genre:
    """Return a genre profile. 'all' picks one at random (needs an rng)."""
    key = name.lower().strip()
    key = {"isan_indie": "isan", "อีสาน": "isan", "อีสานอินดี้": "isan", "isaan": "isan"}.get(key, key)
    if key in ("all", "random", "any"):
        if rng is None:
            import random as _r
            rng = _r.Random()
        key = rng.choice(GENRE_NAMES)
    if key not in GENRES:
        raise ValueError(f"Unknown genre '{name}'. Choose one of: all, {', '.join(GENRE_NAMES)}")
    return GENRES[key]

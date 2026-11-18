"""Basic music theory helpers: scales, chords, progressions."""

from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    # Isan (Thai north-eastern) pentatonic modes used for phin / khaen / wot
    "isan_noi": [0, 2, 5, 7, 9],   # C D F G A  (lai noi flavour)
    "isan_yai": [0, 3, 5, 7, 10],  # C Eb F G Bb (lai yai flavour, = minor pentatonic)
}


def pent_for(scale: str) -> str:
    """Pentatonic scale that fits `scale` (minor-ish -> minor pentatonic, else major)."""
    steps = SCALES[scale]
    if 3 in steps and 4 not in steps:
        return "pentatonic_minor"
    return "pentatonic_major"

CHORD_TYPES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}


def note_name_to_pc(name: str) -> int:
    """'C' -> 0, 'F#' -> 6. Accepts flats like 'Bb'."""
    name = name.strip()
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    name = flats.get(name, name)
    return NOTE_NAMES.index(name)


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def scale_notes(root_pc: int, scale: str, low: int, high: int) -> list[int]:
    """All MIDI notes of `scale` rooted at pitch class `root_pc` in [low, high]."""
    steps = SCALES[scale]
    return [
        n
        for n in range(low, high + 1)
        if (n - root_pc) % 12 in steps
    ]


def degree_to_chord(root_pc: int, scale: str, degree: int, seventh: bool = False) -> tuple[int, str]:
    """Return (chord_root_pc, chord_type) for a 1-based scale degree (diatonic)."""
    steps = SCALES[scale]
    if len(steps) < 7:
        # fall back to the parent major/minor for pentatonic/blues scales
        steps = SCALES["major"] if 4 in steps else SCALES["minor"]
    idx = (degree - 1) % 7
    chord_root = (root_pc + steps[idx]) % 12
    third = (steps[(idx + 2) % 7] - steps[idx]) % 12
    fifth = (steps[(idx + 4) % 7] - steps[idx]) % 12
    if third == 4 and fifth == 7:
        kind = "maj7" if seventh else "maj"
    elif third == 3 and fifth == 7:
        kind = "min7" if seventh else "min"
    elif third == 3 and fifth == 6:
        kind = "dim"
    elif third == 4 and fifth == 8:
        kind = "maj"
    else:
        kind = "maj"
    if seventh and kind == "maj":
        # dominant 7th if the 7th of the scale is a minor 7th above the root
        sev = (steps[(idx + 6) % 7] - steps[idx]) % 12
        kind = "dom7" if sev == 10 else "maj7"
    return chord_root, kind


def chord_midi(root_pc: int, kind: str, octave_base: int = 48) -> list[int]:
    """Chord tones as MIDI notes, root placed at the lowest note >= octave_base."""
    root = octave_base + ((root_pc - octave_base) % 12)
    return [root + i for i in CHORD_TYPES[kind]]

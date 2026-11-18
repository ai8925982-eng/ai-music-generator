import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimusicgen.composer import compose  # noqa: E402
from aimusicgen.genres import GENRES  # noqa: E402


def _bar_vec(notes, b):
    v = np.zeros(16)
    for n in notes:
        t = n.start - b * 4
        if 0 <= t < 4:
            v[int(round(t * 4)) % 16] += 1
    return v


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 if na == 0 and nb == 0 else (0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb)))


def _track(song, name):
    return [t for t in song.tracks if t.name == name][0]


def test_isan_is_deterministic():
    a = compose("isan", seed=21, bars=24)
    b = compose("isan", seed=21, bars=24)
    for ta, tb in zip(a.tracks, b.tracks):
        assert [(n.start, n.pitch, n.velocity) for n in ta.notes] == [(n.start, n.pitch, n.velocity) for n in tb.notes]


def test_phin_riff_repeats_like_a_written_part():
    song = compose("isan", seed=3, bars=32)
    notes = _track(song, "Phin").notes
    sims = [_cos(_bar_vec(notes, b), _bar_vec(notes, b - 4)) for b in range(8, song.bars - 4)]
    assert np.mean(sims) > 0.6          # random note placement scores ~0.2


def test_bass_is_locked_to_the_kick():
    song = compose("isan", seed=3, bars=16)
    kick_steps = {i for i, c in enumerate(GENRES["isan"].kick) if c != "."}
    on_kick = off = 0
    for n in _track(song, "Bass").notes:
        step = (n.start % 4) / 0.25
        if abs(step - round(step)) < 1e-6 and int(round(step)) in kick_steps:
            on_kick += 1
        else:
            off += 1
    assert on_kick > 3 * off            # only the approach notes may fall elsewhere


def test_no_notes_outside_midi_range():
    song = compose("isan", seed=8, bars=32)
    for t in song.tracks:
        for n in t.notes:
            assert 0 <= n.pitch <= 127 and 1 <= n.velocity <= 127 and n.duration > 0

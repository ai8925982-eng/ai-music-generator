import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimusicgen import fx, instruments as I  # noqa: E402
from aimusicgen.audio_io import write_wav  # noqa: E402
from aimusicgen.composer import CHORD_TYPES, _vocal_chords, compose  # noqa: E402
from aimusicgen.genres import GENRES, get_genre  # noqa: E402
from aimusicgen.pitch import segment_notes, track_pitch  # noqa: E402
from aimusicgen.synth import SR, master_bus, render_stereo  # noqa: E402
from aimusicgen.theory import degree_to_chord  # noqa: E402
from aimusicgen.vocal import build_guide, load_vocal  # noqa: E402
from test_pipeline import make_phrased_vocal  # noqa: E402


def sung(melody, sr=44100):
    rng = np.random.default_rng(0)
    parts = []
    for m, d in melody:
        n = int(d * sr)
        t = np.arange(n) / sr
        if m is None:
            parts.append(rng.standard_normal(n) * 0.002)
            continue
        f = 440 * 2 ** ((m - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))
        ph = 2 * np.pi * np.cumsum(f) / sr
        s = sum(a * np.sin((h + 1) * ph) for h, a in enumerate([1, .7, .5, .35, .25, .15, .1, .08])) * 0.25
        s *= np.minimum(1, t * 40) * np.minimum(1, (d - t) * 30)
        parts.append(s + rng.standard_normal(n) * 0.003)
    return np.concatenate(parts)


def test_pitch_tracker_finds_every_sung_note():
    mel = [(57, .5), (60, .5), (62, 1), (None, .5), (64, .5), (62, .5), (60, 1), (57, 1.5), (None, 1)] * 3
    pt = track_pitch(sung(mel), 44100)
    got = [n.midi for n in segment_notes(pt)]
    assert got == [m for m, _ in mel if m is not None]


def test_repeated_syllables_on_one_pitch_are_split():
    # "na na na na": same pitch, loudness dips between syllables
    x = sung([(60, 0.3)] * 6)
    assert len(segment_notes(track_pitch(x, 44100))) >= 5


def _f0(y, true, sr=SR):
    y = y[int(0.05 * sr):int(0.45 * sr)]
    y = y - y.mean()
    ac = np.correlate(y, y, "full")[len(y) - 1:]
    lag = np.arange(int(sr / (true * 1.4)), int(sr / (true * 0.7)))
    k = lag[np.argmax(ac[lag])]
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    return sr / (k + 0.5 * (a - c) / (a - 2 * b + c))


@pytest.mark.parametrize("voice,pitches", [("guitar", [45, 57, 72]), ("phin", [57, 72, 79]),
                                           ("bass", [28, 40]), ("piano", [36, 60, 84])])
def test_instruments_are_in_tune(voice, pitches):
    for p in pitches:
        y = I.VOICES[voice](p, 1.0, 90, 0)
        true = 440 * 2 ** ((p - 69) / 12)
        cents = 1200 * np.log2(_f0(y, true) / true)
        assert abs(cents) < 8, (voice, p, cents)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.02


def test_round_robin_variants_differ():
    a, b = I.phin(64, 0.5, 90, 0), I.phin(64, 0.5, 90, 1)
    n = min(len(a), len(b))
    assert not np.allclose(a[:n], b[:n])


def test_fft_convolve_matches_direct():
    rng = np.random.default_rng(1)
    x, h = rng.standard_normal(5000), rng.standard_normal(700)
    assert np.abs(fx.fft_convolve(x, h) - np.convolve(x, h)[:5000]).max() < 1e-9


def test_body_resonance_is_gentle():
    ir = fx.body_ir(fx.BODIES["Guitar"])
    gain = np.abs(np.fft.rfft(ir, 1 << 16)).max()
    assert gain < 4.0     # < ~12 dB boost; guards against the earlier 30-40 dB bass blow-up


def test_isan_bus_is_balanced():
    song = compose("isan", bars=16, seed=5)
    bus = master_bus(render_stereo(song, get_genre("isan"), SR), get_genre("isan"), SR)
    assert np.abs(bus).max() < 1.2
    spec = np.abs(np.fft.rfft(bus[: SR * 20, 0])) ** 2
    fr = np.fft.rfftfreq(SR * 20, 1 / SR)
    low = spec[(fr >= 20) & (fr < 150)].sum() / spec.sum()
    high = spec[(fr >= 2000) & (fr < 16000)].sum() / spec.sum()
    assert low < 0.5 and high > 0.15


def test_chords_follow_the_melody(tmp_path):
    p = tmp_path / "v.wav"
    make_phrased_vocal(p)
    v = load_vocal(str(p))
    bars = 16
    g = build_guide(v, v.bpm, bars, v.grid_offset_s)
    chords = _vocal_chords(v.root_pc, v.scale, g, GENRES["isan"].progressions[0], bars, False)
    ok = tot = 0
    for i, m in enumerate(g.melody):
        if m < 0 or i // 8 >= bars:
            continue
        r, k = chords[i // 8]
        ok += (m % 12) in {(r + x) % 12 for x in CHORD_TYPES[k]}
        tot += 1
    template = [degree_to_chord(v.root_pc, v.scale, d, False) for d in GENRES["isan"].progressions[0]]
    ok_t = sum((m % 12) in {(template[(i // 8) % 4][0] + x) % 12 for x in CHORD_TYPES[template[(i // 8) % 4][1]]}
               for i, m in enumerate(g.melody) if m >= 0 and i // 8 < bars)
    # passing tones in a real melody are fine; the point is that the fitted chords beat the fixed template clearly
    assert ok / tot > 0.55 and ok / tot > ok_t / tot + 0.1, (ok / tot, ok_t / tot)

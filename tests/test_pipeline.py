import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimusicgen.audio_io import read_audio, write_wav  # noqa: E402
from aimusicgen.composer import compose  # noqa: E402
from aimusicgen.genres import GENRE_NAMES  # noqa: E402
from aimusicgen.midi_io import read_midi, write_midi  # noqa: E402
from aimusicgen.pipeline import GenerateOptions, generate  # noqa: E402
from aimusicgen.synth import SR  # noqa: E402


def make_vocal(path, bpm=100, notes=(69, 72, 76, 72, 69, 67, 64, 67) * 4, sr=44100):
    spb = 60 / bpm
    sig = np.zeros(int(len(notes) * spb * sr) + sr)
    for i, n in enumerate(notes):
        t = np.arange(int(spb * sr)) / sr
        f = 440 * 2 ** ((n - 69) / 12)
        s = np.sin(2 * np.pi * f * t) * np.minimum(1, t * 50) * np.exp(-t * 2) * 0.5
        st = int(i * spb * sr)
        sig[st:st + len(s)] += s
    write_wav(path, sig, sr)
    return sig


@pytest.mark.parametrize("genre", GENRE_NAMES)
def test_every_genre_composes(genre):
    song = compose(genre, bars=16, seed=42)
    assert song.tracks
    assert all(n.pitch in range(128) for t in song.tracks for n in t.notes)


def test_same_seed_same_song():
    a = compose("pop", seed=11, bars=16)
    b = compose("pop", seed=11, bars=16)
    assert [(n.start, n.pitch) for t in a.tracks for n in t.notes] == \
           [(n.start, n.pitch) for t in b.tracks for n in t.notes]


def test_midi_roundtrip(tmp_path):
    song = compose("rock", seed=3, bars=8)
    p = write_midi(tmp_path / "x.mid", song.tracks, song.bpm)
    tracks, bpm = read_midi(p)
    assert abs(bpm - song.bpm) < 0.5
    assert sum(len(t.notes) for t in tracks) == sum(len(t.notes) for t in song.tracks)


def test_audio_and_midi_outputs(tmp_path):
    res = generate(GenerateOptions(genre="all", output_dir=str(tmp_path), seed=5, bars=8))
    assert res.midi_path.exists() and res.audio_path.exists()
    audio, sr = read_audio(res.audio_path)
    assert sr == SR and audio.shape[1] == 2 and np.abs(audio).max() > 0.05
    info = json.loads(res.info_path.read_text(encoding="utf-8"))
    assert info["mode"] == "audio_midi"


def test_midi_only_mode(tmp_path):
    res = generate(GenerateOptions(output_dir=str(tmp_path), seed=5, bars=8, mode="midi"))
    assert res.midi_path.exists() and res.audio_path is None


def test_vocal_is_preserved(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    make_vocal(vocal_path)
    res = generate(GenerateOptions(genre="lofi", output_dir=str(tmp_path / "o"), seed=2,
                                   vocal_path=str(vocal_path), duck_db=0.0))
    # vocal info follows the vocal
    assert res.vocal is not None and res.vocal.bpm is not None
    assert abs(res.song.bpm - res.vocal.bpm) < 1
    mix, sr = read_audio(res.audio_path)
    acc, _ = read_audio(res.accompaniment_path)
    voc, vsr = read_audio(vocal_path)
    n = min(len(voc), len(mix), len(acc))
    # mix - scaled accompaniment must correlate ~1.0 with the original vocal
    # (least-squares fit of the mix onto [vocal, accompaniment])
    A = np.stack([voc[:n, 0], acc[:n, 0]], axis=1)
    coef, *_ = np.linalg.lstsq(A, mix[:n, 0], rcond=None)
    residual = mix[:n, 0] - A @ coef
    assert coef[0] > 0.05
    assert np.sqrt(np.mean(residual ** 2)) < 0.02 * max(np.sqrt(np.mean(mix[:n, 0] ** 2)), 1e-6) + 0.01


def test_bad_mode():
    with pytest.raises(ValueError):
        generate(GenerateOptions(mode="nope"))


def make_phrased_vocal(path, bpm=96, start=1.3, sr=44100):
    """A minor phrases: 2 bars of singing, 1 bar of rest, repeated; starts off the downbeat of the file."""
    spb = 60 / bpm
    phrase = [69, 72, 71, 69, 67, 64, 67, 69, 72, 74, 72, 69, 71, 69, 67, 64]
    sig = np.zeros(int(sr * (start + spb * 4 * 12 + 2)))
    t0 = start
    for _ in range(4):
        for n in phrase:
            d = int(spb / 2 * sr)
            t = np.arange(d) / sr
            f = 440 * 2 ** ((n - 69) / 12)
            s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.minimum(1, t * 60) * np.minimum(1, (d / sr - t) * 60) * 0.4
            a = int(t0 * sr)
            sig[a:a + d] += s[:len(sig) - a]
            t0 += spb / 2
        t0 += spb * 4
    write_wav(path, sig, sr)
    return start


def test_isan_has_all_instruments(tmp_path):
    song = compose("isan", seed=4, bars=16)
    names = {t.name for t in song.tracks}
    assert {"Phin", "Khaen", "Wot", "Guitar", "Piano", "Synth", "Drums", "Bass"} <= names


def test_isan_fits_vocal(tmp_path):
    from aimusicgen.vocal import build_guide
    vp = tmp_path / "v.wav"
    start = make_phrased_vocal(vp)
    res = generate(GenerateOptions(genre="isan", output_dir=str(tmp_path / "o"), seed=9, vocal_path=str(vp)))
    v = res.vocal
    assert abs(res.song.bpm - 96) < 1.5                      # tempo locked to the vocal
    assert v.key_name == "A minor"
    assert abs(res.song.offset_s - start) < 0.1              # bar line aligned to the vocal
    guide = build_guide(v, res.song.bpm, res.song.bars, res.song.offset_s)
    spb = 60 / res.song.bpm
    for name in ("Wot", "Pong"):                             # melodic parts stay out of the singer's way
        tr = [t for t in res.song.tracks if t.name == name][0]
        hits = sum(1 for n in tr.notes if guide.activity[int((n.start - res.song.offset_s / spb) * 2)])
        assert hits == 0
    audio, _ = read_audio(res.audio_path)
    assert np.abs(audio).max() <= 1.0

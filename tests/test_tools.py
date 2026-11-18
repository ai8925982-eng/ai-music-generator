import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aimusicgen import audio_tools as T  # noqa: E402
from aimusicgen.audio_io import read_audio, write_wav  # noqa: E402
from aimusicgen.project import Project, TrackState  # noqa: E402

SR = 22050


def tone(f, dur=2.0):
    t = np.arange(int(dur * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def f0(y):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.argmax(sp) * SR / len(y)


@pytest.mark.parametrize("rate", [0.75, 1.5])
def test_stretch_changes_length_not_pitch(rate):
    y = T.time_stretch(tone(440), rate)
    assert abs(len(y) / (2 * SR) - 1 / rate) < 0.03
    assert abs(f0(y[SR // 4:-SR // 4]) - 440) < 8


def test_pitch_shift_keeps_length():
    y = T.pitch_shift(tone(440), 12)
    assert len(y) == 2 * SR and abs(f0(y[SR // 4:-SR // 4]) - 880) < 12


def test_edit_ops():
    x = tone(220)
    rest, clip = T.cut(x, SR, 0.5, 1.0)
    assert len(rest) + len(clip) == len(x)
    assert len(T.paste(rest, SR, clip, 0.5)) == len(x)
    f = T.fade(x, SR, 0.1, 0.1)
    assert f[0] == 0 and abs(f[-1]) < 1e-3
    assert len(T.replace_section(x, SR, 0.5, 1.0, tone(330, 1.0))) == len(x)


def test_chord_detection():
    t = np.arange(4 * SR) / SR
    am = sum(np.sin(2 * np.pi * f * t) for f in (220, 261.63, 329.63)).astype(np.float32)
    c = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0)).astype(np.float32)
    ch = T.detect_chords(np.concatenate([am, c]), SR, 1.0)
    assert ch[0][2] == "Am" and ch[-1][2] == "C"


def test_export_and_project(tmp_path):
    x = np.stack([tone(440, 1), tone(660, 1)], axis=1)
    p = T.export_audio(x, SR, tmp_path / "a.wav", 48000, 24)
    y, sr = read_audio(p)
    assert sr == 48000 and abs(len(y) - 48000) < 5
    proj = Project("t", {"genre": "isan"}, [TrackState("a", str(p), -6, 0.5), TrackState("b", str(p), mute=True)])
    proj.save(tmp_path / "x.aimproj")
    back = Project.load(tmp_path / "x.aimproj")
    assert back.tracks[0].volume_db == -6 and back.options["genre"] == "isan"
    assert back.mixdown().shape[1] == 2
    with pytest.raises(ValueError):
        Project.load(tmp_path / "missing.aimproj")

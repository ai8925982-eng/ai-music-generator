import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimusicgen.sf2 import SoundFont  # noqa: E402
from sf2_builder import build_sf2  # noqa: E402


@pytest.fixture(scope="module")
def sf(tmp_path_factory):
    return SoundFont(build_sf2(tmp_path_factory.mktemp("sf") / "test.sf2"))


def _f0(y, sr=44100, lo=100, hi=1500):
    y = y[int(0.05 * sr):int(0.35 * sr)]
    ac = np.correlate(y, y, "full")[len(y) - 1:]
    lag = np.arange(int(sr / hi), int(sr / lo))
    return sr / lag[np.argmax(ac[lag])]


def test_reads_presets(sf):
    assert (0, 0) in sf.presets and (128, 0) in sf.presets
    assert sf.preset_names[(0, 0)] == "Sine"


@pytest.mark.parametrize("pitch", [60, 67, 72])
def test_notes_are_in_tune(sf, pitch):
    y = sf.render(0, pitch, 0.6, 100)
    true = 440 * 2 ** ((pitch - 69) / 12)
    assert abs(1200 * np.log2(_f0(y) / true)) < 15


def test_loop_sustains_longer_than_the_sample(sf):
    y = sf.render(0, 60, 1.5, 100)       # source sample is only 0.11 s long
    assert len(y) > 1.4 * 44100
    assert np.abs(y[int(1.0 * 44100):int(1.2 * 44100)]).max() > 0.05


def test_release_fades_out(sf):
    y = sf.render(0, 60, 0.5, 100)
    assert np.abs(y[-10:]).max() < 0.02                         # ends silent, no click
    assert np.abs(y[int(0.3 * 44100):int(0.4 * 44100)]).max() > 0.1   # still sounding before note-off
    assert len(y) < int(0.65 * 44100)                           # and does not ring on after the release


def test_velocity_scales_level(sf):
    assert np.abs(sf.render(0, 60, 0.3, 120)).max() > 1.5 * np.abs(sf.render(0, 60, 0.3, 50)).max()


def test_drum_kit_one_shot(sf):
    k = sf.render(0, 36, 0.1, 110, drums=True)
    assert k is not None and np.abs(k).max() > 0.1
    assert sf.render(0, 50, 0.1, 110, drums=True) is None      # no zone for that key


def test_rejects_other_files(tmp_path):
    p = tmp_path / "x.sf2"
    p.write_bytes(b"not a soundfont")
    with pytest.raises(ValueError):
        SoundFont(p)


def test_pipeline_uses_soundfont(tmp_path):
    from aimusicgen.pipeline import GenerateOptions, generate
    p = build_sf2(tmp_path / "t.sf2")
    o = GenerateOptions(genre="isan", output_dir=str(tmp_path), name="s", bars=4, seed=1, mode="audio",
                        soundfont=str(p))
    r = generate(o, lambda *a: None)
    assert r.audio_path and Path(r.audio_path).exists()

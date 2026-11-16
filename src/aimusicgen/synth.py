"""Numpy synthesizer: renders Track notes to stereo float32 audio (no samplerate deps)."""

from __future__ import annotations

import numpy as np

from .composer import CLAP, HAT, KICK, OPEN_HAT, RIDE, SNARE
from .midi_io import Track
from .theory import midi_to_hz

SR = 44100


def _env(n: int, a: float, d: float, s: float, r: float, sr: int = SR) -> np.ndarray:
    """ADSR envelope of n samples (release happens inside n)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    nr = min(nr, n // 2)
    na = min(na, n - nr)
    nd = min(nd, max(0, n - nr - na))
    ns = max(0, n - na - nd - nr)
    parts = [
        np.linspace(0, 1, na, endpoint=False) if na else np.zeros(0),
        np.linspace(1, s, nd, endpoint=False) if nd else np.zeros(0),
        np.full(ns, s),
        np.linspace(s, 0, nr) if nr else np.zeros(0),
    ]
    env = np.concatenate(parts)
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n].astype(np.float32)


def _osc(kind: str, freq: float, n: int, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    ph = 2 * np.pi * freq * t
    if kind == "sine":
        return np.sin(ph)
    if kind == "square":
        return np.sign(np.sin(ph)) * 0.6
    if kind == "saw":
        # band-limited-ish saw: sum of a few harmonics
        out = np.zeros(n)
        for h in range(1, 9):
            if freq * h > sr / 2.2:
                break
            out += np.sin(ph * h) / h
        return out * 0.6
    if kind == "organ":
        return (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph) + 0.15 * np.sin(4 * ph)) * 0.55
    if kind == "bell":
        return (np.sin(ph) + 0.5 * np.sin(2.76 * ph) + 0.25 * np.sin(5.4 * ph)) * 0.6
    if kind == "pluck":
        out = np.zeros(n)
        for h in range(1, 7):
            out += np.sin(ph * h) / (h ** 1.2) * np.exp(-t * (3 + h * 2))
        return out
    return np.sin(ph)


TIMBRE = {
    # kind: (osc, attack, decay, sustain, release, detune cents for chorus voices)
    "sine": ("sine", 0.01, 0.1, 0.8, 0.12, 0),
    "saw": ("saw", 0.01, 0.12, 0.7, 0.1, 6),
    "square": ("square", 0.005, 0.1, 0.6, 0.08, 4),
    "pluck": ("pluck", 0.002, 0.2, 0.3, 0.1, 0),
    "bell": ("bell", 0.002, 0.4, 0.2, 0.3, 0),
    "organ": ("organ", 0.01, 0.05, 0.9, 0.08, 0),
    "pad": ("saw", 0.25, 0.3, 0.7, 0.4, 9),
    "bass": ("saw", 0.005, 0.1, 0.8, 0.06, 0),
}


def _voice(timbre: str, pitch: int, dur_s: float, vel: int, sr: int = SR) -> np.ndarray:
    osc, a, d, s, r, detune = TIMBRE.get(timbre, TIMBRE["sine"])
    n = int((dur_s + r) * sr)
    f = midi_to_hz(pitch)
    sig = _osc(osc, f, n, sr)
    if detune:
        sig = sig + _osc(osc, f * 2 ** (detune / 1200), n, sr) + _osc(osc, f * 2 ** (-detune / 1200), n, sr)
        sig /= 2.2
    if timbre == "bass":
        sig = sig * 0.7 + 0.5 * np.sin(2 * np.pi * f * np.arange(n) / sr)  # add sub
        sig = _lowpass(sig, 900, sr)
    env = _env(n, a, d, s, r, sr)
    return (sig * env * (vel / 127.0) ** 1.3).astype(np.float32)


def _lowpass(x: np.ndarray, cutoff: float, sr: int = SR) -> np.ndarray:
    """One-pole lowpass (cheap, vectorised via cumulative recursion in chunks)."""
    alpha = 1 - np.exp(-2 * np.pi * cutoff / sr)
    # IIR via lfilter-free approach: exponential moving average using cumsum trick
    n = len(x)
    if n == 0:
        return x
    out = np.empty_like(x)
    acc = 0.0
    # Python loop is too slow for long signals, but notes are short (<~2 s).
    for i in range(n):
        acc += alpha * (x[i] - acc)
        out[i] = acc
    return out


def _noise_hit(dur: float, hp: float, decay: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0.0)  # crude highpass
    if hp > 1:
        x = np.diff(x, prepend=0.0)
    return (x * np.exp(-np.arange(n) / sr * decay)).astype(np.float32)


def _drum(pitch: int, vel: int, sr: int = SR) -> np.ndarray:
    v = (vel / 127.0) ** 1.2
    if pitch == KICK:
        n = int(0.35 * sr)
        t = np.arange(n) / sr
        f = 45 + 110 * np.exp(-t * 28)
        ph = 2 * np.pi * np.cumsum(f) / sr
        return (np.sin(ph) * np.exp(-t * 11) * v * 1.1).astype(np.float32)
    if pitch == SNARE:
        n = int(0.22 * sr)
        t = np.arange(n) / sr
        body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 28) * 0.5
        noise = _noise_hit(0.22, 1, 22, sr, 1)[:n]
        return ((body + noise * 0.7) * v).astype(np.float32)
    if pitch == CLAP:
        out = np.zeros(int(0.25 * sr), dtype=np.float32)
        for k, off in enumerate((0, 0.012, 0.025)):
            h = _noise_hit(0.12, 1, 30, sr, 10 + k)
            o = int(off * sr)
            out[o:o + len(h)] += h[: len(out) - o] * 0.6
        return out * v
    if pitch in (HAT, OPEN_HAT, RIDE):
        dur, dec = (0.06, 70) if pitch == HAT else (0.3, 14)
        return _noise_hit(dur, 2, dec, sr, 3) * v * 0.45
    return _noise_hit(0.1, 1, 40, sr, 4) * v * 0.3


def render_track(track: Track, bpm: float, timbre: str, total_s: float, sr: int = SR) -> np.ndarray:
    """Render one track to a mono float32 buffer of total_s seconds."""
    spb = 60.0 / bpm
    buf = np.zeros(int((total_s + 2.0) * sr), dtype=np.float32)
    cache: dict[tuple, np.ndarray] = {}
    for n in track.notes:
        start = int(n.start * spb * sr)
        if start >= len(buf):
            continue
        if track.is_drums:
            key = (n.pitch, n.velocity)
            snd = cache.get(key)
            if snd is None:
                snd = cache[key] = _drum(n.pitch, n.velocity, sr)
        else:
            dur = n.duration * spb
            key = (n.pitch, round(dur, 3), n.velocity // 4)
            snd = cache.get(key)
            if snd is None:
                snd = cache[key] = _voice(timbre, n.pitch, dur, n.velocity, sr)
        end = min(len(buf), start + len(snd))
        buf[start:end] += snd[: end - start]
    return buf[: int(total_s * sr)]


def render_stereo(song, genre, sr: int = SR) -> dict[str, np.ndarray]:
    """Render all tracks. Returns {track name: stereo (n, 2) array} already panned."""
    timbres = {"Bass": genre.bass_timbre, "Chords": genre.chord_timbre, "Lead": genre.lead_timbre}
    pans = {"Drums": 0.0, "Bass": 0.0, "Chords": -0.25, "Lead": 0.25}
    gains = {"Drums": 0.9, "Bass": 0.8, "Chords": 0.45, "Lead": 0.4}
    stems = {}
    total = song.seconds + 1.5
    for tr in song.tracks:
        mono = render_track(tr, song.bpm, timbres.get(tr.name, "sine"), total, sr)
        mono = mono * gains.get(tr.name, 0.5)
        p = pans.get(tr.name, 0.0)
        left = np.sqrt(0.5 * (1 - p))
        right = np.sqrt(0.5 * (1 + p))
        stems[tr.name] = np.stack([mono * left * np.sqrt(2), mono * right * np.sqrt(2)], axis=1)
    return stems

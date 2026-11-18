"""Numpy synthesizer: renders Track notes to stereo float32 audio (no samplerate deps)."""

from __future__ import annotations

import zlib

import numpy as np

from .composer import CHING, CLAP, CONGA_HI, CONGA_LO, HAT, KICK, OPEN_HAT, RIDE, SHAKER, SNARE
from . import instruments as _I
from .fx import BODIES, compress, fft_convolve, highpass, make_reverb_ir, presence
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
    if kind == "phin":
        # Isan 3-string lute: bright nasal pluck, slight pitch scoop, buzzy upper partials
        scoop = 1.0 + 0.012 * np.exp(-t * 70)
        php = 2 * np.pi * freq * np.cumsum(scoop) / sr
        out = np.zeros(n)
        amps = [1.0, 0.8, 0.9, 0.6, 0.55, 0.35, 0.3, 0.2, 0.12, 0.1]
        for h, a in enumerate(amps, start=1):
            if freq * h > sr / 2.2:
                break
            out += a * np.sin(php * h) * np.exp(-t * (4 + h * 3.2))
        return out * 0.5
    if kind == "khaen":
        # free-reed mouth organ: rich harmonics + a little breath
        amps = [1.0, 0.85, 0.7, 0.5, 0.45, 0.3, 0.25, 0.2]
        out = np.zeros(n)
        for h, a in enumerate(amps, start=1):
            if freq * h > sr / 2.2:
                break
            out += a * np.sin(ph * h + 0.3 * h)
        rng = np.random.default_rng(int(freq))
        breath = np.convolve(rng.standard_normal(n), np.ones(6) / 6, mode="same")
        return out * 0.22 + breath * 0.02
    if kind == "wot":
        # Isan panpipe/flute: breathy sine with vibrato
        vib = 0.6 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1.0, t * 3)
        p2 = ph + vib
        rng = np.random.default_rng(int(freq) + 7)
        breath = np.convolve(rng.standard_normal(n), np.ones(10) / 10, mode="same")
        return (np.sin(p2) + 0.3 * np.sin(2 * p2) + 0.08 * np.sin(3 * p2)) * 0.7 + breath * 0.06
    if kind == "piano":
        out = np.zeros(n)
        for h in range(1, 8):
            fh = freq * h * np.sqrt(1 + 0.0004 * h * h)
            if fh > sr / 2.2:
                break
            out += np.sin(2 * np.pi * fh * t) / (h ** 1.1) * np.exp(-t * (1.2 + h * 1.1))
        click = np.exp(-t * 400) * np.sin(2 * np.pi * freq * 3 * t) * 0.15
        return (out + click) * 0.9
    if kind == "guitar":
        out = np.zeros(n)
        for h in range(1, 9):
            if freq * h > sr / 2.2:
                break
            out += np.sin(ph * h) / (h ** 1.3) * np.exp(-t * (2.2 + h * 1.6))
        return out * 0.9
    if kind == "pong":
        # Pong lang wooden xylophone
        return (np.sin(ph) * np.exp(-t * 12) + 0.4 * np.sin(3.9 * ph) * np.exp(-t * 40)) * 0.9
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
    # Isan indie instruments
    "phin": ("phin", 0.002, 0.25, 0.3, 0.08, 0),
    "khaen": ("khaen", 0.06, 0.1, 0.9, 0.12, 5),
    "wot": ("wot", 0.04, 0.1, 0.85, 0.12, 0),
    "piano": ("piano", 0.003, 0.3, 0.4, 0.3, 0),
    "guitar": ("guitar", 0.002, 0.3, 0.35, 0.12, 0),
    "pong": ("pong", 0.001, 0.2, 0.15, 0.05, 0),
}


def _voice(timbre: str, pitch: int, dur_s: float, vel: int, sr: int = SR, variant: int = 0) -> np.ndarray:
    if timbre in _I.VOICES:
        return _I.VOICES[timbre](pitch, dur_s, vel, variant)
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


_DRUM_CONSTS = dict(KICK=KICK, SNARE=SNARE, CLAP=CLAP, HAT=HAT, OPEN_HAT=OPEN_HAT, RIDE=RIDE,
                    CONGA_HI=CONGA_HI, CONGA_LO=CONGA_LO, SHAKER=SHAKER, CHING=CHING)


def _drum(pitch: int, vel: int, sr: int = SR, variant: int = 0) -> np.ndarray:
    return _I.drum(pitch, vel, variant, _DRUM_CONSTS, sr)


def _drum_basic(pitch: int, vel: int, sr: int = SR) -> np.ndarray:
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
    if pitch in (CONGA_HI, CONGA_LO):
        f0 = 330 if pitch == CONGA_HI else 220
        n = int(0.22 * sr)
        t = np.arange(n) / sr
        f = f0 * (1 + 0.25 * np.exp(-t * 40))
        ph = 2 * np.pi * np.cumsum(f) / sr
        return (np.sin(ph) * np.exp(-t * 16) * v * 0.8).astype(np.float32)
    if pitch == SHAKER:
        return _noise_hit(0.07, 2, 55, sr, 5) * v * 0.3
    if pitch == CHING:
        # Thai finger cymbals: short metallic ring
        n = int(0.16 * sr)
        t = np.arange(n) / sr
        x = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((3150, 1.0), (4300, 0.6), (5750, 0.4), (7300, 0.25)))
        return (x * np.exp(-t * 28) * v * 0.18).astype(np.float32)
    if pitch in (HAT, OPEN_HAT, RIDE):
        dur, dec = (0.06, 70) if pitch == HAT else (0.3, 14)
        return _noise_hit(dur, 2, dec, sr, 3) * v * 0.45
    return _noise_hit(0.1, 1, 40, sr, 4) * v * 0.3


# General-MIDI programs used when a SoundFont is loaded (drums use bank 128, note numbers already match GM)
GM_PROGRAM = {"Phin": 105, "Khaen": 21, "Wot": 75, "Guitar": 25, "Piano": 0, "Synth": 89, "Pong": 13,
              "Bass": 33, "Chords": 0, "Lead": 80}
SF_GAIN = {"Phin": 1.0, "Khaen": 0.9, "Wot": 0.9, "Guitar": 1.0, "Piano": 1.0, "Synth": 0.8, "Pong": 0.9,
           "Bass": 1.0, "Chords": 0.9, "Lead": 0.9, "Drums": 1.0, "Perc": 1.0}


def render_track(track: Track, bpm: float, timbre: str, total_s: float, sr: int = SR,
                 humanize: bool = True, sf=None) -> np.ndarray:
    """Render one track to a mono float32 buffer. With `humanize`, every note gets its own small timing and
    velocity deviation (notes struck together share one, so chords stay tight) and repeated notes use
    round-robin variants, so nothing sounds copy-pasted."""
    spb = 60.0 / bpm
    buf = np.zeros(int((total_s + 2.0) * sr), dtype=np.float32)
    cache: dict[tuple, np.ndarray] = {}
    seed = zlib.crc32(track.name.encode())
    sigma = 0.0035 if (track.is_drums or track.name == "Bass") else 0.008
    for idx, n in enumerate(track.notes):
        jit, vs = 0.0, 1.0
        if humanize:
            rr = np.random.default_rng((seed * 1000003 + int(round(n.start * 480))) % (2 ** 32))
            jit = float(rr.normal(0, sigma)) + (0.0 if track.is_drums else 0.004)   # melodic parts sit slightly behind
            vs = 1.0 + float(rr.normal(0, 0.05))
        start = max(0, int((n.start * spb + jit) * sr))
        if start >= len(buf):
            continue
        vel = int(np.clip(n.velocity * vs, 1, 127))
        variant = (n.pitch + idx) % 3
        if track.is_drums:
            key = (n.pitch, vel // 3, variant)
            snd = cache.get(key)
            if snd is None:
                snd = sf.render(0, n.pitch, 0.1, vel, sr, drums=True) if sf is not None else None
                if snd is None:
                    snd = _drum(n.pitch, vel, sr, variant)
                elif sf is not None:
                    snd = snd * SF_GAIN["Drums"]
                cache[key] = snd
        else:
            dur = n.duration * spb
            key = (n.pitch, round(dur, 2), vel // 4, variant)
            snd = cache.get(key)
            if snd is None:
                prog = GM_PROGRAM.get(track.name)
                if sf is not None and prog is not None and sf.has(prog):
                    snd = sf.render(prog, n.pitch, dur, vel, sr)
                    if snd is not None:
                        snd = snd * SF_GAIN.get(track.name, 1.0)
                if snd is None:
                    snd = _voice(timbre, n.pitch, dur, vel, sr, variant)
                cache[key] = snd
        end = min(len(buf), start + len(snd))
        buf[start:end] += snd[: end - start]
    return buf[: int(total_s * sr)]


TRACK_TIMBRE = {"Phin": "phin", "Khaen": "khaen", "Wot": "wot", "Guitar": "guitar",
                "Piano": "piano", "Synth": "pad", "Pong": "pong"}
PANS = {"Drums": 0.0, "Perc": 0.15, "Bass": 0.0, "Chords": -0.25, "Lead": 0.25,
        "Phin": -0.3, "Khaen": 0.3, "Wot": 0.2, "Guitar": -0.4, "Piano": 0.1, "Synth": 0.0, "Pong": 0.35}
GAINS = {"Drums": 0.29, "Perc": 0.55, "Bass": 0.36, "Chords": 0.22, "Lead": 0.40,
         "Phin": 0.55, "Khaen": 0.4, "Wot": 0.45, "Guitar": 0.5, "Piano": 0.5, "Synth": 0.3, "Pong": 0.4}


# Isan-indie balance: phin / khaen / pong up front, bass and drums tucked back
ISAN_GAINS = {"Drums": 0.32, "Perc": 0.57, "Bass": 0.33, "Phin": 1.3, "Khaen": 0.31, "Wot": 0.29,
              "Guitar": 0.83, "Piano": 1.7, "Synth": 0.31, "Pong": 0.72}


def render_stereo(song, genre, sr: int = SR, sf=None) -> dict[str, np.ndarray]:
    """Render all tracks. Returns {track name: stereo (n, 2) array} already panned."""
    timbres = {"Bass": genre.bass_timbre, "Chords": genre.chord_timbre, "Lead": genre.lead_timbre, **TRACK_TIMBRE}
    gains = {**GAINS, **ISAN_GAINS} if genre.extra.get("isan") else GAINS
    stems = {}
    total = song.seconds + 1.5
    for tr in song.tracks:
        mono = render_track(tr, song.bpm, timbres.get(tr.name, "sine"), total, sr, sf=sf)
        mono = mono * gains.get(tr.name, 0.5)
        p = PANS.get(tr.name, 0.0)
        left = np.sqrt(0.5 * (1 - p))
        right = np.sqrt(0.5 * (1 + p))
        stems[tr.name] = np.stack([mono * float(left * np.sqrt(2)), mono * float(right * np.sqrt(2))], axis=1).astype(np.float32)
    return stems


# How much of each instrument is sent to the room reverb
SENDS = {"Drums": 0.10, "Perc": 0.10, "Bass": 0.0, "Chords": 0.2, "Lead": 0.25, "Phin": 0.22,
         "Khaen": 0.28, "Wot": 0.38, "Guitar": 0.2, "Piano": 0.3, "Synth": 0.25, "Pong": 0.3}


def master_bus(stems: dict[str, np.ndarray], genre, sr: int = SR) -> np.ndarray:
    """Instrument bodies -> sum -> room reverb -> high-pass -> glue compression. Returns stereo float32."""
    n = max(len(x) for x in stems.values())
    dry = np.zeros((n, 2), dtype=np.float32)
    send = np.zeros(n, dtype=np.float32)
    for name, st in stems.items():
        st = np.pad(st, ((0, n - len(st)), (0, 0)))
        modes = BODIES.get(name)
        if modes:
            ir = _body_ir_cached(name)
            st = np.stack([fft_convolve(st[:, 0], ir), fft_convolve(st[:, 1], ir)], axis=1).astype(np.float32)
        dry += st
        send += (st.mean(axis=1) * SENDS.get(name, 0.15)).astype(np.float32)
    rt60 = 1.5 if genre.extra.get("isan") or genre.name in ("ambient", "classical") else 1.0
    ir = make_reverb_ir(rt60=rt60, sr=sr)
    wet = np.stack([fft_convolve(send, ir[:, 0]), fft_convolve(send, ir[:, 1])], axis=1).astype(np.float32)
    del send
    out = highpass(dry + wet, sr)
    out = presence(out, sr)
    out = compress(out, sr)
    return out.astype(np.float32)


_BODY_CACHE: dict[str, np.ndarray] = {}


def _body_ir_cached(name: str) -> np.ndarray:
    from .fx import body_ir
    if name not in _BODY_CACHE:
        _BODY_CACHE[name] = body_ir(BODIES[name])
    return _BODY_CACHE[name]

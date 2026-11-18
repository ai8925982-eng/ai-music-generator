"""Physically-inspired instrument voices (numpy only, no samples).

String instruments use Karplus-Strong waveguides, the piano uses detuned-string modal synthesis,
khaen / wot model reed and air-jet behaviour (breath, pitch scoop, vibrato), drums have layered
transients.  Every voice takes a `variant` so repeated notes are never identical (round-robin).
"""

from __future__ import annotations

import zlib

import numpy as np

SR = 44100


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _lp(x: np.ndarray, a: float) -> np.ndarray:
    """One-pole lowpass with coefficient a in (0,1]; vectorised by repeated smoothing (cheap for short bursts)."""
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        out[i] = acc
    return out


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same") if k > 1 else x


def _adsr_tail(n: int, release_s: float, sr: int = SR) -> np.ndarray:
    """Gain curve: 1 until the release stage, then a smooth fade to 0 at sample n."""
    r = max(8, min(n // 2, int(release_s * sr)))
    g = np.ones(n)
    g[n - r:] = np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    return g


def _warp_pitch(x: np.ndarray, start_ratio: float, tau: float, sr: int = SR) -> np.ndarray:
    """Time-varying pitch: starts at `start_ratio` x the pitch and relaxes to 1 with time constant tau (s)."""
    n = len(x)
    t = np.arange(n) / sr
    ratio = 1.0 + (start_ratio - 1.0) * np.exp(-t / tau)
    pos = np.cumsum(ratio) - ratio[0]
    return np.interp(pos, np.arange(n), x)


# ---------------------------------------------------------------------------
# Karplus-Strong string
# ---------------------------------------------------------------------------

def ks_string(freq: float, dur_s: float, t60: float, pluck_pos: float = 0.18, bright: float = 0.6,
              seed=0, sr: int = SR) -> np.ndarray:
    """Plucked string: noise burst fed through a damped delay line (block-vectorised).

    freq     fundamental (Hz)       t60    seconds for the sound to fall 60 dB
    pluck_pos  0..0.5 position along the string (small = bright / nasal)
    bright   0..1 excitation brightness
    """
    n = int(dur_s * sr)
    N = sr / freq
    Ni = int(np.floor(N))                    # two taps (Ni, Ni+1) blended by b -> average delay Ni + b = N
    b = float(np.clip(N - Ni, 0.0, 1.0))
    if Ni < 4:
        Ni = 4
    g = 10 ** (-3.0 / max(t60 * freq, 1.0))  # per-loop gain for the requested T60
    rng = _rng("ks", seed)
    exc = rng.uniform(-1, 1, Ni + 2)
    a_lp = 0.15 + 0.8 * bright               # brightness of the pluck
    exc = _lp(exc, a_lp)
    k = max(1, int(pluck_pos * Ni))          # pluck-position comb: removes harmonics that have a node there
    exc = exc - np.concatenate([np.zeros(k), exc[:-k]])
    exc /= max(np.abs(exc).max(), 1e-9)
    y = np.zeros(n + 2 * Ni + 4)
    y[:Ni + 1] = exc[:Ni + 1]
    i = Ni + 1
    while i < n + Ni:
        j = min(i + Ni, len(y))
        seg = np.arange(i, j)
        y[seg] = g * ((1 - b) * y[seg - Ni] + b * y[seg - Ni - 1])
        i = j
    y = y[Ni:Ni + n]
    return y - y.mean()


def _vel_gain(vel: int) -> float:
    return (vel / 127.0) ** 1.2


def guitar(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    t60 = float(np.clip(5.0 * (110.0 / f) ** 0.5, 0.9, 4.5))
    ring = dur_s + 0.35
    y = ks_string(f, ring, t60, pluck_pos=0.2 + 0.03 * (variant % 3), bright=0.35 + 0.5 * vel / 127.0,
                  seed=("g", pitch, variant), sr=sr)
    # warm "wood" body: slight low-mid emphasis via a short smoothed copy
    y = y + 0.5 * _smooth(y, 6)
    n = len(y)
    t = np.arange(n) / sr
    click = _rng("gc", pitch, variant).standard_normal(int(0.004 * sr)) * 0.08
    y[:len(click)] += click
    y *= _adsr_tail(n, 0.12, sr)
    return (y * _vel_gain(vel) * 0.8).astype(np.float32)


def phin(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    """Isan 3-string lute: bright, nasal, twangy; strike near the bridge, pitch scoop after the pluck."""
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    t60 = float(np.clip(2.2 * (220.0 / f) ** 0.4, 0.5, 2.2))
    ring = dur_s + 0.25
    y = ks_string(f, ring, t60, pluck_pos=0.09 + 0.015 * (variant % 3), bright=0.75 + 0.2 * vel / 127.0,
                  seed=("p", pitch, variant), sr=sr)
    y = _warp_pitch(y, 1.008, 0.05, sr)
    n = len(y)
    t = np.arange(n) / sr
    # nasal formant-ish emphasis (resonant body around 1.1 kHz) + fret buzz transient
    nas = _smooth(y, 3) - _smooth(y, 18)
    y = y * 0.7 + nas * 0.9
    buzz = _rng("pb", pitch, variant).standard_normal(int(0.012 * sr)) * np.exp(-np.arange(int(0.012 * sr)) / (0.003 * sr)) * 0.12
    y[:len(buzz)] += buzz
    y *= _adsr_tail(n, 0.07, sr)
    return (np.tanh(y * 1.4) * _vel_gain(vel) * 0.7).astype(np.float32)


def bass(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    """Fingered electric bass: KS string, mellow pickup tone plus a little sub and drive."""
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    ring = dur_s + 0.1
    y = ks_string(f, ring, 2.2, pluck_pos=0.28, bright=0.18 + 0.2 * vel / 127.0, seed=("b", pitch, variant), sr=sr)
    y = _smooth(y, 3)
    t = np.arange(len(y)) / sr
    y = y + 0.35 * np.sin(2 * np.pi * f * t) * np.exp(-t / 1.2)
    y = np.tanh(y * 1.6)
    y *= _adsr_tail(len(y), 0.09, sr)
    return (y * _vel_gain(vel) * 0.9).astype(np.float32)


# ---------------------------------------------------------------------------
# Piano: hammer-excited partials, 3 slightly detuned strings per note
# ---------------------------------------------------------------------------

def piano(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    v = vel / 127.0
    ring = dur_s + 0.4
    n = int(ring * sr)
    t = np.arange(n) / sr
    rng = _rng("pn", pitch, variant)
    B = 0.00012 + 0.0008 * (max(pitch - 40, 0) / 60.0) ** 2      # inharmonicity grows with pitch
    tau_slow = 7.0 * (220.0 / f) ** 0.55
    out = np.zeros(n)
    n_part = int(min(14, (sr / 2.3) // f))
    for h in range(1, n_part + 1):
        fh = f * h * np.sqrt(1 + B * h * h)
        hammer = abs(np.sin(np.pi * h * 0.125)) + 0.12                      # strike point at 1/8 of the string
        amp = hammer / (h ** 0.9) * np.exp(-h * (1.15 - 0.85 * v))        # harder strike = brighter
        decay_fast = np.exp(-t / (0.35 / (1 + 0.15 * h) * tau_slow ** 0.5))
        decay_slow = np.exp(-t / (tau_slow / (1 + 0.45 * h)))
        env = 0.55 * decay_fast + 0.45 * decay_slow
        for s in range(3):                                                   # 3 unison strings, tiny detune -> beating
            cents = (s - 1) * rng.uniform(0.5, 1.6)
            ph0 = rng.uniform(0, 2 * np.pi)
            out += amp / 3.0 * env * np.sin(2 * np.pi * fh * 2 ** (cents / 1200) * t + ph0)
    thump = _smooth(rng.standard_normal(int(0.02 * sr)), 14) * np.exp(-np.arange(int(0.02 * sr)) / (0.006 * sr))
    out[:len(thump)] += thump * (0.25 + 0.5 * v)
    out *= np.minimum(1.0, t / 0.002)
    out *= _adsr_tail(n, 0.22, sr)
    return (out * (v ** 1.1) * 0.55).astype(np.float32)


# ---------------------------------------------------------------------------
# Khaen (free reed) and wot (air-jet pipe)
# ---------------------------------------------------------------------------

def _drift(n: int, depth_cents: float, rate_hz: float, rng, sr: int = SR) -> np.ndarray:
    """Slowly wandering pitch (cents) -> frequency ratio curve."""
    k = max(2, int(sr / rate_hz))
    pts = rng.standard_normal(n // k + 3)
    curve = np.interp(np.arange(n), np.arange(len(pts)) * k, pts)
    return 2 ** (curve * depth_cents / 1200)


def khaen(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    """Free-reed mouth organ: buzzy reed spectrum, pipe resonance, breath, slight pitch drift, 'chiff' on entry."""
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    ring = dur_s + 0.12
    n = int(ring * sr)
    t = np.arange(n) / sr
    rng = _rng("kh", pitch, variant)
    ratio = _drift(n, 2.5, 1.7, rng, sr)
    ratio *= 1 - 0.008 * np.exp(-t / 0.03)                      # reed starts a hair flat
    ph = 2 * np.pi * np.cumsum(f * ratio) / sr
    amps = [1.0, 0.9, 0.75, 0.55, 0.5, 0.38, 0.3, 0.24, 0.17, 0.12]
    y = np.zeros(n)
    for h, a in enumerate(amps, start=1):
        if f * h > sr / 2.3:
            break
        y += a * np.sin(h * ph + 0.35 * h) * (1.0 + 0.04 * np.sin(2 * np.pi * (0.7 + 0.13 * h) * t + h))
    # pipe resonance: gentle bump in the upper mids
    y = y + 0.6 * (_smooth(y, 5) - _smooth(y, 30))
    air = _smooth(rng.standard_normal(n), 4) - _smooth(rng.standard_normal(n), 40)
    breath = 0.05 + 0.25 * np.exp(-t / 0.05)
    y = y * 0.16 + air * 0.05 * breath
    env = np.minimum(1.0, t / 0.045) * (0.92 + 0.08 * np.sin(2 * np.pi * 4.6 * t))
    y *= env * _adsr_tail(n, 0.09, sr)
    return (y * (0.35 + 0.65 * vel / 127.0)).astype(np.float32)


def wot(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    """Blown tube: soft sine core, airy band around the pitch, scoop up into the note, delayed vibrato."""
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    ring = dur_s + 0.15
    n = int(ring * sr)
    t = np.arange(n) / sr
    rng = _rng("wo", pitch, variant)
    vib_depth = 0.006 * np.clip((t - 0.18) / 0.3, 0, 1)             # vibrato fades in after the attack
    ratio = (1 + vib_depth * np.sin(2 * np.pi * (5.1 + 0.4 * rng.random()) * t)) * _drift(n, 3.0, 2.3, rng, sr)
    ratio *= 2 ** (-35 * np.exp(-t / 0.045) / 1200)                 # scoop from ~35 cents flat
    ph = 2 * np.pi * np.cumsum(f * ratio) / sr
    y = np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    noise = _smooth(rng.standard_normal(n), 6)                      # band-limited air, centred on the pitch
    air = noise * np.cos(ph) * 2.0
    swell = 0.18 + 0.7 * np.exp(-t / 0.06)
    y = y * 0.55 + air * 0.11 * swell
    env = np.minimum(1.0, t / 0.05) * (0.95 + 0.05 * np.sin(2 * np.pi * 3.1 * t))
    y *= env * _adsr_tail(n, 0.11, sr)
    return (y * (0.3 + 0.7 * vel / 127.0) ** 1.1).astype(np.float32)


def pong(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    """Pong lang (hanging wooden xylophone): hollow bar, 1 : 3 : 6 partials, mallet click."""
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    n = int((0.5 + 0.2 * (220.0 / f)) * sr)
    t = np.arange(n) / sr
    rng = _rng("po", pitch, variant)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.22)
         + 0.45 * np.sin(2 * np.pi * f * 3.0 * 1.003 * t + 1.0) * np.exp(-t / 0.07)
         + 0.2 * np.sin(2 * np.pi * f * 6.1 * t + 2.0) * np.exp(-t / 0.03))
    click = _smooth(rng.standard_normal(int(0.006 * sr)), 3) * np.exp(-np.arange(int(0.006 * sr)) / (0.0015 * sr))
    y[:len(click)] += click * 0.5
    y *= _adsr_tail(n, 0.05, sr)
    return (y * _vel_gain(vel) * 0.7).astype(np.float32)


def pad(pitch: int, dur_s: float, vel: int, variant: int, sr: int = SR) -> np.ndarray:
    """Soft analogue-style pad: 4 detuned voices, few harmonics, slow swell and slow shimmer."""
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    ring = dur_s + 0.6
    n = int(ring * sr)
    t = np.arange(n) / sr
    rng = _rng("pd", pitch, variant)
    y = np.zeros(n)
    for cents in (-9, -3, 4, 10):
        fv = f * 2 ** (cents / 1200)
        ph0 = rng.uniform(0, 6.28)
        for h in range(1, 7):
            if fv * h > sr / 2.3:
                break
            y += np.sin(2 * np.pi * fv * h * t + ph0 * h) / (h ** 1.7)
    y *= 0.2 * (0.85 + 0.15 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6)))
    env = np.minimum(1.0, t / 0.45)
    y *= env * _adsr_tail(n, 0.55, sr)
    return (y * _vel_gain(vel)).astype(np.float32)


# ---------------------------------------------------------------------------
# Drums and percussion (variant = round-robin index)
# ---------------------------------------------------------------------------

def _noise(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n)


def _hp(x: np.ndarray, k: int = 8) -> np.ndarray:
    return x - _smooth(x, k)


def drum(pitch: int, vel: int, variant: int, consts: dict, sr: int = SR) -> np.ndarray:
    KICK, SNARE, CLAP = consts["KICK"], consts["SNARE"], consts["CLAP"]
    HAT, OPEN_HAT, RIDE = consts["HAT"], consts["OPEN_HAT"], consts["RIDE"]
    CONGA_HI, CONGA_LO, SHAKER, CHING = consts["CONGA_HI"], consts["CONGA_LO"], consts["SHAKER"], consts["CHING"]
    v = (vel / 127.0) ** 1.15
    rng = _rng("dr", pitch, variant)
    jit = 1 + rng.uniform(-0.025, 0.025)

    if pitch == KICK:
        n = int(0.42 * sr)
        t = np.arange(n) / sr
        f = (46 + 120 * np.exp(-t / 0.028)) * jit
        body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.16)
        sub = np.sin(2 * np.pi * 44 * jit * t) * np.exp(-t / 0.25) * 0.35
        click = _hp(_noise(n, rng)[:int(0.01 * sr)], 6) * np.exp(-np.arange(int(0.01 * sr)) / (0.0018 * sr)) * (0.2 + 0.4 * v)
        y = body + sub
        y[:len(click)] += click
        return (np.tanh(y * (1.2 + v)) * v * 0.95).astype(np.float32)
    if pitch == SNARE:
        n = int(0.28 * sr)
        t = np.arange(n) / sr
        tone = (np.sin(2 * np.pi * 185 * jit * t) * np.exp(-t / 0.07) + 0.5 * np.sin(2 * np.pi * 330 * jit * t) * np.exp(-t / 0.045))
        wires = _hp(_noise(n, rng), 5) * np.exp(-t / (0.09 + 0.05 * v))
        stick = _hp(_noise(n, rng)[:int(0.004 * sr)], 3) * 0.8
        y = tone * 0.55 + wires * 0.7
        y[:len(stick)] += stick
        return (np.tanh(y * 1.4) * v * 0.9).astype(np.float32)
    if pitch == CLAP:
        out = np.zeros(int(0.3 * sr))
        for k, off in enumerate((0, 0.011, 0.023, 0.036)):
            m = int(0.16 * sr)
            h = _hp(_noise(m, rng), 6) * np.exp(-np.arange(m) / sr / (0.012 if k < 3 else 0.07))
            o = int(off * sr)
            out[o:o + m] += h[:len(out) - o] * (0.5 if k < 3 else 0.8)
        return (out * v * 0.6).astype(np.float32)
    if pitch in (CONGA_HI, CONGA_LO):
        f0 = (340 if pitch == CONGA_HI else 225) * jit
        n = int(0.3 * sr)
        t = np.arange(n) / sr
        f = f0 * (1 + 0.18 * np.exp(-t / 0.02))
        tone = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.09) + 0.3 * np.sin(2 * np.pi * f0 * 2.3 * t) * np.exp(-t / 0.03)
        slap = _hp(_noise(int(0.012 * sr), rng), 4) * np.exp(-np.arange(int(0.012 * sr)) / (0.003 * sr)) * 0.5
        tone[:len(slap)] += slap
        return (tone * v * 0.8).astype(np.float32)
    if pitch == SHAKER:
        n = int(0.09 * sr)
        t = np.arange(n) / sr
        y = _hp(_noise(n, rng), 5) * np.minimum(1.0, t / 0.012) * np.exp(-t / 0.028)
        return (y * v * 0.45).astype(np.float32)
    if pitch == CHING:
        n = int(0.26 * sr)
        t = np.arange(n) / sr
        parts = ((3150, 1.0, 0.07), (4300, 0.65, 0.05), (5750, 0.45, 0.035), (7300, 0.3, 0.025), (2380, 0.5, 0.09))
        y = sum(a * np.sin(2 * np.pi * f * jit * t + k) * np.exp(-t / d) for k, (f, a, d) in enumerate(parts))
        y = y + _hp(_noise(n, rng), 4) * np.exp(-t / 0.006) * 0.4
        return (y * v * 0.2).astype(np.float32)
    if pitch in (HAT, OPEN_HAT, RIDE):
        n = int((0.07 if pitch == HAT else 0.5) * sr)
        t = np.arange(n) / sr
        ratios = (2.0, 3.0, 4.16, 5.43, 6.79, 8.21)
        y = sum(np.sign(np.sin(2 * np.pi * 205.3 * r * jit * t)) for r in ratios) / 6.0
        y = _hp(y, 6) * 0.6 + _hp(_noise(n, rng), 5) * 0.5
        y *= np.exp(-t / (0.022 if pitch == HAT else 0.18)) * np.minimum(1.0, t / 0.0008)
        return (y * v * 0.45).astype(np.float32)
    return (_hp(_noise(int(0.1 * sr), rng), 4) * v * 0.2).astype(np.float32)


# timbre name -> pitched voice function(pitch, dur_s, vel, variant)
VOICES = {"guitar": guitar, "phin": phin, "bass": bass, "piano": piano,
          "khaen": khaen, "wot": wot, "pong": pong, "pad": pad}

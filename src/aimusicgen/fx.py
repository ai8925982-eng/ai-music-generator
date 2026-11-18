"""Mix-bus effects for the accompaniment: body resonance, room reverb, glue compressor.
(The vocal is never routed through any of this.)"""

from __future__ import annotations

import numpy as np

SR = 44100


def fft_convolve(x: np.ndarray, ir: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """Overlap-add FFT convolution of a long signal with a short impulse response (same length as x)."""
    n, m = len(x), len(ir)
    nfft = 1 << int(np.ceil(np.log2(block + m)))
    H = np.fft.rfft(ir, nfft)
    out = np.zeros(n + m)
    for s in range(0, n, block):
        seg = x[s:s + block]
        y = np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)
        out[s:s + len(seg) + m - 1] += y[:len(seg) + m - 1]
    return out[:n]


def room_ir(rt60: float = 1.3, predelay: float = 0.018, seed: int = 7, sr: int = SR) -> np.ndarray:
    """Stereo room impulse response (2, n): early reflections + diffuse tail that darkens as it decays."""
    rng = np.random.default_rng(seed)
    n = int((rt60 * 1.1 + predelay) * sr)
    t = np.arange(n) / sr
    ir = np.zeros((2, n))
    for ch in range(2):
        noise = rng.standard_normal(n)
        low = np.convolve(noise, np.ones(10) / 10, mode="same")
        high = noise - low
        tail = low * np.exp(-6.9 * t / rt60) + 0.5 * high * np.exp(-6.9 * t / (0.45 * rt60))
        tail[: int(predelay * sr)] = 0
        taps = [0.007, 0.011, 0.017, 0.023, 0.031, 0.043, 0.057]
        for k, tp in enumerate(taps):
            idx = int((predelay + tp * (1.0 + 0.12 * ch)) * sr)
            if idx < n:
                tail[idx] += (0.9 ** k) * (1 if (k + ch) % 2 == 0 else -1) * 0.9
        ir[ch] = tail
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True)).clip(1e-9)
    return ir.astype(np.float32)


def body_ir(modes: list[tuple[float, float, float]], length: float = 0.18, sr: int = SR,
            max_boost: float = 1.4) -> np.ndarray:
    """Short IR modelling an instrument body: list of (freq Hz, decay s, relative gain).
    The direct sound stays at 1 and the resonances are scaled so the response never rises more than
    (1 + max_boost) x (~ +7.6 dB) at its strongest mode, whatever mode list is given."""
    n = int(length * sr)
    t = np.arange(n) / sr
    m = np.zeros(n)
    for f, d, g in modes:
        m += g * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
    peak = float(np.abs(np.fft.rfft(m, 1 << 16)).max())
    ir = max_boost * m / max(peak, 1e-9)
    ir[0] += 1.0
    return ir.astype(np.float32)


BODIES = {
    "Guitar": [(95, 0.07, 1.6), (205, 0.05, 1.0), (400, 0.03, 0.7), (1100, 0.015, 0.4)],
    "Phin": [(180, 0.05, 1.0), (520, 0.04, 1.2), (1250, 0.02, 0.8)],
    "Piano": [(70, 0.12, 0.8), (140, 0.08, 0.7), (280, 0.05, 0.5)],
    "Bass": [(60, 0.08, 1.0), (120, 0.05, 0.6)],
    "Khaen": [(900, 0.03, 0.6), (1800, 0.02, 0.4)],
}


def compress(x: np.ndarray, sr: int = SR, thresh_db: float = -18.0, ratio: float = 2.0,
             attack_ms: float = 20.0, release_ms: float = 180.0, makeup_db: float = 2.0) -> np.ndarray:
    """Gentle feed-forward bus compressor (stereo-linked), block-based envelope follower."""
    hop = 256
    n = len(x)
    frames = n // hop
    if frames < 2:
        return x
    mono = np.abs(x[: frames * hop]).max(axis=1).reshape(frames, hop).max(axis=1)
    a = np.exp(-hop / (sr * attack_ms / 1000.0))
    r = np.exp(-hop / (sr * release_ms / 1000.0))
    env = np.zeros(frames)
    acc = 0.0
    for i, v in enumerate(mono):
        coef = a if v > acc else r
        acc = coef * acc + (1 - coef) * v
        env[i] = acc
    db = 20 * np.log10(np.maximum(env, 1e-6))
    over = np.maximum(db - thresh_db, 0.0)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    g = 10 ** (gain_db / 20.0)
    centers = np.arange(frames) * hop + hop / 2
    gs = np.interp(np.arange(n), centers, g)
    return x * gs[:, None]


def make_reverb_ir(rt60: float = 1.3, sr: int = SR) -> np.ndarray:
    """Stereo room IR shaped (n, 2) for the send bus."""
    return room_ir(rt60, sr=sr).T.copy()


def _fft_shape(x: np.ndarray, gain_fn, sr: int) -> np.ndarray:
    """Zero-phase magnitude shaping of a (n, ch) signal over its whole length."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    h = gain_fn(np.maximum(f, 1e-3)).astype(np.float32)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.fft.irfft(np.fft.rfft(x[:, c]) * h, n)
    return out


def highpass(x: np.ndarray, sr: int = SR, cutoff: float = 32.0) -> np.ndarray:
    """2nd-order-style high-pass (removes sub rumble that only eats headroom)."""
    return _fft_shape(x, lambda f: (f / cutoff) ** 2 / np.sqrt(1.0 + (f / cutoff) ** 4), sr)


def presence(x: np.ndarray, sr: int = SR, db: float = 1.8) -> np.ndarray:
    """Small, wide lift around 3 kHz so plucked / reed instruments stay clear next to a vocal."""
    amt = 10 ** (db / 20) - 1.0
    return _fft_shape(x, lambda f: 1.0 + amt * np.exp(-0.5 * (np.log2(f / 3000.0) / 0.9) ** 2), sr)

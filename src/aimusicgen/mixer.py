"""Mixing & mastering helpers. The vocal path is never processed - only the
accompaniment is ducked/limited around it."""

from __future__ import annotations

import numpy as np


def soft_limit(x: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """tanh-style soft clipper that keeps peaks below `ceiling`."""
    return ceiling * np.tanh(x / ceiling)


def peak_normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p < 1e-9:
        return x
    return x * (10 ** (peak_db / 20) / p)


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20 * np.log10(max(r, 1e-9))


def envelope(x_mono: np.ndarray, sr: int, attack_ms: float = 15, release_ms: float = 250, hop: int = 441) -> np.ndarray:
    """Smoothed amplitude envelope, returned at sample rate (linear interpolation)."""
    n_frames = max(1, len(x_mono) // hop)
    trimmed = np.abs(x_mono[: n_frames * hop]).reshape(n_frames, hop).max(axis=1)
    a = np.exp(-hop / (sr * attack_ms / 1000.0))
    r = np.exp(-hop / (sr * release_ms / 1000.0))
    env = np.zeros(n_frames)
    acc = 0.0
    for i, v in enumerate(trimmed):
        coef = a if v > acc else r
        acc = coef * acc + (1 - coef) * v
        env[i] = acc
    centers = np.arange(n_frames) * hop + hop / 2
    return np.interp(np.arange(len(x_mono)), centers, env).astype(np.float32)


def duck_under(accomp: np.ndarray, vocal: np.ndarray, sr: int, amount_db: float = 4.0) -> np.ndarray:
    """Lower the accompaniment by up to `amount_db` while the vocal is singing."""
    n = min(len(accomp), len(vocal))
    env = envelope(vocal[:n].mean(axis=1), sr)
    ref = np.percentile(env[env > 1e-4], 90) if np.any(env > 1e-4) else 1.0
    depth = np.clip(env / max(ref, 1e-6), 0, 1)
    gain = 10 ** (-amount_db * depth / 20.0)
    out = accomp.copy()
    out[:n] *= gain[:, None]
    return out


def mix(accomp: np.ndarray, vocal: np.ndarray | None, sr: int, accomp_gain_db: float = 0.0,
        vocal_gain_db: float = 0.0, duck_db: float = 4.0) -> np.ndarray:
    """Return a mastered stereo mix. Vocal samples are only scaled by `vocal_gain_db` (0 dB = untouched)."""
    acc = accomp * 10 ** (accomp_gain_db / 20)
    if vocal is None:
        return peak_normalize(soft_limit(acc), -1.0)

    length = max(len(acc), len(vocal))
    acc = np.pad(acc, ((0, length - len(acc)), (0, 0)))
    voc = np.pad(vocal, ((0, length - len(vocal)), (0, 0))) * 10 ** (vocal_gain_db / 20)
    if duck_db > 0:
        acc = duck_under(acc, voc, sr, duck_db)

    # keep accompaniment ~7 dB under the vocal in loudness so the voice stays in front
    voc_active = voc[np.abs(voc).max(axis=1) > 0.01]
    if voc_active.size:
        target = rms_db(voc_active) - 7.0
        cur = rms_db(acc[np.abs(acc).max(axis=1) > 0.001]) if np.any(np.abs(acc) > 0.001) else target
        acc = acc * 10 ** ((target - cur) / 20)
    out = acc + voc
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:  # scale everything together (vocal stays untouched relative to accompaniment balance)
        out = out * (0.98 / peak)
    return out.astype(np.float32)

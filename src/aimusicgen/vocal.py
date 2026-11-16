"""Vocal preservation: analyse a vocal file (tempo, key) so the accompaniment fits it,
and keep the vocal itself bit-for-bit untouched (only gain-matched at mix time)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import read_audio, resample
from .theory import NOTE_NAMES

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class VocalInfo:
    path: str
    audio: np.ndarray  # (n, 2) float32, original samplerate
    sr: int
    seconds: float
    bpm: float | None
    root_pc: int | None
    scale: str | None  # 'major' | 'minor'
    key_confidence: float
    start_offset_s: float  # time before the vocal becomes audible

    @property
    def key_name(self) -> str:
        if self.root_pc is None:
            return "unknown"
        return f"{NOTE_NAMES[self.root_pc]} {self.scale}"


def _mono(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=1)


def estimate_tempo(mono: np.ndarray, sr: int, lo: float = 60, hi: float = 180) -> float | None:
    """Onset-envelope autocorrelation tempo estimate. Returns None if the vocal is too arrhythmic."""
    target = 11025
    step = max(1, sr // target)
    x = mono[::step]
    sr2 = sr / step
    hop = 256
    frames = len(x) // hop
    if frames < 64:
        return None
    mag = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) for i in range(frames)])
    onset = np.maximum(0, np.diff(np.log1p(mag * 50)))
    onset -= onset.mean()
    if not np.any(onset):
        return None
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    fps = sr2 / hop
    lags = np.arange(len(ac))
    bpms = 60.0 * fps / np.maximum(lags, 1)
    mask = (bpms >= lo) & (bpms <= hi)
    if not mask.any():
        return None
    # mild prior favouring ~110 BPM so we avoid half/double-time errors
    prior = np.exp(-0.5 * (np.log2(bpms[mask] / 110.0) / 0.9) ** 2)
    score = ac[mask] * prior
    best = int(np.argmax(score))
    if ac[mask][best] <= 0:
        return None
    return float(round(bpms[mask][best]))


def estimate_key(mono: np.ndarray, sr: int) -> tuple[int | None, str | None, float]:
    """Chroma-based key estimate (Krumhansl-Schmuckler). Returns (root_pc, scale, confidence 0..1)."""
    n = 8192
    hop = 4096
    if len(mono) < n:
        return None, None, 0.0
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    valid = (freqs > 80) & (freqs < 2000)
    pcs = (np.round(12 * np.log2(freqs[valid] / 440.0)) + 69) % 12
    pcs = pcs.astype(int)
    chroma = np.zeros(12)
    for i in range(0, len(mono) - n, hop):
        seg = mono[i:i + n] * win
        mag = np.abs(np.fft.rfft(seg))[valid]
        if mag.max() < 1e-4:
            continue
        np.add.at(chroma, pcs, mag ** 2)
    if chroma.sum() <= 0:
        return None, None, 0.0
    chroma /= chroma.sum()
    best = (-2.0, 0, "major")
    scores = []
    for root in range(12):
        for scale, prof in (("major", _MAJOR), ("minor", _MINOR)):
            c = np.corrcoef(chroma, np.roll(prof, root))[0, 1]
            scores.append(c)
            if c > best[0]:
                best = (c, root, scale)
    scores = sorted(scores, reverse=True)
    conf = float(np.clip((scores[0] - scores[1]) * 8 + max(0.0, scores[0]) * 0.5, 0, 1))
    return best[1], best[2], conf


def first_audible_time(mono: np.ndarray, sr: int, thresh_db: float = -40.0) -> float:
    win = int(0.02 * sr)
    thr = 10 ** (thresh_db / 20)
    for i in range(0, len(mono) - win, win):
        if np.sqrt(np.mean(mono[i:i + win] ** 2)) > thr:
            return i / sr
    return 0.0


def load_vocal(path: str) -> VocalInfo:
    """Load a vocal file untouched and analyse it. The audio itself is never modified."""
    audio, sr = read_audio(path)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    audio = audio.astype(np.float32)
    mono = _mono(audio)
    bpm = estimate_tempo(mono, sr)
    root, scale, conf = estimate_key(mono, sr)
    return VocalInfo(
        path=str(path), audio=audio, sr=sr, seconds=len(audio) / sr, bpm=bpm,
        root_pc=root, scale=scale, key_confidence=conf,
        start_offset_s=first_audible_time(mono, sr),
    )

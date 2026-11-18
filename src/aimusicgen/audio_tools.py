"""Audio editing and analysis tools (numpy only): time stretch, pitch shift, fades, cut/copy/paste,
chord detection from audio, and export to WAV/FLAC/MP3 (via soundfile) at 44.1/48/96 kHz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio_io import read_audio, resample, write_wav

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------- editing ----------
def time_stretch(x: np.ndarray, rate: float, n_fft: int = 2048) -> np.ndarray:
    """Phase-vocoder time stretch. rate > 1 = faster/shorter. x is (n,) or (n, ch); pitch is unchanged."""
    if x.ndim == 2:
        return np.stack([time_stretch(x[:, c], rate, n_fft) for c in range(x.shape[1])], axis=1)
    if abs(rate - 1.0) < 1e-6 or len(x) < n_fft * 2:
        return x.astype(np.float32)
    hop = n_fft // 4
    win = np.hanning(n_fft)
    pad = np.concatenate([np.zeros(n_fft // 2), x, np.zeros(n_fft)])
    frames = 1 + (len(pad) - n_fft) // hop
    spec = np.fft.rfft(np.stack([pad[i * hop:i * hop + n_fft] * win for i in range(frames)]), axis=1)
    steps = np.arange(0, frames - 1, rate)
    omega = 2 * np.pi * hop * np.arange(spec.shape[1]) / n_fft
    phase = np.angle(spec[0])
    out = np.zeros(int(len(steps) * hop + n_fft))
    norm = np.zeros_like(out)
    for k, s in enumerate(steps):
        i = int(s)
        f = s - i
        mag = (1 - f) * np.abs(spec[i]) + f * np.abs(spec[i + 1])
        if k:
            dphi = np.angle(spec[i + 1]) - np.angle(spec[i]) - omega
            dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
            phase = phase + omega + dphi
        seg = np.fft.irfft(mag * np.exp(1j * phase), n_fft) * win
        out[k * hop:k * hop + n_fft] += seg
        norm[k * hop:k * hop + n_fft] += win ** 2
    out = out / np.maximum(norm, 1e-3)
    return out[n_fft // 2: n_fft // 2 + int(len(x) / rate)].astype(np.float32)


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch, keep duration: stretch by the pitch ratio, then resample back to the original length."""
    if abs(semitones) < 1e-6:
        return x.astype(np.float32)
    ratio = 2.0 ** (semitones / 12.0)
    y = time_stretch(x, 1.0 / ratio)
    idx = np.linspace(0, len(y) - 1, len(x))
    if y.ndim == 1:
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    return np.stack([np.interp(idx, np.arange(len(y)), y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)


def fade(x: np.ndarray, sr: int, fade_in_s: float = 0.0, fade_out_s: float = 0.0) -> np.ndarray:
    y = x.astype(np.float32).copy()
    shape = (-1,) + (1,) * (y.ndim - 1)
    a, b = min(int(fade_in_s * sr), len(y)), min(int(fade_out_s * sr), len(y))
    if a:
        y[:a] *= np.linspace(0, 1, a, dtype=np.float32).reshape(shape)
    if b:
        y[len(y) - b:] *= np.linspace(1, 0, b, dtype=np.float32).reshape(shape)
    return y


def cut(x: np.ndarray, sr: int, start_s: float, end_s: float):
    """Remove [start, end). Returns (remaining, removed_clip)."""
    a, b = int(start_s * sr), int(end_s * sr)
    return np.concatenate([x[:a], x[b:]]), x[a:b].copy()


def copy(x: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    return x[int(start_s * sr):int(end_s * sr)].copy()


def paste(x: np.ndarray, sr: int, clip: np.ndarray, at_s: float) -> np.ndarray:
    a = int(at_s * sr)
    return np.concatenate([x[:a], clip, x[a:]])


def replace_section(x: np.ndarray, sr: int, start_s: float, end_s: float, new: np.ndarray, xfade_s: float = 0.02) -> np.ndarray:
    """Swap [start, end) for `new` (stretched to fit) with a short crossfade on both edges."""
    a, b = int(start_s * sr), int(end_s * sr)
    n = b - a
    new = time_stretch(new, len(new) / max(n, 1)) if len(new) != n else new
    new = np.pad(new, [(0, max(0, n - len(new)))] + [(0, 0)] * (new.ndim - 1))[:n]
    y = x.copy()
    y[a:b] = new
    return fade_edges(y, a, b, int(xfade_s * sr), x)


def fade_edges(y, a, b, k, orig):
    k = min(k, (b - a) // 2)
    if k < 1:
        return y
    ramp = np.linspace(0, 1, k, dtype=np.float32).reshape((-1,) + (1,) * (y.ndim - 1))
    y[a:a + k] = orig[a:a + k] * (1 - ramp) + y[a:a + k] * ramp
    y[b - k:b] = y[b - k:b] * (1 - ramp) + orig[b - k:b] * ramp
    return y


# ---------- analysis ----------
_MAJ = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], float)
_MIN = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], float)
_TEMPLATES = [(f"{NOTE_NAMES[r]}", np.roll(_MAJ, r)) for r in range(12)] + \
             [(f"{NOTE_NAMES[r]}m", np.roll(_MIN, r)) for r in range(12)]


def chroma(mono: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 2048) -> np.ndarray:
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    ok = (freqs > 55) & (freqs < 2000)
    pc = np.round(12 * np.log2(freqs[ok] / 440.0) + 69).astype(int) % 12
    n = max(0, (len(mono) - n_fft) // hop + 1)
    out = np.zeros((n, 12))
    for i in range(n):
        mag = np.abs(np.fft.rfft(mono[i * hop:i * hop + n_fft] * win))[ok]
        np.add.at(out[i], pc, mag)
    return out


def detect_chords(x: np.ndarray, sr: int, beat_s: float | None = None) -> list[tuple[float, float, str]]:
    """Chord per beat (or per 0.5 s): list of (start_s, end_s, label) with equal neighbours merged."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    hop = 2048
    ch = chroma(mono, sr, hop=hop)
    if not len(ch):
        return []
    seg = max(1, int(round((beat_s or 0.5) * sr / hop)))
    out: list[tuple[float, float, str]] = []
    for i in range(0, len(ch), seg):
        v = ch[i:i + seg].sum(axis=0)
        if v.sum() < 1e-6:
            continue
        v = v / np.linalg.norm(v)
        label = max(_TEMPLATES, key=lambda t: float(v @ (t[1] / np.linalg.norm(t[1]))))[0]
        s, e = i * hop / sr, min(len(mono), (i + seg) * hop + 4096) / sr
        if out and out[-1][2] == label:
            out[-1] = (out[-1][0], e, label)
        else:
            out.append((s, e, label))
    return out


# ---------- export ----------
def export_audio(x: np.ndarray, sr: int, path: str | Path, out_sr: int = 44100, bits: int = 24) -> Path:
    """Write WAV / FLAC / MP3 at 44.1 / 48 / 96 kHz. bits: 16 / 24 / 32 (32 = float, WAV only)."""
    path = Path(path)
    y = resample(x, sr, out_sr) if sr != out_sr else x
    ext = path.suffix.lower()
    if ext in (".wav", ""):
        return write_wav(path.with_suffix(".wav"), y, out_sr, bits)
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".flac":
        sf.write(str(path), y, out_sr, subtype="PCM_24" if bits >= 24 else "PCM_16")
    elif ext == ".mp3":
        sf.write(str(path), y, out_sr, format="MP3")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return path

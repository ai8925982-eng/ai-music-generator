"""Audio file IO. WAV via the standard library (no deps); other formats via
optional `soundfile` / `ffmpeg` if available."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def _read_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    return a.reshape(-1, ch), sr


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (samples shape (n, channels) float32 in [-1, 1], samplerate)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".wav", ".wave"):
        try:
            return _read_wav_stdlib(path)
        except (wave.Error, ValueError):
            pass  # e.g. float WAV -> fall through to soundfile / ffmpeg
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tmp.wav"
            subprocess.run([ffmpeg, "-y", "-i", str(path), "-c:a", "pcm_s24le", str(out)],
                           check=True, capture_output=True)
            return _read_wav_stdlib(out)
    raise RuntimeError(
        f"Cannot read '{path.name}'. Use a .wav file, or `pip install soundfile`, or install ffmpeg for mp3/flac/m4a."
    )


def write_wav(path: str | Path, audio: np.ndarray, sr: int, bits: int = 16) -> Path:
    """Write float audio (n, ch) or (n,) to a PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    a = np.clip(a, -1.0, 1.0)
    if bits == 16:
        pcm = (a * 32767.0).astype("<i2").tobytes()
        sw = 2
    elif bits == 24:
        v = (a * 8388607.0).astype(np.int32)
        b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8)
        pcm = b.tobytes()
        sw = 3
    else:
        raise ValueError("bits must be 16 or 24")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(pcm)
    return path


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler (good enough for analysis & previews)."""
    if sr_in == sr_out:
        return audio
    n_out = int(round(audio.shape[0] * sr_out / sr_in))
    x_old = np.linspace(0, 1, audio.shape[0], endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    cols = [np.interp(x_new, x_old, audio[:, c]) for c in range(audio.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)

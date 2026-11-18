"""Monophonic pitch tracking (YIN) + note segmentation for sung vocals. numpy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRACK_SR = 16000
WIN = 512          # YIN integration window (32 ms)
FRAME = 1024
HOP = 160          # 10 ms
TAU_MAX = 400      # lowest pitch ~ 40 Hz at 16 kHz; limited by fmin below


@dataclass
class PitchTrack:
    times: np.ndarray    # seconds, one per frame
    midi: np.ndarray     # float MIDI note per frame, NaN where unvoiced
    rms: np.ndarray      # frame loudness
    conf: np.ndarray     # 0..1 voicing confidence


@dataclass
class SungNote:
    start: float         # seconds
    end: float
    midi: int            # rounded pitch
    rms: float


def _to_track_sr(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == TRACK_SR:
        return mono.astype(np.float32)
    n_out = int(len(mono) * TRACK_SR / sr)
    # cheap anti-alias: box filter when downsampling by > 1.5x, then linear interpolation
    k = int(sr / TRACK_SR)
    x = mono
    if k > 1:
        m = (len(x) // k) * k
        x = x[:m].reshape(-1, k).mean(axis=1)
        sr_eff = sr / k
    else:
        sr_eff = sr
    xi = np.arange(n_out) * (sr_eff / TRACK_SR)
    return np.interp(xi, np.arange(len(x)), x).astype(np.float32)


def track_pitch(mono: np.ndarray, sr: int, fmin: float = 80.0, fmax: float = 1000.0,
                thresh: float = 0.18) -> PitchTrack:
    """YIN pitch tracker, vectorised over frames."""
    x = _to_track_sr(mono, sr)
    n_frames = max(0, (len(x) - FRAME) // HOP + 1)
    if n_frames <= 0:
        z = np.zeros(0)
        return PitchTrack(z, z, z, z)
    tau_min = max(2, int(TRACK_SR / fmax))
    tau_max = min(TAU_MAX, int(TRACK_SR / fmin))
    times = (np.arange(n_frames) * HOP + FRAME / 2) / TRACK_SR
    midi = np.full(n_frames, np.nan)
    rms = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    chunk = 1500
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        idx = (np.arange(c0, c1)[:, None] * HOP + np.arange(FRAME)[None, :])
        fr = x[idx].astype(np.float64)
        fr -= fr.mean(axis=1, keepdims=True)
        rms[c0:c1] = np.sqrt(np.mean(fr[:, 320:704] ** 2, axis=1))   # 24 ms around the frame centre: sharp enough to see syllable dips
        A = np.fft.rfft(fr[:, :WIN], n=2048, axis=1)
        B = np.fft.rfft(fr, n=2048, axis=1)
        r = np.fft.irfft(np.conj(A) * B, n=2048, axis=1)[:, :tau_max + 1]
        sq = fr ** 2
        cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(sq, axis=1)], axis=1)
        e1 = cs[:, WIN][:, None]
        taus = np.arange(tau_max + 1)
        e2 = cs[:, WIN + taus] - cs[:, taus]
        d = e1 + e2 - 2 * r
        d[:, 0] = 0
        cum = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:][None, :] / np.maximum(cum, 1e-12)
        cm = cmnd[:, tau_min:]
        below = cm < thresh
        has = below.any(axis=1)
        first = np.argmax(below, axis=1)
        gmin = np.argmin(cm, axis=1)
        pick = np.where(has, first, gmin)
        # slide down to the local minimum following the first dip below threshold
        for _ in range(40):
            nxt = np.minimum(pick + 1, cm.shape[1] - 1)
            move = has & (cm[np.arange(len(pick)), nxt] < cm[np.arange(len(pick)), pick])
            if not move.any():
                break
            pick = np.where(move, nxt, pick)
        tau = pick + tau_min
        rows = np.arange(len(tau))
        # parabolic interpolation
        t0 = np.clip(tau, 1, tau_max - 1)
        y0, y1, y2 = cmnd[rows, t0 - 1], cmnd[rows, t0], cmnd[rows, t0 + 1]
        den = (y0 - 2 * y1 + y2)
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        tau_f = t0 + np.clip(shift, -1, 1)
        f0 = TRACK_SR / tau_f
        best = cmnd[rows, t0]
        voiced = (best < 0.35) & (rms[c0:c1] > 0.006)
        conf[c0:c1] = np.clip(1.0 - best / 0.35, 0, 1) * voiced
        m = 69 + 12 * np.log2(np.maximum(f0, 1.0) / 440.0)
        midi[c0:c1] = np.where(voiced, m, np.nan)
    # remove isolated voiced / unvoiced blips and octave jumps against a running median
    midi = _clean(midi)
    return PitchTrack(times, midi, rms, conf)


def _clean(midi: np.ndarray, k: int = 5) -> np.ndarray:
    out = midi.copy()
    n = len(out)
    pad = k // 2
    for i in range(n):
        if np.isnan(out[i]):
            continue
        lo, hi = max(0, i - 8), min(n, i + 9)
        w = midi[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) >= 3:
            med = np.median(w)
            diff = out[i] - med
            if abs(diff) > 9:  # octave / harmonic error -> fold toward neighbourhood median
                out[i] = out[i] - 12 * np.round(diff / 12)
    # drop voiced runs shorter than 4 frames (40 ms)
    v = ~np.isnan(out)
    i = 0
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            if j - i < 4:
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    # light median smoothing inside voiced runs
    sm = out.copy()
    for i in range(n):
        if np.isnan(out[i]):
            continue
        lo, hi = max(0, i - pad), min(n, i + pad + 1)
        w = out[lo:hi]
        w = w[~np.isnan(w)]
        sm[i] = np.median(w)
    return sm


def _split_on_dips(rms: np.ndarray, i: int, j: int, win: int = 14, depth: float = 0.6) -> list[int]:
    """Frame indices inside [i, j] where loudness dips enough to mark a new syllable of the same pitch."""
    cuts: list[int] = []
    last = i
    for k in range(i + 3, j - 2):
        if k - last < 6:
            continue
        left = rms[max(last, k - win):k].max() if k > last else 0.0
        right = rms[k + 1:min(j + 1, k + win)].max()
        if rms[k] <= rms[k - 1] and rms[k] <= rms[k + 1] and rms[k] < depth * min(left, right):
            cuts.append(k)
            last = k
    return cuts


def segment_notes(pt: PitchTrack, min_len: float = 0.07, tol: float = 0.65) -> list[SungNote]:
    """Group frames into sung notes. A new note starts when the pitch moves > `tol` semitones from the
    running median, or when loudness dips and re-attacks (repeated syllables on the same pitch)."""
    notes: list[SungNote] = []
    n = len(pt.midi)
    i = 0
    dt = pt.times[1] - pt.times[0] if n > 1 else 0.01
    bias = 0.012  # frame-centre vs. actual onset
    rms = np.convolve(pt.rms, np.ones(3) / 3, mode="same") if n else pt.rms
    while i < n:
        if np.isnan(pt.midi[i]):
            i += 1
            continue
        j = i
        vals = [pt.midi[i]]
        while j + 1 < n and not np.isnan(pt.midi[j + 1]) and abs(pt.midi[j + 1] - np.median(vals)) <= tol:
            j += 1
            vals.append(pt.midi[j])
        bounds = [i] + _split_on_dips(rms, i, j) + [j + 1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (b - a) * dt >= min_len:
                seg = pt.midi[a:b]
                notes.append(SungNote(float(pt.times[a] - dt / 2 - bias), float(pt.times[b - 1] + dt / 2),
                                      int(round(float(np.median(seg)))), float(np.mean(pt.rms[a:b]))))
        i = j + 1
    return notes

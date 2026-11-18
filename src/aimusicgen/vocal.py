"""Vocal preservation: analyse a vocal file (tempo, key) so the accompaniment fits it,
and keep the vocal itself bit-for-bit untouched (only gain-matched at mix time)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio_io import read_audio, resample
from .pitch import PitchTrack, SungNote, segment_notes, track_pitch
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
    grid_offset_s: float = 0.0  # where the first bar line falls (from onset analysis)
    tempo_conf: float = 0.0     # 0..1, how rhythmic / reliable the detected tempo is
    tempo_stable: bool = True   # False if the singer's tempo drifts noticeably
    notes: list = field(default_factory=list)   # SungNote list
    pitch: PitchTrack | None = None
    register: float = 60.0      # median sung MIDI note

    def grid_for(self, bpm: float) -> float:
        """Bar-line offset (seconds) for a user-forced tempo."""
        return _grid_for_tempo(self.notes, bpm, self.seconds)

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
    """Load a vocal file untouched and analyse it (pitch, notes, key, tempo, bar line)."""
    audio, sr = read_audio(path)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    audio = audio.astype(np.float32)
    mono = _mono(audio)
    seconds = len(audio) / sr
    pt = track_pitch(mono, sr)
    notes = segment_notes(pt)
    reg = float(np.median([n.midi for n in notes])) if notes else 60.0

    if len(notes) >= 8:
        bpm, grid, tconf, stable = estimate_tempo_from_notes(notes, seconds)
        root, scale, kconf = key_from_notes(notes)
    else:  # almost nothing sung: fall back to the signal-level estimators
        bpm = estimate_tempo(mono, sr)
        bpm, grid = refine_tempo_and_grid(mono, sr, bpm)
        tconf, stable = 0.2, True
        root, scale, kconf = estimate_key(mono, sr)
    return VocalInfo(
        path=str(path), audio=audio, sr=sr, seconds=seconds, bpm=bpm,
        root_pc=root, scale=scale, key_confidence=kconf,
        start_offset_s=first_audible_time(mono, sr), grid_offset_s=grid,
        tempo_conf=tconf, tempo_stable=stable, notes=notes, pitch=pt, register=reg,
    )


# ---------------------------------------------------------------------------
# Key from the notes that were actually sung
# ---------------------------------------------------------------------------

def _phrase_final(notes: list[SungNote], gap: float = 0.3) -> set[int]:
    out = set()
    for i, n in enumerate(notes):
        if i + 1 == len(notes) or notes[i + 1].start - n.end > gap:
            out.add(i)
    return out


def key_from_notes(notes: list[SungNote]) -> tuple[int | None, str | None, float]:
    hist = np.zeros(12)
    finals = _phrase_final(notes)
    for i, n in enumerate(notes):
        w = (n.end - n.start) * (n.rms ** 0.5 if n.rms > 0 else 1.0)
        hist[n.midi % 12] += w * (2.0 if i in finals else 1.0)
    if hist.sum() <= 0:
        return None, None, 0.0
    hist /= hist.sum()
    scores = []
    for root in range(12):
        for scale, prof in (("major", _MAJOR), ("minor", _MINOR)):
            p = np.roll(prof, root)
            c = float(np.corrcoef(hist, p)[0, 1])
            # reward a strong tonic / fifth, penalise out-of-scale mass (pentatonic-friendly)
            steps = [0, 2, 4, 5, 7, 9, 11] if scale == "major" else [0, 2, 3, 5, 7, 8, 10]
            outside = sum(hist[(root + k) % 12] for k in range(12) if k not in steps)
            c += 0.35 * hist[root] + 0.1 * hist[(root + 7) % 12] - 0.8 * outside
            scores.append((c, root, scale))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    conf = float(np.clip((best[0] - second[0]) * 6 + 0.3, 0, 1))
    return best[1], best[2], conf


# ---------------------------------------------------------------------------
# Tempo / bar line from sung-note onsets
# ---------------------------------------------------------------------------

def _onset_table(notes: list[SungNote]):
    t = np.array([n.start for n in notes])
    dur = np.array([n.end - n.start for n in notes])
    rms = np.array([n.rms for n in notes])
    w = 0.6 + 0.4 * rms / max(rms.max(), 1e-9)
    gap_before = np.diff(t, prepend=-10.0) - np.concatenate([[0.0], dur[:-1]])
    phrase_start = gap_before > 0.35
    return t, dur, w, phrase_start


def _emphasis(w, dur, phrase_start):
    """Singers put phrase starts and held notes on strong beats: use that only to settle beat/off-beat phase."""
    return w * (1.0 + 2.0 * phrase_start) * (1.0 + 0.5 * (dur > 0.45))


def _score_grid(t, w, bpm_arr, n_phase=24):
    """score[b, p] = how well onsets sit on a (beat + 8th) grid with period 60/bpm and phase p."""
    out = np.zeros((len(bpm_arr), n_phase))
    for i, bpm in enumerate(bpm_arr):
        spb = 60.0 / bpm
        for j in range(n_phase):
            ph = j / n_phase * spb
            x = (t - ph) / spb * 2.0
            r = np.round(x)
            err = np.abs(x - r) * spb / 2.0
            on = (r % 2 == 0)
            k = np.exp(-0.5 * (err / 0.035) ** 2) * np.where(on, 1.0, 0.7)
            out[i, j] = float((k * w).sum() / w.sum())
    return out


def estimate_tempo_from_notes(notes: list[SungNote], seconds: float, lo: float = 60, hi: float = 180):
    t, dur, w, pstart = _onset_table(notes)
    coarse = np.arange(lo, hi + 1e-6, 0.5)
    sc = _score_grid(t, w, coarse)
    best_per = sc.max(axis=1) - sc.mean(axis=1)   # lift over chance: denser grids score higher by luck
    prior = np.exp(-0.5 * (np.log2(coarse / 100.0) / 0.75) ** 2)  # singers most often sit 80-130 BPM
    adj = best_per * (0.6 + 0.4 * prior)
    b0 = float(coarse[int(np.argmax(adj))])
    # octave errors: a 8th-note grid also fits at 2x / 0.5x tempo -> prefer the one in the usual singing range
    pref_lo, pref_hi = 70.0, 135.0
    if not (pref_lo <= b0 <= pref_hi):
        for cand in (b0 / 2.0, b0 * 2.0):
            if lo <= cand <= hi and pref_lo <= cand <= pref_hi:
                j = int(np.argmin(np.abs(coarse - cand)))
                if adj[j] >= 0.55 * adj.max():
                    b0 = float(coarse[j])
                    break
    fine = np.arange(b0 - 0.6, b0 + 0.6001, 0.05)
    scf = _score_grid(t, w, fine, n_phase=48)
    bi, pj = np.unravel_index(int(np.argmax(scf)), scf.shape)
    bpm = float(fine[bi])
    sc_phase = _score_grid(t, _emphasis(w, dur, pstart), np.array([bpm]), n_phase=48)[0]
    phase_s = int(np.argmax(sc_phase)) / 48 * 60.0 / bpm
    z = (best_per.max() - best_per.mean()) / max(best_per.std(), 1e-9)
    conf = float(np.clip((z - 2.0) / 4.0, 0.0, 1.0))

    # drift: does the same tempo explain both halves of the performance?
    stable = True
    half = len(t) // 2
    if half >= 12:
        tempos = []
        for sl in (slice(0, half), slice(half, None)):
            cand = np.arange(bpm * 0.94, bpm * 1.06, 0.1)
            s2 = _score_grid(t[sl], w[sl], cand, n_phase=16)
            tempos.append(float(cand[int(np.argmax(s2.max(axis=1)))]))
        stable = abs(tempos[0] - tempos[1]) / bpm < 0.04
    grid = _grid_from_phase(t, dur, pstart, bpm, phase_s)
    return round(bpm, 2), grid, conf, stable


def _grid_from_phase(t, dur, pstart, bpm: float, phase_s: float) -> float:
    spb = 60.0 / bpm
    beat_idx = np.round((t - phase_s) / spb).astype(int)
    on_grid = np.abs((t - phase_s) / spb - beat_idx) < 0.15
    score = np.zeros(4)
    for bi, ok, ps, d in zip(beat_idx, on_grid, pstart, dur):
        if not ok:
            continue
        score[bi % 4] += (3.0 if ps else 1.0) + min(d / spb, 2.0) * 0.5
    m = int(np.argmax(score))
    return float((phase_s + m * spb) % (4 * spb))


def _grid_for_tempo(notes, bpm: float, seconds: float) -> float:
    if len(notes) < 8:
        return 0.0
    t, dur, w, pstart = _onset_table(notes)
    sc = _score_grid(t, _emphasis(w, dur, pstart), np.array([bpm]), n_phase=48)[0]
    phase_s = int(np.argmax(sc)) / 48 * 60.0 / bpm
    return _grid_from_phase(t, dur, pstart, bpm, phase_s)


def refine_tempo_and_grid(mono: np.ndarray, sr: int, bpm_guess: float | None,
                          beats_per_bar: int = 4, search: float = 4.0):
    """Signal-level fallback (used only when almost nothing was sung)."""
    return _refine_signal(mono, sr, bpm_guess, beats_per_bar, search)


def _onset_envelope(mono: np.ndarray, sr: int, hop: int = 256) -> tuple[np.ndarray, float]:
    target = 11025
    step = max(1, sr // target)
    x = mono[::step]
    sr2 = sr / step
    frames = len(x) // hop
    if frames < 32:
        return np.zeros(0), 1.0
    mag = np.sqrt(np.mean(x[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    onset = np.maximum(0, np.diff(np.log1p(mag * 50), prepend=0.0))
    onset = np.convolve(onset, np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2), mode="same")
    return onset, sr2 / hop


def _refine_signal(mono: np.ndarray, sr: int, bpm_guess: float | None,
                          beats_per_bar: int = 4, search: float = 4.0) -> tuple[float | None, float]:
    """Refine BPM to ~0.05 and find where the first bar line falls (seconds).

    Returns (bpm, grid_offset_s) with 0 <= grid_offset_s < one bar.
    """
    onset, fps = _onset_envelope(mono, sr)
    if bpm_guess is None or len(onset) == 0:
        return bpm_guess, 0.0
    best = (-1.0, bpm_guess, 0.0, 0)
    n = len(onset)
    cands = [bpm_guess] if search <= 0 else np.arange(bpm_guess - search, bpm_guess + search + 0.001, 0.1)
    for bpm in cands:
        period = 60.0 * fps / bpm
        n_beats = int((n - 1) / period)
        if n_beats < 8:
            continue
        k = np.arange(n_beats)
        for phi in np.linspace(0, period, 16, endpoint=False):
            idx = k * period + phi
            idx = idx[idx < n - 1]
            vals = np.interp(idx, np.arange(n), onset)
            sc = float(vals.mean())
            if sc > best[0]:
                best = (sc, float(bpm), phi / fps, len(vals))
    _, bpm, phase_s, _ = best
    spb = 60.0 / bpm
    # which of the 4 beats carries the bar line? pick the one with strongest onsets
    t = np.arange(0, len(mono) / sr, spb) + phase_s
    idx = np.clip(t * fps, 0, n - 1)
    vals = np.interp(idx, np.arange(n), onset)
    scores = [vals[m::beats_per_bar].sum() for m in range(beats_per_bar)]
    m = int(np.argmax(scores))
    return round(bpm, 2), float((phase_s + m * spb) % (beats_per_bar * spb))



def _chroma_of(mono: np.ndarray, sr: int) -> np.ndarray | None:
    n = 4096
    if len(mono) < n:
        return None
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    valid = (freqs > 80) & (freqs < 2000)
    pcs = ((np.round(12 * np.log2(freqs[valid] / 440.0)) + 69) % 12).astype(int)
    chroma = np.zeros(12)
    for i in range(0, len(mono) - n, n // 2):
        mag = np.abs(np.fft.rfft(mono[i:i + n] * win))[valid]
        if mag.max() < 1e-4:
            continue
        np.add.at(chroma, pcs, mag ** 2)
    if chroma.sum() <= 0:
        return None
    return chroma / chroma.sum()


@dataclass
class VocalGuide:
    """What the composer needs to know about the vocal, on the song's bar / 8th-note grid."""
    offset_s: float            # time of the first bar line
    activity: np.ndarray       # bool, one entry per 8th note (True = vocal is singing)
    bar_chroma: np.ndarray     # (bars, 12) sung pitch-class weight in each bar (strong beats emphasised)
    energy: np.ndarray         # per bar 0..1 vocal loudness
    melody: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # MIDI note per 8th, -1 = none
    register: float = 60.0     # median sung MIDI note

    def recent_phrase(self, slot: int, max_notes: int = 6) -> list[tuple[int, int, int]]:
        """Notes of the phrase that ended just before `slot` as (start_slot_rel_to_phrase_start, length_slots, midi)."""
        m = self.melody
        i = min(slot, len(m)) - 1
        while i >= 0 and m[i] < 0:
            i -= 1
        if i < 0:
            return []
        end = i
        gaps = 0
        while i >= 0 and gaps < 2:
            gaps = gaps + 1 if m[i] < 0 else 0
            i -= 1
        start = i + 1
        notes: list[list[int]] = []
        for k in range(start, end + 1):
            if m[k] < 0:
                continue
            if notes and notes[-1][2] == m[k] and notes[-1][0] + notes[-1][1] == k:
                notes[-1][1] += 1
            else:
                notes.append([k - start, 1, int(m[k])])
        return [tuple(n) for n in notes[-max_notes:]]


def build_guide(v: VocalInfo, bpm: float, bars: int, offset_s: float) -> VocalGuide:
    mono = v.audio.mean(axis=1)
    sr = v.sr
    spb = 60.0 / bpm
    slot = spb / 2.0
    n_slots = bars * 8
    pt = v.pitch if v.pitch is not None else track_pitch(mono, sr)

    # loudness per 8th note -> activity
    rms = np.zeros(n_slots)
    for i in range(n_slots):
        a = int((offset_s + i * slot) * sr)
        b = int((offset_s + (i + 1) * slot) * sr)
        if a >= len(mono):
            break
        seg = mono[max(0, a):max(a + 1, min(b, len(mono)))]
        rms[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
    ref = np.percentile(rms[rms > 1e-4], 90) if np.any(rms > 1e-4) else 1.0
    activity = rms > max(0.12 * ref, 1e-3)

    # sung pitch per 8th note
    melody = np.full(n_slots, -1, dtype=int)
    if len(pt.times):
        for i in range(n_slots):
            t0 = offset_s + i * slot
            lo = int(np.searchsorted(pt.times, t0))
            hi = int(np.searchsorted(pt.times, t0 + slot))
            seg = pt.midi[lo:hi]
            ok = ~np.isnan(seg)
            if hi > lo and ok.mean() >= 0.4:
                melody[i] = int(round(float(np.median(seg[ok]))))
    activity |= melody >= 0
    for i in range(1, n_slots - 1):          # breaths between words don't count as free space
        if not activity[i] and activity[i - 1] and activity[i + 1]:
            activity[i] = True

    bar_len = 4 * spb
    chroma = np.zeros((bars, 12))
    energy = np.zeros(bars)
    for b in range(bars):
        a_s, e_s = offset_s + b * bar_len, offset_s + (b + 1) * bar_len
        if a_s >= v.seconds:
            break
        seg = mono[max(0, int(a_s * sr)):int(e_s * sr)]
        energy[b] = min(1.0, float(np.sqrt(np.mean(seg ** 2))) / max(ref, 1e-6)) if len(seg) else 0.0
        if len(pt.times):
            lo, hi = int(np.searchsorted(pt.times, a_s)), int(np.searchsorted(pt.times, e_s))
            for k in range(lo, hi):
                if np.isnan(pt.midi[k]):
                    continue
                pos = ((pt.times[k] - a_s) / spb) % 4.0
                w = pt.rms[k] * (1.6 if (pos < 0.5 or 2.0 <= pos < 2.5) else 1.0)
                chroma[b, int(round(pt.midi[k])) % 12] += w
        s = chroma[b].sum()
        if s > 0:
            chroma[b] /= s
        else:
            c = _chroma_of(seg, sr)
            if c is not None and energy[b] > 0.15:
                chroma[b] = c
    return VocalGuide(offset_s, activity, chroma, energy, melody, v.register)

"""End-to-end pipeline: compose -> MIDI -> render -> (preserve vocal) -> mix -> files."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import resample, write_wav
from .composer import BEATS_PER_BAR, Song, compose
from .genres import get_genre
from .midi_io import write_midi
from .mixer import mix, peak_normalize, soft_limit
from .synth import SR, render_stereo
from .theory import NOTE_NAMES, note_name_to_pc
from .vocal import VocalInfo, load_vocal

Progress = Callable[[float, str], None]


@dataclass
class GenerateOptions:
    genre: str = "all"
    output_dir: str = "output"
    name: str = "song"
    bars: int = 32
    bpm: float | None = None
    key: str | None = None  # e.g. 'C', 'F#'
    scale: str | None = None
    seed: int | None = None
    mode: str = "audio_midi"  # 'audio', 'midi', 'audio_midi'
    vocal_path: str | None = None  # preserve this vocal untouched
    lead: bool | None = None  # None = auto (off when a vocal is given)
    export_stems: bool = False
    bits: int = 16
    duck_db: float = 4.0
    vocal_gain_db: float = 0.0
    follow_vocal: bool = True  # take BPM/key from the vocal when not set manually


@dataclass
class GenerateResult:
    song: Song
    midi_path: Path | None = None
    audio_path: Path | None = None
    accompaniment_path: Path | None = None
    stems: list[Path] = field(default_factory=list)
    info_path: Path | None = None
    vocal: VocalInfo | None = None
    seconds_taken: float = 0.0


def _noop(p: float, m: str) -> None:
    pass


def generate(opts: GenerateOptions, progress: Progress = _noop) -> GenerateResult:
    t0 = time.time()
    out = Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = opts.mode.lower()
    if mode not in ("audio", "midi", "audio_midi"):
        raise ValueError("mode must be 'audio', 'midi' or 'audio_midi'")

    vocal: VocalInfo | None = None
    bpm, key_pc, scale, bars = opts.bpm, None, opts.scale, opts.bars
    if opts.key:
        key_pc = note_name_to_pc(opts.key)

    if opts.vocal_path:
        progress(0.02, "Analysing vocal (tempo / key)...")
        vocal = load_vocal(opts.vocal_path)
        if opts.follow_vocal:
            if bpm is None and vocal.bpm:
                bpm = vocal.bpm
            if key_pc is None and vocal.root_pc is not None and vocal.key_confidence >= 0.15:
                key_pc, scale = vocal.root_pc, scale or vocal.scale
        use_bpm = bpm or 100.0
        sec_per_bar = BEATS_PER_BAR * 60.0 / use_bpm
        bars = max(opts.bars, int(math.ceil((vocal.seconds + 2.0) / sec_per_bar)))
        bpm = use_bpm

    progress(0.10, "Composing...")
    lead = opts.lead if opts.lead is not None else (vocal is None)
    song = compose(opts.genre, bpm, key_pc, scale, bars, opts.seed, lead)
    genre = get_genre(song.genre)
    res = GenerateResult(song=song, vocal=vocal)
    base = out / opts.name

    if mode in ("midi", "audio_midi"):
        progress(0.25, "Writing MIDI...")
        res.midi_path = write_midi(base.with_suffix(".mid"), song.tracks, song.bpm)

    if mode in ("audio", "audio_midi"):
        progress(0.35, "Rendering instruments...")
        stems = render_stereo(song, genre, SR)
        accomp = sum(stems.values())
        accomp = peak_normalize(soft_limit(accomp * 0.8), -3.0)
        if opts.export_stems:
            for name, st in stems.items():
                p = write_wav(out / f"{opts.name}_stem_{name.lower()}.wav", peak_normalize(st, -3.0), SR, opts.bits)
                res.stems.append(p)

        progress(0.75, "Mixing...")
        voc_arr = None
        if vocal is not None:
            voc_arr = vocal.audio
            if vocal.sr != SR:
                voc_arr = resample(voc_arr, vocal.sr, SR)
        res.accompaniment_path = write_wav(out / f"{opts.name}_accompaniment.wav", accomp, SR, opts.bits)
        final = mix(accomp, voc_arr, SR, duck_db=opts.duck_db, vocal_gain_db=opts.vocal_gain_db)
        res.audio_path = write_wav(out / f"{opts.name}.wav", final, SR, opts.bits)

    res.info_path = out / f"{opts.name}_info.json"
    res.info_path.write_text(json.dumps({
        "genre": song.genre, "bpm": song.bpm, "key": song.key_name, "bars": song.bars,
        "seconds": round(song.seconds, 2), "seed": song.seed, "mode": mode,
        "vocal": None if vocal is None else {
            "file": Path(vocal.path).name, "seconds": round(vocal.seconds, 2),
            "detected_bpm": vocal.bpm, "detected_key": vocal.key_name,
            "key_confidence": round(vocal.key_confidence, 2), "preserved_untouched": True,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    res.seconds_taken = time.time() - t0
    progress(1.0, "Done")
    return res

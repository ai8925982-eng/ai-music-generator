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
from .synth import SR, master_bus, render_stereo
from .theory import NOTE_NAMES, note_name_to_pc
from .vocal import VocalInfo, build_guide, load_vocal

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
    grid_shift_ms: float = 0.0  # manual nudge of the music against the vocal (+ = later)
    follow_vocal: bool = True  # take BPM/key from the vocal when not set manually
    soundfont: str = "none"  # "none" = built-in synthesis, "auto" = first .sf2 found in ./soundfonts, or a path to .sf2


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
    guide = None
    off_s = 0.0
    bpm, key_pc, scale, bars = opts.bpm, None, opts.scale, opts.bars
    if opts.key:
        key_pc = note_name_to_pc(opts.key)

    if opts.vocal_path:
        progress(0.02, "Analysing vocal (tempo / key / phrasing)...")
        vocal = load_vocal(opts.vocal_path)
        if vocal.bpm and vocal.tempo_conf < 0.25:
            progress(0.06, f"Warning: tempo of the vocal is unclear ({vocal.bpm:g} BPM guessed). "
                           "If the music feels off, set the BPM yourself.")
        elif vocal.bpm and not vocal.tempo_stable:
            progress(0.06, "Warning: the singer's tempo drifts; a fixed-tempo backing may slip in places.")
        if opts.follow_vocal:
            if bpm is None and vocal.bpm:
                bpm = vocal.bpm
                off_s = vocal.grid_offset_s
            elif bpm is not None and vocal.bpm:
                off_s = vocal.grid_for(bpm)
            if key_pc is None and vocal.root_pc is not None and vocal.key_confidence >= 0.15:
                key_pc, scale = vocal.root_pc, scale or vocal.scale
        use_bpm = bpm or 100.0
        sec_per_bar = BEATS_PER_BAR * 60.0 / use_bpm
        if opts.grid_shift_ms:
            off_s = (off_s + opts.grid_shift_ms / 1000.0) % sec_per_bar
        bars = max(opts.bars, int(math.ceil((vocal.seconds - off_s + 2.0) / sec_per_bar)))
        bpm = use_bpm
        if opts.follow_vocal:
            guide = build_guide(vocal, bpm, bars, off_s)

    progress(0.10, "Composing...")
    lead = opts.lead if opts.lead is not None else (vocal is None)
    song = compose(opts.genre, bpm, key_pc, scale, bars, opts.seed, lead, guide)
    if off_s > 0:
        shift = off_s * song.bpm / 60.0  # seconds -> beats
        for tr in song.tracks:
            for n in tr.notes:
                n.start += shift
        song.offset_s = off_s
    genre = get_genre(song.genre)
    res = GenerateResult(song=song, vocal=vocal)
    base = out / opts.name

    if mode in ("midi", "audio_midi"):
        progress(0.25, "Writing MIDI...")
        res.midi_path = write_midi(base.with_suffix(".mid"), song.tracks, song.bpm)

    if mode in ("audio", "audio_midi"):
        progress(0.35, "Rendering instruments...")
        sf = None
        if opts.soundfont and opts.soundfont != "none":
            from .sf2 import SoundFont, find_soundfonts
            path = opts.soundfont
            if path == "auto":
                found = find_soundfonts()
                path = str(found[0]) if found else None
            if path:
                try:
                    sf = SoundFont(path)
                    progress(0.36, f"SoundFont: {sf.name}")
                except Exception as e:  # bad file -> built-in synthesis
                    progress(0.36, f"SoundFont could not be loaded ({e}); using built-in sounds")
            elif opts.soundfont == "auto":
                progress(0.36, "No .sf2 found in ./soundfonts; using built-in sounds")
        stems = render_stereo(song, genre, SR, sf=sf)
        accomp = master_bus(stems, genre, SR)
        accomp = peak_normalize(soft_limit(accomp * 0.8), -3.0)
        if opts.export_stems:
            for name, st in stems.items():
                p = write_wav(out / f"{opts.name}_stem_{name.lower()}.wav", peak_normalize(st, -3.0), SR, opts.bits)
                res.stems.append(p)
        del stems  # free memory before mixing

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
        "grid_offset_s": round(song.offset_s, 3),
        "vocal": None if vocal is None else {
            "file": Path(vocal.path).name, "seconds": round(vocal.seconds, 2),
            "detected_bpm": vocal.bpm, "detected_key": vocal.key_name,
            "tempo_confidence": round(vocal.tempo_conf, 2), "tempo_stable": vocal.tempo_stable,
            "key_confidence": round(vocal.key_confidence, 2), "preserved_untouched": True,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    res.seconds_taken = time.time() - t0
    progress(1.0, "Done")
    return res

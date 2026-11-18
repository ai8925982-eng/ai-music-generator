"""Command line interface:  python -m aimusicgen  (no args opens the GUI)."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .genres import GENRE_NAMES
from .pipeline import GenerateOptions, generate
from .theory import NOTE_NAMES, SCALES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aimusicgen",
        description="AI Music Generator - create music as audio + MIDI, optionally keeping your vocal untouched.",
    )
    p.add_argument("--version", action="version", version=f"aimusicgen {__version__}")
    p.add_argument("--gui", action="store_true", help="open the graphical interface")
    p.add_argument("-g", "--genre", default="all", help=f"all | {' | '.join(GENRE_NAMES)}  (default: all = random)")
    p.add_argument("-o", "--output", default="output", help="output folder (default: ./output)")
    p.add_argument("-n", "--name", default="song", help="output file name without extension")
    p.add_argument("--mode", choices=["audio", "midi", "audio_midi"], default="audio_midi",
                   help="what to export (default: audio_midi)")
    p.add_argument("--vocal", metavar="FILE", help="vocal file (wav; mp3/flac need soundfile or ffmpeg) to PRESERVE untouched")
    p.add_argument("--bars", type=int, default=32, help="song length in bars (default 32)")
    p.add_argument("--bpm", type=float, help="tempo (default: random for genre / detected from vocal)")
    p.add_argument("--key", choices=NOTE_NAMES, help="key root note (default: random / detected from vocal)")
    p.add_argument("--scale", choices=sorted(SCALES), help="scale (default: depends on genre / vocal)")
    p.add_argument("--seed", type=int, help="random seed (same seed + options = same song)")
    p.add_argument("--lead", choices=["on", "off"], help="generated lead melody (default: off when a vocal is given)")
    p.add_argument("--stems", action="store_true", help="also export one WAV per instrument")
    p.add_argument("--bits", type=int, choices=[16, 24], default=16, help="WAV bit depth")
    p.add_argument("--duck", type=float, default=4.0, help="dB the accompaniment ducks under the vocal (0 = off)")
    p.add_argument("--vocal-gain", type=float, default=0.0, help="vocal gain in dB (0 = untouched)")
    p.add_argument("--soundfont", default="none", help="'none', 'auto' (first .sf2 in ./soundfonts) or a path to a .sf2 file")
    p.add_argument("--shift-ms", type=float, default=0.0, help="nudge the music against the vocal in ms (+ = later)")
    p.add_argument("--list-genres", action="store_true", help="print supported genres and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_genres:
        print("\n".join(["all"] + GENRE_NAMES))
        return 0
    if args.gui or (argv is None and len(sys.argv) == 1):
        from .gui import run_gui
        run_gui()
        return 0

    opts = GenerateOptions(
        genre=args.genre, output_dir=args.output, name=args.name, bars=args.bars, bpm=args.bpm,
        key=args.key, scale=args.scale, seed=args.seed, mode=args.mode, vocal_path=args.vocal,
        lead=None if args.lead is None else args.lead == "on", export_stems=args.stems,
        bits=args.bits, grid_shift_ms=args.shift_ms, soundfont=args.soundfont, duck_db=args.duck, vocal_gain_db=args.vocal_gain,
    )

    def show(p: float, m: str) -> None:
        print(f"[{int(p * 100):3d}%] {m}")

    try:
        res = generate(opts, show)
    except Exception as e:  # friendly error for end users
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    s = res.song
    print(f"\nGenre: {s.genre} | {s.bpm:g} BPM | Key: {s.key_name} | {s.bars} bars (~{s.seconds:.0f}s) | seed {s.seed}")
    if res.vocal:
        v = res.vocal
        print(f"Vocal preserved: {v.seconds:.1f}s, detected {v.bpm or '?'} BPM, key {v.key_name}")
    for label, path in (("MIDI", res.midi_path), ("Audio", res.audio_path),
                        ("Accompaniment", res.accompaniment_path), ("Info", res.info_path)):
        if path:
            print(f"{label}: {path}")
    for p in res.stems:
        print(f"Stem: {p}")
    print(f"Done in {res.seconds_taken:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Editing / analysis commands:  python -m aimusicgen.tools_cli <command> ...
stretch, pitch, chords, convert, separate, mixdown"""

from __future__ import annotations

import argparse
import sys

from . import audio_tools as T
from .audio_io import read_audio


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aimusicgen-tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("stretch", "pitch", "convert"):
        s = sub.add_parser(name)
        s.add_argument("input"); s.add_argument("output")
        if name == "stretch":
            s.add_argument("--rate", type=float, required=True, help="2.0 = twice as fast")
        if name == "pitch":
            s.add_argument("--semitones", type=float, required=True)
        s.add_argument("--sr", type=int, default=44100, choices=[44100, 48000, 96000])
        s.add_argument("--bits", type=int, default=24, choices=[16, 24, 32])
    sub.add_parser("chords").add_argument("input")
    s = sub.add_parser("separate"); s.add_argument("input"); s.add_argument("--out", default="stems")
    s.add_argument("--six", action="store_true"); s.add_argument("--device")
    s = sub.add_parser("mixdown"); s.add_argument("project"); s.add_argument("output")
    a = p.parse_args(argv)
    try:
        if a.cmd in ("stretch", "pitch", "convert"):
            x, sr = read_audio(a.input)
            if a.cmd == "stretch":
                x = T.time_stretch(x, a.rate)
            elif a.cmd == "pitch":
                x = T.pitch_shift(x, a.semitones)
            print("Wrote", T.export_audio(x, sr, a.output, a.sr, a.bits))
        elif a.cmd == "chords":
            x, sr = read_audio(a.input)
            for s0, e0, c in T.detect_chords(x, sr):
                print(f"{s0:7.2f} - {e0:7.2f}  {c}")
        elif a.cmd == "separate":
            from .stems import separate
            for k, v in separate(a.input, a.out, a.six, a.device, lambda f, m: print(m)).items():
                print(k, "->", v)
        elif a.cmd == "mixdown":
            from .project import Project
            print("Wrote", T.export_audio(Project.load(a.project).mixdown(), 44100, a.output))
    except Exception as e:  # readable error instead of a traceback
        print("Error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

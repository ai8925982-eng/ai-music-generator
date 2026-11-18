"""Project files (.aimproj = JSON): generation options, track list with volume/pan/mute/solo, and file paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERSION = 1


@dataclass
class TrackState:
    name: str
    path: str
    volume_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    offset_s: float = 0.0


@dataclass
class Project:
    name: str = "untitled"
    options: dict = field(default_factory=dict)
    tracks: list[TrackState] = field(default_factory=list)
    midi: str = ""
    notes: str = ""

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": VERSION, **asdict(self)}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)          # never leaves a half-written project
        return p

    @staticmethod
    def load(path: str | Path) -> "Project":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Cannot open project: {e}") from e
        if d.get("version", 0) > VERSION:
            raise ValueError("Project was saved by a newer version")
        return Project(name=d.get("name", "untitled"), options=d.get("options", {}),
                       tracks=[TrackState(**t) for t in d.get("tracks", [])], midi=d.get("midi", ""), notes=d.get("notes", ""))

    def mixdown(self, sr: int = 44100):
        """Sum the non-muted tracks (solo wins) with volume / pan / offset -> stereo float32."""
        import numpy as np
        from .audio_io import read_audio, resample
        solos = [t for t in self.tracks if t.solo]
        parts = []
        for t in (solos or [t for t in self.tracks if not t.mute]):
            a, rate = read_audio(t.path)
            a = resample(a, rate, sr)
            a = a if a.shape[1] == 2 else np.repeat(a[:, :1], 2, axis=1)
            g = 10 ** (t.volume_db / 20)
            a = a * np.array([np.sqrt(0.5 * (1 - t.pan)), np.sqrt(0.5 * (1 + t.pan))], dtype=np.float32) * np.sqrt(2) * g
            parts.append((int(t.offset_s * sr), a))
        if not parts:
            return np.zeros((0, 2), np.float32)
        out = np.zeros((max(o + len(a) for o, a in parts), 2), np.float32)
        for o, a in parts:
            out[o:o + len(a)] += a
        return out

"""AI stem separation through Demucs (PyTorch). Demucs is NOT bundled: install `pip install demucs` (needs PyTorch;
a GPU makes it much faster). Models are downloaded on first use. Output: vocals/drums/bass/other (htdemucs) or, with
the 6-source model, also guitar and piano."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

MODELS = {"4": "htdemucs", "6": "htdemucs_6s"}


def available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def separate(audio_path: str, out_dir: str, six: bool = False, device: str | None = None, progress=lambda f, m: None) -> dict[str, Path]:
    if not available():
        raise RuntimeError("Demucs is not installed. Run:  pip install demucs  (PyTorch is installed with it)")
    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(src)
    model = MODELS["6" if six else "4"]
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir), "--filename", "{stem}.{ext}"]
    if device:
        cmd += ["-d", device]
    cmd.append(str(src))
    progress(0.05, f"Separating with {model} (first run downloads the model)...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Demucs failed:\n" + (r.stderr or r.stdout)[-800:])
    folder = Path(out_dir) / model / src.stem
    if not folder.exists():                       # layout of --filename differs between demucs versions
        folder = Path(out_dir) / model
    stems = {p.stem: p for p in sorted(folder.glob("*.wav"))}
    if not stems:
        raise RuntimeError(f"Demucs produced no stems in {folder}")
    progress(1.0, "Done")
    return stems

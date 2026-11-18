"""Entry point for run_gui.bat and the standalone .exe. No arguments -> GUI, arguments -> command line."""

import multiprocessing
import os
import sys
import traceback
from pathlib import Path

if getattr(sys, "frozen", False):
    # windowed exe has no console: make print() harmless
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def _log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / "AIMusicGenerator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _report_crash(exc_text: str) -> None:
    """Write the error to a log file and, for the GUI, show it in a message box."""
    try:
        (_log_dir() / "error.log").write_text(exc_text, encoding="utf-8")
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "AI Music Generator",
            "The program hit an unexpected error.\n\n" + exc_text.strip().splitlines()[-1]
            + f"\n\nDetails were saved to:\n{_log_dir() / 'error.log'}")
        root.destroy()
    except Exception:
        pass


def run() -> int:
    from aimusicgen.cli import main
    return main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        code = run()
    except SystemExit as e:
        raise
    except Exception:
        text = traceback.format_exc()
        if len(sys.argv) == 1:      # GUI launch: no console to read, so show a dialog
            _report_crash(text)
        else:
            print(text, file=sys.stderr)
        code = 1
    raise SystemExit(code)

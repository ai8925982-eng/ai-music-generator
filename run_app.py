"""Entry point used by run_gui.bat and PyInstaller. No args -> GUI, args -> CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aimusicgen.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from multiprocessing import freeze_support
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit_ml.pipeline import main


if __name__ == "__main__":
    freeze_support()
    main()

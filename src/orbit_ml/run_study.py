from __future__ import annotations

import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ['PYTHONPATH'] = str(SRC)


if __name__ == "__main__":
    from orbit_ml.pipeline import main

    freeze_support()
    main()

"""Run AF2 input trimming and renaming from a checked-out repository."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from catcongraph.structure.af2_preprocess import main


if __name__ == "__main__":
    main()

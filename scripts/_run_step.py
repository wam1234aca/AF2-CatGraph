"""Shared step launcher; workflow logic remains in catcongraph."""

from __future__ import annotations

from pathlib import Path
import sys


def run(public_step: str) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from catcongraph.cli import main

    main([f"step{public_step}", *sys.argv[1:]])

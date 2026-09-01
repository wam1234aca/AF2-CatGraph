from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    """Sort embedded numbers numerically (rank2 before rank10)."""
    text = str(value).replace("\\", "/").casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(input_dir: str | Path, patterns: Iterable[str]) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.rglob(pattern))

    unique = sorted({p.resolve() for p in files if p.is_file()}, key=natural_sort_key)
    return unique

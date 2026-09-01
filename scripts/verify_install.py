#!/usr/bin/env python3
"""Verify runtime imports and the external GED executable after installation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODULES = {
    "AF2-CatGraph": "catcongraph",
    "NumPy": "numpy",
    "pandas": "pandas",
    "SciPy": "scipy",
    "Matplotlib": "matplotlib",
    "NetworkX": "networkx",
    "RDKit": "rdkit",
    "OpenPyXL": "openpyxl",
    "PyYAML": "yaml",
    "Biopython": "Bio",
    "Streamlit": "streamlit",
}


RUNTIME_REEXEC_MARKER = "AF2_CATGRAPH_VERIFY_RUNTIME_READY"


def ensure_conda_runtime() -> None:
    """Prefer the active Conda C++ runtime before importing compiled modules."""
    if not sys.platform.startswith("linux") or os.environ.get(RUNTIME_REEXEC_MARKER) == "1":
        return

    library_dir = Path(sys.prefix) / "lib"
    if not (library_dir / "libstdc++.so.6").is_file():
        return

    current = dict(os.environ)
    existing = [
        entry
        for entry in current.get("LD_LIBRARY_PATH", "").split(":")
        if entry and Path(entry) != library_dir
    ]
    desired_library_path = ":".join([str(library_dir), *existing])
    changed = (
        desired_library_path != current.get("LD_LIBRARY_PATH", "")
        or bool(current.get("LD_PRELOAD"))
    )
    if not changed:
        return

    current["LD_LIBRARY_PATH"] = desired_library_path
    current.pop("LD_PRELOAD", None)
    current[RUNTIME_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        current,
    )


def module_version(module: object) -> str:
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    ensure_conda_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ged", default="", help="Path to the compiled ged executable")
    args = parser.parse_args()

    failures: list[str] = []
    print("AF2-CatGraph installation check")
    print(f"Python: {sys.version.split()[0]}")
    for label, name in MODULES.items():
        try:
            module = importlib.import_module(name)
            print(f"{label}: {module_version(module)}")
        except Exception as exc:  # pragma: no cover - executed in user environments
            failures.append(f"{label}: {exc}")

    try:
        from openbabel import openbabel

        print(f"OpenBabel Python: {openbabel.OBReleaseVersion()}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"OpenBabel Python binding: {exc}")

    try:
        print(f"PLIP: {importlib.metadata.version('plip')}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"PLIP metadata: {exc}")
    plip_command = shutil.which("plip")
    if plip_command:
        print(f"PLIP executable: {plip_command}")
    else:
        failures.append("PLIP executable was not found on PATH")

    if args.ged:
        ged = Path(args.ged).expanduser().resolve()
        if not ged.is_file():
            failures.append(f"GED executable does not exist: {ged}")
        else:
            try:
                result = subprocess.run(
                    [str(ged), "-h"], capture_output=True, text=True, timeout=30,
                    check=False,
                )
                if result.returncode not in {0, 1}:
                    failures.append(
                        f"GED help returned {result.returncode}: "
                        f"{(result.stderr or result.stdout).strip()[:300]}"
                    )
                else:
                    print(f"GED executable: {ged}")
            except Exception as exc:  # pragma: no cover
                failures.append(f"GED executable: {exc}")

    if failures:
        print("\nInstallation check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Installation check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

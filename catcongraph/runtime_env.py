"""Linux runtime-library isolation for AF2-CatGraph processes."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping


RUNTIME_REEXEC_MARKER = "AF2_CATGRAPH_RUNTIME_LIBS_READY"
ORIGINAL_LD_LIBRARY_PATH = "AF2_CATGRAPH_ORIGINAL_LD_LIBRARY_PATH"
ORIGINAL_LD_PRELOAD = "AF2_CATGRAPH_ORIGINAL_LD_PRELOAD"


def environment_with_runtime_libraries(
    base: Mapping[str, str] | None = None,
    *,
    prefix: str | Path | None = None,
    platform: str | None = None,
) -> dict[str, str]:
    """Return an environment that searches the selected Conda library first."""
    environment = dict(os.environ if base is None else base)
    active_platform = sys.platform if platform is None else platform
    if not active_platform.startswith("linux"):
        return environment

    separator = ":"
    runtime_prefix = Path(prefix or sys.prefix).expanduser()
    library_dir = runtime_prefix / "lib"
    if not (library_dir / "libstdc++.so.6").exists():
        return environment

    existing = [
        item for item in environment.get("LD_LIBRARY_PATH", "").split(separator)
        if item and Path(item) != library_dir
    ]
    environment["LD_LIBRARY_PATH"] = separator.join([str(library_dir), *existing])
    environment.pop("LD_PRELOAD", None)
    return environment


def environment_without_runtime_libraries(
    base: Mapping[str, str] | None = None,
    *,
    prefix: str | Path | None = None,
    platform: str | None = None,
) -> dict[str, str]:
    """Return a browser-safe environment without AF2-CatGraph runtime injection.

    Compiled scientific packages on older Linux systems may need Conda's newer
    ``libstdc++``.  Firefox must not inherit that override because it can then
    load an incompatible Mesa/WebGL stack.  Restore the environment that
    existed before the CLI re-executed, including values saved by installers
    from earlier AF2-CatGraph releases.
    """

    environment = dict(os.environ if base is None else base)
    active_platform = sys.platform if platform is None else platform
    if not active_platform.startswith("linux"):
        return environment

    saved_library_path = environment.get("AF2_CATGRAPH_SAVED_LD_LIBRARY_PATH")
    saved_preload = environment.get("AF2_CATGRAPH_SAVED_LD_PRELOAD")
    original_library_path = environment.get(ORIGINAL_LD_LIBRARY_PATH)
    original_preload = environment.get(ORIGINAL_LD_PRELOAD)

    if saved_library_path is not None:
        if saved_library_path:
            environment["LD_LIBRARY_PATH"] = saved_library_path
        else:
            environment.pop("LD_LIBRARY_PATH", None)
    elif original_library_path is not None:
        if original_library_path:
            environment["LD_LIBRARY_PATH"] = original_library_path
        else:
            environment.pop("LD_LIBRARY_PATH", None)
    else:
        library_dir = Path(prefix or sys.prefix).expanduser() / "lib"
        remaining = [
            item for item in environment.get("LD_LIBRARY_PATH", "").split(":")
            if item and Path(item) != library_dir
        ]
        if remaining:
            environment["LD_LIBRARY_PATH"] = ":".join(remaining)
        else:
            environment.pop("LD_LIBRARY_PATH", None)

    restored_preload = saved_preload if saved_preload is not None else original_preload
    if restored_preload:
        environment["LD_PRELOAD"] = restored_preload
    else:
        environment.pop("LD_PRELOAD", None)

    for key in (
        RUNTIME_REEXEC_MARKER,
        ORIGINAL_LD_LIBRARY_PATH,
        ORIGINAL_LD_PRELOAD,
        "AF2_CATGRAPH_SAVED_LD_LIBRARY_PATH",
        "AF2_CATGRAPH_SAVED_LD_PRELOAD",
    ):
        environment.pop(key, None)
    return environment


def ensure_current_process_runtime() -> None:
    """Re-execute the CLI before compiled Python modules are loaded."""
    if not sys.platform.startswith("linux"):
        return
    current = dict(os.environ)
    desired = environment_with_runtime_libraries(current, prefix=sys.prefix)
    changed = (
        desired.get("LD_LIBRARY_PATH", "") != current.get("LD_LIBRARY_PATH", "")
        or desired.get("LD_PRELOAD") != current.get("LD_PRELOAD")
    )
    if not changed or current.get(RUNTIME_REEXEC_MARKER) == "1":
        return
    desired[ORIGINAL_LD_LIBRARY_PATH] = current.get("LD_LIBRARY_PATH", "")
    desired[ORIGINAL_LD_PRELOAD] = current.get("LD_PRELOAD", "")
    desired[RUNTIME_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", "catcongraph.cli", *sys.argv[1:]],
        desired,
    )

from pathlib import Path

from catcongraph.gui import _command_runtime_prefix
from catcongraph.runtime_env import environment_with_runtime_libraries


def _fake_linux_prefix(tmp_path: Path) -> Path:
    prefix = tmp_path / "conda-env"
    library_dir = prefix / "lib"
    library_dir.mkdir(parents=True)
    (library_dir / "libstdc++.so.6").write_text("test", encoding="utf-8")
    return prefix


def test_runtime_environment_places_conda_library_first(tmp_path: Path) -> None:
    prefix = _fake_linux_prefix(tmp_path)
    environment = environment_with_runtime_libraries(
        {
            "LD_LIBRARY_PATH": "/lib64:/opt/vendor/lib",
            "LD_PRELOAD": "/lib64/libstdc++.so.6",
        },
        prefix=prefix,
        platform="linux",
    )

    assert environment["LD_LIBRARY_PATH"] == (
        f"{prefix / 'lib'}:/lib64:/opt/vendor/lib"
    )
    assert "LD_PRELOAD" not in environment


def test_runtime_environment_is_unchanged_off_linux(tmp_path: Path) -> None:
    prefix = _fake_linux_prefix(tmp_path)
    original = {"LD_LIBRARY_PATH": "/lib64", "LD_PRELOAD": "example.so"}

    assert environment_with_runtime_libraries(
        original, prefix=prefix, platform="win32"
    ) == original


def test_direct_python_command_selects_its_own_environment(tmp_path: Path) -> None:
    python = tmp_path / "selected-env" / "bin" / "python"

    assert _command_runtime_prefix([str(python), "-m", "catcongraph.cli"]) == (
        tmp_path / "selected-env"
    )

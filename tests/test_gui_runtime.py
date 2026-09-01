from __future__ import annotations

import socket
from pathlib import Path

from catcongraph.gui_runtime import (
    build_identity,
    deployment_revision,
    launch_streamlit_gui,
    port_is_available,
    source_fingerprint,
    streamlit_command,
)
from catcongraph.runtime_env import environment_without_runtime_libraries


def _minimal_repository(tmp_path: Path) -> Path:
    package = tmp_path / "catcongraph"
    static = package / "static"
    static.mkdir(parents=True)
    (package / "gui.py").write_text("print('first')\n", encoding="utf-8")
    (package / "gui_config.py").write_text("VALUE = 1\n", encoding="utf-8")
    (static / "3Dmol-min.js").write_text("window.viewer = 1;\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    return tmp_path


def test_source_fingerprint_changes_with_source_bytes(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    first = source_fingerprint(repository)
    (repository / "catcongraph" / "gui.py").write_text("print('second')\n", encoding="utf-8")
    second = source_fingerprint(repository)
    assert first != second


def test_build_identity_prefers_deployment_revision(tmp_path: Path, monkeypatch) -> None:
    repository = _minimal_repository(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    identity = build_identity(repository, "2.2.4")
    assert identity.version == "2.2.4"
    assert identity.revision == "0123456789ab"
    assert identity.source_root == str(repository.resolve())


def test_deployment_revision_has_stable_precedence() -> None:
    assert deployment_revision({"SOURCE_VERSION": "later", "GITHUB_SHA": "first"}) == "first"


def test_port_probe_detects_an_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not port_is_available("127.0.0.1", port)


def test_streamlit_command_enables_source_watching(tmp_path: Path) -> None:
    command = streamlit_command(tmp_path / "catcongraph" / "gui.py", "127.0.0.1", 8501)
    assert "--server.runOnSave=true" in command
    assert "--server.fileWatcherType=auto" in command
    assert "--server.headless=true" in command
    assert command[command.index("--server.port") + 1] == "8501"


def test_launcher_refuses_to_hide_an_old_server_on_an_occupied_port(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    script = repository / "catcongraph" / "gui.py"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        try:
            launch_streamlit_gui(repository, script, "2.2.4", "127.0.0.1", port)
        except SystemExit as exc:
            assert "No second server was started" in str(exc)
        else:
            raise AssertionError("launcher unexpectedly started a second server")


def test_browser_environment_restores_pre_runtime_values(tmp_path: Path) -> None:
    prefix = tmp_path / "conda-env"
    browser_environment = environment_without_runtime_libraries(
        {
            "LD_LIBRARY_PATH": f"{prefix / 'lib'}:/system/custom/lib",
            "LD_PRELOAD": "/conda/injected.so",
            "AF2_CATGRAPH_ORIGINAL_LD_LIBRARY_PATH": "/system/custom/lib",
            "AF2_CATGRAPH_ORIGINAL_LD_PRELOAD": "",
            "AF2_CATGRAPH_RUNTIME_LIBS_READY": "1",
        },
        prefix=prefix,
        platform="linux",
    )
    assert browser_environment["LD_LIBRARY_PATH"] == "/system/custom/lib"
    assert "LD_PRELOAD" not in browser_environment
    assert "AF2_CATGRAPH_RUNTIME_LIBS_READY" not in browser_environment


def test_browser_environment_supports_previous_installer_hooks(tmp_path: Path) -> None:
    prefix = tmp_path / "conda-env"
    browser_environment = environment_without_runtime_libraries(
        {
            "LD_LIBRARY_PATH": str(prefix / "lib"),
            "AF2_CATGRAPH_SAVED_LD_LIBRARY_PATH": "",
            "AF2_CATGRAPH_SAVED_LD_PRELOAD": "",
        },
        prefix=prefix,
        platform="linux",
    )
    assert "LD_LIBRARY_PATH" not in browser_environment
    assert "LD_PRELOAD" not in browser_environment

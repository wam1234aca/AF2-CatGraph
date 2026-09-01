"""Runtime identity and safe single-instance launching for AF2-CatGraph.

The browser itself is not a reliable indication of which source tree is serving
the Streamlit app: an old process can keep a port open, or an environment can
still point at an older editable installation.  This module makes that state
observable and prevents the launcher from silently starting a second server on
another port.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import secrets
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

from catcongraph.runtime_env import (
    environment_with_runtime_libraries,
    environment_without_runtime_libraries,
)


RUNTIME_DIRECTORY = ".catcongraph"
SERVER_RECORD = "gui-server.json"
REVISION_ENVIRONMENT_KEYS = (
    "CATCONGRAPH_REVISION",
    "GITHUB_SHA",
    "RENDER_GIT_COMMIT",
    "RAILWAY_GIT_COMMIT_SHA",
    "SOURCE_VERSION",
)


@dataclass(frozen=True)
class BuildIdentity:
    """A reproducible description of the Python source serving the UI."""

    version: str
    source_fingerprint: str
    source_root: str
    revision: str | None

    @property
    def label(self) -> str:
        revision = f" | git {self.revision}" if self.revision else ""
        return f"AF2-CatGraph | source {self.source_fingerprint}{revision}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_revision(repository_root: Path) -> str | None:
    """Return the checkout revision when Git metadata is available.

    A source archive does not have a ``.git`` directory, so this is deliberately
    best-effort and never makes starting the GUI depend on Git being installed.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def deployment_revision(
    environment: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
) -> str | None:
    """Get an explicit deployment revision, preferring CI-provided values."""

    env = os.environ if environment is None else environment
    for key in REVISION_ENVIRONMENT_KEYS:
        value = env.get(key, "").strip()
        if value:
            return value[:12]
    return _git_revision(repository_root) if repository_root is not None else None


def _fingerprint_paths(repository_root: Path) -> list[Path]:
    """Return every shipped UI/application source file in deterministic order."""

    package = repository_root / "catcongraph"
    paths = sorted(path for path in package.rglob("*.py") if path.is_file())
    for extra in (repository_root / "pyproject.toml", package / "static" / "3Dmol-min.js"):
        if extra.is_file():
            paths.append(extra)
    return sorted(set(paths), key=lambda path: path.relative_to(repository_root).as_posix())


def source_fingerprint(repository_root: Path) -> str:
    """Hash the exact source bytes used by the Studio, not file timestamps."""

    digest = hashlib.sha256()
    for path in _fingerprint_paths(repository_root):
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def build_identity(repository_root: Path, version: str) -> BuildIdentity:
    repository_root = repository_root.resolve()
    return BuildIdentity(
        version=version,
        source_fingerprint=source_fingerprint(repository_root),
        source_root=str(repository_root),
        revision=deployment_revision(repository_root=repository_root),
    )


def _runtime_dir(repository_root: Path) -> Path:
    return repository_root / RUNTIME_DIRECTORY


def server_record_path(repository_root: Path) -> Path:
    return _runtime_dir(repository_root) / SERVER_RECORD


def read_server_record(repository_root: Path) -> dict[str, Any] | None:
    """Read a launcher record without trusting malformed or stale JSON."""

    path = server_record_path(repository_root)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def _write_server_record(repository_root: Path, record: Mapping[str, Any]) -> None:
    runtime_dir = _runtime_dir(repository_root)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = server_record_path(repository_root)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _remove_own_server_record(repository_root: Path, pid: int) -> None:
    record = read_server_record(repository_root)
    if record and record.get("pid") == pid:
        try:
            server_record_path(repository_root).unlink()
        except FileNotFoundError:
            pass


def _pid_is_running(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_command(pid: int) -> str | None:
    """Read a local process command when the platform exposes it.

    Linux is the deployment target for most Streamlit services.  On other
    systems returning ``None`` simply means ``--replace`` will refuse to kill a
    process it cannot verify.
    """

    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return None


def _record_matches_gui(record: Mapping[str, Any], script_path: Path) -> bool:
    pid = record.get("pid")
    command = _process_command(pid) if isinstance(pid, int) else None
    if command:
        return "streamlit" in command and str(script_path.resolve()) in command
    # Windows does not expose /proc. The launcher-owned record is still scoped
    # to this exact script and PID, so it can safely identify its own server.
    recorded_script = str(record.get("script_path", "")).strip()
    try:
        return Path(recorded_script).resolve() == script_path.resolve()
    except OSError:
        return False


def port_is_available(address: str, port: int) -> bool:
    """Return whether a new TCP listener can bind this exact address and port."""

    try:
        infos = socket.getaddrinfo(address, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Invalid GUI address {address!r}: {exc}") from exc
    for family, socktype, proto, _, sockaddr in infos:
        with socket.socket(family, socktype, proto) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(sockaddr)
            except OSError:
                return False
    return True


def _stop_recorded_server(repository_root: Path, script_path: Path) -> bool:
    """Stop only the server this launcher can positively identify as its own."""

    record = read_server_record(repository_root)
    if not record:
        return False
    pid = record.get("pid")
    if not isinstance(pid, int) or not _pid_is_running(pid):
        _remove_own_server_record(repository_root, pid if isinstance(pid, int) else -1)
        return False
    if not _record_matches_gui(record, script_path):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + 8
    while _pid_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _pid_is_running(pid):
        return False
    _remove_own_server_record(repository_root, pid)
    return True


def streamlit_command(script_path: Path, address: str, port: int) -> list[str]:
    """Use Streamlit's watcher; it reloads code rather than relying on browser cache."""

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path.resolve()),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.runOnSave=true",
        "--server.fileWatcherType=auto",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]


def _browser_url(address: str, port: int) -> str:
    host = "127.0.0.1" if address in {"0.0.0.0", "::", "[::]"} else address
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _wait_until_listening(address: str, port: int, process: subprocess.Popen[Any], timeout: float = 15.0) -> bool:
    host = "127.0.0.1" if address in {"0.0.0.0", "::", "[::]"} else address
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def launch_browser_with_clean_environment(url: str) -> bool:
    """Open the Linux browser without Conda C++/graphics-library overrides."""

    if not sys.platform.startswith("linux"):
        return False
    environment = environment_without_runtime_libraries(prefix=sys.prefix)
    if not (environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")):
        return False

    candidates: list[list[str]] = []
    firefox = shutil.which("firefox", path=environment.get("PATH"))
    if firefox:
        candidates.append([firefox, "--new-tab", url])
    xdg_open = shutil.which("xdg-open", path=environment.get("PATH"))
    if xdg_open:
        candidates.append([xdg_open, url])

    for command in candidates:
        try:
            subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            continue
        return True
    return False


def gui_status(repository_root: Path, version: str, address: str, port: int) -> dict[str, Any]:
    """Return JSON-safe diagnostics for support and deployment checks."""

    identity = build_identity(repository_root, version)
    record = read_server_record(repository_root)
    recorded_pid = record.get("pid") if record else None
    return {
        "build": asdict(identity),
        "build_label": identity.label,
        "address": address,
        "port": port,
        "port_available": port_is_available(address, port),
        "launcher_record": record,
        "recorded_process_running": _pid_is_running(recorded_pid) if isinstance(recorded_pid, int) else False,
    }


def launch_streamlit_gui(
    repository_root: Path,
    script_path: Path,
    version: str,
    address: str,
    port: int,
    *,
    replace: bool = False,
    startup_config: str | Path | None = None,
    open_browser: bool = True,
) -> int:
    """Start one verifiable Streamlit process, never a silent second instance."""

    repository_root = repository_root.resolve()
    script_path = script_path.resolve()
    if not 1 <= port <= 65535:
        raise SystemExit(f"GUI port must be between 1 and 65535, got {port}.")
    if not port_is_available(address, port):
        if not replace or not _stop_recorded_server(repository_root, script_path):
            record = read_server_record(repository_root)
            hint = ""
            if record:
                hint = f" Recorded launcher PID: {record.get('pid')}; build: {record.get('build_label', 'unknown')}."
            raise SystemExit(
                f"Port {address}:{port} is already in use. No second server was started, so the browser cannot "
                f"silently keep showing the old instance.{hint} Stop that server, or rerun with "
                "`af2-catgraph gui --replace` when it was launched by AF2-CatGraph."
            )
        if not port_is_available(address, port):
            raise SystemExit(f"The previous AF2-CatGraph GUI did not stop cleanly; {address}:{port} is still occupied.")

    identity = build_identity(repository_root, version)
    command = streamlit_command(script_path, address, port)
    environment = environment_with_runtime_libraries(prefix=sys.prefix)
    environment["CATCONGRAPH_BUILD_LABEL"] = identity.label
    environment["CATCONGRAPH_SOURCE_FINGERPRINT"] = identity.source_fingerprint
    environment["CATCONGRAPH_GUI_STATE_TOKEN"] = secrets.token_hex(16)
    if startup_config:
        environment["CATCONGRAPH_GUI_CONFIG"] = str(Path(startup_config).expanduser().resolve())
    else:
        environment.pop("CATCONGRAPH_GUI_CONFIG", None)
    process = subprocess.Popen(command, cwd=repository_root, env=environment)
    record = {
        "pid": process.pid,
        "address": address,
        "port": port,
        "script_path": str(script_path),
        "started_at": utc_now(),
        "command": command,
        "build": asdict(identity),
        "build_label": identity.label,
    }
    _write_server_record(repository_root, record)
    try:
        url = _browser_url(address, port)
        if open_browser and _wait_until_listening(address, port, process):
            if not launch_browser_with_clean_environment(url):
                print(f"Open this URL in a browser: {url}", flush=True)
        return process.wait()
    finally:
        _remove_own_server_record(repository_root, process.pid)


def refresh_running_build(repository_root: Path, identity: BuildIdentity) -> None:
    """Keep the diagnostic record current after Streamlit hot-reloads the script."""

    record = read_server_record(repository_root)
    if not record or record.get("pid") != os.getpid():
        return
    record.update({"build": asdict(identity), "build_label": identity.label, "last_reload_at": utc_now()})
    _write_server_record(repository_root, record)

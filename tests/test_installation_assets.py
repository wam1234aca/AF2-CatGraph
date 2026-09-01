import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_recorded_environment_contains_clean_tested_stack() -> None:
    environment = yaml.safe_load(
        (ROOT / "environment.tested.yml").read_text(encoding="utf-8")
    )
    dependencies = [item for item in environment["dependencies"] if isinstance(item, str)]

    for required in (
        "python=3.10.13",
        "numpy=1.26.4",
        "pandas=2.2.3",
        "scipy=1.15.2",
        "rdkit=2024.09.6",
        "openbabel=3.1.1",
        "plip=2.3.1",
    ):
        assert required in dependencies
    assert not any("rdkit-pypi" in item.lower() for item in dependencies)


def test_one_click_installer_uses_tested_environment_and_verifier() -> None:
    installer = (ROOT / "install_af2_catgraph.sh").read_text(encoding="utf-8")

    assert "requirements-conda-tested.txt" in installer
    assert "--override-channels" in installer
    assert 'streamlit==1.59.2' in installer
    assert "pip install --no-deps --editable" in installer
    assert "external/Graph_Edit_Distance" in installer
    assert "git clone" not in installer
    assert (ROOT / "external" / "Graph_Edit_Distance" / "makefile").is_file()
    assert (ROOT / "external" / "Graph_Edit_Distance" / "LICENSE.md").is_file()
    assert "scripts/verify_install.py" in installer
    assert "CONDA_REMOTE_CONNECT_TIMEOUT_SECS" in installer
    assert "CONDA_REMOTE_READ_TIMEOUT_SECS" in installer
    assert "CONDA_REMOTE_MAX_RETRIES" in installer
    assert "AF2_CATGRAPH_CONDA_CREATE_ATTEMPTS" in installer
    assert "ColabFold" in installer
    assert "activate.d" not in installer
    assert "deactivate.d" not in installer


def test_tested_conda_spec_matches_recorded_environment() -> None:
    specifications = set(
        line.strip()
        for line in (ROOT / "requirements-conda-tested.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    assert {
        "python=3.10.13",
        "numpy=1.26.4",
        "rdkit=2024.09.6",
        "openbabel=3.1.1",
        "plip=2.3.1",
    } <= specifications


def test_compatible_conda_spec_includes_linux_cpp_runtime() -> None:
    specifications = set(
        line.strip()
        for line in (ROOT / "requirements-conda-compatible.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    assert "libstdcxx-ng>=14.2" in specifications


def test_direct_verifier_prefers_conda_cpp_runtime(tmp_path: Path, monkeypatch) -> None:
    script = ROOT / "scripts" / "verify_install.py"
    specification = importlib.util.spec_from_file_location(
        "af2_catgraph_verify_install_test", script
    )
    assert specification is not None and specification.loader is not None
    verifier = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(verifier)

    prefix = tmp_path / "conda-env"
    library_dir = prefix / "lib"
    library_dir.mkdir(parents=True)
    (library_dir / "libstdc++.so.6").write_text("test", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_execve(executable, arguments, environment) -> None:
        captured["executable"] = executable
        captured["arguments"] = arguments
        captured["environment"] = environment
        raise RuntimeError("captured verifier re-execution")

    monkeypatch.setattr(verifier.sys, "prefix", str(prefix))
    monkeypatch.setattr(verifier.sys, "platform", "linux")
    monkeypatch.setattr(verifier.os, "execve", fake_execve)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/lib64")
    monkeypatch.setenv("LD_PRELOAD", "/lib64/libstdc++.so.6")
    monkeypatch.delenv(verifier.RUNTIME_REEXEC_MARKER, raising=False)

    try:
        verifier.ensure_conda_runtime()
    except RuntimeError as error:
        assert str(error) == "captured verifier re-execution"
    else:  # pragma: no cover - execve is replaced above
        raise AssertionError("Verifier did not re-execute with the Conda runtime")

    environment = captured["environment"]
    assert environment["LD_LIBRARY_PATH"] == f"{library_dir}:/lib64"
    assert "LD_PRELOAD" not in environment
    assert environment[verifier.RUNTIME_REEXEC_MARKER] == "1"

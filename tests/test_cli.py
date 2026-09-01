from catcongraph import cli


def test_step10_timing_is_forwarded(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call_step(command: str, argv: list[str]) -> None:
        captured["command"] = command
        captured["argv"] = argv

    monkeypatch.setattr(cli, "ensure_current_process_runtime", lambda: None)
    monkeypatch.setattr(cli, "_call_step", fake_call_step)
    cli.main([
        "step10", "--config", "project.yaml", "--timing", "after_clash",
    ])

    assert captured == {
        "command": "step10",
        "argv": ["--config", "project.yaml", "--timing", "after_clash"],
    }

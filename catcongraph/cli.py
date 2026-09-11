from __future__ import annotations

import argparse
import json

from catcongraph.runtime_env import ensure_current_process_runtime


def _call_step(command: str, argv: list[str]) -> None:
    if command == "step10":
        from catcongraph.steps.step10_optional_ligand_trimming import main as run_optional_step10

        run_optional_step10(argv)
        return
    from catcongraph.workflow import run_step

    config = argv[argv.index("--config") + 1]
    public_step = command.removeprefix("step")
    max_structures = int(argv[argv.index("--max-structures") + 1]) if "--max-structures" in argv else None
    run_step(public_step, config=config, max_structures=max_structures, clean=("--clean" in argv))


def main(argv: list[str] | None = None) -> None:
    ensure_current_process_runtime()
    parser = argparse.ArgumentParser(prog="af2-catgraph")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create a minimal project config")
    p_init.add_argument("--target-id", required=True)
    p_init.add_argument("--out", required=True)
    p_init.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run", help="Run a workflow stage")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--stage", choices=["prepare", "interactions", "classify", "visualize", "analysis", "all"], default="all")
    p_run.add_argument("--from-step", default=None)
    p_run.add_argument("--to-step", default=None)
    p_run.add_argument("--max-structures", type=int, default=None)
    p_run.add_argument("--clean", action="store_true")

    p_preflight = sub.add_parser("preflight", help="Validate inputs, executables, and docking compatibility")
    p_preflight.add_argument("--config", required=True)
    p_preflight.add_argument("--max-structures", type=int, default=None)

    p_batch = sub.add_parser("batch", help="Run many enzyme projects from a CSV/YAML manifest")
    p_batch.add_argument("--manifest", required=True)
    p_batch.add_argument("--stage", choices=["prepare", "interactions", "classify", "visualize", "analysis", "all"], default="all")
    p_batch.add_argument("--max-structures", type=int, default=None)
    p_batch.add_argument("--clean", action="store_true")
    p_batch.add_argument("--preflight-only", action="store_true")
    p_batch.add_argument("--stop-on-error", action="store_true")

    p_gui = sub.add_parser("gui", help="Launch the local browser interface (requires AF2-CatGraph[gui])")
    p_gui.add_argument("--address", default="127.0.0.1", help="Bind address, e.g. 0.0.0.0 for a server")
    p_gui.add_argument("--port", type=int, default=8501, help="Browser interface port")
    p_gui.add_argument("--replace", action="store_true", help="Stop the recorded AF2-CatGraph GUI on this port before starting")
    p_gui.add_argument("--fresh", action="store_true", help="Replace the recorded GUI and start with a new, clean UI state")
    p_gui.add_argument("--config", default=None, help="Load this YAML when the new GUI server starts")
    p_gui.add_argument("--check", action="store_true", help="Check GUI runtime status, then exit")
    p_gui.add_argument("--no-browser", action="store_true", help="Start the GUI server without opening a local browser")

    for cmd, help_text in [
        ("step01", "AF2 protein preparation and global pLDDT filtering"),
        ("step02", "Template-guided ligand/cofactor/metal placement"),
        ("step03", "Protein-ligand VDW clash filtering"),
        ("step04", "Catalytic-region pLDDT QC"),
        ("step05", "Run PLIP and build interaction/GED graph files"),
        ("step06", "Four-criterion key-residue screening"),
        ("step07", "GED topology classification"),
        ("step08", "Ligand contact-unit grouping and candidate selection"),
        ("step09", "Class-level visualization and summary"),
        ("step10", "Optional ligand-atom selection/trimming"),
    ]:
        p = sub.add_parser(cmd, help=help_text)
        p.add_argument("--config", required=True)
        if cmd == "step10":
            p.add_argument("--timing", choices=["before_docking", "after_clash"], default=None)
        if cmd in {"step02", "step05", "step06", "step07"}:
            p.add_argument("--max-structures", type=int, default=0)
        if cmd in {"step02", "step05", "step06", "step07", "step08", "step09"}:
            p.add_argument("--clean", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "init":
        from catcongraph.workflow import init_project
        cfg = init_project(args.target_id, args.out, force=args.force)
        print(f"Created project config: {cfg}")
        return

    if args.command == "run":
        from catcongraph.workflow import run_workflow
        run_workflow(
            config=args.config,
            stage=args.stage,
            from_step=args.from_step,
            to_step=args.to_step,
            max_structures=args.max_structures,
            clean=args.clean,
        )
        return

    if args.command == "preflight":
        from catcongraph.preflight import run_preflight
        report = run_preflight(args.config, max_structures=args.max_structures)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["errors"]:
            raise SystemExit(1)
        return

    if args.command == "batch":
        from catcongraph.batch import run_batch
        report = run_batch(
            args.manifest,
            stage=args.stage,
            max_structures=args.max_structures,
            clean=args.clean,
            preflight_only=args.preflight_only,
            continue_on_error=not args.stop_on_error,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["n_failed"]:
            raise SystemExit(1)
        return

    if args.command == "gui":
        from catcongraph.gui import launch_gui, runtime_status
        if args.check:
            print(json.dumps(runtime_status(address=args.address, port=args.port), indent=2, ensure_ascii=False))
            return
        launch_gui(
            address=args.address, port=args.port,
            replace=(args.replace or args.fresh), startup_config=args.config,
            open_browser=not args.no_browser,
        )
        return

    step_argv = ["--config", args.config]
    if getattr(args, "timing", None):
        step_argv += ["--timing", args.timing]
    if hasattr(args, "max_structures") and args.max_structures:
        step_argv += ["--max-structures", str(args.max_structures)]
    if getattr(args, "clean", False):
        step_argv += ["--clean"]
    _call_step(args.command, step_argv)


if __name__ == "__main__":
    main()

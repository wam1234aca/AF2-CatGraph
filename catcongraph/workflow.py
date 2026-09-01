from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Optional

INTERNAL_STEP_MODULES = {
    "01": "catcongraph.steps.step01_af2_prepare",
    "02": "catcongraph.steps.step02_template_guided_docking",
    "03": "catcongraph.steps.step03_clash_filter",
    "05": "catcongraph.steps.step05_catalytic_region_plddt_qc",
    "06": "catcongraph.steps.step06_plip_interaction_graphs",
    "07": "catcongraph.steps.step07_key_residue_screening",
    "08": "catcongraph.steps.step08_ged_classification",
    "09": "catcongraph.steps.step09_chemical_equivalence",
    "10": "catcongraph.steps.step10_class_visualization",
}

# Public workflow numbering. Historical internal module names are implementation
# details and are not exposed as selectable scientific modes.
PUBLIC_TO_INTERNAL = {
    "01": "01", "02": "02", "03": "03",
    "04": "05", "05": "06", "06": "07", "07": "08",
    "08": "09", "09": "10",
}

STAGES = {
    "prepare": ["01", "02", "03", "04"],
    "interactions": ["05"],
    "classify": ["06", "07", "08"],
    "visualize": ["09"],
    "analysis": ["05", "06", "07", "08", "09"],
    "all": ["01", "02", "03", "04", "05", "06", "07", "08", "09"],
}

ORDER = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]


def normalize_step(step: str) -> str:
    s = str(step).lower().replace("step", "").replace("_", "").replace("-", "")
    if s.isdigit():
        return f"{int(s):02d}"
    return s


def select_steps(stage: str, from_step: str | None = None, to_step: str | None = None) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    steps = list(STAGES[stage])
    if from_step:
        start = normalize_step(from_step)
        steps = [s for s in steps if ORDER.index(s) >= ORDER.index(start)]
    if to_step:
        end = normalize_step(to_step)
        steps = [s for s in steps if ORDER.index(s) <= ORDER.index(end)]
    return steps


def run_step(step: str, config: str, max_structures: int | None = None, clean: bool = False) -> None:
    step = normalize_step(step)
    if step not in PUBLIC_TO_INTERNAL:
        raise ValueError(f"Unsupported step: {step}")

    print(f"\n===== AF2-CatGraph Step{step} =====", flush=True)
    internal_step = PUBLIC_TO_INTERNAL[step]

    # Optional ligand-atom selection is outside the numbered main workflow.
    # When enabled, run it immediately before the step that consumes its output
    # so users never need a contradictory extra numbered command.
    if step in {"02", "04"}:
        from catcongraph.config import load_yaml_config
        optional = (load_yaml_config(config).get("step10_optional", {}) or {})
        timing = str(optional.get("timing", "before_docking")).strip().lower()
        required_timing = "before_docking" if step == "02" else "after_clash"
        if bool(optional.get("enabled", False)) and timing == required_timing:
            from catcongraph.steps.step10_optional_ligand_trimming import run_step10_optional
            print(f"Running optional ligand-atom selection ({timing}).", flush=True)
            print(json.dumps(run_step10_optional(config, timing=timing), indent=2, ensure_ascii=False))

    if internal_step == "09":
        from catcongraph.steps.step09_chemical_equivalence import run_step09
        from catcongraph.steps.step09_candidate_export import export_step09_candidates
        report = run_step09(config, clean_output=clean)
        try:
            report["candidate_export"] = export_step09_candidates(config, step09_report=report)
        except Exception as exc:
            report["candidate_export"] = {"status": "failed", "error": str(exc)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    module = importlib.import_module(INTERNAL_STEP_MODULES[internal_step])
    argv = ["--config", config]
    if internal_step in {"02", "06", "07", "08"} and max_structures is not None and max_structures > 0:
        argv += ["--max-structures", str(max_structures)]
    if clean and internal_step in {"02", "06", "07", "08", "10"}:
        argv += ["--clean"]
    module.main(argv)


def run_workflow(
    config: str,
    stage: str = "all",
    from_step: str | None = None,
    to_step: str | None = None,
    max_structures: int | None = None,
    clean: bool = False,
) -> None:
    for step in select_steps(stage, from_step=from_step, to_step=to_step):
        run_step(step, config=config, max_structures=max_structures, clean=clean)


def init_project(target_id: str, out_dir: str | Path, force: bool = False) -> Path:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} already exists and is not empty. Use --force to overwrite config files.")
    out.mkdir(parents=True, exist_ok=True)
    template = f"""# AF2-CatGraph minimal project configuration
# Edit the REQUIRED fields first. Advanced options are documented in
# docs/CONFIG_REFERENCE.md.

config_schema: public_01_09

project:
  target_id: "{target_id}"
  output_root: "results/{target_id}"

inputs:
  # REQUIRED: AF2/ColabFold protein-only PDB files.
  af2_dir: "data/{target_id}/01_af2"

  # REQUIRED for Step02 template-guided ligand placement.
  # For multiple substrates/cofactors/metals, use ligand_pdbs. The list order is
  # the Step02 docking order: ligand 1, then ligand 2 into the resulting complex, etc.
  template_protein_pdb: "data/{target_id}/02_templates/template_protein.pdb"
  ligand_pdb: "data/{target_id}/02_templates/ligand.pdb"
  # ligand_pdbs:
  #   - "data/{target_id}/02_templates/substrate.pdb"
  #   - "data/{target_id}/02_templates/cofactor.pdb"
  #   - "data/{target_id}/02_templates/metal.pdb"

step01:
  global_plddt_min: 80.0

step02:
  output_naming: "short_pdb_rank"
  multi_ligand_mode: "sequential"
  input_source: "step01"
  alignment_mode: "auto"
  general_alignment:
    min_sequence_identity: 0.70
    min_alignment_coverage: 0.70
    min_matched_ca: 30

step03:
  clash:
    clash_max: 10
    delta: 0.5

step04:
  catalytic_region:
    # List all non-protein residue names that define the catalytic region.
    ligand_resnames: ["LIG"]
    distance_cutoff: 6.0
    mean_plddt_min: 90.0
    quantile10_plddt_min: 80.0

step05:
  plip_executable: "plip"
  run_plip: true
  save_intermediate_files: true

step06:
  conservation_file: "data/{target_id}/07_conservation/{target_id}_consurf.zip"
  ligand:
    include_resnames: ["LIG"]
  thresholds:
    consurf_grade_min: 8
    residue_ligand_distance_max: 3.0
    occurrence_frequency_min: 0.80
    require_graph_occurrence: true
  node_grouping:
    enabled: true
    contact_definition: "graph"
    required_residue_source: "auto"

step07:
  ged:
    executable: "/path/to/Graph_Edit_Distance/ged"
    threshold: 1
    max_workers: 2

step08:
  enabled: true
  canonicalization:
    # universal_rules: automatically apply the 12 Supplementary Table S4 rules.
    # conservative_manual: keep atom-level labels unless manual units are supplied.
    # guided_review remains available for backward-compatible review workflows.
    contact_unit_mode: "universal_rules"
    equivalence_mode: "standard"
    use_builtin_ligand_units: false
    manual_ligand_units: {{}}
    recommendations:
      enabled: true
      require_acceptance: false
      accepted_recommendation_ids: []
      rejected_recommendation_ids: []
    auto_detect_ligand_units: true

step09:
  plot:
    formats: ["svg", "png"]
  layout:
    # Keep this in exactly the same order as step04.catalytic_region.ligand_resnames.
    entity_order: ["LIG"]
    entity_spacing: 6.2

step10_optional:
  enabled: false
  timing: before_docking
  atom_selection:
    rules: []
"""
    cfg_path = out / "config.yaml"
    if force or not cfg_path.exists():
        cfg_path.write_text(template, encoding="utf-8")
    return cfg_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="catcongraph.workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run an AF2-CatGraph workflow stage")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--stage", choices=sorted(STAGES), default="all")
    p_run.add_argument("--from-step", default=None)
    p_run.add_argument("--to-step", default=None)
    p_run.add_argument("--max-structures", type=int, default=None)
    p_run.add_argument("--clean", action="store_true")

    p_init = sub.add_parser("init", help="Create a minimal project config")
    p_init.add_argument("--target-id", required=True)
    p_init.add_argument("--out", required=True)
    p_init.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_workflow(
            config=args.config,
            stage=args.stage,
            from_step=args.from_step,
            to_step=args.to_step,
            max_structures=args.max_structures,
            clean=args.clean,
        )
    elif args.command == "init":
        cfg = init_project(args.target_id, args.out, force=args.force)
        print(f"Created project config: {cfg}")


if __name__ == "__main__":
    main()

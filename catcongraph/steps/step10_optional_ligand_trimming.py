"""Optional ligand-atom selection outside the main Step01--09 flow.

Two explicit timings are supported:

``before_docking``
    Trim template ligand/cofactor PDBs, then Step02 automatically consumes the
    generated copies.
``after_clash``
    Trim only complexes that passed Step03, then continue with public Step04.

Both paths preserve a trace table and valid remapped PDB CONECT records.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from catcongraph.config import load_yaml_config
from catcongraph.steps.step02b_ligand_atom_trim import run_step02b
from catcongraph.steps.step04_ligand_atom_trimming import trim_one_complex


def _target_id(config: dict[str, Any]) -> str:
    return str((config.get("project", {}) or {}).get("target_id", "target"))


def _output_root(config: dict[str, Any]) -> Path:
    project = config.get("project", {}) or {}
    return Path(str(project.get("output_root", f"results/{_target_id(config)}")))


def _ligand_paths(config: dict[str, Any]) -> list[Path]:
    inputs = config.get("inputs", {}) or {}
    raw = inputs.get("ligand_pdbs") or inputs.get("ligand_pdb") or []
    values = raw if isinstance(raw, list) else [raw]
    paths = [Path(str(item)) for item in values if str(item).strip()]
    if not paths:
        raise ValueError("Step10 before_docking needs inputs.ligand_pdb or inputs.ligand_pdbs.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Template ligand PDB(s) not found: " + ", ".join(missing))
    return paths


def _run_before_docking(config: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    target_id = _target_id(config)
    out_dir = _output_root(config) / str(cfg.get("output_subdir", "10_optional_ligand_atom_trimming"))
    ligand_dir = out_dir / "template_ligands"
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    for directory in (ligand_dir, tables_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    atom_cfg = cfg.get("atom_selection", {}) or {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for order, source in enumerate(_ligand_paths(config), start=1):
        output = ligand_dir / f"{order:02d}_{source.stem}_trimmed{source.suffix}"
        try:
            summary, trace = trim_one_complex(source, output, f"template_ligand_{order:02d}", atom_cfg)
            pd.DataFrame([item.__dict__ for item in trace]).to_csv(
                tables_dir / f"{target_id}_step10_template_ligand_{order:02d}_atom_trace.csv", index=False
            )
            rows.append({
                "order": order,
                "source_ligand_pdb_path": str(source.resolve()),
                "trimmed_ligand_pdb_path": str(output.resolve()),
                "kept_atom_lines": summary["kept_atom_lines"],
                "removed_atom_lines": summary["removed_atom_lines"],
                "status": "ok",
                "error_message": "",
            })
        except Exception as exc:
            errors.append({"order": order, "source_ligand_pdb_path": str(source), "status": "failed", "error_message": str(exc)})

    index = tables_dir / f"{target_id}_step10_pre_docking_ligand_index.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    failed = tables_dir / f"{target_id}_step10_pre_docking_failures.csv"
    pd.DataFrame(errors).to_csv(failed, index=False)
    report = {
        "target_id": target_id,
        "optional_step": "ligand_atom_selection",
        "timing": "before_docking",
        "status": "completed" if not errors else "failed",
        "output_dir": str(out_dir),
        "n_input_ligands": len(rows) + len(errors),
        "n_trimmed_ligands": len(rows),
        "n_failed": len(errors),
        "tables": {"pre_docking_ligand_index": str(index), "failures": str(failed)},
        "next_action": "Run Step02. It will automatically use this ordered trimmed-ligand index.",
    }
    (reports_dir / f"{target_id}_step10_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise RuntimeError("Step10 before_docking failed for one or more ligands; inspect " + str(failed))
    return report


def _run_after_clash(config: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    target_id = _target_id(config)
    output_root = _output_root(config)
    clash_index = output_root / "03_clash_filter" / "tables" / f"{target_id}_step03_passed_complex_index.csv"
    if not clash_index.exists():
        raise FileNotFoundError(
            "Step10 after_clash needs the Step03 passed-complex index. Run Step03 first: " + str(clash_index)
        )

    # Reuse the hardened batch trimmer, but supply an explicit Step03 index and
    # a Step10 output directory so this optional operation never becomes a
    # hidden part of the main pipeline.
    local = deepcopy(config)
    local_cfg = deepcopy(cfg)
    local_cfg.update({
        "enabled": True,
        "input_complex_table": str(clash_index),
        "input_complex_column": "complex_pdb_path",
        "output_subdir": str(cfg.get("output_subdir", "10_optional_ligand_atom_trimming")),
    })
    local["step02b"] = local_cfg
    report = run_step02b(local)
    report.update({
        "optional_step": "ligand_atom_selection",
        "timing": "after_clash",
        "next_action": "Run public Step04 (catalytic-region pLDDT QC); it will prefer this Step10 trimmed-complex index.",
    })
    return report


def run_step10_optional(config_path: str | Path, timing: str | None = None) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    cfg = config.get("step10_optional", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return {
            "target_id": _target_id(config), "optional_step": "ligand_atom_selection", "status": "skipped",
            "message": "step10_optional.enabled is false. Optional ligand-atom selection was not run.",
        }
    selected_timing = str(timing or cfg.get("timing", "before_docking")).strip().lower()
    if selected_timing not in {"before_docking", "after_clash"}:
        raise ValueError("step10_optional.timing must be before_docking or after_clash")
    return _run_before_docking(config, cfg) if selected_timing == "before_docking" else _run_after_clash(config, cfg)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optional ligand-atom selection outside Step01--09.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timing", choices=["before_docking", "after_clash"], default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run_step10_optional(args.config, timing=args.timing), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

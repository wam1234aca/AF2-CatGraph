#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step05: catalytic-region pLDDT quality control.

This QC is intended to run after Step03 clash filtering and, optionally, after
Step04 ligand/core trimming.  It filters complexes whose local catalytic region
around the selected substrate atoms is not confidently predicted.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import argparse
import json

import pandas as pd
import yaml
from catcongraph.config import load_yaml_config

from catcongraph.config import public_output_subdir
from catcongraph.qc.catalytic_plddt import compute_catalytic_region_plddt


def load_config(config_path: Path | str) -> dict[str, Any]:
    return load_yaml_config(config_path)


def _target_id(config: dict[str, Any]) -> str:
    return str(config.get("project", {}).get("target_id", "target"))


def _output_root(config: dict[str, Any]) -> Path:
    project = config.get("project", {})
    return Path(project.get("output_root", f"results/{_target_id(config)}"))


def resolve_step05_input_table(config: dict[str, Any]) -> Path:
    """Resolve the complex index table used for catalytic-region pLDDT QC.

    Auto mode prefers Step04 only when ``step04.enabled: true`` and the Step04
    output index already exists.  Otherwise it uses Step03-passed complexes.
    This avoids accidentally consuming stale Step04 outputs when the optional
    trimming step is disabled.
    """
    target_id = _target_id(config)
    root = _output_root(config)
    step05 = config.get("step05", {}) or {}
    requested = step05.get("input_complex_table", "auto")

    if requested and requested != "auto":
        return Path(requested)

    candidates: list[Path] = []

    prefer_step04 = bool(step05.get("prefer_step04_if_enabled", True)) and bool(
        (config.get("step04", {}) or {}).get("enabled", False)
    )
    if prefer_step04:
        candidates.append(root / "04_ligand_atom_trimming" / "tables" / f"{target_id}_step04_trimmed_complex_index.csv")

    # Public Step10 can instead trim only Step03-passed complexes.  It is not
    # part of the mandatory workflow, but when explicitly run it must feed the
    # following pLDDT QC rather than leaving a stale untrimmed branch active.
    optional_trim = config.get("step10_optional", {}) or {}
    if bool(optional_trim.get("enabled", False)) and str(optional_trim.get("timing", "")).lower() == "after_clash":
        optional_dir = root / str(optional_trim.get("output_subdir", "10_optional_ligand_atom_trimming"))
        candidates.append(optional_dir / "tables" / f"{target_id}_step02b_trimmed_complex_index.csv")

    candidates.extend(
        [
            root / "03_clash_filter" / "tables" / f"{target_id}_step03_passed_complex_index.csv",
            root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_complex_index.csv",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find Step05 input complex table. Run Step03 first, or set step05.input_complex_table. "
        f"Checked: {', '.join(str(x) for x in candidates)}"
    )


def _infer_complex_path_column(df: pd.DataFrame, configured: str = "auto") -> str:
    if configured and configured != "auto":
        if configured not in df.columns:
            raise KeyError(f"Configured complex path column '{configured}' not found in input table.")
        return configured

    candidates = [
        "complex_pdb_path",
        "trimmed_complex_pdb_path",
        "trimmed_complex_pdb",
        "output_complex_pdb",
        "complex_path",
        "pdb_path",
        "path",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot infer complex PDB path column. Available columns: {list(df.columns)}")


def _resolve_existing_path(path_text: str, config_path: Path) -> Path:
    p = Path(str(path_text)).expanduser()
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, config_path.parent / p]
    if config_path.parent.parent.name == "projects":
        candidates.append(config_path.parent.parent.parent / p)
    candidates.append(p)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / p).resolve()


def _infer_plddt_pdb_path(row: pd.Series, configured: str, complex_path: Path, config_path: Path) -> Optional[Path]:
    """Find the AF2/ColabFold protein PDB used as the pLDDT source.

    The docked complex defines which residues are within the ligand pocket, but
    pLDDT should come from the corresponding AF2 model when that path is present
    in the upstream index table.  If no such column exists, Step05 safely falls
    back to the complex PDB for backward compatibility.
    """
    if configured and configured != "auto":
        if configured not in row.index:
            raise KeyError(f"Configured pLDDT PDB column '{configured}' not found in input table.")
        value = row[configured]
        if pd.notna(value) and str(value).strip():
            return _resolve_existing_path(str(value), config_path)
        return None

    candidates = [
        "af2_pdb_path",
        "af2_model_pdb_path",
        "standardized_pdb_path",
        "protein_pdb_path",
        "input_protein_pdb_path",
        "source_pdb_path",
        "model_pdb_path",
    ]
    for col in candidates:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            candidate = _resolve_existing_path(str(row[col]), config_path)
            if candidate.exists() and candidate.resolve() != complex_path.resolve():
                return candidate
    return None


def _infer_structure_id(row: pd.Series, pdb_path: Path) -> str:
    for col in ["standard_structure_id", "structure_id", "id", "model_id"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    stem = pdb_path.stem
    for suffix in ["_trimmed_complex", "_complex", "_trimmed"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def run_step05(config_path: Path | str) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    target_id = _target_id(config)
    root = _output_root(config)
    step05 = config.get("step05", {}) or {}

    enabled = bool(step05.get("enabled", True))
    if not enabled:
        return {
            "target_id": target_id,
            "status": "skipped",
            "message": "step05.enabled is false. Catalytic-region pLDDT QC was not run.",
        }

    output_dir = root / public_output_subdir(
        step05.get("output_subdir"), "04_catalytic_region_plddt_qc"
    )
    tables_dir = output_dir / "tables"
    lists_dir = output_dir / "lists"
    reports_dir = output_dir / "reports"
    for directory in (tables_dir, lists_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_table = resolve_step05_input_table(config)
    df = pd.read_csv(input_table)
    path_col = _infer_complex_path_column(df, step05.get("input_complex_column", "auto"))

    # Keep the user-facing config readable by accepting both the concise
    # top-level keys and the older nested ``step05.catalytic_region`` section.
    region_cfg = step05.get("catalytic_region", {}) or {}
    radius = float(step05.get("radius_angstrom", region_cfg.get("distance_cutoff", region_cfg.get("radius_angstrom", 6.0))))
    mean_min = float(step05.get("mean_plddt_min", region_cfg.get("mean_plddt_min", 90.0)))
    p10_min = float(
        step05.get(
            "p10_plddt_min",
            region_cfg.get("quantile10_plddt_min", region_cfg.get("p10_plddt_min", 80.0)),
        )
    )
    residue_atom = str(step05.get("residue_plddt_atom", region_cfg.get("plddt_atom", "CA")))
    strict_gt = bool(step05.get("strict_greater_than", region_cfg.get("strict_greater_than", True)))
    ligand_selection = dict(step05.get("ligand_selection", {}) or {})
    if "include_resnames" not in ligand_selection and region_cfg.get("ligand_resnames"):
        ligand_selection["include_resnames"] = region_cfg.get("ligand_resnames")
    if "exclude_resnames" not in ligand_selection and region_cfg.get("exclude_resnames"):
        ligand_selection["exclude_resnames"] = region_cfg.get("exclude_resnames")
    ignore_h = bool(step05.get("ignore_hydrogen_for_distance", region_cfg.get("ignore_hydrogen_for_distance", True)))

    summary_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        pdb_path = _resolve_existing_path(str(row[path_col]), config_path)
        sid = _infer_structure_id(row, pdb_path)
        base = row.to_dict()

        try:
            if not pdb_path.exists():
                raise FileNotFoundError(f"Complex PDB not found: {pdb_path}")
            plddt_path = _infer_plddt_pdb_path(
                row,
                str(step05.get("plddt_pdb_column", "auto")),
                pdb_path,
                config_path,
            )

            result, residues = compute_catalytic_region_plddt(
                pdb_path,
                standard_structure_id=sid,
                radius_angstrom=radius,
                mean_plddt_min=mean_min,
                p10_plddt_min=p10_min,
                residue_plddt_atom=residue_atom,
                ligand_selection=ligand_selection,
                ignore_hydrogen_for_distance=ignore_h,
                strict_greater_than=strict_gt,
                plddt_pdb_path=plddt_path,
            )
            base.update(asdict(result))
            summary_rows.append(base)
            residue_rows.extend(residues)
        except Exception as exc:
            base.update(
                {
                    "complex_pdb_path": str(pdb_path),
                    "plddt_pdb_path": "",
                    "standard_structure_id": sid,
                    "radius_angstrom": radius,
                    "ligand_atom_count": None,
                    "protein_atom_count": None,
                    "catalytic_residue_count": 0,
                    "catalytic_mean_plddt": None,
                    "catalytic_median_plddt": None,
                    "catalytic_p10_plddt": None,
                    "catalytic_min_plddt": None,
                    "catalytic_max_plddt": None,
                    "mean_plddt_min": mean_min,
                    "p10_plddt_min": p10_min,
                    "pass_catalytic_region_plddt": False,
                    "status": "runtime_failed",
                    "error_message": str(exc),
                }
            )
            summary_rows.append(base)

    summary = pd.DataFrame(summary_rows)
    if "pass_catalytic_region_plddt" not in summary.columns:
        summary["pass_catalytic_region_plddt"] = False

    passed = summary[summary["pass_catalytic_region_plddt"] == True].copy()
    failed = summary[summary["pass_catalytic_region_plddt"] != True].copy()
    residues_df = pd.DataFrame(residue_rows)

    summary_path = tables_dir / f"{target_id}_step04_catalytic_region_plddt_summary.csv"
    residues_path = tables_dir / f"{target_id}_step04_catalytic_region_residues.csv"
    passed_path = tables_dir / f"{target_id}_step04_passed_complex_index.csv"
    failed_path = tables_dir / f"{target_id}_step04_failed_complex_index.csv"

    summary.to_csv(summary_path, index=False)
    residues_df.to_csv(residues_path, index=False)
    passed.to_csv(passed_path, index=False)
    failed.to_csv(failed_path, index=False)

    (lists_dir / f"{target_id}_step04_passed_structure_ids.txt").write_text(
        "\n".join(passed["standard_structure_id"].astype(str).tolist()) + ("\n" if len(passed) else ""),
        encoding="utf-8",
    )
    (lists_dir / f"{target_id}_step04_failed_structure_ids.txt").write_text(
        "\n".join(failed["standard_structure_id"].astype(str).tolist()) + ("\n" if len(failed) else ""),
        encoding="utf-8",
    )

    comparator = ">" if strict_gt else ">="
    report = {
        "target_id": target_id,
        "status": "success",
        "config_path": str(config_path),
        "input_complex_table": str(input_table),
        "input_complex_column": path_col,
        "plddt_pdb_column": str(step05.get("plddt_pdb_column", "auto")),
        "output_dir": str(output_dir),
        "n_input_complexes": int(len(df)),
        "n_passed": int(len(passed)),
        "n_failed": int(len(failed)),
        "rule": f"mean catalytic-region pLDDT {comparator} {mean_min:g} and p10 catalytic-region pLDDT {comparator} {p10_min:g}",
        "radius_angstrom": radius,
        "residue_plddt_atom": residue_atom,
        "ligand_selection": ligand_selection,
        "files": {
            "summary": str(summary_path),
            "residues": str(residues_path),
            "passed": str(passed_path),
            "failed": str(failed_path),
        },
    }

    (reports_dir / f"{target_id}_step04_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# Step05 catalytic-region pLDDT QC summary: {target_id}",
        "",
        f"- Input complexes: {len(df)}",
        f"- Passed complexes: {len(passed)}",
        f"- Failed complexes: {len(failed)}",
        f"- Catalytic region: protein residues with any atom within `{radius:g} Å` of selected ligand atoms",
        f"- pLDDT per residue: `{residue_atom}` atom, with residue atom-mean fallback if missing",
        "- pLDDT source: AF2/ColabFold PDB from `step05.plddt_pdb_column` or upstream AF2 path columns when available; complex PDB fallback otherwise.",
        f"- Pass rule: `mean pLDDT {comparator} {mean_min:g}` and `10th percentile pLDDT {comparator} {p10_min:g}`",
        "",
        "## Output files",
        "",
        f"- `{summary_path}`",
        f"- `{residues_path}`",
        f"- `{passed_path}`",
        f"- `{failed_path}`",
        "",
    ]
    (reports_dir / f"{target_id}_step04_summary.md").write_text("\n".join(md), encoding="utf-8")

    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CatConGraph Step05: catalytic-region pLDDT QC.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    args = parser.parse_args(argv)

    report = run_step05(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

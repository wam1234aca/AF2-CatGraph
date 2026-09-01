#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step03: protein-ligand clash detection and filtering."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import argparse
import json
import pandas as pd
from catcongraph.config import load_yaml_config

from catcongraph.qc.clash import compute_clashes_for_complex, normalize_clash_method


def load_config(config_path: Path | str) -> dict:
    return load_yaml_config(config_path)


def _target_id(config: dict) -> str:
    return str(config.get("project", {}).get("target_id", "target"))


def _output_root(config: dict) -> Path:
    project = config.get("project", {})
    return Path(project.get("output_root", f"results/{_target_id(config)}"))


def resolve_step03_input_table(config: dict) -> Path:
    """Find Step02 complex index by default."""
    target_id = _target_id(config)
    step03 = config.get("step03", {})
    requested = step03.get("input_complex_table", "auto")

    if requested and requested != "auto":
        return Path(requested)

    root = _output_root(config)
    candidates = [
        root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_complex_index.csv",
        root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_template_guided_docking_complex_index.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find Step02 complex index table. Set step03.input_complex_table in config.yaml. "
        f"Checked: {', '.join(str(x) for x in candidates)}"
    )


def _infer_complex_path_column(df: pd.DataFrame, configured: str = "auto") -> str:
    if configured and configured != "auto":
        if configured not in df.columns:
            raise KeyError(f"Configured complex path column '{configured}' not found in input table.")
        return configured

    candidates = [
        "complex_pdb_path",
        "output_complex_pdb",
        "complex_path",
        "pdb_path",
        "path",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot infer complex PDB path column. Available columns: {list(df.columns)}")


def _infer_structure_id(row: pd.Series, pdb_path: Path) -> str:
    for col in ["standard_structure_id", "structure_id", "id", "model_id"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    stem = pdb_path.stem
    for suffix in ["_complex", "_trimmed_complex", "_trimmed"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _clash_param(clash_cfg: dict, step03: dict, name: str, aliases: list[str], default):
    """Read Step03 clash options with a few clear backward-compatible aliases."""
    if name in clash_cfg:
        return clash_cfg[name]
    for alias in aliases:
        if alias in clash_cfg:
            return clash_cfg[alias]
        if alias in step03:
            return step03[alias]
    return default


def run_step03(config_path: Path | str) -> Dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    target_id = _target_id(config)
    root = _output_root(config)

    step03 = config.get("step03", {})
    clash_cfg = step03.get("clash", {}) or {}
    ligand_cfg = step03.get("ligand_selection", {}) or {}
    clash_max = int(_clash_param(clash_cfg, step03, "clash_max", ["max_clashes", "max_clash_count"], 10))
    delta = float(_clash_param(clash_cfg, step03, "delta", ["vdw_delta"], 0.5))
    clash_method = normalize_clash_method(
        _clash_param(
            clash_cfg,
            step03,
            "method",
            ["mode", "algorithm", "compatibility_mode"],
            "collision_py_exact",
        )
    )
    neighbor_search_cutoff = float(
        _clash_param(
            clash_cfg,
            step03,
            "neighbor_search_cutoff",
            ["cutoff", "search_cutoff"],
            5.0,
        )
    )

    output_dir = root / step03.get("output_subdir", "03_clash_filter")
    tables_dir = output_dir / "tables"
    lists_dir = output_dir / "lists"
    reports_dir = output_dir / "reports"
    for d in [tables_dir, lists_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    input_table = resolve_step03_input_table(config)
    df = pd.read_csv(input_table)
    path_col = _infer_complex_path_column(df, step03.get("input_complex_column", "auto"))

    summary_rows = []
    pair_rows = []
    failed_runtime_rows = []

    for _, row in df.iterrows():
        pdb_path = Path(str(row[path_col]))
        structure_id = _infer_structure_id(row, pdb_path)

        try:
            if not pdb_path.exists():
                raise FileNotFoundError(f"Complex PDB not found: {pdb_path}")

            result, pairs = compute_clashes_for_complex(
                complex_pdb_path=pdb_path,
                standard_structure_id=structure_id,
                clash_max=clash_max,
                delta=delta,
                include_hetatm_as_ligand=bool(ligand_cfg.get("include_hetatm", True)),
                ligand_resnames=ligand_cfg.get("include_resnames", None),
                exclude_resnames=ligand_cfg.get("exclude_resnames", ["HOH", "WAT"]),
                ignore_hydrogen=bool(clash_cfg.get("ignore_hydrogen", False)),
                ignore_water=bool(clash_cfg.get("ignore_water", True)),
                method=clash_method,
                neighbor_search_cutoff=neighbor_search_cutoff,
            )

            base = row.to_dict()
            base.update(asdict(result))
            base["clash_method"] = clash_method
            base["neighbor_search_cutoff"] = neighbor_search_cutoff
            summary_rows.append(base)
            pair_rows.extend(asdict(p) for p in pairs)

        except Exception as exc:
            base = row.to_dict()
            base.update(
                {
                    "complex_pdb_path": str(pdb_path),
                    "standard_structure_id": structure_id,
                    "protein_atom_count": 0,
                    "ligand_atom_count": 0,
                    "clash_count": None,
                    "clash_score": None,
                    "min_distance": None,
                    "min_clash_distance": None,
                    "clash_max": clash_max,
                    "clash_method": clash_method,
                    "neighbor_search_cutoff": neighbor_search_cutoff,
                    "pass_clash": False,
                    "status": "runtime_failed",
                    "error_message": str(exc),
                }
            )
            summary_rows.append(base)
            failed_runtime_rows.append(base)

    summary = pd.DataFrame(summary_rows)

    if "pass_clash" not in summary.columns:
        summary["pass_clash"] = False

    passed = summary[summary["pass_clash"] == True].copy()
    failed = summary[summary["pass_clash"] != True].copy()

    summary_path = tables_dir / f"{target_id}_step03_clash_summary.csv"
    pairs_path = tables_dir / f"{target_id}_step03_clash_pairs.csv"
    passed_path = tables_dir / f"{target_id}_step03_passed_complex_index.csv"
    failed_path = tables_dir / f"{target_id}_step03_failed_complex_index.csv"

    summary.to_csv(summary_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pairs_path, index=False)
    passed.to_csv(passed_path, index=False)
    failed.to_csv(failed_path, index=False)

    (lists_dir / f"{target_id}_step03_passed_structure_ids.txt").write_text(
        "\n".join(passed["standard_structure_id"].astype(str).tolist()) + ("\n" if len(passed) else ""),
        encoding="utf-8",
    )
    (lists_dir / f"{target_id}_step03_failed_structure_ids.txt").write_text(
        "\n".join(failed["standard_structure_id"].astype(str).tolist()) + ("\n" if len(failed) else ""),
        encoding="utf-8",
    )

    report = {
        "target_id": target_id,
        "config_path": str(config_path),
        "input_complex_table": str(input_table),
        "input_complex_column": path_col,
        "output_dir": str(output_dir),
        "n_input_complexes": int(len(df)),
        "n_passed": int(len(passed)),
        "n_failed": int(len(failed)),
        "clash_max": clash_max,
        "delta": delta,
        "clash_method": clash_method,
        "neighbor_search_cutoff": neighbor_search_cutoff,
        "collision_py_exact_semantics": (
            {
                "parser": "Bio.PDB.PDBParser(QUIET=True)",
                "protein_side": "blank hetfield plus water hetfield W",
                "ligand_side": "nonblank non-water hetfield",
                "hydrogens_included": True,
                "legacy_radius_fallback_angstrom": 1.5,
                "strict_pair_rule": "distance < protein_radius + ligand_radius - delta",
                "pass_rule": "clash_count <= clash_max",
                "ligand_selection_config_ignored": True,
                "ignore_hydrogen_config_ignored": True,
                "ignore_water_config_ignored": True,
            }
            if clash_method == "collision_py_exact"
            else None
        ),
        "ignore_hydrogen": bool(clash_cfg.get("ignore_hydrogen", False)),
        "ignore_water": bool(clash_cfg.get("ignore_water", True)),
        "files": {
            "summary": str(summary_path),
            "pairs": str(pairs_path),
            "passed": str(passed_path),
            "failed": str(failed_path),
        },
    }

    (reports_dir / f"{target_id}_step03_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# Step03 clash filter summary: {target_id}",
        "",
        f"- Input complexes: {len(df)}",
        f"- Passed complexes: {len(passed)}",
        f"- Failed complexes: {len(failed)}",
        f"- Clash method: `{clash_method}`",
        f"- Neighbor-search cutoff: `{neighbor_search_cutoff} Å`",
        f"- Clash pass rule: `clash_count <= {clash_max}`",
        f"- VDW clash rule: `distance < protein_vdw + ligand_vdw - {delta}`",
        "",
        "## Output files",
        "",
        f"- `{summary_path}`",
        f"- `{pairs_path}`",
        f"- `{passed_path}`",
        f"- `{failed_path}`",
        "",
    ]
    (reports_dir / f"{target_id}_step03_summary.md").write_text("\n".join(md), encoding="utf-8")

    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CatConGraph Step03: protein-ligand clash detection and filtering.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    args = parser.parse_args(argv)

    report = run_step03(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

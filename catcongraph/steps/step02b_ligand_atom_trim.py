from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from catcongraph.config import get_nested, load_yaml_config, require_nested
from catcongraph.io.paths import ensure_dir, list_files
from catcongraph.ligand.atom_trimming import trim_complex_pdb


def _auto_step02_complex_index(output_root: Path, target_id: str) -> Path:
    return (
        output_root
        / "02_template_guided_docking"
        / "tables"
        / f"{target_id}_step02_complex_index.csv"
    )


def _records_from_complex_table(table_path: Path, complex_column: str) -> pd.DataFrame:
    df = pd.read_csv(table_path)
    if df.empty:
        raise ValueError(f"Input complex table is empty: {table_path}")
    if "standard_structure_id" not in df.columns:
        raise ValueError(f"Missing required column 'standard_structure_id' in {table_path}")
    if complex_column not in df.columns:
        raise ValueError(
            f"Missing complex PDB column {complex_column!r} in {table_path}. "
            f"Available columns: {list(df.columns)}"
        )
    # Preserve upstream columns (especially Step02's ligand-order manifest)
    # through post-clash trimming.  Downstream QC/PLIP then has an auditable
    # record of which normal or pre-docking-trimmed ligand order produced each
    # analysed complex.
    out = df.copy()
    out["source_complex_pdb"] = out[complex_column]
    return out


def _records_from_complex_dir(complex_dir: Path, patterns: list[str]) -> pd.DataFrame:
    files = list_files(complex_dir, patterns)
    rows = []
    for p in files:
        stem = p.stem
        standard_id = stem
        for suffix in ("_complex", "_trimmed_complex"):
            if standard_id.endswith(suffix):
                standard_id = standard_id[: -len(suffix)]
        rows.append({"standard_structure_id": standard_id, "source_complex_pdb": str(p)})
    return pd.DataFrame(rows)


def run_step02b(config: dict[str, Any]) -> dict[str, Any]:
    """Run optional ligand atom trimming after Step 02.

    This step is intended for long substrates/cofactors where only a reaction-core
    subset should be used in downstream interaction extraction. It never modifies
    Step 02 outputs in place; it writes a new trimmed-complex directory and index.
    """
    target_id = require_nested(config, ["project", "target_id"])
    output_root = Path(get_nested(config, ["project", "output_root"], f"results/{target_id}"))

    step_cfg = get_nested(config, ["step02b"], {}) or {}
    selection_cfg = step_cfg.get("atom_selection", {}) or {}

    enabled = bool(step_cfg.get("enabled", True))
    step_dir = output_root / str(step_cfg.get("output_subdir", "02b_ligand_atom_trimming"))
    complex_dir = ensure_dir(step_dir / "complexes")
    table_dir = ensure_dir(step_dir / "tables")
    list_dir = ensure_dir(step_dir / "lists")
    report_dir = ensure_dir(step_dir / "reports")

    if not enabled:
        report = {
            "target_id": target_id,
            "step": "02b_ligand_atom_trimming",
            "status": "disabled",
            "message": "step02b.enabled is false; no trimming was performed.",
            "output_dir": str(step_dir),
        }
        with (report_dir / f"{target_id}_step02b_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        return report

    input_table_setting = step_cfg.get("input_complex_table", "auto")
    input_complex_dir = step_cfg.get("input_complex_dir")
    complex_column = str(step_cfg.get("input_complex_column", "complex_pdb_path"))

    if input_complex_dir:
        records_df = _records_from_complex_dir(
            Path(input_complex_dir),
            [str(x) for x in step_cfg.get("file_patterns", ["*.pdb"])],
        )
        input_source = str(input_complex_dir)
    else:
        if input_table_setting in (None, "", "auto"):
            input_table = _auto_step02_complex_index(output_root, target_id)
        else:
            input_table = Path(str(input_table_setting))
        if not input_table.exists():
            raise FileNotFoundError(
                f"Step 02b input complex table not found: {input_table}. "
                "Run Step 02 first, or set step02b.input_complex_table / step02b.input_complex_dir."
            )
        records_df = _records_from_complex_table(input_table, complex_column)
        input_source = str(input_table)

    rules = selection_cfg.get("rules", step_cfg.get("rules", [])) or []
    if not isinstance(rules, list):
        raise ValueError("step02b.atom_selection.rules must be a list of rule mappings")

    overwrite = bool(step_cfg.get("overwrite", True))
    output_suffix = str(step_cfg.get("output_complex_suffix", "_trimmed_complex.pdb"))

    default_action = str(selection_cfg.get("default_action", "keep"))
    remove_waters = bool(selection_cfg.get("remove_waters", False))
    drop_conect_records = bool(selection_cfg.get("drop_conect_records", False))
    renumber_atoms = bool(selection_cfg.get("renumber_atoms", True))
    fix_duplicate_ligand_atom_names = bool(selection_cfg.get("fix_duplicate_ligand_atom_names", False))

    summary_rows: list[dict[str, Any]] = []
    atom_trace_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []

    for _, row in records_df.iterrows():
        standard_id = str(row["standard_structure_id"])
        source_complex = Path(str(row["source_complex_pdb"]))
        output_complex = complex_dir / f"{standard_id}{output_suffix}"
        inherited_order = {
            key: row[key]
            for key in [
                "ligand_order_manifest_path", "active_ligand_pdb_order",
                "input_ligand_resname_order", "ligand_input_branch",
            ]
            if key in row.index and pd.notna(row[key]) and str(row[key]).strip()
        }

        if output_complex.exists() and not overwrite:
            summary_rows.append(
                {
                    "standard_structure_id": standard_id,
                    "source_complex_pdb": str(source_complex),
                    "trimmed_complex_pdb": str(output_complex),
                    "complex_pdb_path": str(output_complex),
                    "n_atoms_input": None,
                    "n_atoms_output": None,
                    "n_protein_atoms_input": None,
                    "n_hetatm_input": None,
                    "n_hetatm_output": None,
                    "n_hetatm_removed": None,
                    "n_rule_matched_atoms": None,
                    "n_rules_matched": None,
                    "status": "skipped_exists",
                    "error_message": "",
                    **inherited_order,
                }
            )
            continue

        result = trim_complex_pdb(
            input_pdb=source_complex,
            output_pdb=output_complex,
            standard_structure_id=standard_id,
            rules=rules,
            default_action=default_action,
            remove_waters=remove_waters,
            drop_conect_records=drop_conect_records,
            renumber_atoms=renumber_atoms,
            fix_duplicate_ligand_atom_names=fix_duplicate_ligand_atom_names,
            remark_lines=[
                f"Optional ligand atom trimming for {standard_id}",
                f"Source complex: {source_complex}",
                "This file is for downstream interaction analysis; Step02 original complex is unchanged.",
            ],
        )

        result_dict = asdict(result)
        traces = result_dict.pop("atom_trace_rows", [])
        residues = result_dict.pop("ligand_residue_summary_rows", [])
        result_dict["complex_pdb_path"] = result_dict["trimmed_complex_pdb"]
        result_dict.update(inherited_order)
        summary_rows.append(result_dict)
        atom_trace_rows.extend(traces)
        residue_rows.extend(residues)

    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(atom_trace_rows)
    residue_df = pd.DataFrame(residue_rows)

    summary_table = table_dir / f"{target_id}_step02b_ligand_atom_trim_summary.csv"
    trace_table = table_dir / f"{target_id}_step02b_ligand_atom_trace.csv"
    residue_table = table_dir / f"{target_id}_step02b_ligand_residue_summary.csv"
    index_table = table_dir / f"{target_id}_step02b_trimmed_complex_index.csv"
    failed_table = table_dir / f"{target_id}_step02b_failed_structures.csv"

    summary_df.to_csv(summary_table, index=False)
    trace_df.to_csv(trace_table, index=False)
    residue_df.to_csv(residue_table, index=False)

    ok_df = summary_df[summary_df["status"].isin(["ok", "skipped_exists"])].copy() if "status" in summary_df.columns else pd.DataFrame()
    failed_df = summary_df[~summary_df["status"].isin(["ok", "skipped_exists"])].copy() if "status" in summary_df.columns else pd.DataFrame()

    index_columns = [
        "standard_structure_id",
        "source_complex_pdb",
        "trimmed_complex_pdb",
        "complex_pdb_path",
        "n_atoms_input",
        "n_atoms_output",
        "n_hetatm_input",
        "n_hetatm_output",
        "n_hetatm_removed",
        "status",
    ]
    index_columns.extend([key for key in [
        "ligand_order_manifest_path", "active_ligand_pdb_order",
        "input_ligand_resname_order", "ligand_input_branch",
    ] if key in ok_df.columns])
    if not ok_df.empty:
        ok_df[index_columns].to_csv(index_table, index=False)
    else:
        pd.DataFrame(columns=index_columns).to_csv(index_table, index=False)
    failed_df.to_csv(failed_table, index=False)

    (list_dir / f"{target_id}_step02b_trimmed_complex_structure_ids.txt").write_text(
        "\n".join(ok_df["standard_structure_id"].astype(str).tolist()) + ("\n" if not ok_df.empty else ""),
        encoding="utf-8",
    )

    report = {
        "target_id": target_id,
        "step": "02b_ligand_atom_trimming",
        "input_source": input_source,
        "n_input_complexes": int(len(records_df)),
        "n_trimmed_or_existing_complexes": int(len(ok_df)),
        "n_failed": int(len(failed_df)),
        "n_rules": int(len(rules)),
        "default_action": default_action,
        "output_dir": str(step_dir),
        "main_outputs": {
            "summary_table": str(summary_table),
            "atom_trace_table": str(trace_table),
            "ligand_residue_summary_table": str(residue_table),
            "trimmed_complex_index": str(index_table),
            "complex_dir": str(complex_dir),
        },
        "notes": [
            "Step 02b is optional and never overwrites Step 02 complexes.",
            "It is intended for long substrates/cofactors where only reaction-core atoms should be used downstream.",
            "The atom trace table records every HETATM atom kept or removed and the exact rule responsible.",
            "Downstream PLIP / interaction steps can consume step02b_trimmed_complex_index.csv instead of the Step 02 complex index.",
            "Atom-name based presets are lightweight and do not require RDKit; explicit keep_atom_names can be used for any ligand.",
        ],
    }
    with (report_dir / f"{target_id}_step02b_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    summary_lines = [
        f"# Step 02b summary: {target_id}",
        "",
        "## Scope",
        "",
        "Step 02b optionally removes selected atoms from long ligands before downstream interaction analysis.",
        "The original Step 02 complexes are kept unchanged.",
        "",
        "## Results",
        "",
        f"- Input complexes: {len(records_df)}",
        f"- Trimmed/existing complexes: {len(ok_df)}",
        f"- Failed complexes: {len(failed_df)}",
        f"- Selection rules: {len(rules)}",
        "",
        "## Main outputs",
        "",
        f"- Summary table: `{summary_table}`",
        f"- Atom-level trace table: `{trace_table}`",
        f"- Ligand-residue summary table: `{residue_table}`",
        f"- Trimmed complex index: `{index_table}`",
        f"- Trimmed complexes: `{complex_dir}`",
        "",
    ]
    (report_dir / f"{target_id}_step02b_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 02b: optional ligand atom trimming before downstream analysis.")
    parser.add_argument("--config", required=True, help="Project-level YAML config file.")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    report = run_step02b(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

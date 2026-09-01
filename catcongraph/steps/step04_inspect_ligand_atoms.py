#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect ligand atom names before optional trimming."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import argparse
import json
import pandas as pd
from catcongraph.config import load_yaml_config

from catcongraph.qc.clash import parse_pdb_atoms, WATER_RESNAMES
from catcongraph.steps.step04_ligand_atom_trimming import (
    _target_id,
    _output_root,
    resolve_step04_input_table,
    _infer_complex_path_column,
    _infer_structure_id,
)


def load_config(config_path: Path | str) -> dict:
    return load_yaml_config(config_path)


def run_inspect(config_path: Path | str, max_structures: Optional[int] = 1) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    target_id = _target_id(config)
    root = _output_root(config)
    step04 = config.get("step04", {})

    output_dir = root / step04.get("output_subdir", "04_ligand_atom_trimming")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_table = resolve_step04_input_table(config)
    df = pd.read_csv(input_table)
    if max_structures is not None and max_structures > 0:
        df = df.head(max_structures)

    path_col = _infer_complex_path_column(df, step04.get("input_complex_column", "auto"))

    atom_rows = []
    for _, row in df.iterrows():
        pdb_path = Path(str(row[path_col]))
        sid = _infer_structure_id(row, pdb_path)
        atoms = parse_pdb_atoms(pdb_path)
        for atom in atoms:
            if atom.record != "HETATM":
                continue
            if atom.resname.upper() in WATER_RESNAMES:
                continue
            atom_rows.append(
                {
                    "standard_structure_id": sid,
                    "complex_pdb_path": str(pdb_path),
                    "record": atom.record,
                    "residue_name": atom.resname,
                    "chain_id": atom.chain_id,
                    "residue_number": atom.resseq,
                    "atom_serial": atom.serial,
                    "atom_name": atom.atom_name,
                    "element": atom.element,
                    "x": atom.x,
                    "y": atom.y,
                    "z": atom.z,
                }
            )

    atom_df = pd.DataFrame(atom_rows)
    atom_inventory_path = tables_dir / f"{target_id}_step04_ligand_atom_inventory.csv"
    residue_inventory_path = tables_dir / f"{target_id}_step04_ligand_residue_inventory.csv"
    atom_df.to_csv(atom_inventory_path, index=False)

    if len(atom_df):
        residue = (
            atom_df.groupby(["residue_name", "chain_id", "residue_number"], dropna=False)
            .agg(
                n_atoms=("atom_name", "count"),
                atom_names=("atom_name", lambda x: ",".join(map(str, x))),
                atom_names_for_yaml=("atom_name", lambda x: "\n".join([f'          - "{a}"' for a in x])),
            )
            .reset_index()
        )
    else:
        residue = pd.DataFrame(columns=["residue_name", "chain_id", "residue_number", "n_atoms", "atom_names", "atom_names_for_yaml"])

    residue.to_csv(residue_inventory_path, index=False)

    report = {
        "target_id": target_id,
        "input_complex_table": str(input_table),
        "n_structures_inspected": int(len(df)),
        "n_ligand_atoms": int(len(atom_df)),
        "atom_inventory": str(atom_inventory_path),
        "residue_inventory": str(residue_inventory_path),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect ligand atom names before Step04 trimming.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    parser.add_argument("--max-structures", type=int, default=1, help="Number of complexes to inspect. Use 0 for all.")
    args = parser.parse_args(argv)

    max_structures = None if args.max_structures == 0 else args.max_structures
    run_inspect(args.config, max_structures=max_structures)


if __name__ == "__main__":
    main()

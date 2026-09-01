#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Catalytic-region pLDDT quality control utilities.

AlphaFold/ColabFold writes pLDDT values into the PDB B-factor column.  This
module scores the protein residues around a selected substrate/ligand region,
usually before PLIP/interaction graph extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np

from catcongraph.io.pdb import PDBAtomRecord, parse_pdb_atoms


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}

WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD", "SOL"}


@dataclass(frozen=True)
class CatalyticRegionPLDDTResult:
    complex_pdb_path: str
    standard_structure_id: str
    radius_angstrom: float
    ligand_atom_count: int
    protein_atom_count: int
    catalytic_residue_count: int
    catalytic_mean_plddt: float | None
    catalytic_median_plddt: float | None
    catalytic_p10_plddt: float | None
    catalytic_min_plddt: float | None
    catalytic_max_plddt: float | None
    mean_plddt_min: float
    p10_plddt_min: float
    pass_catalytic_region_plddt: bool
    status: str
    error_message: str = ""
    plddt_pdb_path: str = ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _norm(value: Any) -> str:
    return str(value).strip().upper()


def _value_set(value: Any, *, upper: bool = True) -> set[str]:
    out: set[str] = set()
    for item in _as_list(value):
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        out.add(text.upper() if upper else text)
    return out


def _residue_key(atom: PDBAtomRecord) -> tuple[str, int, str, str]:
    return (
        atom.chain_id.strip() or "_",
        int(atom.residue_number),
        atom.insertion_code.strip() or "",
        atom.residue_name.strip().upper(),
    )


def _residue_label(key: tuple[str, int, str, str]) -> str:
    chain, resseq, icode, resname = key
    suffix = icode if icode else ""
    return f"{resname}{resseq}{suffix}:{chain}"


def _is_hydrogen(atom: PDBAtomRecord) -> bool:
    name = atom.atom_name.strip().upper().lstrip("0123456789")
    element = atom.raw_line[76:78].strip().upper() if len(atom.raw_line) >= 78 else ""
    return element == "H" or name.startswith("H")


def select_ligand_atoms(
    atoms: Sequence[PDBAtomRecord],
    *,
    include_resnames: Sequence[str] | None = None,
    exclude_resnames: Sequence[str] = ("HOH", "WAT"),
    chains: Sequence[str] | None = None,
    resseqs: Sequence[int | str] | None = None,
    atom_names: Sequence[str] | None = None,
    include_hetatm: bool = True,
    ignore_hydrogen: bool = True,
) -> list[PDBAtomRecord]:
    """Select substrate/ligand atoms used to define the catalytic region."""
    include_set = _value_set(include_resnames)
    exclude_set = _value_set(exclude_resnames) | WATER_RESNAMES
    chain_set = {str(x).strip() for x in _as_list(chains) if str(x).strip()}
    resseq_set = {str(x).strip() for x in _as_list(resseqs) if str(x).strip()}
    atom_name_set = _value_set(atom_names)

    selected: list[PDBAtomRecord] = []
    for atom in atoms:
        rec = atom.record_name.strip().upper()
        resname = atom.residue_name.strip().upper()
        if include_hetatm:
            if rec != "HETATM":
                continue
        elif rec not in {"ATOM", "HETATM"}:
            continue

        if resname in exclude_set:
            continue
        if include_set and resname not in include_set:
            continue
        if chain_set and (atom.chain_id.strip() or "_") not in chain_set:
            continue
        if resseq_set and str(atom.residue_number).strip() not in resseq_set:
            continue
        if atom_name_set and atom.atom_name.strip().upper() not in atom_name_set:
            continue
        if ignore_hydrogen and _is_hydrogen(atom):
            continue
        selected.append(atom)
    return selected


def _is_protein_atom(atom: PDBAtomRecord) -> bool:
    return atom.record_name.strip().upper() == "ATOM" and atom.residue_name.strip().upper() in STANDARD_AA


def _near_ligand_residue_keys(
    protein_atoms: Sequence[PDBAtomRecord],
    ligand_atoms: Sequence[PDBAtomRecord],
    radius: float,
    *,
    ignore_hydrogen: bool = True,
) -> set[tuple[str, int, str, str]]:
    if not protein_atoms or not ligand_atoms:
        return set()

    filtered_protein = [a for a in protein_atoms if not (ignore_hydrogen and _is_hydrogen(a))]
    filtered_ligand = [a for a in ligand_atoms if not (ignore_hydrogen and _is_hydrogen(a))]
    if not filtered_protein or not filtered_ligand:
        return set()

    p_coords = np.array([[a.x, a.y, a.z] for a in filtered_protein], dtype=float)
    l_coords = np.array([[a.x, a.y, a.z] for a in filtered_ligand], dtype=float)
    radius2 = float(radius) * float(radius)

    keys: set[tuple[str, int, str, str]] = set()
    for lig_xyz in l_coords:
        diff = p_coords - lig_xyz[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        mask = d2 <= radius2
        if np.any(mask):
            for idx in np.where(mask)[0]:
                keys.add(_residue_key(filtered_protein[int(idx)]))
    return keys


def _residue_base_key(key: tuple[str, int, str, str]) -> tuple[int, str, str]:
    chain, resseq, icode, resname = key
    return (int(resseq), str(icode or ""), str(resname).upper())


def _residue_plddt_rows(
    plddt_protein_atoms: Sequence[PDBAtomRecord],
    residue_keys: set[tuple[str, int, str, str]],
    *,
    preferred_atom_name: str = "CA",
    plddt_pdb_path: str = "",
) -> list[dict[str, Any]]:
    """Return pLDDT rows for residue keys defined in the docked complex.

    The catalytic region is detected in the docked complex, but the pLDDT can
    be read from the corresponding AF2/ColabFold protein PDB.  Exact
    chain/residue matches are preferred; if docking changed/blanked chain IDs,
    a conservative chain-insensitive (resname, resseq, insertion code) fallback
    is used and recorded in the output table.
    """
    atoms_by_exact: dict[tuple[str, int, str, str], list[PDBAtomRecord]] = {}
    atoms_by_base: dict[tuple[int, str, str], list[PDBAtomRecord]] = {}
    for atom in plddt_protein_atoms:
        key = _residue_key(atom)
        atoms_by_exact.setdefault(key, []).append(atom)
        atoms_by_base.setdefault(_residue_base_key(key), []).append(atom)

    rows: list[dict[str, Any]] = []
    preferred = str(preferred_atom_name or "CA").strip().upper()
    for key in sorted(residue_keys, key=lambda k: (k[0], k[1], k[2], k[3])):
        atoms = atoms_by_exact.get(key)
        match_mode = "exact"
        if not atoms:
            atoms = atoms_by_base.get(_residue_base_key(key), [])
            match_mode = "chain_insensitive" if atoms else "missing_in_plddt_source"
        preferred_atoms = [a for a in atoms if a.atom_name.strip().upper() == preferred]
        score_atoms = preferred_atoms if preferred_atoms else atoms
        values = [float(a.b_factor) for a in score_atoms if a.b_factor is not None]
        chain, resseq, icode, resname = key
        rows.append(
            {
                "residue_label": _residue_label(key),
                "residue_name": resname,
                "chain_id": chain,
                "residue_number": int(resseq),
                "insertion_code": icode,
                "plddt": float(np.mean(values)) if values else math.nan,
                "plddt_source": preferred if preferred_atoms else ("residue_atom_mean_fallback" if atoms else "missing"),
                "plddt_match_mode": match_mode,
                "plddt_pdb_path": plddt_pdb_path,
                "n_atoms_in_region": len(atoms),
                "n_atoms_used_for_plddt": len(score_atoms),
            }
        )
    return rows

def compute_catalytic_region_plddt(
    complex_pdb_path: str | Path,
    *,
    standard_structure_id: str = "",
    radius_angstrom: float = 6.0,
    mean_plddt_min: float = 90.0,
    p10_plddt_min: float = 80.0,
    residue_plddt_atom: str = "CA",
    ligand_selection: Mapping[str, Any] | None = None,
    ignore_hydrogen_for_distance: bool = True,
    strict_greater_than: bool = True,
    plddt_pdb_path: str | Path | None = None,
) -> tuple[CatalyticRegionPLDDTResult, list[dict[str, Any]]]:
    """Compute local pLDDT QC for protein residues near selected ligand atoms.

    The catalytic region is defined as all standard amino-acid residues with at
    least one protein atom within ``radius_angstrom`` of any selected ligand
    atom.  Each residue contributes one pLDDT value, preferably from its CA atom
    because AlphaFold pLDDT is residue-level.  If CA is missing, the residue's
    atom-level B-factor mean is used as a fallback.
    """
    pdb_path = Path(complex_pdb_path)
    if not standard_structure_id:
        standard_structure_id = pdb_path.stem.replace("_trimmed_complex", "").replace("_complex", "")

    ligand_cfg = dict(ligand_selection or {})
    atoms = parse_pdb_atoms(pdb_path, include_hetatm=True, atom_name=None)
    protein_atoms = [a for a in atoms if _is_protein_atom(a)]
    plddt_source = Path(plddt_pdb_path) if plddt_pdb_path else pdb_path
    if plddt_source != pdb_path and plddt_source.exists():
        plddt_atoms = parse_pdb_atoms(plddt_source, include_hetatm=True, atom_name=None)
        plddt_protein_atoms = [a for a in plddt_atoms if _is_protein_atom(a)]
    else:
        plddt_source = pdb_path
        plddt_protein_atoms = protein_atoms
    ligand_atoms = select_ligand_atoms(
        atoms,
        include_resnames=ligand_cfg.get("include_resnames"),
        exclude_resnames=ligand_cfg.get("exclude_resnames", ["HOH", "WAT"]),
        chains=ligand_cfg.get("chains") or ligand_cfg.get("chain"),
        resseqs=ligand_cfg.get("resseqs") or ligand_cfg.get("resseq"),
        atom_names=ligand_cfg.get("atom_names"),
        include_hetatm=bool(ligand_cfg.get("include_hetatm", True)),
        ignore_hydrogen=bool(ligand_cfg.get("ignore_hydrogen", True)),
    )

    status = "passed"
    error_message = ""
    residue_rows: list[dict[str, Any]] = []
    pass_qc = False
    mean_value: float | None = None
    median_value: float | None = None
    p10_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None

    if not ligand_atoms:
        status = "failed"
        error_message = "No selected ligand/substrate atoms were found. Check step05.ligand_selection."
    elif not protein_atoms:
        status = "failed"
        error_message = "No standard protein ATOM records were found."
    else:
        region_keys = _near_ligand_residue_keys(
            protein_atoms,
            ligand_atoms,
            float(radius_angstrom),
            ignore_hydrogen=bool(ignore_hydrogen_for_distance),
        )
        residue_rows = _residue_plddt_rows(
            plddt_protein_atoms,
            region_keys,
            preferred_atom_name=residue_plddt_atom,
            plddt_pdb_path=str(plddt_source),
        )
        values = np.array([r["plddt"] for r in residue_rows if not math.isnan(float(r["plddt"]))], dtype=float)
        if values.size == 0:
            status = "failed"
            error_message = f"No protein residues were found within {float(radius_angstrom):g} Å of selected ligand atoms."
        else:
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            p10_value = float(np.percentile(values, 10))
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            if strict_greater_than:
                pass_qc = bool(mean_value > float(mean_plddt_min) and p10_value > float(p10_plddt_min))
            else:
                pass_qc = bool(mean_value >= float(mean_plddt_min) and p10_value >= float(p10_plddt_min))
            status = "passed" if pass_qc else "failed"

    for row in residue_rows:
        row.update(
            {
                "standard_structure_id": standard_structure_id,
                "complex_pdb_path": str(pdb_path),
                "plddt_pdb_path": str(plddt_source),
                "radius_angstrom": float(radius_angstrom),
            }
        )

    result = CatalyticRegionPLDDTResult(
        complex_pdb_path=str(pdb_path),
        standard_structure_id=standard_structure_id,
        radius_angstrom=float(radius_angstrom),
        ligand_atom_count=int(len(ligand_atoms)),
        protein_atom_count=int(len(protein_atoms)),
        catalytic_residue_count=int(len(residue_rows)),
        catalytic_mean_plddt=mean_value,
        catalytic_median_plddt=median_value,
        catalytic_p10_plddt=p10_value,
        catalytic_min_plddt=min_value,
        catalytic_max_plddt=max_value,
        mean_plddt_min=float(mean_plddt_min),
        p10_plddt_min=float(p10_plddt_min),
        pass_catalytic_region_plddt=bool(pass_qc),
        status=status,
        error_message=error_message,
        plddt_pdb_path=str(plddt_source),
    )
    return result, residue_rows

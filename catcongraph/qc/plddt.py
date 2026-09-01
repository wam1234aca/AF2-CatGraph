from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from catcongraph.io.pdb import parse_pdb_atoms


@dataclass(frozen=True)
class GlobalPLDDTResult:
    n_atoms: int
    n_residues: int
    global_mean_plddt: float
    global_median_plddt: float
    global_p10_plddt: float
    global_min_plddt: float
    global_max_plddt: float


def compute_global_plddt(
    pdb_path: str | Path,
    *,
    atom_name: str = "CA",
    include_hetatm: bool = False,
) -> GlobalPLDDTResult:
    """Compute global pLDDT from the PDB B-factor column.

    The default uses CA atoms so each protein residue contributes once.
    If no matching CA atoms are found, all ATOM records are used as a fallback.
    """
    ca_records = parse_pdb_atoms(pdb_path, include_hetatm=include_hetatm, atom_name=atom_name)
    all_records = parse_pdb_atoms(pdb_path, include_hetatm=include_hetatm, atom_name=None)

    records = ca_records if ca_records else all_records
    if not records:
        raise ValueError(f"No usable ATOM records found in PDB: {pdb_path}")

    values = np.array([r.b_factor for r in records], dtype=float)
    residues = {r.residue_key for r in records}

    return GlobalPLDDTResult(
        n_atoms=len(all_records),
        n_residues=len(residues),
        global_mean_plddt=float(np.mean(values)),
        global_median_plddt=float(np.median(values)),
        global_p10_plddt=float(np.percentile(values, 10)),
        global_min_plddt=float(np.min(values)),
        global_max_plddt=float(np.max(values)),
    )

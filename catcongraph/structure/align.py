from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from catcongraph.io.pdb import parse_pdb_atoms, write_transformed_pdb


@dataclass(frozen=True)
class AlignmentResult:
    n_matched_atoms: int
    rmsd_before: float
    rmsd_after: float
    output_pdb: str | None


def _kabsch_row_vectors(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation R and translation t mapping moving to reference.

    Coordinates are treated as row vectors:
        transformed = moving @ R + t
    """
    moving_centroid = moving.mean(axis=0)
    reference_centroid = reference.mean(axis=0)

    p = moving - moving_centroid
    q = reference - reference_centroid

    covariance = p.T @ q
    v, _, wt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(v @ wt))
    correction = np.diag([1.0, 1.0, d])
    rotation = v @ correction @ wt
    translation = reference_centroid - moving_centroid @ rotation
    return rotation, translation


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def align_pdb_to_reference(
    moving_pdb: str | Path,
    reference_pdb: str | Path,
    output_pdb: str | Path,
    *,
    atom_name: str = "CA",
    min_matched_atoms: int = 50,
) -> AlignmentResult:
    """Align one protein PDB to a reference using matched atom keys.

    Matching is based on chain ID, residue number, insertion code, and atom name.
    This is suitable for AF2 ensembles of the same target sequence.
    """
    moving_pdb = Path(moving_pdb)
    reference_pdb = Path(reference_pdb)
    output_pdb = Path(output_pdb)

    moving_atoms = parse_pdb_atoms(moving_pdb, include_hetatm=False, atom_name=atom_name)
    reference_atoms = parse_pdb_atoms(reference_pdb, include_hetatm=False, atom_name=atom_name)

    moving_by_key = {a.atom_key: a for a in moving_atoms}
    reference_by_key = {a.atom_key: a for a in reference_atoms}

    matched_keys = sorted(set(moving_by_key).intersection(reference_by_key))
    if len(matched_keys) < min_matched_atoms:
        raise ValueError(
            f"Too few matched {atom_name} atoms for alignment: "
            f"{len(matched_keys)} < {min_matched_atoms}. "
            f"Moving={moving_pdb}, Reference={reference_pdb}"
        )

    moving_coords = np.array([[moving_by_key[k].x, moving_by_key[k].y, moving_by_key[k].z] for k in matched_keys], dtype=float)
    reference_coords = np.array([[reference_by_key[k].x, reference_by_key[k].y, reference_by_key[k].z] for k in matched_keys], dtype=float)

    rmsd_before = _rmsd(moving_coords, reference_coords)
    rotation, translation = _kabsch_row_vectors(moving_coords, reference_coords)
    aligned_coords = moving_coords @ rotation + translation
    rmsd_after = _rmsd(aligned_coords, reference_coords)

    all_moving_atoms = parse_pdb_atoms(moving_pdb, include_hetatm=True, atom_name=None)
    transformed_by_line_index: dict[int, tuple[float, float, float]] = {}
    for atom in all_moving_atoms:
        xyz = np.array([atom.x, atom.y, atom.z], dtype=float)
        new_xyz = xyz @ rotation + translation
        transformed_by_line_index[atom.line_index] = (float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]))

    write_transformed_pdb(moving_pdb, output_pdb, transformed_by_line_index)

    return AlignmentResult(
        n_matched_atoms=len(matched_keys),
        rmsd_before=rmsd_before,
        rmsd_after=rmsd_after,
        output_pdb=str(output_pdb),
    )

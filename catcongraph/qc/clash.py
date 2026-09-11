#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Protein-ligand clash detection utilities.

The default rule follows the user's legacy `colision.py` logic:

    distance < ligand_vdw_radius + protein_vdw_radius - delta

The module is intentionally RDKit-free and only depends on NumPy, so it is
stable on Linux/HPC environments. It reads coordinates directly from PDB
ATOM/HETATM records.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}

WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD"}

# Conservative VDW radii in Å. This is enough for PDB protein/ligand screening.
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "MG": 1.73,
    "MN": 1.79,
    "FE": 1.72,
    "CO": 1.67,
    "NI": 1.63,
    "CU": 1.40,
    "ZN": 1.39,
    "CA": 2.31,
    "NA": 2.27,
    "K": 2.75,
}


# Exact table from the user-provided colision.py.
#
# Important: this intentionally preserves the original mixed-case ``Cl`` and
# ``Br`` keys. colision.py uppercases Bio.PDB's element before lookup, so
# chlorine and bromine fall back to 1.50 Å in that script. The same seemingly
# unusual behavior is retained here because article reproduction requires
# numerical identity, not a silently "corrected" chemistry table.
COLISION_PY_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "METAL": 1.50,
}


def normalize_clash_method(value: object) -> str:
    raw = str(value or "collision_py_exact").strip().lower()
    aliases = {
        "standard": "collision_py_exact",
        "catcongraph_studio_v1": "collision_py_exact",
        "collision_py": "collision_py_exact",
        "colision_py": "collision_py_exact",
        "colision_py_exact": "collision_py_exact",
        "legacy": "collision_py_exact",
        "legacy_exact": "collision_py_exact",
        "manuscript": "collision_py_exact",
        "article": "collision_py_exact",
        "catcongraph": "catcongraph_vdw",
        "current": "catcongraph_vdw",
        "optimized_vdw": "catcongraph_vdw",
    }
    method = aliases.get(raw, raw)
    if method not in {"collision_py_exact", "catcongraph_vdw"}:
        raise ValueError(
            "clash method must be 'standard' or 'catcongraph_vdw'; "
            f"got {value!r}"
        )
    return method


@dataclass
class PDBAtom:
    record: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain_id: str
    resseq: str
    icode: str
    x: float
    y: float
    z: float
    occupancy: Optional[float]
    bfactor: Optional[float]
    element: str
    line_index: int
    raw_line: str

    @property
    def residue_key(self) -> str:
        return f"{self.resname}:{self.chain_id}:{self.resseq}{self.icode}".strip()


@dataclass
class ClashPair:
    complex_pdb_path: str
    standard_structure_id: str
    protein_atom_serial: int
    protein_atom_name: str
    protein_resname: str
    protein_chain: str
    protein_resseq: str
    ligand_atom_serial: int
    ligand_atom_name: str
    ligand_resname: str
    ligand_chain: str
    ligand_resseq: str
    distance: float
    protein_vdw_radius: float
    ligand_vdw_radius: float
    clash_cutoff_distance: float
    overlap: float


@dataclass
class ClashResult:
    complex_pdb_path: str
    standard_structure_id: str
    protein_atom_count: int
    ligand_atom_count: int
    clash_count: int
    clash_score: float
    min_distance: Optional[float]
    min_clash_distance: Optional[float]
    clash_max: int
    pass_clash: bool
    status: str
    error_message: str = ""
    weighted_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def n_clashes(self) -> int:
        """Backward-compatible alias used by Step02 ligand adjustment."""
        return int(self.clash_count)

    @property
    def collision_score(self) -> float:
        """Backward-compatible alias used by Step02 ligand adjustment."""
        return float(self.clash_score)


def guess_element(atom_name: str, element_field: str = "") -> str:
    """Guess chemical element from PDB element field or atom name."""
    elem = (element_field or "").strip().upper()
    if elem:
        if len(elem) >= 2 and elem[:2] in VDW_RADII:
            return elem[:2]
        return elem[0]

    name = atom_name.strip().upper()
    if not name:
        return ""

    # PDB atom names can begin with a digit, e.g. 1HG.
    name = name.lstrip("0123456789")
    if len(name) >= 2 and name[:2] in VDW_RADII:
        return name[:2]
    return name[0]


def vdw_radius(element: str, default: float = 1.70) -> float:
    return float(VDW_RADII.get((element or "").upper(), default))


def parse_pdb_atoms(pdb_path: Path | str, keep_altloc: Sequence[str] = ("", "A")) -> List[PDBAtom]:
    """Parse ATOM/HETATM records from a PDB file."""
    pdb_path = Path(pdb_path)
    atoms: List[PDBAtom] = []

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            record = line[0:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue

            altloc = line[16:17].strip()
            if altloc not in set(keep_altloc):
                continue

            try:
                serial = int(line[6:11])
            except Exception:
                serial = len(atoms) + 1

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception as exc:
                raise ValueError(f"Failed to parse coordinates in {pdb_path}, line {idx+1}: {line.rstrip()}") from exc

            try:
                occupancy = float(line[54:60])
            except Exception:
                occupancy = None

            try:
                bfactor = float(line[60:66])
            except Exception:
                bfactor = None

            element_field = line[76:78] if len(line) >= 78 else ""
            element = guess_element(atom_name, element_field)

            atoms.append(
                PDBAtom(
                    record=record,
                    serial=serial,
                    atom_name=atom_name,
                    altloc=altloc,
                    resname=resname,
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element=element,
                    line_index=idx,
                    raw_line=line.rstrip("\n"),
                )
            )

    return atoms


def is_hydrogen(atom: PDBAtom) -> bool:
    return atom.element.upper() == "H" or atom.atom_name.strip().upper().startswith("H")


def split_protein_ligand_atoms(
    atoms: Sequence[PDBAtom],
    include_hetatm_as_ligand: bool = True,
    ligand_resnames: Optional[Sequence[str]] = None,
    exclude_resnames: Sequence[str] = ("HOH", "WAT"),
    ignore_hydrogen: bool = False,
    ignore_water: bool = True,
    protein_resnames: Sequence[str] = tuple(STANDARD_AA),
) -> Tuple[List[PDBAtom], List[PDBAtom]]:
    """Split atoms into protein atoms and ligand atoms.

    Protein atoms are standard amino acid ATOM records by default.
    Ligand atoms are HETATM records, optionally restricted by ligand_resnames.
    Metals/cofactors/substrates are intentionally kept as ligand-side atoms.
    """
    protein_set = set(protein_resnames)
    exclude_set = {x.upper() for x in exclude_resnames}
    ligand_set = {x.upper() for x in ligand_resnames} if ligand_resnames else None

    protein_atoms: List[PDBAtom] = []
    ligand_atoms: List[PDBAtom] = []

    for atom in atoms:
        resname = atom.resname.upper()

        if ignore_hydrogen and is_hydrogen(atom):
            continue
        if ignore_water and resname in WATER_RESNAMES:
            continue
        if resname in exclude_set:
            continue

        if atom.record == "ATOM" and resname in protein_set:
            protein_atoms.append(atom)
            continue

        if include_hetatm_as_ligand and atom.record == "HETATM":
            if ligand_set is None or resname in ligand_set:
                ligand_atoms.append(atom)

    return protein_atoms, ligand_atoms


def _pairwise_clash_search(
    protein_atoms: Sequence[PDBAtom],
    ligand_atoms: Sequence[PDBAtom],
    delta: float = 0.5,
    neighbor_padding: float = 0.0,
) -> Tuple[List[Tuple[int, int, float, float, float, float]], Optional[float]]:
    """Return protein-ligand clash pairs.

    The implementation uses vectorized NumPy chunks. For typical enzyme
    complexes this is fast enough and avoids a SciPy dependency.
    """
    if not protein_atoms or not ligand_atoms:
        return [], None

    p_coords = np.array([[a.x, a.y, a.z] for a in protein_atoms], dtype=float)
    l_coords = np.array([[a.x, a.y, a.z] for a in ligand_atoms], dtype=float)

    p_r = np.array([vdw_radius(a.element) for a in protein_atoms], dtype=float)
    l_r = np.array([vdw_radius(a.element) for a in ligand_atoms], dtype=float)

    pairs: List[Tuple[int, int, float, float, float, float]] = []
    global_min = math.inf

    # Iterate ligand atoms; protein side vectorized.
    for li, coord in enumerate(l_coords):
        diffs = p_coords - coord
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        if len(dists):
            local_min = float(np.min(dists))
            if local_min < global_min:
                global_min = local_min

        cutoffs = p_r + l_r[li] - float(delta)
        mask = dists < cutoffs
        if np.any(mask):
            for pi in np.where(mask)[0]:
                distance = float(dists[pi])
                cutoff = float(cutoffs[pi])
                overlap = float(cutoff - distance)
                pairs.append((pi, li, distance, float(p_r[pi]), float(l_r[li]), overlap))

    return pairs, (None if math.isinf(global_min) else float(global_min))



def _bio_atom_to_pdb_atom(atom, line_index: int = -1) -> PDBAtom:
    """Convert a Bio.PDB Atom proxy into the Step03 audit representation."""
    residue = atom.get_parent()
    chain = residue.get_parent()
    hetfield, resseq, icode = residue.id
    coord = atom.coord
    serial = atom.get_serial_number()
    return PDBAtom(
        record="HETATM" if str(hetfield).strip() else "ATOM",
        serial=int(serial) if serial is not None else 0,
        atom_name=str(atom.get_name()).strip(),
        altloc=str(atom.get_altloc()).strip(),
        resname=str(residue.get_resname()).strip(),
        chain_id=str(chain.id).strip(),
        resseq=str(resseq),
        icode=str(icode).strip(),
        x=float(coord[0]),
        y=float(coord[1]),
        z=float(coord[2]),
        occupancy=(
            float(atom.get_occupancy())
            if atom.get_occupancy() is not None
            else None
        ),
        bfactor=(
            float(atom.get_bfactor())
            if atom.get_bfactor() is not None
            else None
        ),
        element=str(getattr(atom, "element", "") or "").strip().upper(),
        line_index=int(line_index),
        raw_line="",
    )


def _compute_collision_py_exact(
    complex_pdb_path: Path | str,
    standard_structure_id: str,
    clash_max: int,
    delta: float,
    neighbor_search_cutoff: float,
) -> Tuple[ClashResult, List[ClashPair]]:
    """Execute the uploaded colision.py algorithm without semantic changes.

    Exact legacy behavior:
      * Bio.PDB.PDBParser(QUIET=True);
      * residue.id[0].strip() not in ('', 'W') -> ligand;
      * blank hetfield and water hetfield 'W' -> protein side;
      * all atoms, including hydrogen, are retained;
      * exact legacy radius table and fallback radius 1.50 Å;
      * scipy.spatial.KDTree query_ball_point(..., cutoff=5.0);
      * strict ``distance < r_protein + r_ligand - delta``;
      * one count per protein-ligand atom pair;
      * pass when ``clash_count <= clash_max``.
    """
    try:
        from Bio.PDB import PDBParser
    except Exception as exc:
        raise ImportError(
            "collision_py_exact requires Biopython because the original "
            "colision.py uses Bio.PDB.PDBParser."
        ) from exc
    try:
        from scipy.spatial import KDTree
    except Exception as exc:
        raise ImportError(
            "collision_py_exact requires SciPy because the original "
            "colision.py uses scipy.spatial.KDTree."
        ) from exc

    complex_pdb_path = Path(complex_pdb_path)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(complex_pdb_path))

    protein_bio_atoms = []
    ligand_bio_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetfield = residue.id[0].strip()
                if hetfield not in ("", "W"):
                    for atom in residue:
                        ligand_bio_atoms.append(atom)
                else:
                    for atom in residue:
                        protein_bio_atoms.append(atom)

    # This mirrors the original failure mode, but with a more useful message.
    if not protein_bio_atoms:
        raise ValueError(
            f"colision.py exact mode found no protein-side atoms in {complex_pdb_path}"
        )
    if not ligand_bio_atoms:
        raise ValueError(
            f"colision.py exact mode found no ligand-side atoms in {complex_pdb_path}"
        )

    protein_atoms = [
        _bio_atom_to_pdb_atom(atom, index)
        for index, atom in enumerate(protein_bio_atoms)
    ]
    ligand_atoms = [
        _bio_atom_to_pdb_atom(atom, index)
        for index, atom in enumerate(ligand_bio_atoms)
    ]

    protein_coords = []
    protein_radii = []
    for atom in protein_bio_atoms:
        elem = atom.element.strip().upper()
        protein_coords.append(atom.coord)
        protein_radii.append(COLISION_PY_VDW_RADII.get(elem, 1.5))

    ligand_coords = []
    ligand_radii = []
    for atom in ligand_bio_atoms:
        elem = atom.element.strip().upper()
        ligand_coords.append(atom.coord)
        ligand_radii.append(COLISION_PY_VDW_RADII.get(elem, 1.5))

    protein_coords_array = np.asarray(protein_coords, dtype=float)
    kdtree = KDTree(protein_coords_array)

    clash_pairs: List[ClashPair] = []
    clash_score = 0.0
    min_distance: Optional[float] = None
    min_clash_distance: Optional[float] = None

    for ligand_index, (lig_coord, lig_radius) in enumerate(
        zip(ligand_coords, ligand_radii)
    ):
        nearest_distance, _nearest_index = kdtree.query(lig_coord)
        nearest_distance = float(nearest_distance)
        if min_distance is None or nearest_distance < min_distance:
            min_distance = nearest_distance

        indices = kdtree.query_ball_point(
            lig_coord,
            float(neighbor_search_cutoff),
        )
        for protein_index in indices:
            prot_radius = float(protein_radii[protein_index])
            prot_coord = protein_coords_array[protein_index]
            distance = float(np.linalg.norm(lig_coord - prot_coord))
            cutoff_distance = float(
                float(lig_radius) + prot_radius - float(delta)
            )
            if distance < cutoff_distance:
                overlap = float(cutoff_distance - distance)
                clash_score += overlap
                if (
                    min_clash_distance is None
                    or distance < min_clash_distance
                ):
                    min_clash_distance = distance

                protein_atom = protein_atoms[protein_index]
                ligand_atom = ligand_atoms[ligand_index]
                clash_pairs.append(
                    ClashPair(
                        complex_pdb_path=str(complex_pdb_path),
                        standard_structure_id=standard_structure_id,
                        protein_atom_serial=protein_atom.serial,
                        protein_atom_name=protein_atom.atom_name,
                        protein_resname=protein_atom.resname,
                        protein_chain=protein_atom.chain_id,
                        protein_resseq=protein_atom.resseq,
                        ligand_atom_serial=ligand_atom.serial,
                        ligand_atom_name=ligand_atom.atom_name,
                        ligand_resname=ligand_atom.resname,
                        ligand_chain=ligand_atom.chain_id,
                        ligand_resseq=ligand_atom.resseq,
                        distance=distance,
                        protein_vdw_radius=prot_radius,
                        ligand_vdw_radius=float(lig_radius),
                        clash_cutoff_distance=cutoff_distance,
                        overlap=overlap,
                    )
                )

    # Pair order has no effect on the legacy count. Sort only for deterministic
    # audit-table output across SciPy versions.
    clash_pairs.sort(
        key=lambda pair: (
            pair.ligand_atom_serial,
            pair.protein_atom_serial,
            pair.distance,
        )
    )
    clash_count = len(clash_pairs)
    pass_clash = clash_count <= int(clash_max)
    result = ClashResult(
        complex_pdb_path=str(complex_pdb_path),
        standard_structure_id=standard_structure_id,
        protein_atom_count=len(protein_atoms),
        ligand_atom_count=len(ligand_atoms),
        clash_count=clash_count,
        clash_score=float(clash_score),
        min_distance=min_distance,
        min_clash_distance=min_clash_distance,
        clash_max=int(clash_max),
        pass_clash=bool(pass_clash),
        status="passed" if pass_clash else "failed",
    )
    return result, clash_pairs


def calculate_clash_result(
    obstacle_coords: np.ndarray | Sequence[Sequence[float]],
    ligand_coords: np.ndarray | Sequence[Sequence[float]],
    *,
    threshold: float = 2.0,
    complex_pdb_path: str = "",
    standard_structure_id: str = "",
    clash_max: int = 0,
) -> ClashResult:
    """Backward-compatible coordinate clash API for Step02.

    Older Step02 code imports ``calculate_clash_result`` and expects a simple
    coordinate-threshold collision result with these attributes:
    ``n_clashes``, ``collision_score``, ``min_distance``, and
    ``weighted_direction``.  Step03 uses the newer PDB/VDW-based
    ``compute_clashes_for_complex`` API.  Keep both interfaces so adding the
    Step03 QC module never breaks Step02 docking.

    Parameters
    ----------
    obstacle_coords
        Protein and/or already placed ligand heavy-atom coordinates.
    ligand_coords
        Coordinates of the ligand atoms currently being checked.
    threshold
        Two atoms are counted as colliding when ``distance < threshold``.
        This is intentionally the legacy Step02 geometric threshold, not the
        Step03 VDW-radius clash rule.
    """
    obstacles = np.asarray(obstacle_coords, dtype=float)
    ligands = np.asarray(ligand_coords, dtype=float)

    if obstacles.size == 0:
        obstacles = obstacles.reshape((0, 3))
    elif obstacles.ndim == 1:
        obstacles = obstacles.reshape((1, 3))
    if ligands.size == 0:
        ligands = ligands.reshape((0, 3))
    elif ligands.ndim == 1:
        ligands = ligands.reshape((1, 3))

    if obstacles.shape[-1:] != (3,) or ligands.shape[-1:] != (3,):
        raise ValueError(
            "calculate_clash_result expects coordinates with shape (n, 3); "
            f"got {obstacles.shape} and {ligands.shape}"
        )

    threshold = float(threshold)
    clash_count = 0
    clash_score = 0.0
    min_distance: Optional[float] = None
    min_clash_distance: Optional[float] = None
    direction_acc = np.zeros(3, dtype=float)

    if len(obstacles) and len(ligands):
        for ligand_xyz in ligands:
            diffs = ligand_xyz[None, :] - obstacles
            dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
            if dists.size:
                local_min = float(np.min(dists))
                min_distance = local_min if min_distance is None else min(min_distance, local_min)

            mask = dists < threshold
            if np.any(mask):
                for diff, dist in zip(diffs[mask], dists[mask]):
                    distance = float(dist)
                    overlap = float(threshold - distance)
                    clash_count += 1
                    clash_score += overlap
                    min_clash_distance = (
                        distance if min_clash_distance is None else min(min_clash_distance, distance)
                    )
                    norm = float(np.linalg.norm(diff))
                    if norm > 1e-12:
                        direction_acc += (diff / norm) * max(overlap, 1e-12)
                    else:
                        # Deterministic fallback for perfectly overlapping atoms.
                        direction_acc += np.array([0.0, 0.0, 1.0], dtype=float) * max(overlap, 1e-12)

    direction_norm = float(np.linalg.norm(direction_acc))
    if direction_norm > 1e-12:
        weighted_direction = tuple((direction_acc / direction_norm).astype(float).tolist())
    else:
        weighted_direction = (0.0, 0.0, 0.0)

    pass_clash = clash_count <= int(clash_max)
    return ClashResult(
        complex_pdb_path=str(complex_pdb_path),
        standard_structure_id=str(standard_structure_id),
        protein_atom_count=int(len(obstacles)),
        ligand_atom_count=int(len(ligands)),
        clash_count=int(clash_count),
        clash_score=float(clash_score),
        min_distance=min_distance,
        min_clash_distance=min_clash_distance,
        clash_max=int(clash_max),
        pass_clash=bool(pass_clash),
        status="passed" if pass_clash else "failed",
        weighted_direction=weighted_direction,
    )


def compute_clashes_for_complex(
    complex_pdb_path: Path | str,
    standard_structure_id: str = "",
    clash_max: int = 10,
    delta: float = 0.5,
    include_hetatm_as_ligand: bool = True,
    ligand_resnames: Optional[Sequence[str]] = None,
    exclude_resnames: Sequence[str] = ("HOH", "WAT"),
    ignore_hydrogen: bool = False,
    ignore_water: bool = True,
    method: str = "collision_py_exact",
    neighbor_search_cutoff: float = 5.0,
) -> Tuple[ClashResult, List[ClashPair]]:
    """Compute protein-ligand clashes for one complex.

    ``collision_py_exact`` is the article-reproduction default and reproduces
    the uploaded colision.py exactly. ``catcongraph_vdw`` retains the newer,
    chemically expanded AF2-CatGraph implementation.
    """
    complex_pdb_path = Path(complex_pdb_path)
    if not standard_structure_id:
        standard_structure_id = complex_pdb_path.stem.replace("_complex", "").replace("_trimmed", "")

    normalized_method = normalize_clash_method(method)
    if normalized_method == "collision_py_exact":
        return _compute_collision_py_exact(
            complex_pdb_path=complex_pdb_path,
            standard_structure_id=standard_structure_id,
            clash_max=clash_max,
            delta=delta,
            neighbor_search_cutoff=neighbor_search_cutoff,
        )

    atoms = parse_pdb_atoms(complex_pdb_path)
    protein_atoms, ligand_atoms = split_protein_ligand_atoms(
        atoms,
        include_hetatm_as_ligand=include_hetatm_as_ligand,
        ligand_resnames=ligand_resnames,
        exclude_resnames=exclude_resnames,
        ignore_hydrogen=ignore_hydrogen,
        ignore_water=ignore_water,
    )

    pairs_raw, min_distance = _pairwise_clash_search(
        protein_atoms=protein_atoms,
        ligand_atoms=ligand_atoms,
        delta=delta,
    )

    clash_pairs: List[ClashPair] = []
    clash_score = 0.0
    min_clash_distance = None

    for pi, li, distance, p_rad, l_rad, overlap in pairs_raw:
        p = protein_atoms[pi]
        l = ligand_atoms[li]
        clash_score += overlap
        if min_clash_distance is None or distance < min_clash_distance:
            min_clash_distance = distance

        clash_pairs.append(
            ClashPair(
                complex_pdb_path=str(complex_pdb_path),
                standard_structure_id=standard_structure_id,
                protein_atom_serial=p.serial,
                protein_atom_name=p.atom_name,
                protein_resname=p.resname,
                protein_chain=p.chain_id,
                protein_resseq=p.resseq,
                ligand_atom_serial=l.serial,
                ligand_atom_name=l.atom_name,
                ligand_resname=l.resname,
                ligand_chain=l.chain_id,
                ligand_resseq=l.resseq,
                distance=distance,
                protein_vdw_radius=p_rad,
                ligand_vdw_radius=l_rad,
                clash_cutoff_distance=p_rad + l_rad - float(delta),
                overlap=overlap,
            )
        )

    clash_count = len(clash_pairs)
    pass_clash = clash_count <= int(clash_max)
    status = "passed" if pass_clash else "failed"

    result = ClashResult(
        complex_pdb_path=str(complex_pdb_path),
        standard_structure_id=standard_structure_id,
        protein_atom_count=len(protein_atoms),
        ligand_atom_count=len(ligand_atoms),
        clash_count=clash_count,
        clash_score=float(clash_score),
        min_distance=min_distance,
        min_clash_distance=min_clash_distance,
        clash_max=int(clash_max),
        pass_clash=bool(pass_clash),
        status=status,
    )

    return result, clash_pairs

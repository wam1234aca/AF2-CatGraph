from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import math

import numpy as np

from catcongraph.io.pdb import PDBAtomRecord, parse_pdb_atoms, replace_xyz_in_pdb_line
from catcongraph.qc.clash import ClashResult, calculate_clash_result


WATER_RESNAMES = {"HOH", "WAT", "H2O", "SOL"}


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform mapping row-vector coordinates as xyz @ rotation + translation."""

    rotation: np.ndarray
    translation: np.ndarray
    n_matched_atoms: int
    rmsd_before: float
    rmsd_after: float


@dataclass(frozen=True)
class LigandAtomBlock:
    """Selected ligand/cofactor/metal atoms for one sequential docking group."""

    ligand_id: str
    order_index: int
    source: str
    source_path: str
    atoms: list[PDBAtomRecord]
    selection_summary: str
    group_config: Mapping[str, Any]

    @property
    def coords(self) -> np.ndarray:
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=float)


@dataclass(frozen=True)
class LigandGroupResult:
    standard_structure_id: str
    ligand_id: str
    order_index: int
    source: str
    source_path: str
    selection_summary: str
    n_ligand_atoms: int
    initial_clashes: int
    initial_clash_score: float
    initial_min_distance: float | None
    final_clashes: int
    final_clash_score: float
    final_min_distance: float | None
    adjustment_status: str
    adjustment_attempts: int
    total_translation: float
    total_rotation_deg: float
    status: str
    error_message: str


@dataclass(frozen=True)
class PlacementResult:
    standard_structure_id: str
    source_pdb: str
    complex_pdb_path: str
    aligned_protein_pdb_path: str | None
    placed_ligand_pdb_path: str | None
    n_ligand_groups: int
    ligand_order: str
    n_template_ligand_atoms: int
    n_alignment_atoms: int
    alignment_rmsd_before: float
    alignment_rmsd_after: float
    initial_clashes: int
    initial_clash_score: float
    initial_min_distance: float | None
    final_clashes: int
    final_clash_score: float
    final_min_distance: float | None
    ligand_adjustment_status: str
    ligand_adjustment_attempts: int
    total_translation: float
    total_rotation_deg: float
    status: str
    error_message: str
    group_results: list[LigandGroupResult]
    adjustment_trace_rows: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Rigid protein alignment
# -----------------------------------------------------------------------------


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def kabsch_transform(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation and translation mapping moving coordinates to reference coordinates."""
    moving = np.asarray(moving, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError(f"Invalid Kabsch coordinate shapes: {moving.shape} and {reference.shape}")

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


def apply_transform(coords: np.ndarray, transform: RigidTransform | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(transform, RigidTransform):
        rotation, translation = transform.rotation, transform.translation
    else:
        rotation, translation = transform
    coords = np.asarray(coords, dtype=float)
    return coords @ rotation + translation


def _atom_identity_key(atom: PDBAtomRecord, match_by: str) -> tuple[Any, ...]:
    match_by = match_by.lower()
    if match_by == "chain_resseq_atom":
        return (atom.chain_id.strip() or "_", atom.residue_number, atom.insertion_code.strip() or "", atom.atom_name.strip())
    if match_by == "resseq_atom":
        return (atom.residue_number, atom.insertion_code.strip() or "", atom.atom_name.strip())
    raise ValueError(f"Unsupported atom match mode for key matching: {match_by}")


def _filter_atoms(
    atoms: Iterable[PDBAtomRecord],
    *,
    chain_id: str | None = None,
    atom_name: str | None = None,
    record_name: str | None = None,
) -> list[PDBAtomRecord]:
    chain_filter = None if chain_id in (None, "", "null") else str(chain_id).strip()
    atom_filter = None if atom_name in (None, "", "null") else str(atom_name).strip().upper()
    record_filter = None if record_name in (None, "", "null") else str(record_name).strip().upper()

    out: list[PDBAtomRecord] = []
    for atom in atoms:
        if chain_filter is not None and (atom.chain_id.strip() or "_") != chain_filter:
            continue
        if atom_filter is not None and atom.atom_name.strip().upper() != atom_filter:
            continue
        if record_filter is not None and atom.record_name.strip().upper() != record_filter:
            continue
        out.append(atom)
    return out


def build_template_to_target_transform(
    *,
    target_pdb: str | Path,
    template_protein_pdb: str | Path,
    atom_name: str = "CA",
    match_by: str = "resseq_atom",
    template_chain: str | None = None,
    target_chain: str | None = None,
    min_matched_atoms: int = 50,
) -> RigidTransform:
    """Calculate transform that aligns the AF2/target protein onto the template protein.

    The placement uses the established rigid-docking geometry: the AF2
    protein is moved into the template catalytic frame; ligands that are already in
    that frame are then inserted. This is template-guided rigid placement, not
    free flexible docking.
    """
    template_atoms = parse_pdb_atoms(template_protein_pdb, include_hetatm=False, atom_name=atom_name)
    target_atoms = parse_pdb_atoms(target_pdb, include_hetatm=False, atom_name=atom_name)

    template_atoms = _filter_atoms(template_atoms, chain_id=template_chain)
    target_atoms = _filter_atoms(target_atoms, chain_id=target_chain)

    if not template_atoms:
        raise ValueError(f"No template protein atoms found for atom_name={atom_name!r}")
    if not target_atoms:
        raise ValueError(f"No target protein atoms found for atom_name={atom_name!r} in {target_pdb}")

    match_by = match_by.lower()
    if match_by == "order":
        n = min(len(template_atoms), len(target_atoms))
        if n < min_matched_atoms:
            raise ValueError(f"Too few ordered alignment atoms: {n} < {min_matched_atoms}")
        template_matched = template_atoms[:n]
        target_matched = target_atoms[:n]
    elif match_by in {"chain_resseq_atom", "resseq_atom"}:
        template_by_key = {_atom_identity_key(a, match_by): a for a in template_atoms}
        target_by_key = {_atom_identity_key(a, match_by): a for a in target_atoms}
        keys = sorted(set(template_by_key).intersection(target_by_key))
        if len(keys) < min_matched_atoms:
            raise ValueError(
                f"Too few matched alignment atoms with match_by={match_by}: "
                f"{len(keys)} < {min_matched_atoms}. Try match_by='order' or check residue numbering/chain IDs."
            )
        template_matched = [template_by_key[k] for k in keys]
        target_matched = [target_by_key[k] for k in keys]
    else:
        raise ValueError(f"Unsupported alignment.match_by: {match_by}")

    moving = np.array([[a.x, a.y, a.z] for a in target_matched], dtype=float)
    reference = np.array([[a.x, a.y, a.z] for a in template_matched], dtype=float)
    rmsd_before = _rmsd(moving, reference)
    rotation, translation = kabsch_transform(moving, reference)
    aligned = moving @ rotation + translation
    rmsd_after = _rmsd(aligned, reference)
    return RigidTransform(rotation, translation, len(target_matched), rmsd_before, rmsd_after)


# -----------------------------------------------------------------------------
# Ligand/cofactor/metal selection and naming
# -----------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _ligand_group_id(group: Mapping[str, Any], order_index: int) -> str:
    raw = group.get("ligand_id") or group.get("name") or group.get("id") or group.get("resname") or f"LIG{order_index:02d}"
    if isinstance(raw, (list, tuple, set)):
        raw = "_".join(str(x) for x in raw)
    return str(raw).strip() or f"LIG{order_index:02d}"


def _selector_to_text(selector: Mapping[str, Any]) -> str:
    name = selector.get("ligand_id") or selector.get("name") or selector.get("resname") or "ligand"
    resname = selector.get("resname", "*")
    chain = selector.get("chain", "*")
    resseq = selector.get("resseq", "*")
    atoms = selector.get("atom_names", "*")
    return f"{name}(resname={resname}, chain={chain}, resseq={resseq}, atom_names={atoms})"


def _is_hydrogen(atom: PDBAtomRecord) -> bool:
    name = atom.atom_name.strip().upper()
    stripped = name.lstrip("0123456789")
    return stripped.startswith("H")


def _matches_group_selector(atom: PDBAtomRecord, selector: Mapping[str, Any]) -> bool:
    resnames = {str(x).strip().upper() for x in _as_list(selector.get("resname")) if str(x).strip()}
    chain = selector.get("chain", None)
    resseq_values = {int(x) for x in _as_list(selector.get("resseq")) if x is not None and str(x) != ""}
    atom_names = {str(x).strip().upper() for x in _as_list(selector.get("atom_names")) if str(x).strip()}

    if resnames and atom.residue_name.strip().upper() not in resnames:
        return False
    if chain not in (None, "", "*") and (atom.chain_id.strip() or "_") != str(chain).strip():
        return False
    if resseq_values and atom.residue_number not in resseq_values:
        return False
    if atom_names and atom.atom_name.strip().upper() not in atom_names:
        return False
    return True


def _select_from_pdb_atoms(atoms: list[PDBAtomRecord], group: Mapping[str, Any]) -> list[PDBAtomRecord]:
    matched: list[PDBAtomRecord] = []
    seen: set[int] = set()

    selectors = group.get("selectors")
    if selectors:
        for selector in _as_list(selectors):
            if not isinstance(selector, Mapping):
                raise ValueError(f"Ligand selector must be a mapping, got: {selector!r}")
            for atom in atoms:
                if _matches_group_selector(atom, selector) and atom.line_index not in seen:
                    matched.append(atom)
                    seen.add(atom.line_index)
        return matched

    for atom in atoms:
        if _matches_group_selector(atom, group) and atom.line_index not in seen:
            matched.append(atom)
            seen.add(atom.line_index)
    return matched


def select_ligand_group_atoms(
    *,
    group: Mapping[str, Any],
    order_index: int,
    template_complex_pdb: str | Path,
) -> LigandAtomBlock:
    """Select one ligand/cofactor/metal group for sequential placement."""
    ligand_id = _ligand_group_id(group, order_index)
    source = str(group.get("source", group.get("input_mode", "template_hetatm"))).lower()

    if source in {"template", "template_hetatm", "template_complex"}:
        source_path = str(template_complex_pdb)
        all_atoms = parse_pdb_atoms(template_complex_pdb, include_hetatm=True, atom_name=None)
        hetatm_atoms = [a for a in all_atoms if a.record_name.upper() == "HETATM"]
        hetatm_atoms = [a for a in hetatm_atoms if a.residue_name.strip().upper() not in WATER_RESNAMES]
        atoms = _select_from_pdb_atoms(hetatm_atoms, group)
    elif source in {"file", "pdb_file", "ligand_pdb"}:
        ligand_file = group.get("path") or group.get("file") or group.get("ligand_file")
        if not ligand_file:
            raise ValueError(f"Ligand group {ligand_id!r} uses source='file' but has no path/file/ligand_file")
        source_path = str(ligand_file)
        path = Path(source_path)
        if path.suffix.lower() != ".pdb":
            raise ValueError(
                f"Ligand group {ligand_id!r} uses external file {path}. "
                "Current Step 02 accepts PDB ligand files in the template coordinate frame. "
                "Convert SDF/MOL to PDB first, or use template HETATM selection."
            )
        atoms = parse_pdb_atoms(path, include_hetatm=True, atom_name=None)
        atoms = [a for a in atoms if a.record_name.upper() in {"ATOM", "HETATM"}]
        atoms = [a for a in atoms if a.residue_name.strip().upper() not in WATER_RESNAMES]
        # If resname/chain/resseq/atom_names are given, apply them. If not, use all atoms in the file.
        if any(k in group for k in ("resname", "chain", "resseq", "atom_names", "selectors")):
            atoms = _select_from_pdb_atoms(atoms, group)
    else:
        raise ValueError(f"Unsupported ligand group source for {ligand_id!r}: {source!r}")

    if not atoms:
        raise ValueError(
            f"No atoms selected for ligand group {ligand_id!r} from {source_path}. "
            f"Selector: {_selector_to_text(group)}"
        )
    return LigandAtomBlock(
        ligand_id=ligand_id,
        order_index=order_index,
        source=source,
        source_path=source_path,
        atoms=atoms,
        selection_summary=f"{_selector_to_text(group)}: {len(atoms)} atoms",
        group_config=group,
    )


def build_ligand_group_configs(template_ligands: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Build an ordered list of ligand groups from the project config.

    Preferred config key is ``step02.template_ligands.groups``. For backward
    compatibility, old ``selectors`` are treated as ordered groups.
    """
    template_ligands = template_ligands or {}
    if template_ligands.get("groups"):
        groups = list(template_ligands["groups"])
    elif template_ligands.get("selectors"):
        groups = list(template_ligands["selectors"])
    else:
        selection_mode = str(template_ligands.get("selection_mode", "all_hetatm_nonwater")).lower()
        if selection_mode == "all_hetatm_nonwater":
            groups = [
                {
                    "ligand_id": "ALL_HETATM_NONWATER",
                    "source": "template_hetatm",
                    "name": "all_nonwater_hetatm",
                }
            ]
        else:
            raise ValueError(
                "No Step 02 ligand groups were provided. Set step02.template_ligands.groups "
                "or use legacy step02.template_ligands.selectors."
            )

    cleaned: list[Mapping[str, Any]] = []
    for i, group in enumerate(groups, start=1):
        if not isinstance(group, Mapping):
            raise ValueError(f"Ligand group #{i} must be a mapping, got: {group!r}")
        copied = dict(group)
        copied.setdefault("source", template_ligands.get("default_source", "template_hetatm"))
        cleaned.append(copied)
    return cleaned


def _line_with_optional_ligand_identity(
    line: str,
    *,
    output_resname: str | None = None,
    output_chain: str | None = None,
    output_resseq: int | None = None,
    force_hetatm: bool = True,
) -> str:
    """Return a PDB line with optional ligand identity normalization."""
    raw = line.rstrip("\n")
    if len(raw) < 80:
        raw = raw.ljust(80)

    if force_hetatm:
        raw = "HETATM" + raw[6:]

    if output_resname:
        res = str(output_resname).strip().upper()
        if len(res) > 3:
            raise ValueError(
                f"PDB residue names are limited to 3 characters; got output_resname={res!r}. "
                "Use a 3-letter code or move to mmCIF in a future version."
            )
        raw = f"{raw[:17]}{res:>3}{raw[20:]}"
    if output_chain is not None:
        chain = str(output_chain).strip()
        if len(chain) > 1:
            raise ValueError(f"PDB chain IDs are limited to 1 character; got {chain!r}")
        raw = f"{raw[:21]}{chain or ' '}{raw[22:]}"
    if output_resseq is not None:
        raw = f"{raw[:22]}{int(output_resseq):4d}{raw[26:]}"
    return raw


def _replace_serial_in_pdb_line(line: str, serial: int) -> str:
    raw = line.rstrip("\n")
    if len(raw) < 11:
        raw = raw.ljust(11)
    return f"{raw[:6]}{int(serial):5d}{raw[11:]}"


# -----------------------------------------------------------------------------
# Deterministic bounded clash relief: same principle as the original dock.py
# -----------------------------------------------------------------------------


def _axis_angle_to_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + math.sin(angle_rad) * k + (1.0 - math.cos(angle_rad)) * (k @ k)


def _orthogonal_axis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    v1 = direction / norm
    v2 = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(v1, v2))) > 0.9:
        v2 = np.array([0.0, 1.0, 0.0], dtype=float)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return axis / axis_norm


def _clash_result_to_trace_fields(prefix: str, result: ClashResult) -> dict[str, Any]:
    return {
        f"{prefix}_clashes": int(result.n_clashes),
        f"{prefix}_clash_score": float(result.collision_score),
        f"{prefix}_min_distance": result.min_distance,
        f"{prefix}_weighted_direction_x": float(result.weighted_direction[0]),
        f"{prefix}_weighted_direction_y": float(result.weighted_direction[1]),
        f"{prefix}_weighted_direction_z": float(result.weighted_direction[2]),
    }


def _movement_trace_row(
    *,
    attempt: int,
    event: str,
    phase: str,
    action: str,
    before: ClashResult,
    after: ClashResult,
    translation: np.ndarray | None = None,
    rotation_axis: np.ndarray | None = None,
    rotation_deg: float = 0.0,
    rotation_center: np.ndarray | None = None,
    cumulative_translation_vec: np.ndarray | None = None,
    cumulative_rotation_deg: float = 0.0,
    status_after_attempt: str = "continued",
    best_result: ClashResult | None = None,
) -> dict[str, Any]:
    translation = np.zeros(3, dtype=float) if translation is None else np.asarray(translation, dtype=float)
    rotation_axis = np.zeros(3, dtype=float) if rotation_axis is None else np.asarray(rotation_axis, dtype=float)
    rotation_center = np.zeros(3, dtype=float) if rotation_center is None else np.asarray(rotation_center, dtype=float)
    cumulative_translation_vec = (
        np.zeros(3, dtype=float)
        if cumulative_translation_vec is None
        else np.asarray(cumulative_translation_vec, dtype=float)
    )
    best_result = best_result or after

    row: dict[str, Any] = {
        "attempt": int(attempt),
        "event": event,
        "phase": phase,
        "action": action,
        "translation_dx": float(translation[0]),
        "translation_dy": float(translation[1]),
        "translation_dz": float(translation[2]),
        "translation_norm": float(np.linalg.norm(translation)),
        "rotation_axis_x": float(rotation_axis[0]),
        "rotation_axis_y": float(rotation_axis[1]),
        "rotation_axis_z": float(rotation_axis[2]),
        "rotation_deg": float(rotation_deg),
        "rotation_center_x": float(rotation_center[0]),
        "rotation_center_y": float(rotation_center[1]),
        "rotation_center_z": float(rotation_center[2]),
        "cumulative_translation_dx": float(cumulative_translation_vec[0]),
        "cumulative_translation_dy": float(cumulative_translation_vec[1]),
        "cumulative_translation_dz": float(cumulative_translation_vec[2]),
        "cumulative_translation_norm": float(np.linalg.norm(cumulative_translation_vec)),
        "cumulative_rotation_deg": float(cumulative_rotation_deg),
        "status_after_attempt": status_after_attempt,
        "best_clashes_after_attempt": int(best_result.n_clashes),
        "best_clash_score_after_attempt": float(best_result.collision_score),
        "best_min_distance_after_attempt": best_result.min_distance,
    }
    row.update(_clash_result_to_trace_fields("before", before))
    row.update(_clash_result_to_trace_fields("after", after))
    return row


def avoid_ligand_collision_deterministic(
    *,
    obstacle_coords: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_clash_indices: list[int] | None = None,
    threshold: float = 2.0,
    max_attempts: int = 50,
    max_translation: float = 5.0,
    max_rotation_deg: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply minimal deterministic ligand-group translation/rotation to reduce clashes.

    The selected substrate/cofactor/metal group is moved as one rigid body. This
    uses bounded small translations followed by small
    rotations, then a bounded combination. No random sampling or scoring function
    is introduced here.

    The returned ``trace_rows`` field records every deterministic movement step.
    It is intended as a raw-data audit trail for method summaries, supplement
    tables, and reproducibility checks.
    """
    current = np.array(ligand_coords, dtype=float, copy=True)
    if ligand_clash_indices:
        clash_indices = np.array(ligand_clash_indices, dtype=int)
    else:
        clash_indices = np.arange(current.shape[0], dtype=int)

    def clash(coords: np.ndarray) -> ClashResult:
        return calculate_clash_result(obstacle_coords, coords[clash_indices], threshold=threshold)

    best = current.copy()
    best_result = clash(current)
    initial_result = best_result
    trace_rows: list[dict[str, Any]] = []

    if initial_result.n_clashes == 0:
        trace_rows.append(
            _movement_trace_row(
                attempt=0,
                event="initial_check",
                phase="none",
                action="none",
                before=initial_result,
                after=initial_result,
                status_after_attempt="no_initial_clash",
                best_result=initial_result,
            )
        )
        return current, {
            "status": "no_initial_clash",
            "attempts": 0,
            "initial": initial_result,
            "final": initial_result,
            "total_translation": 0.0,
            "total_rotation_deg": 0.0,
            "trace_rows": trace_rows,
        }

    total_translation_vec = np.zeros(3, dtype=float)
    total_rotation_deg = 0.0
    best_total_translation_vec = np.zeros(3, dtype=float)
    best_total_rotation_deg = 0.0
    last_score = float("inf")
    no_improvement_count = 0
    phase = 0
    attempts_performed = 0

    for attempt in range(max_attempts):
        attempt_no = attempt + 1
        result_before = clash(current)
        if result_before.collision_score < best_result.collision_score:
            best_result = result_before
            best = current.copy()
            best_total_translation_vec = total_translation_vec.copy()
            best_total_rotation_deg = total_rotation_deg
        if result_before.n_clashes == 0:
            return current, {
                "status": "resolved",
                "attempts": attempts_performed,
                "initial": initial_result,
                "final": result_before,
                "total_translation": float(np.linalg.norm(total_translation_vec)),
                "total_rotation_deg": float(total_rotation_deg),
                "trace_rows": trace_rows,
            }

        if result_before.collision_score >= last_score:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= 3:
            phase = (phase + 1) % 3
            no_improvement_count = 0
        last_score = result_before.collision_score

        direction = np.array(result_before.weighted_direction, dtype=float)
        if np.linalg.norm(direction) < 1e-12:
            direction = np.array([0.0, 0.0, 1.0], dtype=float)

        can_translate = float(np.linalg.norm(total_translation_vec)) < max_translation - 1e-9
        can_rotate = total_rotation_deg < max_rotation_deg - 1e-9

        translation = np.zeros(3, dtype=float)
        rotation_axis = np.zeros(3, dtype=float)
        rotation_center = np.zeros(3, dtype=float)
        rotation_deg = 0.0
        moved = False
        phase_label = {0: "translation", 1: "rotation", 2: "combined"}.get(phase, str(phase))
        action = "none"

        if phase == 0 and can_translate:
            step_size = min(0.1 * (1.0 + attempt / 10.0), 0.5)
            remaining = max_translation - float(np.linalg.norm(total_translation_vec))
            step_size = max(0.0, min(step_size, remaining))
            translation = direction * step_size
            current = current + translation
            total_translation_vec = total_translation_vec + translation
            moved = True
            action = "translate"
        elif phase == 1 and can_rotate:
            rotation_deg = min(2.0 * (1.0 + attempt / 20.0), 5.0, max_rotation_deg - total_rotation_deg)
            rotation_axis = _orthogonal_axis(direction)
            rotation = _axis_angle_to_rotation_matrix(rotation_axis, math.radians(rotation_deg))
            rotation_center = current.mean(axis=0)
            current = (current - rotation_center) @ rotation + rotation_center
            total_rotation_deg += rotation_deg
            moved = True
            action = "rotate"
        else:
            action_parts: list[str] = []
            if can_translate:
                step_size = min(0.05 * (1.0 + attempt / 20.0), 0.2)
                remaining = max_translation - float(np.linalg.norm(total_translation_vec))
                step_size = max(0.0, min(step_size, remaining))
                translation = direction * step_size
                current = current + translation
                total_translation_vec = total_translation_vec + translation
                moved = True
                action_parts.append("translate")
            if can_rotate:
                rotation_deg = min(1.0 * (1.0 + attempt / 30.0), 3.0, max_rotation_deg - total_rotation_deg)
                rotation_axis = _orthogonal_axis(direction)
                rotation = _axis_angle_to_rotation_matrix(rotation_axis, math.radians(rotation_deg))
                rotation_center = current.mean(axis=0)
                current = (current - rotation_center) @ rotation + rotation_center
                total_rotation_deg += rotation_deg
                moved = True
                action_parts.append("rotate")
            action = "+".join(action_parts) if action_parts else "no_allowed_move"

        attempts_performed = attempt_no if moved else attempts_performed
        result_after = clash(current) if moved else result_before
        if result_after.collision_score < best_result.collision_score:
            best_result = result_after
            best = current.copy()
            best_total_translation_vec = total_translation_vec.copy()
            best_total_rotation_deg = total_rotation_deg

        status_after_attempt = "resolved" if result_after.n_clashes == 0 else ("continued" if moved else "stopped_no_allowed_move")
        trace_rows.append(
            _movement_trace_row(
                attempt=attempt_no,
                event="movement_attempt" if moved else "stop",
                phase=phase_label,
                action=action,
                before=result_before,
                after=result_after,
                translation=translation,
                rotation_axis=rotation_axis,
                rotation_deg=rotation_deg,
                rotation_center=rotation_center,
                cumulative_translation_vec=total_translation_vec,
                cumulative_rotation_deg=total_rotation_deg,
                status_after_attempt=status_after_attempt,
                best_result=best_result,
            )
        )

        if result_after.n_clashes == 0:
            return current, {
                "status": "resolved",
                "attempts": attempt_no,
                "initial": initial_result,
                "final": result_after,
                "total_translation": float(np.linalg.norm(total_translation_vec)),
                "total_rotation_deg": float(total_rotation_deg),
                "trace_rows": trace_rows,
            }
        if not moved:
            break

    return best, {
        "status": "partial_improvement" if best_result.collision_score < initial_result.collision_score else "unresolved",
        "attempts": int(attempts_performed),
        "initial": initial_result,
        "final": best_result,
        "total_translation": float(np.linalg.norm(best_total_translation_vec)),
        "total_rotation_deg": float(best_total_rotation_deg),
        "trace_rows": trace_rows,
    }


# -----------------------------------------------------------------------------
# PDB writing
# -----------------------------------------------------------------------------


def _coords_by_line_index_for_atoms(atoms: list[PDBAtomRecord], coords: np.ndarray) -> dict[int, tuple[float, float, float]]:
    return {
        atom.line_index: (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        for atom, xyz in zip(atoms, np.asarray(coords, dtype=float))
    }


def _write_atom_subset_with_coords(
    *,
    atoms: list[PDBAtomRecord],
    coords: np.ndarray,
    output_pdb: str | Path,
    remark_lines: list[str] | None = None,
    ligand_group_config: Mapping[str, Any] | None = None,
) -> None:
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    with output_pdb.open("w", encoding="utf-8") as out:
        if remark_lines:
            for remark in remark_lines:
                out.write(f"REMARK {remark}\n")
        for serial, (atom, xyz) in enumerate(zip(atoms, np.asarray(coords, dtype=float)), start=1):
            raw = replace_xyz_in_pdb_line(atom.raw_line.rstrip("\n"), float(xyz[0]), float(xyz[1]), float(xyz[2]))
            if ligand_group_config is not None:
                raw = _line_with_optional_ligand_identity(
                    raw,
                    output_resname=ligand_group_config.get("output_resname"),
                    output_chain=ligand_group_config.get("output_chain"),
                    output_resseq=ligand_group_config.get("output_resseq"),
                    force_hetatm=True,
                )
            raw = _replace_serial_in_pdb_line(raw, serial)
            out.write(raw.rstrip("\n") + "\n")
        out.write("END\n")


def write_complex_pdb(
    *,
    target_pdb: str | Path,
    output_pdb: str | Path,
    transform: RigidTransform,
    placed_groups: list[tuple[LigandAtomBlock, np.ndarray]],
    remark_lines: list[str] | None = None,
) -> None:
    """Write aligned target protein plus sequentially placed ligand groups."""
    target_pdb = Path(target_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    target_atoms = parse_pdb_atoms(target_pdb, include_hetatm=False, atom_name=None)
    target_coords = np.array([[a.x, a.y, a.z] for a in target_atoms], dtype=float)
    aligned_target_coords = apply_transform(target_coords, transform)
    target_coord_map = _coords_by_line_index_for_atoms(target_atoms, aligned_target_coords)

    with output_pdb.open("w", encoding="utf-8") as out:
        if remark_lines:
            for remark in remark_lines:
                out.write(f"REMARK {remark}\n")

        serial = 1
        with target_pdb.open("r", encoding="utf-8", errors="replace") as inp:
            for idx, line in enumerate(inp):
                if idx in target_coord_map:
                    x, y, z = target_coord_map[idx]
                    raw = replace_xyz_in_pdb_line(line.rstrip("\n"), x, y, z)
                    raw = _replace_serial_in_pdb_line(raw, serial)
                    out.write(raw.rstrip("\n") + "\n")
                    serial += 1
        out.write("TER\n")

        for block, coords in placed_groups:
            cfg = block.group_config
            out.write(f"REMARK Ligand group {block.order_index}: {block.ligand_id}\n")
            for atom, xyz in zip(block.atoms, np.asarray(coords, dtype=float)):
                raw = replace_xyz_in_pdb_line(atom.raw_line.rstrip("\n"), float(xyz[0]), float(xyz[1]), float(xyz[2]))
                raw = _line_with_optional_ligand_identity(
                    raw,
                    output_resname=cfg.get("output_resname"),
                    output_chain=cfg.get("output_chain"),
                    output_resseq=cfg.get("output_resseq"),
                    force_hetatm=True,
                )
                raw = _replace_serial_in_pdb_line(raw, serial)
                out.write(raw.rstrip("\n") + "\n")
                serial += 1
        out.write("END\n")


# -----------------------------------------------------------------------------
# Main placement function for one AF2 structure
# -----------------------------------------------------------------------------


def _merge_adjustment_config(global_cfg: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(global_cfg or {})
    if isinstance(group_cfg.get("adjustment"), Mapping):
        merged.update(dict(group_cfg["adjustment"]))
    return merged


def _clash_heavy_indices(atoms: list[PDBAtomRecord]) -> list[int]:
    return [i for i, atom in enumerate(atoms) if not _is_hydrogen(atom)]


def _make_group_result(
    *,
    standard_structure_id: str,
    block: LigandAtomBlock,
    initial_clash: ClashResult,
    final_clash: ClashResult,
    adjustment_stats: Mapping[str, Any],
    status: str = "ok",
    error_message: str = "",
) -> LigandGroupResult:
    return LigandGroupResult(
        standard_structure_id=standard_structure_id,
        ligand_id=block.ligand_id,
        order_index=block.order_index,
        source=block.source,
        source_path=block.source_path,
        selection_summary=block.selection_summary,
        n_ligand_atoms=len(block.atoms),
        initial_clashes=initial_clash.n_clashes,
        initial_clash_score=initial_clash.collision_score,
        initial_min_distance=initial_clash.min_distance,
        final_clashes=final_clash.n_clashes,
        final_clash_score=final_clash.collision_score,
        final_min_distance=final_clash.min_distance,
        adjustment_status=str(adjustment_stats.get("status", "unknown")),
        adjustment_attempts=int(adjustment_stats.get("attempts", 0)),
        total_translation=float(adjustment_stats.get("total_translation", 0.0)),
        total_rotation_deg=float(adjustment_stats.get("total_rotation_deg", 0.0)),
        status=status,
        error_message=error_message,
    )


def place_template_ligands_for_one_structure(
    *,
    standard_structure_id: str,
    target_pdb: str | Path,
    template_complex_pdb: str | Path,
    output_complex_pdb: str | Path,
    output_aligned_protein_pdb: str | Path | None = None,
    output_placed_ligand_pdb: str | Path | None = None,
    alignment: Mapping[str, Any] | None = None,
    template_ligands: Mapping[str, Any] | None = None,
    ligand_adjustment: Mapping[str, Any] | None = None,
) -> PlacementResult:
    """Template-guided rigid placement for one AF2 protein structure.

    Multiple substrates/cofactors/metals are placed sequentially according to the
    order of ``step02.template_ligands.groups``. This gives the user explicit
    control over which molecule is placed first, second, third, and so on.
    """
    alignment = alignment or {}
    template_ligands = template_ligands or {}
    ligand_adjustment = ligand_adjustment or {}
    group_results: list[LigandGroupResult] = []
    adjustment_trace_rows: list[dict[str, Any]] = []

    try:
        template_protein_pdb = template_ligands.get("template_protein_pdb") or template_complex_pdb
        transform = build_template_to_target_transform(
            target_pdb=target_pdb,
            template_protein_pdb=template_protein_pdb,
            atom_name=str(alignment.get("atom_name", "CA")),
            match_by=str(alignment.get("match_by", "resseq_atom")),
            template_chain=alignment.get("template_chain"),
            target_chain=alignment.get("target_chain"),
            min_matched_atoms=int(alignment.get("min_matched_atoms", 50)),
        )

        group_configs = build_ligand_group_configs(template_ligands)
        ligand_blocks = [
            select_ligand_group_atoms(group=group, order_index=i, template_complex_pdb=template_complex_pdb)
            for i, group in enumerate(group_configs, start=1)
        ]

        # Aligned target protein heavy atoms are the base obstacle set.
        target_atoms = parse_pdb_atoms(target_pdb, include_hetatm=False, atom_name=None)
        target_heavy = [a for a in target_atoms if not _is_hydrogen(a)]
        target_heavy_coords = np.array([[a.x, a.y, a.z] for a in target_heavy], dtype=float)
        aligned_target_heavy_coords = apply_transform(target_heavy_coords, transform)

        placed_groups: list[tuple[LigandAtomBlock, np.ndarray]] = []
        previous_ligand_heavy_coords: list[np.ndarray] = []

        total_initial_clashes = 0
        total_final_clashes = 0
        total_initial_score = 0.0
        total_final_score = 0.0
        min_initial_distance: float | None = None
        min_final_distance: float | None = None
        total_attempts = 0
        total_translation = 0.0
        total_rotation = 0.0
        adjustment_statuses: list[str] = []

        include_previous_default = bool(ligand_adjustment.get("avoid_previous_ligands", True))

        for block in ligand_blocks:
            group_cfg = block.group_config
            group_adjustment_cfg = _merge_adjustment_config(ligand_adjustment, group_cfg)
            threshold = float(group_adjustment_cfg.get("collision_threshold", 2.0))
            adjustment_enabled = bool(group_adjustment_cfg.get("enabled", True))
            include_previous = bool(group_adjustment_cfg.get("avoid_previous_ligands", include_previous_default))
            obstacle_scope = "protein_plus_previous_ligands" if include_previous and previous_ligand_heavy_coords else "protein_only"

            ligand_coords = block.coords
            heavy_indices = _clash_heavy_indices(block.atoms)
            ligand_heavy_coords = ligand_coords[heavy_indices] if heavy_indices else ligand_coords

            obstacle_parts = [aligned_target_heavy_coords]
            if include_previous and previous_ligand_heavy_coords:
                obstacle_parts.extend(previous_ligand_heavy_coords)
            obstacles = np.vstack([part for part in obstacle_parts if part.size]) if obstacle_parts else aligned_target_heavy_coords

            initial_clash = calculate_clash_result(obstacles, ligand_heavy_coords, threshold=threshold)
            adjusted_coords = ligand_coords.copy()

            if adjustment_enabled and initial_clash.n_clashes > 0:
                adjusted_coords, adjustment_stats = avoid_ligand_collision_deterministic(
                    obstacle_coords=obstacles,
                    ligand_coords=ligand_coords,
                    ligand_clash_indices=heavy_indices,
                    threshold=threshold,
                    max_attempts=int(group_adjustment_cfg.get("max_attempts", 50)),
                    max_translation=float(group_adjustment_cfg.get("max_translation", 5.0)),
                    max_rotation_deg=float(group_adjustment_cfg.get("max_rotation_deg", 30.0)),
                )
                final_heavy_coords = adjusted_coords[heavy_indices] if heavy_indices else adjusted_coords
                final_clash = calculate_clash_result(obstacles, final_heavy_coords, threshold=threshold)
            else:
                adjustment_stats = {
                    "status": "disabled" if not adjustment_enabled else "no_initial_clash",
                    "attempts": 0,
                    "initial": initial_clash,
                    "final": initial_clash,
                    "total_translation": 0.0,
                    "total_rotation_deg": 0.0,
                    "trace_rows": [
                        _movement_trace_row(
                            attempt=0,
                            event="adjustment_disabled" if not adjustment_enabled else "initial_check",
                            phase="none",
                            action="none",
                            before=initial_clash,
                            after=initial_clash,
                            status_after_attempt="disabled" if not adjustment_enabled else "no_initial_clash",
                            best_result=initial_clash,
                        )
                    ],
                }
                final_clash = initial_clash

            for trace_row in adjustment_stats.get("trace_rows", []):
                enriched_trace = dict(trace_row)
                enriched_trace.update(
                    {
                        "standard_structure_id": standard_structure_id,
                        "ligand_id": block.ligand_id,
                        "order_index": block.order_index,
                        "source": block.source,
                        "source_path": block.source_path,
                        "n_ligand_atoms": len(block.atoms),
                        "n_ligand_heavy_atoms": len(heavy_indices),
                        "obstacle_scope": obstacle_scope,
                        "n_obstacle_atoms": int(obstacles.shape[0]) if hasattr(obstacles, "shape") else 0,
                        "collision_threshold": threshold,
                        "max_attempts": int(group_adjustment_cfg.get("max_attempts", 50)),
                        "max_translation": float(group_adjustment_cfg.get("max_translation", 5.0)),
                        "max_rotation_deg": float(group_adjustment_cfg.get("max_rotation_deg", 30.0)),
                        "adjustment_enabled": bool(adjustment_enabled),
                        "final_adjustment_status_for_group": str(adjustment_stats.get("status", "unknown")),
                    }
                )
                adjustment_trace_rows.append(enriched_trace)

            placed_groups.append((block, adjusted_coords))
            adjusted_heavy = adjusted_coords[heavy_indices] if heavy_indices else adjusted_coords
            if adjusted_heavy.size:
                previous_ligand_heavy_coords.append(adjusted_heavy)

            group_result = _make_group_result(
                standard_structure_id=standard_structure_id,
                block=block,
                initial_clash=initial_clash,
                final_clash=final_clash,
                adjustment_stats=adjustment_stats,
            )
            group_results.append(group_result)

            total_initial_clashes += int(initial_clash.n_clashes)
            total_final_clashes += int(final_clash.n_clashes)
            total_initial_score += float(initial_clash.collision_score)
            total_final_score += float(final_clash.collision_score)
            if initial_clash.min_distance is not None:
                min_initial_distance = initial_clash.min_distance if min_initial_distance is None else min(min_initial_distance, initial_clash.min_distance)
            if final_clash.min_distance is not None:
                min_final_distance = final_clash.min_distance if min_final_distance is None else min(min_final_distance, final_clash.min_distance)
            total_attempts += int(adjustment_stats.get("attempts", 0))
            total_translation += float(adjustment_stats.get("total_translation", 0.0))
            total_rotation += float(adjustment_stats.get("total_rotation_deg", 0.0))
            adjustment_statuses.append(str(adjustment_stats.get("status", "unknown")))

        ligand_order = " -> ".join(block.ligand_id for block in ligand_blocks)
        status_priority = ["unresolved", "partial_improvement", "resolved", "no_initial_clash", "disabled"]
        if any(s == "unresolved" for s in adjustment_statuses):
            overall_adjustment_status = "unresolved"
        elif any(s == "partial_improvement" for s in adjustment_statuses):
            overall_adjustment_status = "partial_improvement"
        elif any(s == "resolved" for s in adjustment_statuses):
            overall_adjustment_status = "resolved"
        elif adjustment_statuses:
            overall_adjustment_status = adjustment_statuses[0]
        else:
            overall_adjustment_status = "no_ligand_groups"

        remark_lines = [
            f"Step 02 template-guided rigid docking for {standard_structure_id}",
            f"Template complex: {template_complex_pdb}",
            f"Ligand order: {ligand_order}",
            f"Alignment atoms: {transform.n_matched_atoms}; RMSD after: {transform.rmsd_after:.3f} A",
            f"Initial total clashes: {total_initial_clashes}; final total clashes: {total_final_clashes}",
        ]

        write_complex_pdb(
            target_pdb=target_pdb,
            output_pdb=output_complex_pdb,
            transform=transform,
            placed_groups=placed_groups,
            remark_lines=remark_lines,
        )

        if output_aligned_protein_pdb:
            protein_atoms = parse_pdb_atoms(target_pdb, include_hetatm=False, atom_name=None)
            protein_coords = np.array([[a.x, a.y, a.z] for a in protein_atoms], dtype=float)
            aligned_coords = apply_transform(protein_coords, transform)
            _write_atom_subset_with_coords(
                atoms=protein_atoms,
                coords=aligned_coords,
                output_pdb=output_aligned_protein_pdb,
                remark_lines=[f"Step 02 aligned protein for {standard_structure_id}"],
            )

        if output_placed_ligand_pdb:
            output_placed_ligand_pdb = Path(output_placed_ligand_pdb)
            output_placed_ligand_pdb.parent.mkdir(parents=True, exist_ok=True)
            with output_placed_ligand_pdb.open("w", encoding="utf-8") as out:
                serial = 1
                out.write(f"REMARK Step 02 placed ligand groups for {standard_structure_id}\n")
                out.write(f"REMARK Ligand order: {ligand_order}\n")
                for block, coords in placed_groups:
                    out.write(f"REMARK Ligand group {block.order_index}: {block.ligand_id}\n")
                    for atom, xyz in zip(block.atoms, np.asarray(coords, dtype=float)):
                        raw = replace_xyz_in_pdb_line(atom.raw_line.rstrip("\n"), float(xyz[0]), float(xyz[1]), float(xyz[2]))
                        raw = _line_with_optional_ligand_identity(
                            raw,
                            output_resname=block.group_config.get("output_resname"),
                            output_chain=block.group_config.get("output_chain"),
                            output_resseq=block.group_config.get("output_resseq"),
                            force_hetatm=True,
                        )
                        raw = _replace_serial_in_pdb_line(raw, serial)
                        out.write(raw.rstrip("\n") + "\n")
                        serial += 1
                out.write("END\n")

        return PlacementResult(
            standard_structure_id=standard_structure_id,
            source_pdb=str(target_pdb),
            complex_pdb_path=str(output_complex_pdb),
            aligned_protein_pdb_path=str(output_aligned_protein_pdb) if output_aligned_protein_pdb else None,
            placed_ligand_pdb_path=str(output_placed_ligand_pdb) if output_placed_ligand_pdb else None,
            n_ligand_groups=len(ligand_blocks),
            ligand_order=ligand_order,
            n_template_ligand_atoms=sum(len(block.atoms) for block in ligand_blocks),
            n_alignment_atoms=transform.n_matched_atoms,
            alignment_rmsd_before=transform.rmsd_before,
            alignment_rmsd_after=transform.rmsd_after,
            initial_clashes=total_initial_clashes,
            initial_clash_score=total_initial_score,
            initial_min_distance=min_initial_distance,
            final_clashes=total_final_clashes,
            final_clash_score=total_final_score,
            final_min_distance=min_final_distance,
            ligand_adjustment_status=overall_adjustment_status,
            ligand_adjustment_attempts=total_attempts,
            total_translation=total_translation,
            total_rotation_deg=total_rotation,
            status="ok",
            error_message="",
            group_results=group_results,
            adjustment_trace_rows=adjustment_trace_rows,
        )
    except Exception as exc:
        return PlacementResult(
            standard_structure_id=standard_structure_id,
            source_pdb=str(target_pdb),
            complex_pdb_path=str(output_complex_pdb),
            aligned_protein_pdb_path=str(output_aligned_protein_pdb) if output_aligned_protein_pdb else None,
            placed_ligand_pdb_path=str(output_placed_ligand_pdb) if output_placed_ligand_pdb else None,
            n_ligand_groups=0,
            ligand_order="",
            n_template_ligand_atoms=0,
            n_alignment_atoms=0,
            alignment_rmsd_before=float("nan"),
            alignment_rmsd_after=float("nan"),
            initial_clashes=0,
            initial_clash_score=float("nan"),
            initial_min_distance=None,
            final_clashes=0,
            final_clash_score=float("nan"),
            final_min_distance=None,
            ligand_adjustment_status="error",
            ligand_adjustment_attempts=0,
            total_translation=0.0,
            total_rotation_deg=0.0,
            status="error",
            error_message=str(exc),
            group_results=group_results,
            adjustment_trace_rows=adjustment_trace_rows,
        )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from collections import defaultdict
import math

import numpy as np

from catcongraph.io.pdb import PDBAtomRecord, parse_pdb_atoms


WATER_RESNAMES = {"HOH", "WAT", "H2O", "SOL"}

# Atom-name presets are intentionally PDB-name based. They do not require RDKit,
# so this optional trimming step can run in a lightweight Linux/HPC environment.
# Users can always override these with explicit keep_atom_names in the config.
PRESET_ATOM_NAME_GROUPS: dict[str, list[str]] = {
    # ATP: keep triphosphate plus the ribose C5'/O5' linker atom names.
    "atp_triphosphate_core": [
        "PA", "O1A", "O2A", "O3A",
        "PB", "O1B", "O2B", "O3B",
        "PG", "O1G", "O2G", "O3G",
        "O5'", "C5'", "O5*", "C5*",
    ],
    # ADP: useful when the template ligand is ADP or an ADP-like ATP analog.
    "adp_diphosphate_core": [
        "PA", "O1A", "O2A", "O3A",
        "PB", "O1B", "O2B", "O3B",
        "O5'", "C5'", "O5*", "C5*",
    ],
    # AMP or terminal monophosphate naming variants.
    "amp_phosphate_core": [
        "P", "O1P", "O2P", "O3P", "OP1", "OP2", "OP3",
        "PA", "O1A", "O2A", "O3A",
        "O5'", "C5'", "O5*", "C5*",
    ],
    # dTMP/TMP monophosphate plus the deoxyribose C5'/O5' linker.
    "dtmp_phosphate_core": [
        "P", "O1P", "O2P", "O3P", "OP1", "OP2", "OP3",
        "O5'", "C5'", "O5*", "C5*",
    ],
    # Generic phosphate atom-name aliases for many PDB ligands.
    "generic_phosphate_atoms": [
        "P", "PA", "PB", "PG",
        "O1P", "O2P", "O3P", "OP1", "OP2", "OP3",
        "O1A", "O2A", "O3A",
        "O1B", "O2B", "O3B",
        "O1G", "O2G", "O3G",
    ],
    # Common metal ion atom names. Most metal residues should usually be kept.
    "common_metal_atoms": [
        "MG", "MN", "ZN", "FE", "FE2", "CA", "NA", "K", "CO", "NI", "CU",
    ],
}


@dataclass(frozen=True)
class TrimComplexResult:
    standard_structure_id: str
    source_complex_pdb: str
    trimmed_complex_pdb: str
    n_atoms_input: int
    n_atoms_output: int
    n_protein_atoms_input: int
    n_hetatm_input: int
    n_hetatm_output: int
    n_hetatm_removed: int
    n_rule_matched_atoms: int
    n_rules_matched: int
    status: str
    error_message: str
    atom_trace_rows: list[dict[str, Any]]
    ligand_residue_summary_rows: list[dict[str, Any]]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _norm_atom_name(name: Any) -> str:
    return str(name).strip().upper()


def _norm_resname(name: Any) -> str:
    return str(name).strip().upper()


def _is_hydrogen(atom: PDBAtomRecord) -> bool:
    name = atom.atom_name.strip().upper()
    stripped = name.lstrip("0123456789")
    return stripped.startswith("H")


def _atom_serial(atom: PDBAtomRecord) -> int | None:
    try:
        return int(atom.raw_line[6:11])
    except Exception:
        return None


def _selector_from_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    selector: dict[str, Any] = {}
    if isinstance(rule.get("match"), Mapping):
        selector.update(dict(rule["match"]))
    # Backward-compatible flat selectors.
    for key in ("resname", "chain", "resseq", "insertion_code", "atom_names", "record_name"):
        if key in rule and key not in selector:
            selector[key] = rule[key]
    return selector


def _value_set(value: Any, *, cast: type = str, upper: bool = False) -> set[Any]:
    out: set[Any] = set()
    for x in _as_list(value):
        if x is None:
            continue
        if str(x).strip() == "":
            continue
        v = cast(x)
        if upper and isinstance(v, str):
            v = v.strip().upper()
        out.add(v)
    return out


def _atom_matches_selector(atom: PDBAtomRecord, selector: Mapping[str, Any]) -> bool:
    if not selector:
        return True

    resnames = _value_set(selector.get("resname"), cast=str, upper=True)
    if resnames and _norm_resname(atom.residue_name) not in resnames:
        return False

    chains = _value_set(selector.get("chain"), cast=str, upper=False)
    if chains and "*" not in chains and (atom.chain_id.strip() or "_") not in chains:
        return False

    resseqs = _value_set(selector.get("resseq"), cast=int, upper=False)
    if resseqs and atom.residue_number not in resseqs:
        return False

    icodes = _value_set(selector.get("insertion_code"), cast=str, upper=False)
    if icodes and (atom.insertion_code.strip() or "") not in icodes:
        return False

    atom_names = _value_set(selector.get("atom_names"), cast=str, upper=True)
    if atom_names and _norm_atom_name(atom.atom_name) not in atom_names:
        return False

    records = _value_set(selector.get("record_name"), cast=str, upper=True)
    if records and atom.record_name.strip().upper() not in records:
        return False

    return True


def _rule_id(rule: Mapping[str, Any], index: int) -> str:
    return str(
        rule.get("rule_id")
        or rule.get("ligand_id")
        or rule.get("name")
        or rule.get("resname")
        or f"trim_rule_{index:02d}"
    )


def _expand_atom_names(rule: Mapping[str, Any], key: str) -> set[str]:
    names: set[str] = set()
    names.update(_norm_atom_name(x) for x in _as_list(rule.get(key)) if str(x).strip())

    preset_keys: list[Any] = []
    if key.startswith("keep"):
        preset_keys.extend(_as_list(rule.get("keep_presets")))
        preset_keys.extend(_as_list(rule.get("keep_atom_presets")))
        preset_keys.extend(_as_list(rule.get("preset_atom_groups")))
    elif key.startswith("remove"):
        preset_keys.extend(_as_list(rule.get("remove_presets")))
        preset_keys.extend(_as_list(rule.get("remove_atom_presets")))

    for preset in preset_keys:
        preset_name = str(preset).strip().lower()
        if not preset_name:
            continue
        if preset_name not in PRESET_ATOM_NAME_GROUPS:
            raise ValueError(
                f"Unknown ligand atom-name preset {preset_name!r}. "
                f"Available presets: {sorted(PRESET_ATOM_NAME_GROUPS)}"
            )
        names.update(_norm_atom_name(x) for x in PRESET_ATOM_NAME_GROUPS[preset_name])
    return names


def _distance_cutoff_from_rule(rule: Mapping[str, Any]) -> float | None:
    for key in ("distance_cutoff", "distance_to_protein", "keep_within_distance"):
        value = rule.get(key)
        if value is None or value is False or value == "":
            continue
        # keep_within_distance can be either a number or a boolean.
        if isinstance(value, bool):
            continue
        return float(value)
    return None


def _min_distance_to_protein(atom: PDBAtomRecord, protein_coords: np.ndarray) -> float | None:
    if protein_coords.size == 0:
        return None
    xyz = np.array([atom.x, atom.y, atom.z], dtype=float)
    diff = protein_coords - xyz[None, :]
    d2 = np.sum(diff * diff, axis=1)
    return float(np.sqrt(np.min(d2)))


def _find_first_matching_rule(atom: PDBAtomRecord, rules: list[Mapping[str, Any]]) -> tuple[int, Mapping[str, Any]] | tuple[None, None]:
    for i, rule in enumerate(rules, start=1):
        selector = _selector_from_rule(rule)
        if _atom_matches_selector(atom, selector):
            return i, rule
    return None, None


def _decide_ligand_atom(
    atom: PDBAtomRecord,
    *,
    rule_index: int | None,
    rule: Mapping[str, Any] | None,
    default_action: str,
    protein_coords: np.ndarray,
) -> tuple[bool, str, str, float | None]:
    """Return keep flag, rule_id, reason, min_distance_to_protein."""
    if rule is None:
        if default_action.lower() in {"remove", "drop"}:
            return False, "", "unmatched_hetatm_default_remove", None
        return True, "", "unmatched_hetatm_default_keep", None

    rid = _rule_id(rule, int(rule_index or 0))
    action = str(rule.get("action", "keep_only")).strip().lower()
    keep_names = _expand_atom_names(rule, "keep_atom_names")
    remove_names = _expand_atom_names(rule, "remove_atom_names")
    atom_name = _norm_atom_name(atom.atom_name)

    cutoff = _distance_cutoff_from_rule(rule)
    min_dist = _min_distance_to_protein(atom, protein_coords) if cutoff is not None else None
    within_distance = (min_dist is not None and min_dist <= cutoff) if cutoff is not None else False

    if action in {"keep_all", "keep", "retain_all"}:
        return True, rid, f"rule:{rid}:keep_all", min_dist

    if action in {"remove_ligand", "drop_ligand", "remove_all"}:
        return False, rid, f"rule:{rid}:remove_ligand", min_dist

    if action in {"remove", "drop", "remove_selected"}:
        if remove_names and atom_name in remove_names:
            return False, rid, f"rule:{rid}:remove_atom_name", min_dist
        return True, rid, f"rule:{rid}:not_in_remove_atom_names", min_dist

    if action in {"keep_within_distance", "distance"}:
        if cutoff is None:
            raise ValueError(f"Rule {rid!r} uses action={action!r} but no distance cutoff was provided")
        return bool(within_distance), rid, (
            f"rule:{rid}:within_{cutoff:g}A" if within_distance else f"rule:{rid}:outside_{cutoff:g}A"
        ), min_dist

    if action in {"keep_only", "keep_selected", "crop"}:
        has_name_criterion = bool(keep_names)
        has_distance_criterion = cutoff is not None
        by_name = atom_name in keep_names if has_name_criterion else False

        if not has_name_criterion and not has_distance_criterion:
            # Safe fallback: a malformed keep_only rule should not silently delete an entire ligand.
            return True, rid, f"rule:{rid}:no_keep_criteria_keep_all", min_dist

        keep = by_name or within_distance
        reasons: list[str] = []
        if by_name:
            reasons.append("keep_atom_name")
        if within_distance:
            reasons.append(f"within_{cutoff:g}A")
        if not reasons:
            reasons.append("not_selected")
        return keep, rid, f"rule:{rid}:{'+'.join(reasons)}", min_dist

    raise ValueError(
        f"Unsupported ligand atom trimming action {action!r} in rule {rid!r}. "
        "Supported actions: keep_only, keep_all, remove, remove_ligand, keep_within_distance."
    )


def _replace_serial_in_pdb_line(line: str, serial: int) -> str:
    raw = line.rstrip("\n")
    if len(raw) < 11:
        raw = raw.ljust(11)
    return f"{raw[:6]}{int(serial):5d}{raw[11:]}"


def _replace_atom_name_in_pdb_line(line: str, atom_name: str) -> str:
    raw = line.rstrip("\n")
    if len(raw) < 16:
        raw = raw.ljust(16)
    name = str(atom_name).strip()
    if len(name) > 4:
        name = name[:4]
    return f"{raw[:12]}{name:<4}{raw[16:]}"


def _build_duplicate_atom_name_map(kept_atoms: list[PDBAtomRecord]) -> dict[int, str]:
    """Rename duplicated HETATM atom names across ligand residues, if requested.

    This mirrors the user's previous utility script, but it is disabled by default
    because later ligand contact-unit grouping may intentionally rely on original atom names.
    """
    ligand_residues: dict[tuple[str, int, str, str], list[PDBAtomRecord]] = defaultdict(list)
    for atom in kept_atoms:
        if atom.record_name.upper() != "HETATM":
            continue
        res_id = (
            atom.chain_id.strip() or "_",
            atom.residue_number,
            atom.insertion_code.strip() or "",
            atom.residue_name.strip().upper(),
        )
        ligand_residues[res_id].append(atom)

    if len(ligand_residues) <= 1:
        return {}

    used: set[str] = set()
    rename_map: dict[int, str] = {}

    for i, res_id in enumerate(sorted(ligand_residues.keys())):
        atoms_in_residue = ligand_residues[res_id]
        if i == 0:
            used.update(_norm_atom_name(a.atom_name) for a in atoms_in_residue)
            continue

        for atom in atoms_in_residue:
            original = _norm_atom_name(atom.atom_name)
            if original not in used:
                used.add(original)
                continue

            symbol_guess = "".join(ch for ch in original if ch.isalpha()).strip() or original[:1] or "X"
            symbol_guess = symbol_guess[:2].upper()
            new_name = None
            for serial in range(1, 1000):
                candidate = f"{symbol_guess}{serial}"
                if len(candidate) > 4:
                    candidate = candidate[:4]
                if candidate not in used:
                    new_name = candidate
                    break
            if new_name is not None:
                used.add(new_name)
                rename_map[atom.line_index] = new_name
    return rename_map


def trim_complex_pdb(
    *,
    input_pdb: str | Path,
    output_pdb: str | Path,
    standard_structure_id: str,
    rules: list[Mapping[str, Any]] | None = None,
    default_action: str = "keep",
    remove_waters: bool = False,
    drop_conect_records: bool = True,
    renumber_atoms: bool = True,
    fix_duplicate_ligand_atom_names: bool = False,
    remark_lines: list[str] | None = None,
) -> TrimComplexResult:
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    rules = list(rules or [])

    try:
        atoms = parse_pdb_atoms(input_pdb, include_hetatm=True, atom_name=None)
        atom_by_line_index = {atom.line_index: atom for atom in atoms}
        protein_atoms = [a for a in atoms if a.record_name.upper() == "ATOM"]
        protein_heavy_atoms = [a for a in protein_atoms if not _is_hydrogen(a)]
        protein_coords = np.array([[a.x, a.y, a.z] for a in protein_heavy_atoms], dtype=float)

        keep_by_line_index: dict[int, bool] = {}
        trace_rows: list[dict[str, Any]] = []

        matched_rule_ids: set[str] = set()
        matched_rule_atom_count = 0

        for atom in atoms:
            rec = atom.record_name.upper()
            keep = True
            rid = ""
            reason = "protein_atom"
            min_dist = None

            if rec == "ATOM":
                keep = True
                rid = ""
                reason = "protein_atom"
            elif rec == "HETATM":
                if remove_waters and _norm_resname(atom.residue_name) in WATER_RESNAMES:
                    keep = False
                    rid = ""
                    reason = "water_removed"
                else:
                    rule_index, rule = _find_first_matching_rule(atom, rules)
                    keep, rid, reason, min_dist = _decide_ligand_atom(
                        atom,
                        rule_index=rule_index,
                        rule=rule,
                        default_action=default_action,
                        protein_coords=protein_coords,
                    )
                    if rid:
                        matched_rule_ids.add(rid)
                        matched_rule_atom_count += 1
            else:
                # parse_pdb_atoms only returns ATOM/HETATM; keep a safe default.
                keep = True
                rid = ""
                reason = "non_atom_record"

            keep_by_line_index[atom.line_index] = bool(keep)
            trace_rows.append(
                {
                    "standard_structure_id": standard_structure_id,
                    "source_complex_pdb": str(input_pdb),
                    "record_name": atom.record_name,
                    "atom_serial": _atom_serial(atom),
                    "atom_name_original": atom.atom_name.strip(),
                    "atom_name_output": atom.atom_name.strip(),
                    "residue_name": atom.residue_name.strip(),
                    "chain_id": atom.chain_id.strip() or "_",
                    "residue_number": atom.residue_number,
                    "insertion_code": atom.insertion_code.strip(),
                    "x": atom.x,
                    "y": atom.y,
                    "z": atom.z,
                    "matched_rule_id": rid,
                    "keep_atom": bool(keep),
                    "decision_reason": reason,
                    "min_distance_to_protein": min_dist,
                }
            )

        kept_atoms = [atom for atom in atoms if keep_by_line_index.get(atom.line_index, False)]
        rename_map = _build_duplicate_atom_name_map(kept_atoms) if fix_duplicate_ligand_atom_names else {}

        # Update trace with output atom names after optional deduplication.
        for row in trace_rows:
            line_index = None
            # Find line index by atom serial/residue/atom; atom serial is not guaranteed unique after editing.
            # We update in a second pass below by matching atom objects instead.
        trace_by_line_index = {atom.line_index: row for atom, row in zip(atoms, trace_rows)}
        for atom in atoms:
            if atom.line_index in rename_map and atom.line_index in trace_by_line_index:
                trace_by_line_index[atom.line_index]["atom_name_output"] = rename_map[atom.line_index]
                trace_by_line_index[atom.line_index]["decision_reason"] += ";duplicate_atom_name_renamed"

        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        with input_pdb.open("r", encoding="utf-8", errors="replace") as inp:
            input_lines = inp.readlines()

        out_lines: list[str] = []
        if remark_lines:
            for remark in remark_lines:
                out_lines.append(f"REMARK {remark}\n")

        serial = 1
        for idx, line in enumerate(input_lines):
            record = line[0:6].strip().upper()
            if record in {"ATOM", "HETATM"}:
                if not keep_by_line_index.get(idx, False):
                    continue
                raw = line.rstrip("\n")
                if idx in rename_map:
                    raw = _replace_atom_name_in_pdb_line(raw, rename_map[idx])
                if renumber_atoms:
                    raw = _replace_serial_in_pdb_line(raw, serial)
                    serial += 1
                out_lines.append(raw.rstrip("\n") + "\n")
            elif record == "CONECT" and drop_conect_records:
                continue
            elif record in {"END", "ENDMDL"}:
                continue
            else:
                out_lines.append(line if line.endswith("\n") else line + "\n")
        out_lines.append("END\n")

        with output_pdb.open("w", encoding="utf-8") as out:
            out.writelines(out_lines)

        kept_hetatm = [a for a in kept_atoms if a.record_name.upper() == "HETATM"]
        input_hetatm = [a for a in atoms if a.record_name.upper() == "HETATM"]

        # Per-ligand residue summary for raw-data audit.
        grouped: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
        for atom in atoms:
            if atom.record_name.upper() != "HETATM":
                continue
            row = trace_by_line_index[atom.line_index]
            key = (
                atom.residue_name.strip(),
                atom.chain_id.strip() or "_",
                atom.residue_number,
                atom.insertion_code.strip(),
                row.get("matched_rule_id", ""),
            )
            if key not in grouped:
                grouped[key] = {
                    "standard_structure_id": standard_structure_id,
                    "source_complex_pdb": str(input_pdb),
                    "trimmed_complex_pdb": str(output_pdb),
                    "residue_name": key[0],
                    "chain_id": key[1],
                    "residue_number": key[2],
                    "insertion_code": key[3],
                    "matched_rule_id": key[4],
                    "n_atoms_input": 0,
                    "n_atoms_kept": 0,
                    "n_atoms_removed": 0,
                    "kept_atom_names": [],
                    "removed_atom_names": [],
                }
            grouped[key]["n_atoms_input"] += 1
            if row["keep_atom"]:
                grouped[key]["n_atoms_kept"] += 1
                grouped[key]["kept_atom_names"].append(row["atom_name_output"])
            else:
                grouped[key]["n_atoms_removed"] += 1
                grouped[key]["removed_atom_names"].append(row["atom_name_original"])

        residue_rows: list[dict[str, Any]] = []
        for row in grouped.values():
            row["kept_atom_names"] = ";".join(row["kept_atom_names"])
            row["removed_atom_names"] = ";".join(row["removed_atom_names"])
            residue_rows.append(row)

        return TrimComplexResult(
            standard_structure_id=standard_structure_id,
            source_complex_pdb=str(input_pdb),
            trimmed_complex_pdb=str(output_pdb),
            n_atoms_input=len(atoms),
            n_atoms_output=len(kept_atoms),
            n_protein_atoms_input=len(protein_atoms),
            n_hetatm_input=len(input_hetatm),
            n_hetatm_output=len(kept_hetatm),
            n_hetatm_removed=len(input_hetatm) - len(kept_hetatm),
            n_rule_matched_atoms=matched_rule_atom_count,
            n_rules_matched=len(matched_rule_ids),
            status="ok",
            error_message="",
            atom_trace_rows=trace_rows,
            ligand_residue_summary_rows=residue_rows,
        )
    except Exception as exc:
        return TrimComplexResult(
            standard_structure_id=standard_structure_id,
            source_complex_pdb=str(input_pdb),
            trimmed_complex_pdb=str(output_pdb),
            n_atoms_input=0,
            n_atoms_output=0,
            n_protein_atoms_input=0,
            n_hetatm_input=0,
            n_hetatm_output=0,
            n_hetatm_removed=0,
            n_rule_matched_atoms=0,
            n_rules_matched=0,
            status="error",
            error_message=str(exc),
            atom_trace_rows=[],
            ligand_residue_summary_rows=[],
        )

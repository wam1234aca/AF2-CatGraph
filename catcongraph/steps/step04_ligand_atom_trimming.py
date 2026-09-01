#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step04 optional ligand atom trimming.

This step is intentionally optional. It should normally be run after Step03,
so trimming is applied only to full complexes that passed the clash filter.

The implementation is PDB-rule based and does not require RDKit. Users should
prefer `keep_atom_names` derived from `04_inspect_ligand_atoms.py` instead of
blind presets when preparing manuscript-scale analyses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import re
import pandas as pd
from catcongraph.config import load_yaml_config

from catcongraph.qc.clash import parse_pdb_atoms, PDBAtom, STANDARD_AA, WATER_RESNAMES
from catcongraph.qc.trimming import audit_trimmed_connectivity


PRESET_KEEP_ATOM_NAMES = {
    # Common ATP naming in many PDB files.
    "atp_triphosphate_core": {
        "PA", "PB", "PG",
        "O1A", "O2A", "O3A",
        "O1B", "O2B", "O3B",
        "O1G", "O2G", "O3G",
        "O5'", "C5'", "O5*", "C5*",
    },
    "adp_diphosphate_core": {
        "PA", "PB",
        "O1A", "O2A", "O3A",
        "O1B", "O2B", "O3B",
        "O5'", "C5'", "O5*", "C5*",
    },
    "amp_phosphate_core": {
        "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
        "O5'", "C5'", "O5*", "C5*",
    },
    "dtmp_phosphate_core": {
        "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
        "O5'", "C5'", "O5*", "C5*",
    },
    "generic_phosphate_atoms": {
        "P", "PA", "PB", "PG",
        "OP1", "OP2", "OP3",
        "O1P", "O2P", "O3P",
        "O1A", "O2A", "O3A",
        "O1B", "O2B", "O3B",
        "O1G", "O2G", "O3G",
        "O5'", "C5'", "O5*", "C5*",
    },
    "common_metal_atoms": {"MG", "MN", "ZN", "FE", "CU", "NI", "CO", "CA", "NA", "K"},
}


@dataclass
class AtomDecision:
    standard_structure_id: str
    complex_pdb_path: str
    output_pdb_path: str
    line_index: int
    record: str
    atom_serial: int
    atom_name_original: str
    residue_name: str
    chain_id: str
    residue_number: str
    element: str
    is_ligand_candidate: bool
    keep_atom: bool
    matched_rule_id: str
    decision_reason: str


def load_config(config_path: Path | str) -> dict:
    return load_yaml_config(config_path)


def _target_id(config: dict) -> str:
    return str(config.get("project", {}).get("target_id", "target"))


def _output_root(config: dict) -> Path:
    project = config.get("project", {})
    return Path(project.get("output_root", f"results/{_target_id(config)}"))


def resolve_step04_input_table(config: dict) -> Path:
    """Prefer Step03-passed complexes; fall back to Step02 complexes if explicitly allowed."""
    target_id = _target_id(config)
    step04 = config.get("step04", {})
    requested = step04.get("input_complex_table", "auto")

    if requested and requested != "auto":
        return Path(requested)

    root = _output_root(config)
    candidates = [
        root / "03_clash_filter" / "tables" / f"{target_id}_step03_passed_complex_index.csv",
        root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_complex_index.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find Step04 input complex table. Run Step03 first or set step04.input_complex_table. "
        f"Checked: {', '.join(str(x) for x in candidates)}"
    )


def _infer_complex_path_column(df: pd.DataFrame, configured: str = "auto") -> str:
    if configured and configured != "auto":
        if configured not in df.columns:
            raise KeyError(f"Configured complex path column '{configured}' not found in input table.")
        return configured

    for col in ["complex_pdb_path", "output_complex_pdb", "complex_path", "pdb_path", "path"]:
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


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _norm_set(value) -> set:
    return {str(x).strip().upper() for x in _as_list(value) if str(x).strip()}


def _read_bool(value, default: bool = False, field_name: str = "boolean") -> bool:
    """Parse config booleans without treating typos like 'ture' as True."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid {field_name}: {value!r}. Use true/false, not a free-text string.")


def is_ligand_candidate(atom: PDBAtom, cfg: dict) -> bool:
    resname = atom.resname.upper()
    if atom.record != "HETATM":
        return False
    if resname in WATER_RESNAMES:
        return False
    exclude = _norm_set(cfg.get("exclude_resnames", ["HOH", "WAT"]))
    if resname in exclude:
        return False
    return True


def rule_matches_atom(rule: dict, atom: PDBAtom) -> bool:
    match = rule.get("match", {}) or {}

    if "resname" in match:
        if atom.resname.upper() not in _norm_set(match["resname"]):
            return False
    if "chain" in match:
        chains = _norm_set(match["chain"])
        if chains and atom.chain_id.upper() not in chains:
            return False
    if "resseq" in match:
        resseqs = {str(x).strip() for x in _as_list(match["resseq"])}
        if resseqs and atom.resseq.strip() not in resseqs:
            return False
    if "atom_names" in match:
        names = _norm_set(match["atom_names"])
        if names and atom.atom_name.upper() not in names:
            return False

    return True


def collect_keep_names(rule: dict) -> set:
    keep = _norm_set(rule.get("keep_atom_names", []))
    for preset in _as_list(rule.get("keep_presets", [])):
        keep.update(x.upper() for x in PRESET_KEEP_ATOM_NAMES.get(str(preset), set()))
    return keep


def decide_atom(atom: PDBAtom, rules: list, atom_cfg: dict) -> Tuple[bool, str, str]:
    """Return keep_atom, matched_rule_id, decision_reason."""
    if not is_ligand_candidate(atom, atom_cfg):
        return True, "", "non_ligand_or_excluded"

    default_action = str(atom_cfg.get("default_action", "keep")).lower()

    for rule in rules:
        if not rule_matches_atom(rule, atom):
            continue

        rule_id = str(rule.get("rule_id", "unnamed_rule"))
        action = str(rule.get("action", "keep_only")).lower()

        if action == "keep_all":
            return True, rule_id, "rule_keep_all"

        if action == "remove_all":
            return False, rule_id, "rule_remove_all"

        if action == "remove_atom_names":
            remove_names = _norm_set(rule.get("remove_atom_names", []))
            if atom.atom_name.upper() in remove_names:
                return False, rule_id, "rule_remove_atom_name"
            return True, rule_id, "rule_remove_atom_names_not_selected"

        if action == "keep_only":
            keep_names = collect_keep_names(rule)
            if atom.atom_name.upper() in keep_names:
                return True, rule_id, "rule_keep_only_selected"
            return False, rule_id, "rule_keep_only_not_selected"

        raise ValueError(f"Unknown trimming action '{action}' in rule '{rule_id}'")

    if default_action == "keep":
        return True, "", "default_keep"
    if default_action == "remove":
        return False, "", "default_remove"

    raise ValueError(f"Unknown default_action: {default_action}")


def _validate_keep_only_rules(trace_rows: List[AtomDecision], atom_cfg: dict) -> None:
    """Fail fast when a keep_only rule matched a ligand residue but kept zero atoms.

    This prevents a common silent failure: a preset atom list does not match the
    actual ligand atom names, so an ATP/TMP/substrate residue is completely
    removed and only metals remain.  Users can disable this with
    atom_selection.fail_on_empty_keep_only: false, but the safe default is to
    stop and show the mismatching atom names in the Step04 failed table.
    """
    if not _read_bool(atom_cfg.get("fail_on_empty_keep_only", True), True, "atom_selection.fail_on_empty_keep_only"):
        return

    grouped: Dict[Tuple[str, str, str, str], List[AtomDecision]] = {}
    for row in trace_rows:
        if not row.is_ligand_candidate or not row.matched_rule_id:
            continue
        if not row.decision_reason.startswith("rule_keep_only"):
            continue
        key = (row.matched_rule_id, row.residue_name, row.chain_id, row.residue_number)
        grouped.setdefault(key, []).append(row)

    errors = []
    for (rule_id, resname, chain_id, resseq), rows in grouped.items():
        if rows and not any(r.keep_atom for r in rows):
            atom_names = ",".join(sorted({r.atom_name_original for r in rows}))
            errors.append(
                f"rule {rule_id!r} matched ligand residue {resname}:{chain_id or '_'}:{resseq} "
                f"but kept 0 atoms. Actual atom names: {atom_names}"
            )
    if errors:
        raise ValueError("; ".join(errors))


def trim_one_complex(
    complex_pdb_path: Path,
    output_pdb_path: Path,
    standard_structure_id: str,
    atom_cfg: dict,
) -> Tuple[Dict, List[AtomDecision]]:
    rules = atom_cfg.get("rules", []) or []

    atoms = parse_pdb_atoms(complex_pdb_path)
    decisions_by_line = {}
    trace_rows: List[AtomDecision] = []

    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)

    for atom in atoms:
        keep, rule_id, reason = decide_atom(atom, rules, atom_cfg)
        decisions_by_line[atom.line_index] = keep
        trace_rows.append(
            AtomDecision(
                standard_structure_id=standard_structure_id,
                complex_pdb_path=str(complex_pdb_path),
                output_pdb_path=str(output_pdb_path),
                line_index=atom.line_index,
                record=atom.record,
                atom_serial=atom.serial,
                atom_name_original=atom.atom_name,
                residue_name=atom.resname,
                chain_id=atom.chain_id,
                residue_number=atom.resseq,
                element=atom.element,
                is_ligand_candidate=is_ligand_candidate(atom, atom_cfg),
                keep_atom=bool(keep),
                matched_rule_id=rule_id,
                decision_reason=reason,
            )
        )

    _validate_keep_only_rules(trace_rows, atom_cfg)

    # A trimming operation must not erase the substrate topology used by
    # Step06/Step09/Step10.  CONECT records are therefore preserved by default
    # and rewritten below after removed atoms and optional serial renumbering.
    drop_conect = _read_bool(atom_cfg.get("drop_conect_records", False), False, "atom_selection.drop_conect_records")
    renumber_atoms = _read_bool(atom_cfg.get("renumber_atoms", True), True, "atom_selection.renumber_atoms")

    kept_atom_lines = 0
    removed_atom_lines = 0
    output_lines = []

    serial_mapping: Dict[int, int] = {}
    next_serial = 1
    for atom in atoms:
        if not decisions_by_line.get(atom.line_index, True):
            continue
        serial_mapping[int(atom.serial)] = next_serial if renumber_atoms else int(atom.serial)
        if renumber_atoms:
            next_serial += 1

    def rewritten_conect(line: str) -> List[str]:
        """Keep only bonds whose two atoms survived, with remapped serials."""
        numbers = [int(value) for value in re.findall(r"\d+", line[6:])]
        if len(numbers) < 2 or numbers[0] not in serial_mapping:
            return []
        source = serial_mapping[numbers[0]]
        neighbors = [serial_mapping[value] for value in numbers[1:] if value in serial_mapping]
        if not neighbors:
            return []
        return [
            "CONECT" + f"{source:5d}" + "".join(f"{neighbor:5d}" for neighbor in neighbors[index:index + 4])
            for index in range(0, len(neighbors), 4)
        ]

    with complex_pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            rec = line[0:6].strip()
            if rec in {"ATOM", "HETATM"}:
                keep = decisions_by_line.get(idx, True)
                if not keep:
                    removed_atom_lines += 1
                    continue
                kept_atom_lines += 1
                if renumber_atoms:
                    old_serial = int(line[6:11])
                    line = f"{line[:6]}{serial_mapping[old_serial]:5d}{line[11:]}"
                output_lines.append(line.rstrip("\n"))
            elif rec == "CONECT":
                if not drop_conect:
                    output_lines.extend(rewritten_conect(line))
            else:
                output_lines.append(line.rstrip("\n"))

    output_pdb_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    summary = {
        "standard_structure_id": standard_structure_id,
        "input_complex_pdb_path": str(complex_pdb_path),
        "trimmed_complex_pdb_path": str(output_pdb_path),
        "kept_atom_lines": kept_atom_lines,
        "removed_atom_lines": removed_atom_lines,
        "status": "success",
        "error_message": "",
    }

    return summary, trace_rows


def run_step04(config_path: Path | str) -> Dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    target_id = _target_id(config)
    root = _output_root(config)

    step04 = config.get("step04", {})
    enabled = _read_bool(step04.get("enabled", False), False, "step04.enabled")
    if not enabled:
        return {
            "target_id": target_id,
            "status": "skipped",
            "message": "step04.enabled is false. This optional trimming step was not run.",
        }

    output_dir = root / step04.get("output_subdir", "04_ligand_atom_trimming")
    complexes_dir = output_dir / "complexes"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for d in [complexes_dir, tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    input_table = resolve_step04_input_table(config)
    df = pd.read_csv(input_table)
    path_col = _infer_complex_path_column(df, step04.get("input_complex_column", "auto"))
    atom_cfg = step04.get("atom_selection", {}) or {}

    index_rows = []
    trace_rows = []
    failed_rows = []
    connectivity_audit_rows = []

    for _, row in df.iterrows():
        pdb_path = Path(str(row[path_col]))
        sid = _infer_structure_id(row, pdb_path)
        out_pdb = complexes_dir / f"{sid}_trimmed_complex.pdb"

        try:
            if not pdb_path.exists():
                raise FileNotFoundError(f"Input complex PDB not found: {pdb_path}")
            summary, trace = trim_one_complex(
                complex_pdb_path=pdb_path,
                output_pdb_path=out_pdb,
                standard_structure_id=sid,
                atom_cfg=atom_cfg,
            )
            base = row.to_dict()
            base.update(summary)
            base["complex_pdb_path"] = str(out_pdb)
            index_rows.append(base)
            trace_rows.extend(asdict(t) for t in trace)
            connectivity_audit_rows.append(
                audit_trimmed_connectivity(
                    pdb_path,
                    [asdict(t) for t in trace],
                    structure_id=sid,
                )
            )
        except Exception as exc:
            base = row.to_dict()
            base.update(
                {
                    "standard_structure_id": sid,
                    "input_complex_pdb_path": str(pdb_path),
                    "trimmed_complex_pdb_path": "",
                    "complex_pdb_path": "",
                    "kept_atom_lines": None,
                    "removed_atom_lines": None,
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            failed_rows.append(base)

    index = pd.DataFrame(index_rows)
    trace_df = pd.DataFrame(trace_rows)
    failed = pd.DataFrame(failed_rows)
    connectivity_audit = pd.DataFrame(connectivity_audit_rows)

    # Residue summary is useful for quickly checking over-trimming.
    residue_summary = pd.DataFrame()
    if len(trace_df):
        residue_summary = (
            trace_df[trace_df["is_ligand_candidate"] == True]
            .groupby(["standard_structure_id", "residue_name", "chain_id", "residue_number"], dropna=False)
            .agg(
                total_atoms=("atom_name_original", "count"),
                kept_atoms=("keep_atom", "sum"),
                removed_atoms=("keep_atom", lambda x: int((~x.astype(bool)).sum())),
                atom_names=("atom_name_original", lambda x: ",".join(map(str, x))),
                kept_atom_names=("atom_name_original", lambda x: ",".join(map(str, trace_df.loc[x.index][trace_df.loc[x.index, "keep_atom"] == True]["atom_name_original"]))),
                removed_atom_names=("atom_name_original", lambda x: ",".join(map(str, trace_df.loc[x.index][trace_df.loc[x.index, "keep_atom"] == False]["atom_name_original"]))),
            )
            .reset_index()
        )

    index_path = tables_dir / f"{target_id}_step04_trimmed_complex_index.csv"
    summary_path = tables_dir / f"{target_id}_step04_ligand_atom_trim_summary.csv"
    trace_path = tables_dir / f"{target_id}_step04_ligand_atom_trace.csv"
    residue_path = tables_dir / f"{target_id}_step04_ligand_residue_summary.csv"
    failed_path = tables_dir / f"{target_id}_step04_failed_structures.csv"
    connectivity_audit_path = tables_dir / f"{target_id}_step04_connectivity_audit.csv"

    index.to_csv(index_path, index=False)
    index.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    residue_summary.to_csv(residue_path, index=False)
    failed.to_csv(failed_path, index=False)
    connectivity_audit.to_csv(connectivity_audit_path, index=False)

    report = {
        "target_id": target_id,
        "status": "success",
        "optional_step": True,
        "input_complex_table": str(input_table),
        "input_complex_column": path_col,
        "n_input_complexes": int(len(df)),
        "n_trimmed_complexes": int(len(index)),
        "n_failed": int(len(failed)),
        "files": {
            "trimmed_complex_index": str(index_path),
            "atom_trace": str(trace_path),
            "residue_summary": str(residue_path),
            "failed": str(failed_path),
            "connectivity_audit": str(connectivity_audit_path),
        },
    }

    (reports_dir / f"{target_id}_step04_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# Step04 optional ligand atom trimming summary: {target_id}",
        "",
        "> This is an optional analysis-preprocessing step, not a mandatory main-pipeline filter.",
        "",
        f"- Input complexes: {len(df)}",
        f"- Trimmed complexes: {len(index)}",
        f"- Failed complexes: {len(failed)}",
        "",
        "## Output files",
        "",
        f"- `{index_path}`",
        f"- `{trace_path}`",
        f"- `{residue_path}`",
        f"- `{failed_path}`",
        "",
    ]
    (reports_dir / f"{target_id}_step04_summary.md").write_text("\n".join(md), encoding="utf-8")

    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CatConGraph Step04 optional ligand atom trimming.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    args = parser.parse_args(argv)

    report = run_step04(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from catcongraph.config import load_yaml_config
from catcongraph.profiles import workflow_profile
from catcongraph.ligand.docking_general import (
    assess_legacy_compatibility,
    assess_sequence_aware_compatibility,
)


def _resolve(path_value: object, config_path: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, config_path.parent / path]
    if config_path.parent.parent.name == "projects":
        candidates.append(config_path.parent.parent.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _ligand_values(config: Mapping[str, Any]) -> list[object]:
    inputs = config.get("inputs", {}) or {}
    value = inputs.get("ligand_pdbs") or inputs.get("ligand_pdb") or []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def run_preflight(config_path: str | Path, max_structures: Optional[int] = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    project = cfg.get("project", {}) or {}
    inputs = cfg.get("inputs", {}) or {}
    target_id = str(project.get("target_id") or config_path.parent.name)
    errors: list[str] = []
    warnings: list[str] = []

    af2_value = inputs.get("af2_dir")
    template_value = inputs.get("template_protein_pdb")
    if not af2_value:
        errors.append("Missing inputs.af2_dir")
        af2_dir = None
    else:
        af2_dir = _resolve(af2_value, config_path)
        if not af2_dir.is_dir():
            errors.append(f"AF2 input directory not found: {af2_dir}")

    if not template_value:
        errors.append("Missing inputs.template_protein_pdb")
        template = None
    else:
        template = _resolve(template_value, config_path)
        if not template.is_file():
            errors.append(f"Template protein PDB not found: {template}")

    ligands = []
    for value in _ligand_values(cfg):
        if not value:
            continue
        path = _resolve(value, config_path)
        ligands.append(path)
        if not path.is_file():
            errors.append(f"Ligand/cofactor file not found: {path}")
    if not ligands:
        errors.append("No inputs.ligand_pdb or inputs.ligand_pdbs were provided")

    pdbs = []
    if af2_dir and af2_dir.is_dir():
        pdbs = sorted(p for p in af2_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdb")
        if not pdbs:
            errors.append(f"No PDB files found directly in {af2_dir}")
    selected = pdbs[:max_structures] if max_structures else pdbs

    compatibility_rows = []
    step02 = cfg.get("step02", {}) or {}
    alignment_mode = str(step02.get("alignment_mode", "auto") or "auto").strip().lower()
    alignment_cfg = step02.get("general_alignment", {}) or {}
    if template and template.is_file():
        for pdb in selected:
            result = assess_legacy_compatibility(template, pdb)
            sequence_result = assess_sequence_aware_compatibility(
                template, pdb, config=alignment_cfg
            )
            compatibility_rows.append(
                {
                    "pdb": str(pdb),
                    **result.__dict__,
                    "sequence_aware_compatible": sequence_result.compatible,
                    "sequence_aware_reason": sequence_result.reason,
                    "sequence_aware_matched_ca_count": sequence_result.matched_ca_count,
                    "sequence_aware_identity": sequence_result.sequence_identity,
                    "sequence_aware_coverage": sequence_result.alignment_coverage,
                    "sequence_aware_chain_mapping": sequence_result.chain_mapping,
                }
            )
    n_legacy = sum(bool(x["legacy_compatible"]) for x in compatibility_rows)
    allow_unsafe = bool(step02.get("allow_unsafe_legacy_alignment", False))
    for row in compatibility_rows:
        if alignment_mode in {"coordinate_matching", "dock_py_exact", "legacy", "original"}:
            if not row["legacy_compatible"] and not allow_unsafe:
                errors.append(
                    f"{Path(row['pdb']).name}: exact dock.py alignment is unsafe "
                    f"({row['compatibility_reason']})"
                )
        elif alignment_mode in {"auto", "compatible_auto", "sequence_aware", "general"}:
            needs_sequence = alignment_mode in {"sequence_aware", "general"} or not row["legacy_compatible"]
            if needs_sequence and not row["sequence_aware_compatible"]:
                errors.append(
                    f"{Path(row['pdb']).name}: sequence-aware alignment preflight failed "
                    f"({row['sequence_aware_reason']}; identity={row['sequence_aware_identity']:.3f}; "
                    f"coverage={row['sequence_aware_coverage']:.3f}; "
                    f"matched_CA={row['sequence_aware_matched_ca_count']})"
                )
        else:
            errors.append(
                "step02.alignment_mode must be auto, coordinate_matching, or sequence_aware"
            )
            break

    step05_names = set(((cfg.get("step05", {}) or {}).get("catalytic_region", {}) or {}).get("ligand_resnames") or [])
    step07_names = set(((cfg.get("step07", {}) or {}).get("ligand", {}) or {}).get("include_resnames") or [])
    if step05_names and step07_names and step05_names != step07_names:
        warnings.append(
            "step05.catalytic_region.ligand_resnames and step07.ligand.include_resnames differ: "
            f"{sorted(step05_names)} vs {sorted(step07_names)}"
        )

    # Parse the conservation input during preflight.  This catches a website
    # ``new1.gz``/tar download, a damaged archive, or an unsuitable file before
    # the computationally expensive docking and PLIP stages are launched.
    step07 = cfg.get("step07", {}) or {}
    conservation_cfg = step07.get("conservation", {}) or {}
    conservation_value = (
        step07.get("conservation_file")
        or conservation_cfg.get("file")
        or conservation_cfg.get("zip")
    )
    conservation_input: dict[str, Any] = {
        "configured_path": str(conservation_value or ""),
        "status": "not_checked",
    }
    if not conservation_value:
        errors.append(
            "Missing Step07 ConSurf input: set step07.conservation_file to a ConSurf ZIP, TAR.GZ/GZ, or CSV/TSV file."
        )
        conservation_input["status"] = "missing"
    else:
        conservation_path = _resolve(conservation_value, config_path)
        conservation_input["resolved_path"] = str(conservation_path)
        if not conservation_path.is_file():
            errors.append(f"ConSurf conservation input not found: {conservation_path}")
            conservation_input["status"] = "missing_file"
        else:
            try:
                # Deferred import keeps unrelated lightweight CLI commands from
                # importing the Step07 parser unless preflight is requested.
                from catcongraph.steps.step07_key_residue_screening import (
                    assess_consurf_mapping_against_complexes,
                    load_consurf_scores,
                    map_consurf_scores_to_complexes,
                )

                scores, conservation_metadata = load_consurf_scores(conservation_path, conservation_cfg)
                conservation_input.update(conservation_metadata)
                conservation_input["n_consurf_residues"] = int(len(scores))
                mapping_records = pd.DataFrame(
                    [{"structure_id": p.stem, "pdb_path": str(p)} for p in selected]
                )
                if not mapping_records.empty:
                    mapped_scores, mapping_audit, mapping_warnings = map_consurf_scores_to_complexes(
                        scores, mapping_records, conservation_cfg
                    )
                    overlap_audit, overlap_warnings = assess_consurf_mapping_against_complexes(
                        mapped_scores, mapping_records
                    )
                    conservation_input["mapping"] = {**mapping_audit, **overlap_audit}
                    warnings.extend(mapping_warnings + overlap_warnings)
                    mapping_cfg = conservation_cfg.get("mapping", {}) or {}
                    minimum = float(mapping_cfg.get("min_target_coverage", 0.90))
                    target_coverage = mapping_audit.get("target_residue_coverage")
                    conflict_rows = int(mapping_audit.get("n_conflict_rows", 0) or 0)
                    if conflict_rows:
                        message = (
                            f"ConSurf mapping produced {conflict_rows} non-unique source-to-target residue assignments."
                        )
                        if bool(mapping_cfg.get("fail_on_mapping_conflict", True)):
                            errors.append(message + " Resolve chain/offset settings before running Step07.")
                        else:
                            warnings.append(message)
                    if target_coverage is not None and float(target_coverage) < minimum:
                        message = (
                            f"ConSurf target-residue mapping coverage is {float(target_coverage):.1%}, "
                            f"below the configured preflight minimum {minimum:.1%}."
                        )
                        if bool(mapping_cfg.get("fail_on_low_target_coverage", True)):
                            errors.append(message + " Review construct boundaries, chain IDs, or residue offsets.")
                        else:
                            warnings.append(message)
                conservation_input["status"] = "parsed"
            except Exception as exc:
                errors.append(f"Could not parse ConSurf conservation input {conservation_path}: {exc}")
                conservation_input["status"] = "parse_failed"
                conservation_input["parse_error"] = str(exc)

    plip = str((cfg.get("step06", {}) or {}).get("plip_executable", "plip"))
    if not Path(plip).exists() and shutil.which(plip) is None:
        warnings.append(f"PLIP executable was not found during preflight: {plip}")
    ged = str((((cfg.get("step08", {}) or {}).get("ged", {}) or {}).get("executable", "ged")))
    if not Path(ged).exists() and shutil.which(ged) is None:
        warnings.append(f"GED executable was not found during preflight: {ged}")

    report = {
        "target_id": target_id,
        "workflow_profile": workflow_profile(cfg),
        "config": str(config_path),
        "status": "failed" if errors else ("warning" if warnings else "passed"),
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "af2_dir": str(af2_dir) if af2_dir else "",
            "template_protein_pdb": str(template) if template else "",
            "ligands": [str(x) for x in ligands],
            "n_pdbs": len(pdbs),
            "n_pdbs_checked": len(selected),
            "conservation": conservation_input,
        },
        "docking_compatibility": {
            "alignment_mode_configured": alignment_mode,
            "n_legacy_compatible": n_legacy,
            "n_sequence_aware_required": len(compatibility_rows) - n_legacy,
            "n_sequence_aware_compatible": sum(
                bool(x["sequence_aware_compatible"]) for x in compatibility_rows
            ),
            "structures": compatibility_rows,
        },
    }
    output_root = _resolve(project.get("output_root", f"results/{target_id}"), config_path)
    report_dir = output_root / "00_preflight"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{target_id}_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import shutil
import sys
import re
from typing import Any

import pandas as pd

from catcongraph.config import get_nested, load_yaml_config, require_nested
from catcongraph.io.paths import ensure_dir, list_files
from catcongraph.io.pdb import extract_colabfold_metadata
from catcongraph.qc.plddt import compute_global_plddt
from catcongraph.structure.align import align_pdb_to_reference


def _format_standard_id(prefix: str, index: int, width: int) -> str:
    return f"{prefix}_{index:0{width}d}"


def _natural_key(value: object) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def _extract_prediction_index(filename: str, target_id: str | None = None) -> int | None:
    """Extract the AF2/ColabFold prediction number from a filename.

    This is for manuscript traceability: if the AF2 prediction is #100, the
    standardized ID can remain *_0100 instead of becoming *_0001 simply because
    it was the first file encountered after filtering/sorting.

    Preferred patterns include:
      <target>_100_Chain_A...pdb
      <anything>_prediction_100...pdb
      <anything>_model_100...pdb
      <anything>_rank_001...pdb  (fallback only)
    """
    stem = Path(filename).stem
    tid = re.escape(str(target_id or "").strip())
    patterns: list[str] = []
    if tid:
        patterns.extend([rf"^{tid}[_-](\d+)(?:[_-]|$)", rf"^{tid}.*?[_-](\d+)[_-]Chain"])
    patterns.extend([
        r"(?:^|[_-])prediction[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])pred[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])model[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])(\d+)[_-]Chain(?:[_-]|$)",
        r"(?:^|[_-])rank[_-]?0*(\d+)(?:[_-]|$)",
    ])
    for pat in patterns:
        m = re.search(pat, stem, flags=re.I)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    # Conservative final fallback: first standalone integer in the stem.
    m = re.search(r"(?:^|[_-])(\d+)(?:[_-]|$)", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "none":
        return
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


_PLDDT_FILTER_SCOPES = {"input", "source_if_available", "source_required"}


def _normalise_plddt_filter_scope(value: object) -> str:
    """Return a documented Step01 pLDDT-filter scope.

    ``input`` preserves the historical behaviour: score the PDB files selected
    in ``inputs.af2_dir``.  This is useful when that directory is intentionally
    a domain-only input.  The two ``source`` modes use the audit manifest
    produced by Step 0A to recover the *untrimmed* AF2 PDB before applying the
    global quality threshold.  Docking still always receives the trimmed PDB.
    """
    raw = str(value or "source_if_available").strip().lower()
    aliases = {
        "legacy": "input",
        "current": "input",
        "trimmed": "input",
        "source": "source_if_available",
        "original": "source_if_available",
        "full_length": "source_if_available",
        "full_length_required": "source_required",
    }
    scope = aliases.get(raw, raw)
    if scope not in _PLDDT_FILTER_SCOPES:
        choices = ", ".join(sorted(_PLDDT_FILTER_SCOPES))
        raise ValueError(f"step01.plddt_filter_scope must be one of {choices}, got {value!r}")
    return scope


def _manifest_source_paths(
    af2_dir: Path,
    manifest_setting: object,
) -> tuple[dict[Path, Path], dict[str, Path], Path | None, str]:
    """Read the Step 0A audit manifest without guessing source identities.

    The preprocessing manifest is the only reliable relation between a renamed,
    trimmed PDB and its original AF2 PDB.  Name-only matching is kept only as a
    unique fallback for copied directories; ambiguous names are deliberately
    left unresolved rather than silently scoring the wrong prediction.
    """
    raw = str(manifest_setting or "auto").strip()
    if raw.lower() in {"", "auto", "none", "null", "~"}:
        manifest_path = af2_dir / "af2_preprocess_manifest.csv"
    else:
        candidate = Path(raw).expanduser()
        manifest_path = candidate if candidate.is_absolute() else (af2_dir / candidate)
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        return {}, {}, None, "manifest_not_found"

    by_output: dict[Path, Path] = {}
    by_name_candidates: dict[str, set[Path]] = {}
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                output_text = str(row.get("output", "")).strip()
                source_text = str(row.get("source", "")).strip()
                if not output_text or not source_text:
                    continue
                output = Path(output_text).expanduser()
                source = Path(source_text).expanduser()
                if not output.is_absolute():
                    output = (manifest_path.parent / output)
                if not source.is_absolute():
                    source = (manifest_path.parent / source)
                output = output.resolve()
                source = source.resolve()
                by_output[output] = source
                by_name_candidates.setdefault(output.name, set()).add(source)
    except Exception as exc:
        return {}, {}, manifest_path, f"manifest_unreadable: {exc}"

    by_unique_name = {
        name: next(iter(sources))
        for name, sources in by_name_candidates.items()
        if len(sources) == 1
    }
    return by_output, by_unique_name, manifest_path, "ok"


def _plddt_values(prefix: str, result) -> dict[str, object]:
    """Expand a pLDDT result into stable, explicit CSV column names."""
    if result is None:
        return {
            f"{prefix}_n_atoms": None,
            f"{prefix}_n_residues": None,
            f"{prefix}_global_mean_plddt": None,
            f"{prefix}_global_median_plddt": None,
            f"{prefix}_global_p10_plddt": None,
            f"{prefix}_global_min_plddt": None,
            f"{prefix}_global_max_plddt": None,
        }
    return {
        f"{prefix}_n_atoms": result.n_atoms,
        f"{prefix}_n_residues": result.n_residues,
        f"{prefix}_global_mean_plddt": result.global_mean_plddt,
        f"{prefix}_global_median_plddt": result.global_median_plddt,
        f"{prefix}_global_p10_plddt": result.global_p10_plddt,
        f"{prefix}_global_min_plddt": result.global_min_plddt,
        f"{prefix}_global_max_plddt": result.global_max_plddt,
    }


def run_step01(config: dict[str, Any]) -> dict[str, Any]:
    """Run Step 01.

    Step 01 is intentionally protein-only:
    - index raw AF2/ColabFold PDB structures;
    - optionally assign standardized structure IDs;
    - compute global pLDDT from B-factor values;
    - filter by global mean pLDDT;
    - copy or symlink passed PDBs into a standardized folder;
    - optionally align passed protein structures.

    No ligand, metal, cofactor, clash, PLIP, or pocket analysis is performed here.
    """
    target_id = require_nested(config, ["project", "target_id"])
    output_root = Path(get_nested(config, ["project", "output_root"], f"results/{target_id}"))
    step01_dir = output_root / "01_af2_prepare"

    table_dir = ensure_dir(step01_dir / "tables")
    list_dir = ensure_dir(step01_dir / "lists")
    report_dir = ensure_dir(step01_dir / "reports")
    standardized_pdb_dir = ensure_dir(step01_dir / "pdb" / "standardized")
    aligned_pdb_dir = ensure_dir(step01_dir / "pdb" / "aligned")

    af2_dir = Path(require_nested(config, ["inputs", "af2_dir"]))
    patterns = get_nested(config, ["step01", "file_patterns"], ["*.pdb"])
    pdb_files = list_files(af2_dir, patterns)

    number_source = str(get_nested(config, ["structure_ids", "number_source"], "input_order")).lower()
    if number_source in {"filename_prediction_index", "filename_index", "source_index", "af2_index"}:
        pdb_files = sorted(
            pdb_files,
            key=lambda p: (
                _extract_prediction_index(p.name, target_id) is None,
                _extract_prediction_index(p.name, target_id) or 10**12,
                _natural_key(p.name),
            ),
        )

    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {af2_dir} with patterns={patterns}")

    plddt_atom = get_nested(config, ["step01", "plddt_atom"], "CA")
    include_hetatm = bool(get_nested(config, ["step01", "include_hetatm_for_plddt"], False))
    global_plddt_min = float(get_nested(config, ["step01", "global_plddt_min"], 80.0))
    plddt_filter_scope = _normalise_plddt_filter_scope(
        get_nested(config, ["step01", "plddt_filter_scope"], "source_if_available")
    )
    manifest_by_output, manifest_by_name, manifest_path, manifest_status = _manifest_source_paths(
        af2_dir,
        get_nested(config, ["step01", "af2_preprocess_manifest"], "auto"),
    )

    prefix_cfg = get_nested(config, ["structure_ids", "prefix"], None)
    prefix = str(prefix_cfg).format(target_id=target_id) if prefix_cfg else f"{target_id}_AF2"
    width = int(get_nested(config, ["structure_ids", "width"], 4))
    keep_original_names = bool(get_nested(config, ["structure_ids", "keep_original_names"], False))
    # number_source controls how standardized IDs are numbered.  Use
    # filename_prediction_index to preserve AF2 file numbers for article-level
    # traceability, e.g. 6B90_100_Chain_A -> PTP1B_AF2_0100.
    number_source = str(get_nested(config, ["structure_ids", "number_source"], number_source)).lower()
    copy_mode = get_nested(config, ["step01", "standardized_pdb_mode"], "copy")

    rows: list[dict[str, Any]] = []

    for input_order, pdb_path in enumerate(pdb_files, start=1):
        meta = extract_colabfold_metadata(pdb_path.name)
        original_id = str(meta["original_structure_id"])
        source_prediction_index = _extract_prediction_index(pdb_path.name, target_id)
        if keep_original_names:
            standard_id = original_id
            numbering_index = input_order
        elif number_source in {"filename_prediction_index", "filename_index", "source_index", "af2_index"}:
            numbering_index = int(source_prediction_index) if source_prediction_index is not None else input_order
            standard_id = _format_standard_id(prefix, numbering_index, width)
        else:
            numbering_index = input_order
            standard_id = _format_standard_id(prefix, input_order, width)

        input_plddt = None
        input_error = ""
        try:
            input_plddt = compute_global_plddt(
                pdb_path, atom_name=plddt_atom, include_hetatm=include_hetatm
            )
        except Exception as exc:
            input_error = str(exc)

        # ``pdb_path`` is the file used for docking.  If Step 0A created it,
        # the manifest tells us which full-length AF2 file it came from.  Do
        # not infer this from rank numbers: copied/renamed ensembles can reuse
        # ranks and would make that silently wrong.
        resolved_input_path = pdb_path.resolve()
        original_af2_path = manifest_by_output.get(resolved_input_path)
        source_match_kind = "manifest_exact" if original_af2_path else ""
        if original_af2_path is None:
            original_af2_path = manifest_by_name.get(pdb_path.name)
            source_match_kind = "manifest_unique_filename" if original_af2_path else ""

        source_plddt = None
        source_error = ""
        if original_af2_path is not None:
            if original_af2_path == resolved_input_path:
                source_plddt = input_plddt
            elif not original_af2_path.is_file():
                source_error = f"original AF2 PDB listed in manifest is unavailable: {original_af2_path}"
            else:
                try:
                    source_plddt = compute_global_plddt(
                        original_af2_path,
                        atom_name=plddt_atom,
                        include_hetatm=include_hetatm,
                    )
                except Exception as exc:
                    source_error = str(exc)

        filter_plddt = input_plddt
        filter_path = resolved_input_path
        scope_applied = "input"
        plddt_source_status = "input_requested"
        status = "ok" if input_plddt is not None else "error"
        error_message = input_error
        if plddt_filter_scope != "input":
            if source_plddt is not None:
                filter_plddt = source_plddt
                filter_path = original_af2_path or resolved_input_path
                scope_applied = "source"
                plddt_source_status = source_match_kind or "source_path"
                status = "ok"
                error_message = ""
            elif plddt_filter_scope == "source_required":
                filter_plddt = None
                filter_path = None
                scope_applied = "source_unavailable"
                plddt_source_status = "source_required_but_unavailable"
                status = "error"
                error_message = source_error or (
                    "No original AF2 PDB could be resolved from the Step 0A "
                    f"manifest ({manifest_status})."
                )
            else:
                # A normal, untrimmed AF2 directory has no manifest.  Preserve
                # its historical behaviour while recording that the input PDB
                # rather than a recoverable source PDB determined the decision.
                plddt_source_status = "source_unavailable_fell_back_to_input"
                if source_error:
                    error_message = source_error

        pass_global = bool(
            status == "ok"
            and filter_plddt
            and filter_plddt.global_mean_plddt >= global_plddt_min
        )

        standardized_name = f"{standard_id}.pdb"
        standardized_path = standardized_pdb_dir / standardized_name
        if pass_global and status == "ok" and copy_mode != "none":
            _copy_or_link(pdb_path, standardized_path, copy_mode)

        rows.append(
            {
                "input_order": input_order,
                "source_prediction_index": source_prediction_index,
                "numbering_index": numbering_index,
                "target_id": target_id,
                "standard_structure_id": standard_id,
                "original_structure_id": original_id,
                "original_filename": pdb_path.name,
                # ``source_path`` remains the input PDB used by the remainder
                # of the workflow for backward compatibility.  The untrimmed
                # AF2 source is always exposed separately as source_af2_path.
                "source_path": str(pdb_path),
                "input_pdb_path": str(pdb_path),
                "source_af2_path": str(original_af2_path or ""),
                "source_af2_match": source_match_kind or "not_available",
                "plddt_filter_scope_requested": plddt_filter_scope,
                "plddt_filter_scope_applied": scope_applied,
                "plddt_filter_pdb_path": str(filter_path or ""),
                "plddt_source_status": plddt_source_status,
                "standardized_pdb_path": str(standardized_path) if pass_global and copy_mode != "none" else "",
                "rank": meta["rank"],
                "model": meta["model"],
                "seed": meta["seed"],
                # The historical global_* columns remain the scores that
                # actually controlled pass/fail.  Explicit input_/source_
                # columns make a trimmed-versus-full-length comparison auditable.
                "n_atoms": filter_plddt.n_atoms if filter_plddt else None,
                "n_residues": filter_plddt.n_residues if filter_plddt else None,
                "global_mean_plddt": filter_plddt.global_mean_plddt if filter_plddt else None,
                "global_median_plddt": filter_plddt.global_median_plddt if filter_plddt else None,
                "global_p10_plddt": filter_plddt.global_p10_plddt if filter_plddt else None,
                "global_min_plddt": filter_plddt.global_min_plddt if filter_plddt else None,
                "global_max_plddt": filter_plddt.global_max_plddt if filter_plddt else None,
                **_plddt_values("input", input_plddt),
                **_plddt_values("source", source_plddt),
                "global_plddt_threshold": global_plddt_min,
                "pass_global_plddt": pass_global,
                "status": status,
                "error_message": error_message,
            }
        )

    df = pd.DataFrame(rows)
    input_index = df[
        [
            "input_order",
            "source_prediction_index",
            "numbering_index",
            "target_id",
            "standard_structure_id",
            "original_structure_id",
            "original_filename",
            "source_path",
            "input_pdb_path",
            "source_af2_path",
            "source_af2_match",
            "plddt_filter_scope_requested",
            "plddt_filter_scope_applied",
            "plddt_filter_pdb_path",
            "plddt_source_status",
            "rank",
            "model",
            "seed",
        ]
    ].copy()

    passed_df = df[(df["status"] == "ok") & (df["pass_global_plddt"] == True)].copy()
    failed_df = df[~((df["status"] == "ok") & (df["pass_global_plddt"] == True))].copy()

    alignment_enabled = bool(get_nested(config, ["step01", "alignment", "enabled"], False))
    alignment_rows: list[dict[str, Any]] = []
    reference_pdb: Path | None = None

    if alignment_enabled and not passed_df.empty:
        reference_setting = get_nested(config, ["step01", "alignment", "reference"], "auto_best_plddt")
        atom_name = get_nested(config, ["step01", "alignment", "atom_name"], "CA")
        min_matched = int(get_nested(config, ["step01", "alignment", "min_matched_atoms"], 50))

        if reference_setting == "auto_best_plddt":
            best = passed_df.sort_values(["global_mean_plddt", "input_order"], ascending=[False, True]).iloc[0]
            reference_pdb = Path(best["standardized_pdb_path"]) if best["standardized_pdb_path"] else Path(best["source_path"])
        elif reference_setting == "first_passed":
            first = passed_df.sort_values("input_order").iloc[0]
            reference_pdb = Path(first["standardized_pdb_path"]) if first["standardized_pdb_path"] else Path(first["source_path"])
        else:
            reference_pdb = Path(str(reference_setting))

        for _, row in passed_df.iterrows():
            standard_id = row["standard_structure_id"]
            moving_pdb = Path(row["standardized_pdb_path"]) if row["standardized_pdb_path"] else Path(row["source_path"])
            output_pdb = aligned_pdb_dir / f"{standard_id}.pdb"
            try:
                aln = align_pdb_to_reference(
                    moving_pdb,
                    reference_pdb,
                    output_pdb,
                    atom_name=atom_name,
                    min_matched_atoms=min_matched,
                )
                alignment_rows.append(
                    {
                        "standard_structure_id": standard_id,
                        "source_pdb": str(moving_pdb),
                        "aligned_pdb_path": str(output_pdb),
                        "reference_pdb": str(reference_pdb),
                        "alignment_atom": atom_name,
                        "n_matched_atoms": aln.n_matched_atoms,
                        "rmsd_before": aln.rmsd_before,
                        "rmsd_after": aln.rmsd_after,
                        "alignment_status": "ok",
                        "alignment_error": "",
                    }
                )
            except Exception as exc:
                alignment_rows.append(
                    {
                        "standard_structure_id": standard_id,
                        "source_pdb": str(moving_pdb),
                        "aligned_pdb_path": "",
                        "reference_pdb": str(reference_pdb),
                        "alignment_atom": atom_name,
                        "n_matched_atoms": None,
                        "rmsd_before": None,
                        "rmsd_after": None,
                        "alignment_status": "error",
                        "alignment_error": str(exc),
                    }
                )

    alignment_df = pd.DataFrame(alignment_rows)

    # Write tables.
    input_index.to_csv(table_dir / f"{target_id}_step01_input_structure_index.csv", index=False)
    df.to_csv(table_dir / f"{target_id}_step01_global_plddt_summary.csv", index=False)
    passed_df.to_csv(table_dir / f"{target_id}_step01_passed_structure_index.csv", index=False)
    failed_df.to_csv(table_dir / f"{target_id}_step01_failed_structure_index.csv", index=False)
    df[
        [
            "standard_structure_id",
            "source_prediction_index",
            "numbering_index",
            "original_structure_id",
            "original_filename",
            "source_path",
            "input_pdb_path",
            "source_af2_path",
            "plddt_filter_scope_requested",
            "plddt_filter_scope_applied",
            "plddt_filter_pdb_path",
            "standardized_pdb_path",
            "rank",
            "model",
            "seed",
        ]
    ].to_csv(table_dir / f"{target_id}_step01_structure_id_mapping.csv", index=False)

    if not alignment_df.empty:
        alignment_df.to_csv(table_dir / f"{target_id}_step01_alignment_summary.csv", index=False)

    (list_dir / f"{target_id}_step01_passed_structure_ids.txt").write_text(
        "\n".join(passed_df["standard_structure_id"].astype(str).tolist()) + ("\n" if not passed_df.empty else ""),
        encoding="utf-8",
    )
    (list_dir / f"{target_id}_step01_failed_structure_ids.txt").write_text(
        "\n".join(failed_df["standard_structure_id"].astype(str).tolist()) + ("\n" if not failed_df.empty else ""),
        encoding="utf-8",
    )

    report = {
        "target_id": target_id,
        "step": "01_af2_prepare",
        "input_af2_dir": str(af2_dir),
        "n_input_pdb": int(len(df)),
        "n_passed_global_plddt": int(len(passed_df)),
        "n_failed_global_plddt_or_error": int(len(failed_df)),
        "global_plddt_min": global_plddt_min,
        "plddt_filter_scope_requested": plddt_filter_scope,
        "af2_preprocess_manifest": str(manifest_path or ""),
        "af2_preprocess_manifest_status": manifest_status,
        "n_source_af2_paths_resolved": int(df["source_af2_path"].astype(bool).sum()),
        "n_source_af2_scores_used": int((df["plddt_filter_scope_applied"] == "source").sum()),
        "n_input_scores_used": int((df["plddt_filter_scope_applied"] == "input").sum()),
        "global_mean_plddt_min_observed": float(df["global_mean_plddt"].min()) if df["global_mean_plddt"].notna().any() else None,
        "global_mean_plddt_max_observed": float(df["global_mean_plddt"].max()) if df["global_mean_plddt"].notna().any() else None,
        "global_mean_plddt_average": float(df["global_mean_plddt"].mean()) if df["global_mean_plddt"].notna().any() else None,
        "keep_original_names": keep_original_names,
        "structure_id_number_source": number_source,
        "standard_structure_id_prefix": prefix,
        "standardized_pdb_mode": copy_mode,
        "alignment_enabled": alignment_enabled,
        "alignment_reference_pdb": str(reference_pdb) if reference_pdb else None,
        "output_dir": str(step01_dir),
        "notes": [
            "Step 01 is protein-only.",
            "No docking, ligand naming, clash filtering, pocket pLDDT, or PLIP analysis is performed in this step.",
            "Ligand, metal, and cofactor standardization starts after complex generation in Step 02.",
        ],
    }

    with (report_dir / f"{target_id}_step01_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    summary_lines = [
        f"# Step 01 summary: {target_id}",
        "",
        "## Scope",
        "",
        "Step 01 is limited to protein-only AF2/ColabFold structures.",
        "It performs structure indexing, optional standardized naming, optional protein alignment, and global pLDDT filtering.",
        "",
        "It does not perform docking, ligand standardization, clash filtering, pocket pLDDT, PLIP, or interaction-graph analysis.",
        "",
        "## Results",
        "",
        f"- Input PDB files: {report['n_input_pdb']}",
        f"- Passed global pLDDT: {report['n_passed_global_plddt']}",
        f"- Failed or unreadable: {report['n_failed_global_plddt_or_error']}",
        f"- Global pLDDT threshold: {global_plddt_min}",
        f"- Requested pLDDT filter scope: {plddt_filter_scope}",
        f"- Step 0A manifest: {manifest_path or 'not found'} ({manifest_status})",
        f"- Full-length source scores used: {report['n_source_af2_scores_used']}",
        f"- Input/trimmed scores used: {report['n_input_scores_used']}",
        f"- Minimum observed global mean pLDDT: {report['global_mean_plddt_min_observed']}",
        f"- Maximum observed global mean pLDDT: {report['global_mean_plddt_max_observed']}",
        f"- Average observed global mean pLDDT: {report['global_mean_plddt_average']}",
        "",
        "## Main outputs",
        "",
        f"- `{table_dir / f'{target_id}_step01_global_plddt_summary.csv'}`",
        f"- `{table_dir / f'{target_id}_step01_structure_id_mapping.csv'}`",
        f"- `{list_dir / f'{target_id}_step01_passed_structure_ids.txt'}`",
        f"- `{standardized_pdb_dir}`",
    ]
    if alignment_enabled:
        summary_lines.append(f"- `{aligned_pdb_dir}`")
        summary_lines.append(f"- `{table_dir / f'{target_id}_step01_alignment_summary.csv'}`")

    (report_dir / f"{target_id}_step01_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 01: AF2 protein preparation and global pLDDT filtering."
    )
    parser.add_argument("--config", required=True, help="Project-level YAML config, e.g. projects/2PBR/config.yaml")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_yaml_config(args.config)
    report = run_step01(cfg)

    print("Step 01 completed.")
    print(f"Target: {report['target_id']}")
    print(f"Input PDB files: {report['n_input_pdb']}")
    print(f"Passed global pLDDT: {report['n_passed_global_plddt']}")
    print(f"Failed or unreadable: {report['n_failed_global_plddt_or_error']}")
    print(f"Output: {report['output_dir']}")


if __name__ == "__main__":
    main()

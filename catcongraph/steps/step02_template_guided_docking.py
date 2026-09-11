from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import shutil
from typing import Any

import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:  # Allow CLI/help and naming utilities to import.
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]

from catcongraph.config import get_nested, load_yaml_config, require_nested
from catcongraph.io.paths import ensure_dir
from catcongraph.profiles import GENERAL, workflow_profile


DOCKPY_MODES = {
    "dock_py_compatible",
    "dock_py_exact",
    "dock_py_original",
    "article_dockpy",
    "original_dock.py_exact",
    "dock_py_mirror",
}


def _require_docking_dependencies():
    """Load docking dependencies only when Step02 calculations are requested."""
    if Chem is None or AllChem is None:
        raise ModuleNotFoundError(
            "Step02 requires RDKit. Install the AF2-CatGraph environment "
            "from environment.yml before running ligand placement."
        )
    from catcongraph.ligand import dock_py_original, docking_general
    return dock_py_original, docking_general


def _natural_key(path: Path) -> list[Any]:
    """Natural filename sort: 6B90.9.pdb < 6B90.10.pdb."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _clean_name_token(value: str) -> str:
    """Return a filesystem-safe short token for output filenames."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "model"


def _infer_pdb_code_from_stem(stem: str, target_id: str = "") -> str:
    """Infer the leading PDB-like identifier from an AF2/ColabFold filename.

    Examples
    --------
    6B90_1_Chain_A_..._rank_221_... -> 6B90
    6B90.308 -> 6B90

    If the filename does not start with a 4-character PDB code, the first
    underscore/dot-delimited token is used. This keeps the rule useful for
    non-PDB target names while still producing compact filenames.
    """
    stem = str(stem).strip()
    m = re.match(r"^([A-Za-z0-9]{4})(?:[_.-]|$)", stem)
    if m:
        return _clean_name_token(m.group(1).upper())
    first = re.split(r"[_.-]", stem, maxsplit=1)[0] if stem else target_id
    return _clean_name_token(first or target_id or "model")


def _infer_rank_token_from_stem(stem: str, input_order: int) -> str:
    """Infer a compact rank/index token from an AF2/ColabFold filename.

    Priority:
    1. explicit ``rank_221`` / ``rank-221`` / ``rank221`` token;
    2. numeric token immediately after the leading PDB code, e.g. ``6B90_1``;
    3. deterministic input order in the current run.

    The first rule makes files such as ``...rank_221...`` and ``...rank_222...``
    become distinct short names: ``6B90_221_complex.pdb`` and
    ``6B90_222_complex.pdb``.
    """
    stem = str(stem).strip()
    m = re.search(r"(?:^|[_.-])(?:af2)?rank[_.-]?(\d+)(?:[_.-]|$)", stem, flags=re.IGNORECASE)
    if m:
        return str(int(m.group(1)))
    m = re.match(r"^[A-Za-z0-9]{4}[_.-](\d+)(?:[_.-]|$)", stem)
    if m:
        return str(int(m.group(1)))
    return str(int(input_order))


def _make_step02_output_name(
    source_pdb: Path,
    *,
    input_order: int,
    target_id: str,
    step02_cfg: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Build Step02 complex filename and short structure ID.

    ``step02.output_naming`` controls the rule:
    - ``short_pdb_rank`` (default): ``<PDB>_<rank>_complex.pdb``;
    - ``source_stem`` / ``dock_py_exact``: old dock.py basename rule,
      ``<input_pdb_stem>_complex.pdb``.

    The geometry remains the original dock.py mirror; only the output filename
    and downstream structure ID are shortened.
    """
    naming = str(step02_cfg.get("output_naming", "short_pdb_rank") or "short_pdb_rank").strip().lower()
    if naming in {"source_stem", "input_stem", "dock_py_exact", "original"}:
        sid = _clean_name_token(source_pdb.stem)
        return f"{sid}_complex.pdb", sid, "", ""

    pdb_code = _infer_pdb_code_from_stem(source_pdb.stem, target_id=target_id)
    rank_token = _infer_rank_token_from_stem(source_pdb.stem, input_order=input_order)
    sid = f"{pdb_code}_{rank_token}"
    return f"{sid}_complex.pdb", sid, pdb_code, rank_token


def _collect_pdbs_from_af2_dir(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Use inputs.af2_dir exactly like original dock.py uses mutant_folder.

    The original dock.py does:
        pdb_files = [f for f in os.listdir(mutant_folder) if f.endswith(".pdb")]
        out_name = os.path.splitext(pdb_file)[0] + "_complex.pdb"

    This function applies the configured rigid-docking behavior.
    """
    af2_dir_value = require_nested(config, ["inputs", "af2_dir"])
    af2_dir = Path(str(af2_dir_value))
    if not af2_dir.exists():
        raise FileNotFoundError(f"inputs.af2_dir not found: {af2_dir}")
    if not af2_dir.is_dir():
        raise NotADirectoryError(f"inputs.af2_dir is not a directory: {af2_dir}")

    pdbs = sorted(
        [p for p in af2_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdb"],
        key=_natural_key,
    )
    if not pdbs:
        raise ValueError(f"No .pdb files found directly under inputs.af2_dir: {af2_dir}")

    rows = []
    for i, pdb in enumerate(pdbs, start=1):
        rows.append(
            {
                "input_order": i,
                "standard_structure_id": pdb.stem,
                "source_pdb": str(pdb),
                "source_pdb_basename": pdb.name,
                "source_pdb_stem": pdb.stem,
            }
        )
    return pd.DataFrame(rows), f"inputs.af2_dir:{af2_dir}"


def _auto_step01_passed_table(output_root: Path, target_id: str) -> Path:
    candidates = [
        output_root / "01_af2_prepare" / "tables" / f"{target_id}_step01_passed_structure_index.csv",
        output_root / "01_global_plddt_filter" / f"{target_id}_step01_passed_structure_index.csv",
        output_root / "01_global_plddt_filter" / "tables" / f"{target_id}_step01_passed_structure_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _choose_input_pdb(row: pd.Series, preferred_column: str) -> str:
    candidates = [preferred_column]
    for col in [
        "source_pdb",
        "source_pdb_path",
        "original_pdb_path",
        "af2_pdb_path",
        "input_pdb",
        "input_pdb_path",
        "pdb_path",
        "standardized_pdb_path",
        "aligned_pdb_path",
        "source_path",
    ]:
        if col not in candidates:
            candidates.append(col)

    for col in candidates:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    raise ValueError(f"Could not find an input PDB path. Tried columns: {candidates}")


def _collect_pdbs_from_table(config: dict[str, Any], output_root: Path, target_id: str) -> tuple[pd.DataFrame, str]:
    step02_cfg = get_nested(config, ["step02"], {}) or {}
    input_table_setting = step02_cfg.get("input_structure_table", "auto")
    if input_table_setting in (None, "", "auto"):
        input_table = _auto_step01_passed_table(output_root, target_id)
    else:
        input_table = Path(str(input_table_setting))

    if not input_table.exists():
        raise FileNotFoundError(
            f"Step02 input structure table not found: {input_table}. "
            "Either run Step01 again, set step02.input_structure_table, or use step02.input_source: af2_dir."
        )

    input_pdb_column = str(step02_cfg.get("input_pdb_column", "source_pdb"))
    df = pd.read_csv(input_table)
    if df.empty:
        raise ValueError(f"Step02 input structure table is empty: {input_table}")

    rows = []
    for i, row in df.iterrows():
        pdb = Path(_choose_input_pdb(row, input_pdb_column))
        rows.append(
            {
                **row.to_dict(),
                "input_order": i + 1,
                "standard_structure_id": str(row.get("standard_structure_id") or pdb.stem),
                "source_pdb": str(pdb),
                "source_pdb_basename": pdb.name,
                "source_pdb_stem": pdb.stem,
            }
        )

    rows.sort(key=lambda item: _natural_key(Path(str(item["source_pdb_basename"]))))
    for input_order, item in enumerate(rows, start=1):
        item["input_order"] = input_order

    return pd.DataFrame(rows), f"table:{input_table}"


def _load_input_dataframe(config: dict[str, Any], output_root: Path, target_id: str) -> tuple[pd.DataFrame, str]:
    """Pick the Step02 input source.

    Important default:
    - If step02.input_structure_table is explicitly set to a real CSV path, use it.
    - Otherwise, use inputs.af2_dir directly, because dock.py's native input is a
      mutant_folder, not a previous Step01 CSV.

    To force old Step01-table behavior:
        step02:
          input_source: "step01_table"
    """
    step02_cfg = get_nested(config, ["step02"], {}) or {}
    input_source = str(step02_cfg.get("input_source", "") or "").strip().lower()
    input_table_setting = step02_cfg.get("input_structure_table", "auto")

    explicit_table = input_table_setting not in (None, "", "auto")
    if explicit_table or input_source in {"step01", "step01_table", "table", "csv"}:
        return _collect_pdbs_from_table(config, output_root, target_id)

    return _collect_pdbs_from_af2_dir(config)


def _as_path_list(value: Any) -> list[Path]:
    """Normalize a config value into a non-empty list of filesystem paths."""
    if value in (None, "", [], ()):  # type: ignore[comparison-overlap]
        return []
    if isinstance(value, (list, tuple)):
        return [Path(str(v)) for v in value if str(v).strip()]
    return [Path(str(value))]


def _ligand_resnames_in_source(path: Path) -> list[str]:
    """Return residue names in source-file order, without inferring chemistry.

    This is an audit/display manifest only.  Step06 still reads the actual
    selected (possibly trimmed) complex, but retaining this list makes every
    later result explicit about the original user order.
    """
    names: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21:22].strip() or "_"
            resseq = line[22:26].strip() or "0"
            key = (resname, chain, resseq)
            if resname and key not in seen:
                seen.add(key)
                names.append(resname)
    except OSError:
        pass
    return names


def _ligand_order_manifest(config: dict[str, Any], ligand_pdbs: list[Path]) -> list[dict[str, Any]]:
    """Build an immutable ordered record of normal or pre-docking-trim inputs."""
    optional_trim = config.get("step10_optional", {}) or {}
    before_trim = bool(optional_trim.get("enabled", False)) and str(
        optional_trim.get("timing", "before_docking")
    ).strip().lower() == "before_docking"
    source_by_active: dict[str, str] = {}
    if before_trim:
        target_id = str((config.get("project", {}) or {}).get("target_id", "target"))
        output_root = Path(str((config.get("project", {}) or {}).get("output_root", f"results/{target_id}")))
        optional_dir = output_root / str(optional_trim.get("output_subdir", "10_optional_ligand_atom_trimming"))
        index_path = optional_dir / "tables" / f"{target_id}_step10_pre_docking_ligand_index.csv"
        if index_path.exists():
            try:
                idx = pd.read_csv(index_path)
                for _, row in idx.iterrows():
                    active = str(row.get("trimmed_ligand_pdb_path", "")).strip()
                    source = str(row.get("source_ligand_pdb_path", "")).strip()
                    if active:
                        source_by_active[str(Path(active).resolve())] = source
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for order, active in enumerate(ligand_pdbs, start=1):
        resolved = active.resolve()
        source_text = source_by_active.get(str(resolved), str(resolved))
        rows.append({
            "docking_order": order,
            "source_ligand_pdb_path": source_text,
            "active_ligand_pdb_path": str(resolved),
            "selection_branch": "step10_before_docking_trim" if before_trim else "untrimmed_input",
            "residue_names_in_file_order": ";".join(_ligand_resnames_in_source(active)),
        })
    return rows


def _resolve_template_and_ligands(config: dict[str, Any]) -> tuple[Path, list[Path]]:
    """Resolve Step02 template protein and one or more ligand/cofactor files.

    For strict dock.py reproduction, use a single ``inputs.ligand_pdb``.  For
    enzymes with several substrates/cofactors/metals, set ``inputs.ligand_pdbs``
    in the desired docking order.  The workflow applies sequential
    dock.py workflow by docking ligand 1, then ligand 2 into the resulting
    complex, then ligand 3, and so on.  A single pre-combined ligand assembly PDB
    is still supported through ``inputs.ligand_pdb``.
    """
    template_ligands = get_nested(config, ["step02", "template_ligands"], {}) or {}
    template_protein = (
        template_ligands.get("template_protein_pdb")
        or template_ligands.get("template_protein")
        or get_nested(config, ["inputs", "template_protein_pdb"], None)
    )
    ligand_value = (
        template_ligands.get("dock_py_ligand_pdbs")
        or template_ligands.get("ligand_pdbs")
        or get_nested(config, ["inputs", "ligand_pdbs"], None)
        or template_ligands.get("dock_py_ligand_pdb")
        or template_ligands.get("ligand_pdb")
        or get_nested(config, ["inputs", "ligand_pdb"], None)
    )

    if not template_protein or not str(template_protein).strip():
        raise ValueError(
            "Missing dock.py template protein. Set:\n"
            "  inputs.template_protein_pdb: /path/to/template_protein.pdb"
        )

    # Public Step10 may deliberately trim the template substrate before
    # docking.  Its ordered index is the sole source of the generated copies;
    # the original input paths in the YAML remain untouched for provenance.
    optional_trim = config.get("step10_optional", {}) or {}
    optional_enabled = bool(optional_trim.get("enabled", False))
    optional_timing = str(optional_trim.get("timing", "before_docking")).strip().lower()
    if optional_enabled and optional_timing == "before_docking":
        target_id = str((config.get("project", {}) or {}).get("target_id", "target"))
        output_root = Path(str((config.get("project", {}) or {}).get("output_root", f"results/{target_id}")))
        optional_dir = output_root / str(optional_trim.get("output_subdir", "10_optional_ligand_atom_trimming"))
        index_path = optional_dir / "tables" / f"{target_id}_step10_pre_docking_ligand_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(
                "Step10 is configured for before_docking, but its trimmed-ligand index is missing. "
                "Run `catcongraph step10 --config <config>` first: " + str(index_path)
            )
        index = pd.read_csv(index_path)
        if "trimmed_ligand_pdb_path" not in index.columns or index.empty:
            raise ValueError(f"Step10 trimmed-ligand index is empty or malformed: {index_path}")
        ligand_pdbs = [Path(str(value)) for value in index.sort_values("order")["trimmed_ligand_pdb_path"].dropna().tolist()]
    else:
        ligand_pdbs = _as_path_list(ligand_value)
    if not ligand_pdbs:
        raise ValueError(
            "Missing dock.py ligand/cofactor input. Set one of:\n"
            "  inputs.ligand_pdb: /path/to/ligand.pdb\n"
            "  inputs.ligand_pdbs: [/path/to/substrate.pdb, /path/to/cofactor.pdb, /path/to/metal.pdb]"
        )

    template_protein_pdb = Path(str(template_protein))
    if not template_protein_pdb.exists():
        raise FileNotFoundError(f"template_protein_pdb not found: {template_protein_pdb}")

    seen: set[Path] = set()
    out: list[Path] = []
    for ligand_pdb in ligand_pdbs:
        if not ligand_pdb.exists():
            raise FileNotFoundError(f"ligand/cofactor PDB not found: {ligand_pdb}")
        if template_protein_pdb.resolve() == ligand_pdb.resolve():
            raise ValueError("template_protein_pdb and ligand/cofactor PDB cannot be the same file.")
        resolved = ligand_pdb.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(ligand_pdb)

    return template_protein_pdb, out


def _load_ligands(ligand_pdbs: list[Path]) -> list[Chem.Mol]:
    """Load one or more non-protein PDB/SDF/MOL files in user-defined order."""
    dockpy, _docking_general = _require_docking_dependencies()
    mols: list[Chem.Mol] = []
    for ligand_pdb in ligand_pdbs:
        mol = dockpy.load_molecule(str(ligand_pdb))
        if mol is None:
            raise ValueError(f"Could not load ligand/cofactor/metal file: {ligand_pdb}")
        if mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                print(f"Generated a 3D conformer for {ligand_pdb.name}.", flush=True)
            except Exception as exc:
                raise ValueError(f"Could not generate the required 3D conformer for {ligand_pdb}.") from exc
        mols.append(mol)
    return mols


def _combine_ligands_as_assembly(ligand_mols: list[Chem.Mol]) -> Chem.Mol:
    """Combine loaded ligand molecules as disconnected fragments."""
    _require_docking_dependencies()
    if not ligand_mols:
        raise ValueError("No ligand molecules were loaded.")
    ligand = Chem.Mol(ligand_mols[0])
    for mol in ligand_mols[1:]:
        ligand = Chem.CombineMols(ligand, mol)
    return ligand


def _dock_ligands_with_dockpy_order(
    template_protein: Chem.Mol,
    ligand_mols: list[Chem.Mol],
    mutant_protein: Chem.Mol,
    *,
    multi_ligand_mode: str = "sequential_dock_py",
    docking_alignment_mode: str = "dock_py_exact",
    compatibility: Any | None = None,
    general_alignment_cfg: dict[str, Any] | None = None,
    alignment_audit_log: list[dict[str, Any]] | None = None,
) -> Chem.Mol:
    """Dock one or more ligands while preserving dock.py geometry semantics.

    Single-ligand input is exactly the original dock.py call.  Multi-ligand input
    defaults to the sequential workflow used in manuscript production: the first
    ligand is docked into the aligned protein, the second ligand is docked into
    that protein-ligand complex, and so on.  This makes later ligands avoid
    collisions with both the protein and the previously placed ligands.

    ``multi_ligand_mode: rigid_assembly`` remains available for users who have
    intentionally prepared a single rigid multi-entity assembly, but it is not
    the manuscript default.
    """
    if not ligand_mols:
        raise ValueError("No ligand molecules were loaded.")
    dockpy, docking_general = _require_docking_dependencies()

    def dock_one(ligand: Chem.Mol, current: Chem.Mol) -> Chem.Mol:
        if docking_alignment_mode == "sequence_aware":
            if compatibility is None:
                raise ValueError("Sequence-aware docking requires a compatibility assessment.")
            result, audit = docking_general.rigid_docking_sequence_aware(
                template_protein,
                Chem.Mol(ligand),
                current,
                compatibility=compatibility,
                config=general_alignment_cfg,
            )
            if alignment_audit_log is not None:
                alignment_audit_log.append(audit.as_dict())
            return result
        result = dockpy.rigid_docking(template_protein, Chem.Mol(ligand), current)
        if alignment_audit_log is not None and compatibility is not None:
            alignment_audit_log.append(docking_general.legacy_audit(compatibility).as_dict())
        return result

    if len(ligand_mols) == 1:
        return dock_one(ligand_mols[0], mutant_protein)

    mode = str(multi_ligand_mode or "sequential_dock_py").strip().lower()
    if mode in {"rigid_assembly", "assembly", "combined", "combine", "one_assembly"}:
        ligand_assembly = _combine_ligands_as_assembly(ligand_mols)
        return dock_one(ligand_assembly, mutant_protein)

    if mode not in {"sequential", "sequential_dock_py", "dock_py_sequential", "dockpy_sequential"}:
        raise ValueError(
            "step02.multi_ligand_mode must be 'sequential_dock_py' or 'rigid_assembly'. "
            f"Got: {multi_ligand_mode!r}"
        )

    current_complex = mutant_protein
    for idx, ligand in enumerate(ligand_mols, start=1):
        print(f"   ↳ sequential dock.py ligand {idx}/{len(ligand_mols)}", flush=True)
        current_complex = dock_one(ligand, current_complex)
    return current_complex


def _write_tables(
    rows: list[dict[str, Any]],
    *,
    target_id: str,
    table_dir: Path,
    list_dir: Path,
) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    summary_table = table_dir / f"{target_id}_step02_template_guided_docking_summary.csv"
    complex_index_table = table_dir / f"{target_id}_step02_complex_index.csv"
    failed_table = table_dir / f"{target_id}_step02_failed_structures.csv"
    complex_list = list_dir / f"{target_id}_step02_complex_pdbs.txt"

    df = pd.DataFrame(rows)
    df.to_csv(summary_table, index=False)

    if df.empty:
        pd.DataFrame().to_csv(complex_index_table, index=False)
        pd.DataFrame().to_csv(failed_table, index=False)
        complex_list.write_text("", encoding="utf-8")
    else:
        ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
        failed = df[df["status"] != "ok"].copy() if "status" in df.columns else pd.DataFrame()

        preferred = [
            "input_order",
            "standard_structure_id",
            "output_complex_basename",
            "output_pdb_code",
            "output_rank_token",
            "source_pdb_basename",
            "source_pdb_stem",
            "source_pdb",
            "complex_pdb_path",
            "template_protein_pdb_path",
            "ligand_pdb_path",
            "ligand_order_manifest_path",
            "active_ligand_pdb_order",
            "input_ligand_resname_order",
            "ligand_input_branch",
            "ligand_pdb_count",
            "n_ligand_atoms",
            "workflow_profile",
            "docking_alignment_mode",
            "legacy_compatible",
            "compatibility_reason",
            "matched_ca_count",
            "sequence_identity",
            "alignment_coverage",
            "alignment_rmsd",
            "status",
            "error_message",
        ]
        cols = [c for c in preferred if c in ok.columns]
        if not ok.empty:
            ok[cols].to_csv(complex_index_table, index=False)
            complex_list.write_text(
                "\n".join(ok["complex_pdb_path"].astype(str).tolist()) + "\n",
                encoding="utf-8",
            )
        else:
            pd.DataFrame(columns=preferred).to_csv(complex_index_table, index=False)
            complex_list.write_text("", encoding="utf-8")

        failed.to_csv(failed_table, index=False)

    return {
        "summary_table": summary_table,
        "complex_index_table": complex_index_table,
        "failed_table": failed_table,
        "complex_list": complex_list,
    }


def run_step02(
    config: dict[str, Any],
    *,
    max_structures: int | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    dockpy, docking_general = _require_docking_dependencies()
    target_id = require_nested(config, ["project", "target_id"])
    profile = workflow_profile(config)
    output_root = Path(get_nested(config, ["project", "output_root"], f"results/{target_id}"))
    step02_cfg = get_nested(config, ["step02"], {}) or {}

    mode = str(step02_cfg.get("mode", "dock_py_mirror")).lower()
    if mode not in DOCKPY_MODES:
        raise ValueError(f"Unsupported step02.mode={mode!r}. Use 'dock_py_mirror' or 'dock_py_compatible'.")

    step02_dir = output_root / str(step02_cfg.get("output_subdir", "02_template_guided_docking"))
    if clean and step02_dir.exists():
        shutil.rmtree(step02_dir)

    complex_dir = ensure_dir(step02_dir / "complexes")
    table_dir = ensure_dir(step02_dir / "tables")
    list_dir = ensure_dir(step02_dir / "lists")
    report_dir = ensure_dir(step02_dir / "reports")

    structure_df, input_source_desc = _load_input_dataframe(config, output_root, target_id)
    if structure_df.empty:
        raise ValueError("Step02 input dataframe is empty.")

    cfg_max = step02_cfg.get("max_structures")
    if max_structures is None and cfg_max not in (None, "", "none", "None"):
        max_structures = int(cfg_max)
    if max_structures is not None and max_structures > 0:
        structure_df = structure_df.head(int(max_structures)).copy()

    overwrite = bool(step02_cfg.get("overwrite", True))
    template_protein_pdb, ligand_pdbs = _resolve_template_and_ligands(config)
    ligand_pdbs_text = ";".join(str(p) for p in ligand_pdbs)
    ligand_manifest_rows = _ligand_order_manifest(config, ligand_pdbs)
    ligand_manifest_path = table_dir / f"{target_id}_step02_ligand_order_manifest.csv"
    pd.DataFrame(ligand_manifest_rows).to_csv(ligand_manifest_path, index=False)
    active_ligand_order = ";".join(row["active_ligand_pdb_path"] for row in ligand_manifest_rows)
    input_resname_order = ";".join(
        name for row in ligand_manifest_rows for name in str(row["residue_names_in_file_order"]).split(";") if name
    )
    ligand_input_branch = ligand_manifest_rows[0]["selection_branch"] if ligand_manifest_rows else "untrimmed_input"

    rows: list[dict[str, Any]] = []
    table_paths = _write_tables(rows, target_id=target_id, table_dir=table_dir, list_dir=list_dir)

    print("Loading the template protein and ligand/cofactor/metal conformers...", flush=True)
    # Mirror original dock.py: load template and ligand molecules once before looping.
    template_protein = dockpy.load_pdb(str(template_protein_pdb))
    ligand_mols = _load_ligands(ligand_pdbs)
    n_ligand_atoms = int(sum(m.GetNumAtoms() for m in ligand_mols))
    multi_ligand_mode = str(step02_cfg.get("multi_ligand_mode", "sequential_dock_py"))
    alignment_cfg = step02_cfg.get("general_alignment", {}) or {}
    configured_alignment_mode = str(step02_cfg.get("alignment_mode", "auto") or "auto").strip().lower()
    if len(ligand_pdbs) > 1:
        print(
            f"Loaded {len(ligand_pdbs)} non-protein entity files; "
            f"multi-ligand/cofactor docking will use the submitted order in {multi_ligand_mode} mode.",
            flush=True,
        )

    total = len(structure_df)
    used_output_names: set[str] = set()

    for out_i, (_, row) in enumerate(structure_df.iterrows(), start=1):
        source_pdb = Path(str(row["source_pdb"]))
        if not source_pdb.exists():
            error = f"input PDB not found: {source_pdb}"
            result_row = {
                "input_order": int(row.get("input_order", out_i)),
                "standard_structure_id": str(row.get("standard_structure_id", source_pdb.stem)),
                "output_complex_basename": "",
                "output_pdb_code": "",
                "output_rank_token": "",
                "source_pdb_basename": source_pdb.name,
                "source_pdb_stem": source_pdb.stem,
                "source_pdb": str(source_pdb),
                "complex_pdb_path": "",
                "template_protein_pdb_path": str(template_protein_pdb),
                "ligand_pdb_path": ligand_pdbs_text,
                "ligand_order_manifest_path": str(ligand_manifest_path),
                "active_ligand_pdb_order": active_ligand_order,
                "input_ligand_resname_order": input_resname_order,
                "ligand_input_branch": ligand_input_branch,
                "ligand_pdb_count": len(ligand_pdbs),
                "n_ligand_atoms": n_ligand_atoms,
                "workflow_profile": profile,
                "docking_alignment_mode": "not_run",
                "legacy_compatible": False,
                "compatibility_reason": "missing_input_pdb",
                "status": "error",
                "error_message": error,
            }
            rows.append(result_row)
            table_paths = _write_tables(rows, target_id=target_id, table_dir=table_dir, list_dir=list_dir)
            print(f"[Step02 {out_i}/{total}] failed: {error}", flush=True)
            continue

        compatibility = docking_general.assess_legacy_compatibility(template_protein_pdb, source_pdb)
        if configured_alignment_mode in {"auto", "compatible_auto"}:
            # The standard workflow keeps the established coordinate-matching
            # behaviour when an existing config says ``auto``. Other profiles may use the compatibility audit to
            # select the safer sequence-aware method.  An explicit selection
            # below is always respected, regardless of profile.
            docking_alignment_mode = (
                "dock_py_exact"
                if profile != GENERAL or compatibility.legacy_compatible
                else "sequence_aware"
            )
        elif configured_alignment_mode in {
            "coordinate_matching", "dock_py_exact", "legacy", "original"
        }:
            if not compatibility.legacy_compatible and not bool(step02_cfg.get("allow_unsafe_legacy_alignment", False)):
                raise ValueError(
                    f"{source_pdb.name} is not legacy-compatible ({compatibility.reason}). "
                    "Use step02.alignment_mode: sequence_aware, or explicitly set "
                    "allow_unsafe_legacy_alignment: true."
                )
            docking_alignment_mode = "dock_py_exact"
        elif configured_alignment_mode in {"sequence_aware", "general"}:
            docking_alignment_mode = "sequence_aware"
        else:
            raise ValueError(
                "step02.alignment_mode must be auto, coordinate_matching, or sequence_aware"
            )

        out_name, output_structure_id, output_pdb_code, output_rank_token = _make_step02_output_name(
            source_pdb,
            input_order=int(row.get("input_order", out_i)),
            target_id=str(target_id),
            step02_cfg=step02_cfg,
        )
        if out_name in used_output_names:
            # Same compact name appears twice in one run. Do not silently overwrite.
            stem = out_name[:-len("_complex.pdb")] if out_name.endswith("_complex.pdb") else Path(out_name).stem
            output_structure_id = f"{stem}_dup{out_i}"
            out_name = f"{output_structure_id}_complex.pdb"
        used_output_names.add(out_name)

        output_complex = complex_dir / out_name

        if output_complex.exists() and not overwrite:
            result_row = {
                "input_order": int(row.get("input_order", out_i)),
                "standard_structure_id": output_structure_id,
                "output_complex_basename": out_name,
                "output_pdb_code": output_pdb_code,
                "output_rank_token": output_rank_token,
                "source_pdb_basename": source_pdb.name,
                "source_pdb_stem": source_pdb.stem,
                "source_pdb": str(source_pdb),
                "complex_pdb_path": str(output_complex),
                "template_protein_pdb_path": str(template_protein_pdb),
                "ligand_pdb_path": ligand_pdbs_text,
                "ligand_order_manifest_path": str(ligand_manifest_path),
                "active_ligand_pdb_order": active_ligand_order,
                "input_ligand_resname_order": input_resname_order,
                "ligand_input_branch": ligand_input_branch,
                "ligand_pdb_count": len(ligand_pdbs),
                "n_ligand_atoms": n_ligand_atoms,
                "workflow_profile": profile,
                "docking_alignment_mode": docking_alignment_mode,
                "legacy_compatible": compatibility.legacy_compatible,
                "compatibility_reason": compatibility.reason,
                "status": "skipped_exists",
                "error_message": "",
            }
            rows.append(result_row)
            table_paths = _write_tables(rows, target_id=target_id, table_dir=table_dir, list_dir=list_dir)
            print(f"[Step02 {out_i}/{total}] skipped existing: {output_complex}", flush=True)
            continue

        print(f"\nProcessing ensemble structure: {source_pdb.name}", flush=True)
        print(f"[Step02 {out_i}/{total}] dock.py mirror: {source_pdb.name} -> {out_name}", flush=True)

        try:
            mutant_protein = dockpy.load_pdb(str(source_pdb))
            alignment_audits: list[dict[str, Any]] = []
            complex_mol = _dock_ligands_with_dockpy_order(
                template_protein,
                ligand_mols,
                mutant_protein,
                multi_ligand_mode=multi_ligand_mode,
                docking_alignment_mode=docking_alignment_mode,
                compatibility=compatibility,
                general_alignment_cfg=alignment_cfg,
                alignment_audit_log=alignment_audits,
            )

            with Chem.PDBWriter(str(output_complex)) as writer:
                writer.write(complex_mol)

            result_row = {
                "input_order": int(row.get("input_order", out_i)),
                "standard_structure_id": output_structure_id,
                "output_complex_basename": out_name,
                "output_pdb_code": output_pdb_code,
                "output_rank_token": output_rank_token,
                "source_pdb_basename": source_pdb.name,
                "source_pdb_stem": source_pdb.stem,
                "source_pdb": str(source_pdb),
                "complex_pdb_path": str(output_complex),
                "template_protein_pdb_path": str(template_protein_pdb),
                "ligand_pdb_path": ligand_pdbs_text,
                "ligand_order_manifest_path": str(ligand_manifest_path),
                "active_ligand_pdb_order": active_ligand_order,
                "input_ligand_resname_order": input_resname_order,
                "ligand_input_branch": ligand_input_branch,
                "ligand_pdb_count": len(ligand_pdbs),
                "n_ligand_atoms": n_ligand_atoms,
                "status": "ok",
                "error_message": "",
            }
            alignment_audit = alignment_audits[0] if alignment_audits else {}
            print(f"Complex saved: {output_complex}", flush=True)
        except Exception as exc:
            alignment_audit = {}
            result_row = {
                "input_order": int(row.get("input_order", out_i)),
                "standard_structure_id": output_structure_id,
                "output_complex_basename": out_name,
                "output_pdb_code": output_pdb_code,
                "output_rank_token": output_rank_token,
                "source_pdb_basename": source_pdb.name,
                "source_pdb_stem": source_pdb.stem,
                "source_pdb": str(source_pdb),
                "complex_pdb_path": str(output_complex),
                "template_protein_pdb_path": str(template_protein_pdb),
                "ligand_pdb_path": ligand_pdbs_text,
                "ligand_order_manifest_path": str(ligand_manifest_path),
                "active_ligand_pdb_order": active_ligand_order,
                "input_ligand_resname_order": input_resname_order,
                "ligand_input_branch": ligand_input_branch,
                "ligand_pdb_count": len(ligand_pdbs),
                "n_ligand_atoms": n_ligand_atoms,
                "status": "error",
                "error_message": str(exc),
            }
            print(f"[Step02 {out_i}/{total}] failed: {source_pdb.name}; {exc}", flush=True)

        result_row.update(
            {
                "workflow_profile": profile,
                "docking_alignment_mode": docking_alignment_mode,
                "legacy_compatible": compatibility.legacy_compatible,
                "compatibility_reason": compatibility.reason,
                "template_chain_count": compatibility.template_chain_count,
                "target_chain_count": compatibility.target_chain_count,
                "template_ca_count": compatibility.template_ca_count,
                "target_ca_count": compatibility.target_ca_count,
                "matched_ca_count": alignment_audit.get("matched_ca_count", ""),
                "sequence_identity": alignment_audit.get("sequence_identity", ""),
                "alignment_coverage": alignment_audit.get("alignment_coverage", ""),
                "alignment_rmsd": alignment_audit.get("alignment_rmsd", ""),
            }
        )
        rows.append(result_row)
        # Flush after every one structure.
        table_paths = _write_tables(rows, target_id=target_id, table_dir=table_dir, list_dir=list_dir)

    summary_df = pd.DataFrame(rows)
    ok_df = summary_df[summary_df["status"] == "ok"].copy() if not summary_df.empty else pd.DataFrame()
    failed_df = summary_df[summary_df["status"] != "ok"].copy() if not summary_df.empty else pd.DataFrame()

    report = {
        "target_id": target_id,
        "step": "02_template_guided_docking",
        "mode": "dock_py_mirror",
        "workflow_profile": profile,
        "alignment_mode_configured": configured_alignment_mode,
        "n_dock_py_exact": int((summary_df.get("docking_alignment_mode") == "dock_py_exact").sum()) if "docking_alignment_mode" in summary_df else 0,
        "n_sequence_aware": int((summary_df.get("docking_alignment_mode") == "sequence_aware").sum()) if "docking_alignment_mode" in summary_df else 0,
        "input_source": input_source_desc,
        "template_protein_pdb": str(template_protein_pdb),
        "ligand_pdbs": [str(p) for p in ligand_pdbs],
        "ligand_order_manifest": str(ligand_manifest_path),
        "ligand_input_branch": ligand_input_branch,
        "n_input_structures": int(total),
        "n_complexes_written": int(len(ok_df)),
        "n_failed_or_skipped": int(len(failed_df)),
        "output_dir": str(step02_dir),
        "complex_dir": str(complex_dir),
        "main_outputs": {k: str(v) for k, v in table_paths.items()},
        "naming_rule": "short_pdb_rank default: <PDB>_<rank_or_index>_complex.pdb; set step02.output_naming: source_stem for old dock.py basename filenames",
        "multi_ligand_rule": "inputs.ligand_pdbs are docked sequentially by default: ligand 1 -> ligand 2 -> ligand 3, preserving dock.py rigid_docking calls and user-defined ligand order. Set step02.multi_ligand_mode: rigid_assembly only for intentional one-piece assemblies.",
        "multi_ligand_mode": multi_ligand_mode,
    }
    with (report_dir / f"{target_id}_step02_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    summary_lines = [
        f"# Step 02 summary: {target_id}",
        "",
        "Workflow policy: one general, sequence-aware workflow.",
        f"Configured alignment mode: `{configured_alignment_mode}`.",
        "",
        f"Input source: `{input_source_desc}`",
        f"Input structures: {report['n_input_structures']}",
        f"Complexes written: {report['n_complexes_written']}",
        f"Failed or skipped: {report['n_failed_or_skipped']}",
        "",
        "Naming rule:",
        "- Default short rule: `<PDB>_<rank_or_index>_complex.pdb`",
        "- Old basename rule is still available with `step02.output_naming: source_stem`",
        "",
        "Multi-ligand rule:",
        f"- `{report['multi_ligand_rule']}`",
        "",
        "Main outputs:",
        f"- `{table_paths['summary_table']}`",
        f"- `{table_paths['complex_index_table']}`",
        f"- `{table_paths['failed_table']}`",
        f"- Ligand-order manifest: `{ligand_manifest_path}`",
        f"- `{complex_dir}`",
    ]
    (report_dir / f"{target_id}_step02_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if len(ok_df) == 0:
        first_error = ""
        if not failed_df.empty and "error_message" in failed_df.columns:
            first_error = str(failed_df["error_message"].dropna().astype(str).head(1).tolist()[0])
        raise RuntimeError(
            f"Step02 wrote 0 complexes from {total} input structures. First error: {first_error}. "
            f"Failed table: {table_paths['failed_table']}."
        )

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 02: mirror original dock.py geometry and write compact complex filenames.")
    parser.add_argument("--config", required=True, help="Project-level YAML config, e.g. projects/PTP1B/config.yaml")
    parser.add_argument("--max-structures", type=int, default=None, help="Run only the first N structures.")
    parser.add_argument("--clean", action="store_true", help="Delete the Step02 output directory before running.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_yaml_config(args.config)
    report = run_step02(cfg, max_structures=args.max_structures, clean=args.clean)

    print("Step 02 completed.")
    print(f"  target_id: {report['target_id']}")
    print(f"  mode: {report['mode']}")
    print(f"  input source: {report['input_source']}")
    print(f"  input structures: {report['n_input_structures']}")
    print(f"  complexes written: {report['n_complexes_written']}")
    print(f"  output: {report['output_dir']}")
    print(f"  mapping table: {report['main_outputs']['complex_index_table']}")


if __name__ == "__main__":
    main()

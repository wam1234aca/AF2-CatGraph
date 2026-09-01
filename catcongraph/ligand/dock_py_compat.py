from __future__ import annotations

"""Thin CatConGraph adapter around the user's original dock.py.

Geometry rule for this module:
    Do not reimplement, optimize, vectorize, or reinterpret dock.py.

The actual docking functions are imported from ``catcongraph.ligand.dock_py_original``,
which is a direct copy of the uploaded ``dock.py``.  This adapter only does the
workflow work needed by CatConGraph: choose input files from config, call the
original functions once per AF2 structure, write the complex to the Step02 output
folder, and return a table row for later steps.
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import AllChem

from catcongraph.io.pdb import PDBAtomRecord, parse_pdb_atoms
from catcongraph.ligand.template_placement import (
    build_ligand_group_configs,
    select_ligand_group_atoms,
)
from catcongraph.ligand import dock_py_original as dockpy


@dataclass(frozen=True)
class DockPyResult:
    standard_structure_id: str
    source_pdb: str
    complex_pdb_path: str
    ligand_pdb_path: str
    template_protein_pdb_path: str
    n_ligand_atoms: int
    n_alignment_atoms: int
    alignment_rmsd_after: float
    initial_collision_score: float
    initial_clashing_ligand_atoms: int
    final_collision_score: float
    final_clashing_ligand_atoms: int
    ligand_adjustment_status: str
    ligand_adjustment_attempts: int
    total_translation: float
    total_rotation_deg: float
    status: str
    error_message: str
    trace_rows: list[dict[str, Any]]


def _write_records(path: Path, records: Iterable[PDBAtomRecord], force_ter: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(rec.raw_line.rstrip("\n") + "\n")
        if force_ter:
            out.write("TER\n")
        out.write("END\n")


def write_template_protein_from_complex(template_complex_pdb: str | Path, output_path: str | Path) -> Path:
    atoms = parse_pdb_atoms(template_complex_pdb, include_hetatm=False, atom_name=None)
    if not atoms:
        raise ValueError(f"No ATOM protein records found in template complex: {template_complex_pdb}")
    output_path = Path(output_path)
    _write_records(output_path, atoms, force_ter=True)
    return output_path


def write_ligand_from_template_groups(
    *,
    template_complex_pdb: str | Path,
    template_ligands: Mapping[str, Any] | None,
    output_path: str | Path,
) -> Path:
    template_ligands = template_ligands or {}
    group_configs = build_ligand_group_configs(template_ligands)
    all_atoms: list[PDBAtomRecord] = []
    for i, group in enumerate(group_configs, start=1):
        block = select_ligand_group_atoms(group=group, order_index=i, template_complex_pdb=template_complex_pdb)
        all_atoms.extend(block.atoms)
    if not all_atoms:
        raise ValueError("No ligand atoms selected for dock.py-compatible docking.")
    output_path = Path(output_path)
    _write_records(output_path, all_atoms, force_ter=False)
    return output_path


def _as_existing_path(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path




def _count_ca_atoms_in_pdb(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                count += 1
    return count


def _validate_dockpy_inputs(template_protein_pdb: Path, ligand_pdb: Path) -> None:
    n_ca = _count_ca_atoms_in_pdb(template_protein_pdb)
    if n_ca < 20:
        raise ValueError(
            "Invalid dock.py template protein input: "
            f"{template_protein_pdb} contains only {n_ca} CA atoms. "
            "This file must be the article template protein PDB, usually 02_templates/template.pdb. "
            "It looks like template_protein_pdb may have been set to ligand.pdb by mistake."
        )
    if Path(template_protein_pdb).resolve() == Path(ligand_pdb).resolve():
        raise ValueError(
            "Invalid dock.py inputs: template_protein_pdb and dock_py_ligand_pdb are the same file. "
            "Use template.pdb for template_protein_pdb and ligand.pdb for dock_py_ligand_pdb."
        )

def _resolve_template_and_ligand_files(
    *,
    standard_structure_id: str,
    template_complex_pdb: str | Path | None,
    template_ligands: Mapping[str, Any] | None,
    tmp_path: Path,
) -> tuple[Path, Path]:
    template_ligands = template_ligands or {}
    template_protein_cfg = template_ligands.get("template_protein_pdb") if isinstance(template_ligands, Mapping) else None
    ligand_file_cfg = None
    if isinstance(template_ligands, Mapping):
        ligand_file_cfg = template_ligands.get("dock_py_ligand_pdb") or template_ligands.get("ligand_pdb")

    template_protein_pdb = _as_existing_path(template_protein_cfg, "step02.template_ligands.template_protein_pdb")
    ligand_pdb = _as_existing_path(ligand_file_cfg, "step02.template_ligands.dock_py_ligand_pdb")

    if template_protein_pdb is not None and ligand_pdb is not None:
        _validate_dockpy_inputs(template_protein_pdb, ligand_pdb)
        return template_protein_pdb, ligand_pdb

    template_complex_text = "" if template_complex_pdb is None else str(template_complex_pdb).strip()
    if not template_complex_text:
        raise ValueError(
            "dock.py exact mode needs either both files below:\n"
            "  step02.template_ligands.template_protein_pdb\n"
            "  step02.template_ligands.dock_py_ligand_pdb\n"
            "or a valid inputs.template_complex_pdb for fallback extraction."
        )
    template_complex_path = Path(template_complex_text)
    if not template_complex_path.exists():
        raise FileNotFoundError(f"inputs.template_complex_pdb not found: {template_complex_path}")

    if template_protein_pdb is None:
        template_protein_pdb = write_template_protein_from_complex(
            template_complex_path, tmp_path / f"{standard_structure_id}_template_protein.pdb"
        )
    if ligand_pdb is None:
        ligand_pdb = write_ligand_from_template_groups(
            template_complex_pdb=template_complex_path,
            template_ligands=template_ligands,
            output_path=tmp_path / f"{standard_structure_id}_dockpy_ligand.pdb",
        )
    _validate_dockpy_inputs(template_protein_pdb, ligand_pdb)
    return template_protein_pdb, ligand_pdb


def run_dock_py_compatible_for_one_structure(
    *,
    standard_structure_id: str,
    target_pdb: str | Path,
    template_complex_pdb: str | Path | None,
    output_complex_pdb: str | Path,
    template_ligands: Mapping[str, Any] | None = None,
    ligand_adjustment: Mapping[str, Any] | None = None,
    work_dir: str | Path | None = None,
) -> DockPyResult:
    """Run exactly the uploaded dock.py algorithm for one target structure.

    ``ligand_adjustment`` is accepted only for backward-compatible config parsing.
    In this exact mode it is intentionally not used, because the uploaded dock.py
    hard-codes its collision parameters inside ``rigid_docking``.
    """
    output_complex_pdb = Path(output_complex_pdb)
    output_complex_pdb.parent.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []

    try:
        if work_dir is None:
            tmp_ctx = TemporaryDirectory(prefix="catcongraph_dockpy_exact_")
            tmp_path = Path(tmp_ctx.name)
            cleanup = tmp_ctx.cleanup
        else:
            tmp_path = Path(work_dir)
            tmp_path.mkdir(parents=True, exist_ok=True)
            cleanup = lambda: None

        try:
            template_protein_pdb, ligand_pdb = _resolve_template_and_ligand_files(
                standard_structure_id=standard_structure_id,
                template_complex_pdb=template_complex_pdb,
                template_ligands=template_ligands,
                tmp_path=tmp_path,
            )

            # The next block is the original dock.py main loop, parameterized by paths.
            # Do not replace with CatConGraph geometry code.
            # Some RDKit versions used on servers do not accept pathlib.Path
            # objects in MolFromPDBFile / MolFromMolFile.  The original dock.py
            # load functions eventually call those RDKit readers, so keep the
            # geometry untouched but pass plain string paths here.
            template_protein_path = str(template_protein_pdb)
            ligand_path = str(ligand_pdb)
            target_path = str(target_pdb)

            template_protein = dockpy.load_pdb(template_protein_path)
            ligand = dockpy.load_molecule(ligand_path)
            if ligand is None:
                raise ValueError(f"Cannot load ligand file: {ligand_path}")
            if ligand.GetNumConformers() == 0:
                AllChem.EmbedMolecule(ligand, randomSeed=42)
            mutant_protein = dockpy.load_pdb(target_path)

            atom_map = dockpy.create_protein_atom_map(mutant_protein, template_protein) or []
            complex_mol = dockpy.rigid_docking(template_protein, ligand, mutant_protein)

            with Chem.PDBWriter(str(output_complex_pdb)) as writer:
                writer.write(complex_mol)

            return DockPyResult(
                standard_structure_id=standard_structure_id,
                source_pdb=str(target_pdb),
                complex_pdb_path=str(output_complex_pdb),
                ligand_pdb_path=str(ligand_pdb),
                template_protein_pdb_path=str(template_protein_pdb),
                n_ligand_atoms=int(ligand.GetNumAtoms()),
                n_alignment_atoms=int(len(atom_map)),
                alignment_rmsd_after=float("nan"),
                initial_collision_score=float("nan"),
                initial_clashing_ligand_atoms=0,
                final_collision_score=float("nan"),
                final_clashing_ligand_atoms=0,
                ligand_adjustment_status="original_dock.py_exact",
                ligand_adjustment_attempts=0,
                total_translation=0.0,
                total_rotation_deg=0.0,
                status="ok",
                error_message="",
                trace_rows=trace_rows,
            )
        finally:
            cleanup()
    except Exception as exc:
        return DockPyResult(
            standard_structure_id=standard_structure_id,
            source_pdb=str(target_pdb),
            complex_pdb_path=str(output_complex_pdb),
            ligand_pdb_path="",
            template_protein_pdb_path="",
            n_ligand_atoms=0,
            n_alignment_atoms=0,
            alignment_rmsd_after=float("nan"),
            initial_collision_score=float("nan"),
            initial_clashing_ligand_atoms=0,
            final_collision_score=float("nan"),
            final_clashing_ligand_atoms=0,
            ligand_adjustment_status="error",
            ligand_adjustment_attempts=0,
            total_translation=0.0,
            total_rotation_deg=0.0,
            status="error",
            error_message=str(exc),
            trace_rows=trace_rows,
        )

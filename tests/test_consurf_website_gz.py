"""Regression coverage for ConSurf website ``new1.gz``-style downloads."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
import tarfile

import pandas as pd


WEBSITE_GRADES = """\
POS SEQ ATOM SCORE COLOR CONFIDENCE INTERVAL B/E F/S MSA DATA RESIDUE VARIETY
  1  A  -          0.151   5* -0.544,  0.587  7,4 e     4/150 A 50%
 16  T  THR:132:A  1.650   2   0.867,  2.429  3,1 e f  79/150 T 3%
 18  C  CYS:134:A -1.190   9  -1.303, -1.135  9,9 b s  96/150 C 98%
 27  G  GLY:143:A -1.203   9  -1.303, -1.169  9,9 b s 105/150 G 99%
"""

LEGACY_COMPACT_GRADES = """\
 13 K -1.110 9 e f 149/150 K 98%, R 2%
 14 A  0.428 4 e   120/150 A 55%, G 45%
"""


def _write_website_tar_gz(path: Path, text: str = WEBSITE_GRADES) -> None:
    encoded = text.encode("utf-8")
    info = tarfile.TarInfo("nested/1PZT_1_consurf_grades.txt")
    info.size = len(encoded)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(encoded))


def test_legacy_compact_grades_keep_position_based_compatibility() -> None:
    from catcongraph.steps.step07_key_residue_screening import parse_consurf_grades_text

    scores = parse_consurf_grades_text(LEGACY_COMPACT_GRADES, default_chain="B")

    assert list(scores["residue_key"]) == ["LYS13B", "ALA14B"]
    assert list(scores["consurf_grade"]) == [9, 4]
    assert set(scores["residue_mapping_source"]) == {"alignment_position"}


def test_website_tar_gz_uses_pdb_residue_mapping_and_skips_unmapped_positions(tmp_path: Path) -> None:
    from catcongraph.steps.step07_key_residue_screening import load_consurf_scores

    path = tmp_path / "new1.gz"
    _write_website_tar_gz(path)
    scores, metadata = load_consurf_scores(path, {})

    assert metadata["source_type"] == "tar_archive"
    assert metadata["selected_file"] == "nested/1PZT_1_consurf_grades.txt"
    assert metadata["grades_format"] == "website_pdb_mapped"
    assert metadata["n_rows_skipped_unmapped_pdb_position"] == 1
    assert set(scores["residue_key"]) == {"THR132A", "CYS134A", "GLY143A"}
    assert "ALA1A" not in set(scores["residue_key"])
    assert int(scores.loc[scores["residue_key"] == "CYS134A", "consurf_grade"].iloc[0]) == 9
    assert scores.loc[scores["residue_key"] == "GLY143A", "residue_mapping_source"].iloc[0] == "pdb_annotation"


def test_plain_gzip_grades_file_is_also_accepted(tmp_path: Path) -> None:
    from catcongraph.steps.step07_key_residue_screening import load_consurf_scores

    path = tmp_path / "grades.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(WEBSITE_GRADES)
    scores, metadata = load_consurf_scores(path, {})

    assert metadata["source_type"] == "gzip_text"
    assert metadata["selected_file"] == "grades.txt.gz"
    assert list(scores["residue_key"]) == ["THR132A", "CYS134A", "GLY143A"]


def test_key_residue_workflow_accepts_website_tar_gz(tmp_path: Path) -> None:
    """Exercise the actual Step07 workflow, not only the standalone parser."""
    from catcongraph.steps.step07_key_residue_screening import run_step07

    archive_path = tmp_path / "new1.gz"
    _write_website_tar_gz(archive_path)
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  CYS A 134       0.000   0.000   0.000  1.00 90.00           C",
                "ATOM      2  CB  CYS A 134       1.000   0.000   0.000  1.00 90.00           C",
                "HETATM    3  C1  LIG A 501       2.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "complex_index.csv"
    pd.DataFrame([{"structure_id": "s1", "complex_pdb_path": str(complex_pdb)}]).to_csv(index_path, index=False)
    interaction = tmp_path / "05_plip_interaction_graphs" / "plip_outputs" / "s1" / "interaction_output.txt"
    interaction.parent.mkdir(parents=True)
    interaction.write_text(
        "Type: hydrogen_bonds\tLigand: C1\tProtein: CYS134A\tLigandNodeKey: LIG:A:501:C1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  target_id: TEST",
                f"  output_root: {tmp_path}",
                "step07:",
                f"  conservation_file: {archive_path}",
                f"  input_complex_table: {index_path}",
                "  ligand:",
                "    include_resnames: [LIG]",
                "  thresholds:",
                "    consurf_grade_min: 8",
                "    residue_ligand_distance_max: 3.0",
                "    occurrence_frequency_min: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_step07(config_path)

    assert report["conservation"]["source_type"] == "tar_archive"
    assert report["conservation_mapping_audit"]["n_exact_residue_key_matches"] == 1
    assert report["key_residues"] == ["CYS134A"]


def test_preflight_parses_website_tar_gz_before_the_full_workflow(tmp_path: Path) -> None:
    try:
        from catcongraph.preflight import run_preflight
    except ModuleNotFoundError as exc:
        # Preflight imports the existing docking module, whose optional
        # scientific dependency is not present in the lightweight parser test
        # environment.  The normal project environment installs RDKit.
        if exc.name == "rdkit":
            return
        raise

    archive_path = tmp_path / "new1.gz"
    _write_website_tar_gz(archive_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  target_id: TEST",
                f"  output_root: {tmp_path / 'results'}",
                "step07:",
                f"  conservation_file: {archive_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_preflight(config_path)

    conservation = report["inputs"]["conservation"]
    assert conservation["status"] == "parsed"
    assert conservation["source_type"] == "tar_archive"
    assert conservation["n_consurf_residues"] == 3

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pdb(path: Path, sequence: str, start: int = 1) -> None:
    aa3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY"}
    lines = []
    serial = 1
    for i, aa in enumerate(sequence, start=start):
        lines.append(
            f"ATOM  {serial:5d}  CA  {aa3[aa]:>3s} A{i:4d}    {float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C"
        )
        serial += 1
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def _scores(numbers: list[int], sequence: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chain": ["A"] * len(sequence),
            "resseq": numbers,
            "icode": [""] * len(sequence),
            "seq": list(sequence),
            "resname": [{"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY"}[x] for x in sequence],
            "consurf_grade": [9] * len(sequence),
            "consurf_score": [-1.0] * len(sequence),
            "consurf_position": list(range(1, len(sequence) + 1)),
            "residue_key": [f"{ {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY'}[x] }{n}A" for x, n in zip(sequence, numbers)],
        }
    )


def test_new6_style_manual_offset_maps_to_af2_numbering(tmp_path: Path) -> None:
    pdb = tmp_path / "af2.pdb"
    _pdb(pdb, "ACDE", start=1)
    records = pd.DataFrame([{"structure_id": "s1", "pdb_path": str(pdb)}])
    scores = _scores([229, 230, 231, 232], "ACDE")

    from catcongraph.steps.step07_key_residue_screening import map_consurf_scores_to_complexes

    mapped, audit, warnings = map_consurf_scores_to_complexes(
        scores,
        records,
        {"mapping": {"residue_number_offset": {"A": -228}}},
    )

    assert not warnings
    assert audit["n_mapped_rows"] == 4
    assert list(mapped["residue_key"]) == ["ALA1A", "CYS2A", "ASP3A", "GLU4A"]
    assert set(mapped["consurf_mapping_method"]) == {"manual_offset"}


def test_sequence_mapping_handles_truncated_renumbered_construct(tmp_path: Path) -> None:
    pdb = tmp_path / "af2.pdb"
    _pdb(pdb, "CDE", start=1)
    records = pd.DataFrame([{"structure_id": "s1", "pdb_path": str(pdb)}])
    scores = _scores([230, 231, 232, 233, 234], "ACDEG")

    from catcongraph.steps.step07_key_residue_screening import map_consurf_scores_to_complexes

    mapped, audit, _warnings = map_consurf_scores_to_complexes(scores, records, {"mapping": {}})

    assert audit["n_mapped_target_residues"] == 3
    assert audit["target_residue_coverage"] == 1.0
    assert list(mapped.dropna(subset=["residue_key"])["residue_key"]) == ["CYS1A", "ASP2A", "GLU3A"]
    assert set(mapped.loc[mapped["consurf_mapping_status"] == "mapped", "consurf_mapping_method"]) == {"sequence_alignment"}


def test_unrelated_consurf_sequence_is_rejected(tmp_path: Path) -> None:
    pdb = tmp_path / "af2.pdb"
    _pdb(pdb, "ACDE", start=1)
    records = pd.DataFrame([{"structure_id": "s1", "pdb_path": str(pdb)}])
    scores = _scores([1, 2, 3, 4], "GGGG")

    from catcongraph.steps.step07_key_residue_screening import map_consurf_scores_to_complexes

    mapped, audit, warnings = map_consurf_scores_to_complexes(
        scores, records, {"mapping": {"min_sequence_identity": 0.70}}
    )

    assert audit["n_mapped_rows"] == 0
    assert mapped["residue_key"].isna().all()
    assert warnings


def test_sequence_alignment_wins_over_wrong_same_numbered_keys(tmp_path: Path) -> None:
    """A construct with overlapping residue numbers must map by sequence."""
    pdb = tmp_path / "af2.pdb"
    _pdb(pdb, "CDE", start=1)
    records = pd.DataFrame([{"structure_id": "s1", "pdb_path": str(pdb)}])
    # The source was numbered 1..3 but starts with an extra residue in the
    # original construct.  Literal keys would incorrectly map ALA1/ CYS2 /...
    # onto the target; sequence alignment must select CDE -> target 1..3.
    scores = _scores([1, 2, 3, 4], "ACDE")

    from catcongraph.steps.step07_key_residue_screening import map_consurf_scores_to_complexes

    mapped, audit, warnings = map_consurf_scores_to_complexes(scores, records, {"mapping": {}})

    assert not warnings
    assert audit["n_mapped_rows"] == 3
    assert list(mapped.dropna(subset=["residue_key"])["residue_key"]) == ["CYS1A", "ASP2A", "GLU3A"]
    assert set(mapped.loc[mapped["consurf_mapping_status"] == "mapped", "consurf_mapping_method"]) == {"sequence_alignment"}


def test_duplicate_target_mapping_is_rejected(tmp_path: Path) -> None:
    pdb = tmp_path / "af2.pdb"
    _pdb(pdb, "ACDE", start=1)
    records = pd.DataFrame([{"structure_id": "s1", "pdb_path": str(pdb)}])
    # Two source rows intentionally point to the same target residue via a
    # manual offset.  Neither row may be silently retained.
    scores = _scores([0, 0, 2, 3], "AACD")

    from catcongraph.steps.step07_key_residue_screening import map_consurf_scores_to_complexes

    mapped, audit, warnings = map_consurf_scores_to_complexes(
        scores, records, {"mapping": {"residue_number_offset": {"A": 1}}}
    )

    assert audit["n_conflict_rows"] == 2
    # The important invariant is that every mapped target is unique.
    keys = mapped.loc[mapped["consurf_mapping_status"] == "mapped", "residue_key"].dropna().tolist()
    assert len(keys) == len(set(keys))

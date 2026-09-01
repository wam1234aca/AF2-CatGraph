from __future__ import annotations

import csv
from pathlib import Path

import pytest

from catcongraph.structure.af2_preprocess import (
    build_jobs,
    parse_residue_ranges,
    preprocess_af2_directory,
)


def _atom(serial: int, residue: int, atom: str = "CA", chain: str = "A") -> str:
    return f"ATOM  {serial:5d} {atom:<4} ALA {chain}{residue:4d}    {serial:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C  \n"


def test_parse_residue_ranges() -> None:
    ranges = parse_residue_ranges("1-120, 290-300,415")
    assert [(item.start, item.end) for item in ranges] == [(1, 120), (290, 300), (415, 415)]
    with pytest.raises(ValueError):
        parse_residue_ranges("120-1")


def test_rank_sorting_and_contiguous_output_numbering(tmp_path: Path) -> None:
    paths = [tmp_path / "x_rank_010.pdb", tmp_path / "x_rank_002.pdb", tmp_path / "x_ranked_0.pdb"]
    jobs = build_jobs(paths)
    assert [job.source.name for job in jobs] == ["x_ranked_0.pdb", "x_rank_002.pdb", "x_rank_010.pdb"]
    assert [job.rank for job in jobs] == [1, 2, 3]


def test_trim_and_rename_keeps_original_numbering_and_filters_conect(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "model_rank_002.pdb"
    source.parent.mkdir()
    source.write_text(
        "HEADER    TEST\n"
        + _atom(1, 1)
        + _atom(2, 2)
        + _atom(3, 3)
        + "CONECT    1    2    3\n"
        + "END\n",
        encoding="utf-8",
    )
    output = tmp_path / "trimmed"
    rows = preprocess_af2_directory(source.parent, output, "6B90", "1-2")
    assert len(rows) == 1
    result = output / "6B90.AF2rank1.pdb"
    lines = result.read_text(encoding="utf-8").splitlines()
    assert [int(line[22:26]) for line in lines if line.startswith("ATOM")] == [3]
    assert not any(line.startswith("CONECT") for line in lines)
    with (output / "af2_preprocess_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]["removed_residue_count"] == "2"


def test_refuses_output_inside_input_directory(tmp_path: Path) -> None:
    (tmp_path / "model.pdb").write_text(_atom(1, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="outside input_dir"):
        preprocess_af2_directory(tmp_path, tmp_path / "output", "6B90", "1")


def test_gui_uses_the_same_af2_preprocess_command_contract() -> None:
    from catcongraph.gui import _af2_preprocess_command

    command = _af2_preprocess_command(
        "data/raw", "data/trimmed", "6B90", "1-120,290-300",
        chains="A", recursive=False, rank_start=3,
    )
    assert "--input-dir" in command and command[command.index("--input-dir") + 1] == "data/raw"
    assert "--output-dir" in command and command[command.index("--output-dir") + 1] == "data/trimmed"
    assert "--chains" in command and command[command.index("--chains") + 1] == "A"
    assert "--no-recursive" in command

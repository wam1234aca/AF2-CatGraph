from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from catcongraph.steps.step01_af2_prepare import run_step01


def _ca(serial: int, residue: int, plddt: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{residue:4d}    "
        f"{float(residue):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{plddt:6.2f}           C\n"
    )


def _config(tmp_path: Path, af2_dir: Path, output_name: str, scope: str) -> dict:
    return {
        "project": {"target_id": "TEST", "output_root": str(tmp_path / output_name)},
        "inputs": {"af2_dir": str(af2_dir)},
        "step01": {
            "global_plddt_min": 75.0,
            "plddt_filter_scope": scope,
            "af2_preprocess_manifest": "auto",
            "standardized_pdb_mode": "none",
        },
    }


def test_step01_scores_full_length_source_but_keeps_trimmed_input_for_docking(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "rank1.pdb"
    trimmed = tmp_path / "trimmed" / "TEST.AF2rank1.pdb"
    raw.parent.mkdir()
    trimmed.parent.mkdir()
    raw.write_text("".join([_ca(1, 1, 50), _ca(2, 2, 50), _ca(3, 3, 90), _ca(4, 4, 90)]), encoding="utf-8")
    trimmed.write_text("".join([_ca(3, 3, 90), _ca(4, 4, 90)]), encoding="utf-8")
    with (trimmed.parent / "af2_preprocess_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "output"])
        writer.writeheader()
        writer.writerow({"source": str(raw), "output": str(trimmed)})

    report = run_step01(_config(tmp_path, trimmed.parent, "source_scope", "source_if_available"))
    summary = pd.read_csv(Path(report["output_dir"]) / "tables" / "TEST_step01_global_plddt_summary.csv")
    row = summary.iloc[0]
    assert report["n_passed_global_plddt"] == 0
    assert row["source_n_residues"] == 4
    assert row["input_n_residues"] == 2
    assert row["global_mean_plddt"] == 70.0
    assert row["plddt_filter_scope_applied"] == "source"
    assert Path(row["source_af2_path"]) == raw
    assert Path(row["source_path"]) == trimmed

    legacy = run_step01(_config(tmp_path, trimmed.parent, "input_scope", "input"))
    assert legacy["n_passed_global_plddt"] == 1


def test_step01_source_required_does_not_silently_score_a_trimmed_file(tmp_path: Path) -> None:
    af2_dir = tmp_path / "trimmed_without_manifest"
    af2_dir.mkdir()
    (af2_dir / "TEST.AF2rank1.pdb").write_text("".join([_ca(1, 1, 90), _ca(2, 2, 90)]), encoding="utf-8")
    report = run_step01(_config(tmp_path, af2_dir, "required_scope", "source_required"))
    assert report["n_passed_global_plddt"] == 0
    summary = pd.read_csv(Path(report["output_dir"]) / "tables" / "TEST_step01_global_plddt_summary.csv")
    assert summary.iloc[0]["status"] == "error"
    assert summary.iloc[0]["plddt_filter_scope_applied"] == "source_unavailable"

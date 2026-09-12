"""Regression contract for the default v8-compatible Studio path.

The fixture deliberately gives the text topology a conflicting extra atom.  A
valid structured edge CSV must win in ``v8_exact`` mode, exactly as it did in
AF2-CatGraph_v8; the two original GED classes must also merge after O1/O2 are
canonicalized to one phosphate unit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_v8_exact_merges_classes_and_draws_the_v8_topology(tmp_path: Path):
    from catcongraph.steps.step09_chemical_equivalence import run_step09
    from catcongraph.steps.step10_class_visualization import run_step10

    plip_root = tmp_path / "06_plip_interaction_graphs" / "plip_outputs"
    step07_tables = tmp_path / "07_key_residue_screening" / "tables"
    step08_tables = tmp_path / "08_ged_classification" / "tables"
    step07_tables.mkdir(parents=True)
    step08_tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"structure_id": "s1", "pass_key_residue_filter": True},
            {"structure_id": "s2", "pass_key_residue_filter": True},
        ]
    ).to_csv(step07_tables / "PTP1B_step07_passed_complex_index.csv", index=False)
    pd.DataFrame(
        [
            {"class": "class_1", "structure_id": "s1"},
            {"class": "class_2", "structure_id": "s2"},
        ]
    ).to_csv(step08_tables / "PTP1B_step08_graph_classes.csv", index=False)

    for structure_id, atom in [("s1", "O1"), ("s2", "O2")]:
        folder = plip_root / structure_id
        folder.mkdir(parents=True)
        (folder / "interaction_output.txt").write_text(
            "Type: hydrogen_bonds\t"
            f"Ligand: {atom}\tProtein: ASP9A\tLigandNodeKey: LIG:A:501:{atom}\n",
            encoding="utf-8",
        )
        (folder / "substrate_connectivity_edges.csv").write_text(
            "source,target\nLIG:A:501:O1,LIG:A:501:O2\n", encoding="utf-8"
        )
        (folder / "substrate_connectivity.txt").write_text(
            "LIG:A:501:O1 -> LIG:A:501:C99\n", encoding="utf-8"
        )

    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "project:",
                '  target_id: "PTP1B"',
                f'  output_root: "{tmp_path.as_posix()}"',
                "workflow:",
                '  profile: "manuscript_legacy"',
                "step09:",
                "  input_step06_dir: " + (tmp_path / "06_plip_interaction_graphs").as_posix(),
                "  split_nonidentical_classes: false",
                "  fail_if_class_members_nonidentical: false",
                "  canonicalization:",
                '    equivalence_mode: "v8_exact"',
                "    manual_ligand_units:",
                '      Phosphate unit: ["O1", "O2"]',
                "  export_candidates:",
                "    enabled: false",
                "step10:",
                "  plot:",
                '    formats: ["png"]',
                "    dpi: 72",
                "    fig_width: 2.0",
                "    fig_height: 1.6",
                "  layout:",
                '    connectivity_source: "v8_exact"',
                "  export_candidates:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    chemistry = run_step09(config)
    mapping = pd.read_csv(chemistry["tables"]["original_to_merged_classes"])
    canonical = pd.read_csv(chemistry["tables"]["raw_to_canonical_mapping"])
    assert chemistry["n_original_classes"] == 2
    assert chemistry["n_merged_classes"] == 1
    assert set(mapping["merged_class"]) == {"class_1"}
    assert set(canonical["canonical_ligand_unit"]) == {"Phosphate unit"}

    visualization = run_step10(config)
    assert Path(visualization["figures"]["class_grid_png"]).is_file()

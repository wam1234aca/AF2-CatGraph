from __future__ import annotations

from pathlib import Path

from catcongraph.steps.step09_chemical_equivalence import (
    UNIVERSAL_CONTACT_UNIT_RULES,
    build_ligand_unit_recommendations,
    build_substrate_inventory,
    canonicalize_records,
    collect_ligand_units,
    detect_universal_contact_units_from_pdbs,
)
from catcongraph.gui import _build_recommendation_preview


def _hetatm(serial: int, atom: str, element: str, x: float) -> str:
    return (
        f"HETATM{serial:5d} {atom:<4} LIG A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"  1.00 20.00          {element:>2}\n"
    )


def _carboxylate_pdb(path: Path) -> None:
    path.write_text(
        _hetatm(1, "C1", "C", 0.0)
        + _hetatm(2, "O1", "O", 1.2)
        + _hetatm(3, "O2", "O", -1.2)
        + "CONECT    1    2    3\nEND\n",
        encoding="utf-8",
    )


def test_recommendations_are_reviewable_then_applied_only_after_acceptance(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "step09": {
            "canonicalization": {
                "recommendations": {"enabled": True, "consensus_min_fraction": 0.8},
            }
        }
    }
    proposed, mapping, report = build_ligand_unit_recommendations(cfg, [pdb])
    assert len(proposed) == 1
    assert proposed.iloc[0]["status"] == "proposed"
    assert proposed.iloc[0]["rule"] == "carboxylate_or_carbonyl_oxygen_pair"
    assert mapping == {}
    recommendation_id = proposed.iloc[0]["recommendation_id"]

    cfg["step09"]["canonicalization"]["recommendations"]["accepted_recommendation_ids"] = [recommendation_id]
    accepted, mapping, report = build_ligand_unit_recommendations(cfg, [pdb])
    assert accepted.iloc[0]["status"] == "accepted"
    assert mapping["LIG:O1"] == mapping["LIG:O2"]
    assert report["n_accepted_ligand_unit_recommendations"] == 1

    strict_cfg = {
        "workflow": {"profile": "general"},
        "step09": {
            "canonicalization": {
                "equivalence_mode": "strict_chemistry",
                "auto_detect_ligand_units": True,
                "recommendations": {
                    "enabled": True,
                    "require_acceptance": True,
                },
            }
        },
    }
    raw_records = [
        {"raw_type": "hydrogen_bonds", "ligand_atom": "O1", "protein_raw": "ASP9A", "ligand_node_key": "LIG:A:1:O1", "structure_id": "s1", "class": "class_1"},
        {"raw_type": "hydrogen_bonds", "ligand_atom": "O2", "protein_raw": "ASP9A", "ligand_node_key": "LIG:A:1:O2", "structure_id": "s2", "class": "class_2"},
    ]
    unaccepted, _report, _audit = canonicalize_records(
        raw_records,
        strict_cfg,
        "NEW_TARGET",
        [pdb],
    )
    assert unaccepted["canonical_ligand_unit"].nunique() == 2
    strict_cfg["step09"]["canonicalization"]["recommendations"]["accepted_recommendation_ids"] = [recommendation_id]
    canonical, _report, _audit = canonicalize_records(raw_records, strict_cfg, "NEW_TARGET", [pdb])
    assert canonical["canonical_ligand_unit"].nunique() == 1
    assert set(canonical["ligand_unit_mapping_source"]) == {"accepted_recommendation"}


def test_inventory_keeps_optional_smiles_as_an_audited_annotation(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "step09": {
            "canonicalization": {
                "recommendations": {"ligand_smiles": {"LIG": "CC(=O)O"}},
            }
        }
    }
    inventory = build_substrate_inventory(cfg, [pdb])
    assert inventory.iloc[0]["resname"] == "LIG"
    assert inventory.iloc[0]["smiles"] == "CC(=O)O"
    assert inventory.iloc[0]["smiles_status"] in {
        "validated_rdkit", "not_validated_rdkit_unavailable",
    }


def test_gui_config_can_select_strict_chemistry_for_a_new_enzyme() -> None:
    from catcongraph.gui_config import DEFAULT_VALUES, build_config

    values = dict(DEFAULT_VALUES)
    values["equivalence_mode"] = "strict_chemistry"
    config = build_config(values)
    assert config["step08"]["canonicalization"]["equivalence_mode"] == "strict_chemistry"


def test_phosphate_recommendation_contains_only_terminal_oxygens(tmp_path: Path) -> None:
    pdb = tmp_path / "phosphate.pdb"
    pdb.write_text(
        _hetatm(1, "P1", "P", 0.0)
        + _hetatm(2, "O1", "O", 1.5)
        + _hetatm(3, "O2", "O", -1.5)
        + _hetatm(4, "OB", "O", 0.0)
        + _hetatm(5, "C1", "C", 0.0)
        + "CONECT    1    2    3    4\n"
        + "CONECT    4    5\nEND\n",
        encoding="utf-8",
    )
    cfg = {"step09": {"canonicalization": {"recommendations": {"enabled": True}}}}
    table, _mapping, _report = build_ligand_unit_recommendations(cfg, [pdb])
    assert len(table) == 1
    assert table.iloc[0]["rule"] == "phosphate_terminal_oxygen_group"
    assert table.iloc[0]["atoms"] == "O1;O2"
    assert "P1" not in table.iloc[0]["atoms"]
    assert "OB" not in table.iloc[0]["atoms"]


def test_recommendations_require_a_complete_terminal_group(tmp_path: Path) -> None:
    pdb = tmp_path / "single_terminal_oxygen.pdb"
    pdb.write_text(
        _hetatm(1, "C1", "C", 0.0)
        + _hetatm(2, "O1", "O", 1.2)
        + "END\n",
        encoding="utf-8",
    )
    cfg = {"step09": {"canonicalization": {"recommendations": {"enabled": True}}}}
    table, _mapping, _report = build_ligand_unit_recommendations(cfg, [pdb])
    assert table.empty


def test_unmatched_saved_acceptance_is_reported_explicitly(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "step09": {
            "canonicalization": {
                "recommendations": {
                    "enabled": True,
                    "accepted_recommendation_ids": ["ID_FROM_A_DIFFERENT_LIGAND"],
                }
            }
        }
    }

    _table, mapping, report = build_ligand_unit_recommendations(cfg, [pdb])

    assert mapping == {}
    assert report["matched_accepted_recommendation_ids"] == []
    assert report["unmatched_accepted_recommendation_ids"] == [
        "ID_FROM_A_DIFFERENT_LIGAND"
    ]


def test_recommendations_use_audited_local_geometry_when_template_omits_conect(tmp_path: Path) -> None:
    """A template PDB without CONECT can still make a review-only proposal.

    The suggestion remains inactive until its stable ID is accepted, so this
    fallback cannot silently alter a class assignment.
    """
    pdb = tmp_path / "no_conect_carboxylate.pdb"
    pdb.write_text(
        _hetatm(1, "C1", "C", 0.0)
        + _hetatm(2, "O1", "O", 1.2)
        + _hetatm(3, "O2", "O", -1.2)
        + "END\n",
        encoding="utf-8",
    )
    cfg = {"step09": {"canonicalization": {"recommendations": {"enabled": True}}}}
    table, mapping, report = build_ligand_unit_recommendations(cfg, [pdb])
    assert len(table) == 1
    assert table.iloc[0]["status"] == "proposed"
    assert table.iloc[0]["connectivity_evidence"] == "local_geometry_covalent_cutoff"
    assert table.iloc[0]["confidence"] == "moderate_review_required"
    assert mapping == {}
    assert report["n_ligand_unit_recommendations"] == 1


def test_gui_preview_reports_an_empty_result_instead_of_hiding_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing_ligand.pdb"
    config = {
        "inputs": {"ligand_pdb": str(missing)},
        "step09": {"canonicalization": {"recommendations": {"enabled": True}}},
    }
    records, inventory, report, input_audit = _build_recommendation_preview(config)
    assert records == []
    assert inventory == []
    assert report["missing_ligand_paths"] == [str(missing)]
    assert input_audit == [{
        "requested_path": str(missing), "exists": False, "status": "missing_file",
        "hetatm_records": 0, "conect_records": 0,
    }]


def test_new_review_workflow_does_not_apply_generic_v8_auto_mapping(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "workflow": {"profile": "manuscript_legacy"},
        "step09": {
            "canonicalization": {
                "equivalence_mode": "v8_exact",
                "auto_detect_ligand_units": True,
                "recommendations": {"enabled": True, "require_acceptance": True},
            }
        },
    }
    raw_records = [
        {"raw_type": "hydrogen_bonds", "ligand_atom": "O1", "protein_raw": "ASP9A", "ligand_node_key": "LIG:A:1:O1"},
        {"raw_type": "hydrogen_bonds", "ligand_atom": "O2", "protein_raw": "ASP9A", "ligand_node_key": "LIG:A:1:O2"},
    ]
    canonical, _report, _audit = canonicalize_records(raw_records, cfg, "NEW_TARGET", [pdb])
    assert canonical["canonical_ligand_unit"].nunique() == 2


def test_conservative_manual_policy_keeps_only_explicit_units(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "conservative_manual",
                "use_builtin_ligand_units": True,
                "auto_detect_ligand_units": True,
                "manual_ligand_units": {
                    "Reviewed carboxylate O": ["LIG:O1", "LIG:O2"],
                },
                "recommendations": {"enabled": True},
            }
        }
    }

    table, accepted, recommendation_report = build_ligand_unit_recommendations(cfg, [pdb])
    mapping, report, _audit = collect_ligand_units(cfg, "AqTMK", [pdb])

    assert table.empty
    assert accepted == {}
    assert recommendation_report["recommendations_disabled_reason"] == "conservative_manual_policy"
    assert mapping == {
        "LIG:O1": "Reviewed carboxylate O",
        "LIG:O2": "Reviewed carboxylate O",
    }
    assert report["n_builtin_atom_mappings"] == 0
    assert report["n_auto_atom_mappings"] == 0
    assert report["contact_unit_mode"] == "conservative_manual"


def test_supported_target_suggestions_require_acceptance() -> None:
    cfg = {
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "guided_review",
                "equivalence_mode": "standard",
                "recommendations": {"enabled": True, "require_acceptance": True},
            }
        }
    }

    table, _accepted, _recommendation_report = build_ligand_unit_recommendations(
        cfg, [], target_id="AqTMK"
    )
    mapping, report, _audit = collect_ligand_units(cfg, "AqTMK", [])

    assert len(table) == 7
    assert set(table["status"]) == {"recommended"}
    assert mapping == {}
    assert report["n_builtin_atom_mappings"] == 0
    assert report["n_accepted_recommendation_atom_mappings"] == 0
    assert report["n_auto_atom_mappings"] == 0
    assert report["contact_unit_mode"] == "guided_review"


def test_supported_target_suggestions_are_applied_after_acceptance() -> None:
    base = {
        "project": {"target_id": "AqTMK"},
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "guided_review",
                "equivalence_mode": "standard",
                "recommendations": {"enabled": True, "require_acceptance": True},
            }
        },
    }
    table, _accepted, _report = build_ligand_unit_recommendations(
        base, [], target_id="AqTMK"
    )
    accepted_ids = table["recommendation_id"].astype(str).tolist()
    base["step09"]["canonicalization"]["recommendations"][
        "accepted_recommendation_ids"
    ] = accepted_ids

    mapping, report, _audit = collect_ligand_units(base, "AqTMK", [])

    assert len(mapping) == 19
    assert report["n_builtin_atom_mappings"] == 0
    assert report["n_accepted_recommendation_atom_mappings"] == 19


def test_guided_reference_target_match_is_case_insensitive_but_exact() -> None:
    cfg = {
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "guided_review",
                "equivalence_mode": "standard",
                "recommendations": {"enabled": True, "require_acceptance": True},
            }
        }
    }

    lower_table, _accepted, _report = build_ligand_unit_recommendations(
        cfg, [], target_id="aqtmk"
    )
    renamed_table, _accepted, _report = build_ligand_unit_recommendations(
        cfg, [], target_id="AqTMK2"
    )
    lower_mapping, lower_report, _audit = collect_ligand_units(cfg, "aqtmk", [])
    renamed_mapping, renamed_report, _audit = collect_ligand_units(cfg, "AqTMK2", [])

    assert len(lower_table) == 7
    assert renamed_table.empty
    assert lower_mapping == {}
    assert lower_report["builtin_unit_source"] == "applied_after_review_acceptance"
    assert renamed_mapping == {}
    assert renamed_report["builtin_unit_source"] == "none"


def test_supported_target_shows_built_in_suggestions_and_allows_manual_rules(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "atp_like.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "project": {"target_id": "AqTMK"},
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "guided_review",
                "equivalence_mode": "standard",
                "use_builtin_ligand_units": False,
                "manual_ligand_units": {
                    "Unreported refinement": ["LIG:O1", "LIG:O2"],
                },
                "recommendations": {
                    "enabled": True,
                    "require_acceptance": True,
                    "accepted_recommendation_ids": [
                        "LIG_CARBOXYLATE_OR_CARBONYL_OXYGEN_PAIR_O1_O2"
                    ],
                },
            }
        },
    }

    table, accepted, recommendation_report = build_ligand_unit_recommendations(
        cfg, [pdb], target_id="AqTMK"
    )
    mapping, report, _audit = collect_ligand_units(cfg, "AqTMK", [pdb])

    assert len(table) == 7
    assert set(table["status"]) == {"recommended"}
    assert accepted == {}
    assert recommendation_report["recommendations_enabled"] is True
    assert mapping == {
        "LIG:O1": "Unreported refinement",
        "LIG:O2": "Unreported refinement",
    }
    assert report["n_builtin_atom_mappings"] == 0
    assert report["n_manual_atom_mappings"] == 2
    assert report["n_accepted_recommendation_atom_mappings"] == 0


def test_conservative_policy_keeps_atom_identity_but_groups_interaction_types() -> None:
    cfg = {
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "conservative_manual",
                "manual_ligand_units": {},
            }
        }
    }
    raw_records = [
        {
            "raw_type": "hydrogen_bonds",
            "ligand_atom": "O1",
            "ligand_node_key": "LIG:A:1:O1",
            "protein_raw": "ASP9A",
        },
        {
            "raw_type": "salt_bridges",
            "ligand_atom": "O2",
            "ligand_node_key": "LIG:A:1:O2",
            "protein_raw": "ARG10A",
        },
    ]

    canonical, report, _audit = canonicalize_records(
        raw_records, cfg, "NEW14", []
    )

    assert set(canonical["canonical_ligand_unit"]) == {"LIG_1_O1", "LIG_1_O2"}
    assert set(canonical["interaction_supertype"]) == {"Polar contact"}
    assert report["contact_unit_mode"] == "conservative_manual"


def test_gui_config_preserves_contact_unit_review_fields_for_supported_target() -> None:
    from catcongraph.gui_config import DEFAULT_VALUES, build_config

    values = dict(DEFAULT_VALUES)
    values.update({
        "target_id": "AqTMK",
        "manual_ligand_units_yaml": "Extra: [ATP:O9, ATP:O10]",
        "accepted_recommendation_ids_yaml": "[STALE_ACCEPTED_ID]",
        "rejected_recommendation_ids_yaml": "[STALE_REJECTED_ID]",
    })

    config = build_config(values)
    canonicalization = config["step08"]["canonicalization"]

    assert canonicalization["manual_ligand_units"] == {
        "Extra": ["ATP:O9", "ATP:O10"]
    }
    assert canonicalization["recommendations"]["enabled"] is True
    assert canonicalization["recommendations"]["accepted_recommendation_ids"] == [
        "STALE_ACCEPTED_ID"
    ]
    assert canonicalization["recommendations"]["rejected_recommendation_ids"] == [
        "STALE_REJECTED_ID"
    ]


def _component_atom(
    serial: int, atom: str, resname: str, resseq: int, element: str
) -> str:
    return (
        f"HETATM{serial:5d} {atom:<4} {resname:>3} A{resseq:4d}    "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"  1.00 20.00          {element:>2}\n"
    )


def _conect(source: int, *targets: int) -> str:
    return "CONECT" + f"{source:5d}" + "".join(f"{target:5d}" for target in targets) + "\n"


def test_universal_rules_apply_automatically_without_acceptance(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _carboxylate_pdb(pdb)
    cfg = {
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "universal_rules",
                "auto_detect_ligand_units": True,
                "recommendations": {
                    "enabled": True,
                    "require_acceptance": False,
                    "consensus_min_fraction": 0.8,
                    "accepted_recommendation_ids": [],
                },
            }
        }
    }
    records = [
        {"raw_type": "hydrogen_bonds", "ligand_atom": "O1", "protein_raw": "ASP9A", "ligand_node_key": "LIG:A:1:O1"},
        {"raw_type": "hydrogen_bonds", "ligand_atom": "O2", "protein_raw": "ASP9A", "ligand_node_key": "LIG:A:1:O2"},
    ]

    table, mapping, recommendation_report = build_ligand_unit_recommendations(
        cfg, [pdb], target_id="NEW_ENZYME"
    )
    canonical, report, _audit = canonicalize_records(
        records, cfg, "NEW_ENZYME", [pdb]
    )

    assert "applied" in set(table["status"])
    assert mapping["LIG:A:1:O1"] == "Carboxylate oxygen unit"
    assert mapping["LIG:A:1:O2"] == "Carboxylate oxygen unit"
    assert canonical["canonical_ligand_unit"].nunique() == 1
    assert set(canonical["ligand_unit_mapping_source"]) == {"universal_s4_rule"}
    assert recommendation_report["contact_unit_mode"] == "universal_rules"
    assert report["automatic_rule"] == "Supplementary_Table_S4_universal_contact_units"


def test_three_manuscript_cases_use_the_same_universal_detector(tmp_path: Path) -> None:
    cfg = {
        "step09": {
            "canonicalization": {
                "contact_unit_mode": "universal_rules",
                "recommendations": {"enabled": True, "require_acceptance": False},
            }
        }
    }

    gac = tmp_path / "gac_glutamine.pdb"
    gac_atoms = [
        ("N1", "N"), ("C1", "C"), ("O1", "O"), ("C2", "C"), ("C3", "C"),
        ("C4", "C"), ("C5", "C"), ("N2", "N"), ("O2", "O"), ("O3", "O"),
    ]
    gac_edges = [(2, 1), (2, 3), (1, 4), (4, 5), (4, 8), (5, 6), (6, 7), (7, 9), (7, 10)]
    gac.write_text(
        "".join(_component_atom(i, atom, "GLN", 1, element) for i, (atom, element) in enumerate(gac_atoms, 1))
        + "".join(_conect(left, right) for left, right in gac_edges) + "END\n",
        encoding="utf-8",
    )

    aqtmk = tmp_path / "aqtmk_phosphate_core.pdb"
    aq_atoms = [
        ("P1", "ATP", 1, "P"), ("O2", "ATP", 1, "O"), ("O3", "ATP", 1, "O"),
        ("O4", "ATP", 1, "O"), ("O5", "ATP", 1, "O"), ("P2", "ATP", 1, "P"),
        ("O6", "ATP", 1, "O"), ("O7", "ATP", 1, "O"), ("O8", "ATP", 1, "O"),
        ("P3", "ATP", 1, "P"), ("O9", "ATP", 1, "O"), ("O10", "ATP", 1, "O"),
        ("O11", "ATP", 1, "O"), ("P4", "TMP", 2, "P"), ("O12", "TMP", 2, "O"),
        ("O13", "TMP", 2, "O"), ("O14", "TMP", 2, "O"), ("O15", "TMP", 2, "O"),
        ("MG", "MG", 3, "MG"),
    ]
    aq_edges = [
        (1, 2), (1, 3), (1, 4), (1, 5), (5, 6), (6, 7), (6, 8), (6, 9),
        (9, 10), (10, 11), (10, 12), (10, 13),
        (14, 15), (14, 16), (14, 17), (14, 18),
    ]
    aqtmk.write_text(
        "".join(
            _component_atom(i, atom, resname, resseq, element)
            for i, (atom, resname, resseq, element) in enumerate(aq_atoms, 1)
        ) + "".join(_conect(left, right) for left, right in aq_edges) + "END\n",
        encoding="utf-8",
    )

    ptp1b = tmp_path / "ptp1b_substrate.pdb"
    ptp_atoms = [
        ("C1", "C"), ("C2", "C"), ("C3", "C"), ("C4", "C"), ("C5", "C"),
        ("C6", "C"), ("C7", "C"), ("C8", "C"), ("O5", "O"), ("O6", "O"),
        ("P1", "P"), ("O1", "O"), ("O2", "O"), ("O3", "O"), ("O4", "O"),
        ("N1", "N"),
    ]
    ptp_edges = [
        (1, 2), (1, 9), (1, 10), (2, 3), (2, 16),
        (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 3),
        (11, 12), (11, 13), (11, 14), (11, 15), (12, 4),
    ]
    ptp1b.write_text(
        "".join(_component_atom(i, atom, "PTR", 1, element) for i, (atom, element) in enumerate(ptp_atoms, 1))
        + "".join(_conect(left, right) for left, right in ptp_edges) + "END\n",
        encoding="utf-8",
    )

    expected = {
        "GAC": (gac, {"GLN:A:1:O2": "Carboxylate oxygen unit", "GLN:A:1:C2": "Aliphatic carbon unit", "GLN:A:1:N1": "N1 site"}),
        "AqTMK": (aqtmk, {"ATP:A:1:P1": "ATP phosphate 1 unit", "ATP:A:1:O5": "ATP phosphate 1–2 bridging oxygen", "TMP:A:2:P4": "dTMP phosphate unit", "MG:A:3:MG": "Mg2+ ion"}),
        "PTP1B": (ptp1b, {"PTR:A:1:O5": "Carboxylate oxygen unit", "PTR:A:1:P1": "Phosphate unit", "PTR:A:1:O1": "Phosphate unit", "PTR:A:1:C3": "Aromatic ring unit", "PTR:A:1:C1": "Aliphatic carbon unit"}),
    }
    for target_id, (pdb, required) in expected.items():
        mapping, report, audit = collect_ligand_units(cfg, target_id, [pdb])
        for atom_key, unit in required.items():
            assert mapping[atom_key] == unit
        assert report["builtin_unit_source"] == "disabled_in_universal_rules"
        assert report["n_builtin_atom_mappings"] == 0
        assert set(audit["decision"]) == {"apply"}


def test_universal_detector_ignores_protonation_hydrogens(tmp_path: Path) -> None:
    pdb = tmp_path / "protonated_carboxylate.pdb"
    pdb.write_text(
        _component_atom(1, "C1", "CAR", 1, "C")
        + _component_atom(2, "O1", "CAR", 1, "O")
        + _component_atom(3, "O2", "CAR", 1, "O")
        + _component_atom(4, "H", "CAR", 1, "H")
        + "CONECT    1    2    3\n"
        + "CONECT    2    4\nEND\n",
        encoding="utf-8",
    )

    mapping, audit = detect_universal_contact_units_from_pdbs([pdb])
    inventory = build_substrate_inventory({}, [pdb])

    assert mapping["CAR:A:1:O1"] == "Carboxylate oxygen unit"
    assert mapping["CAR:A:1:O2"] == "Carboxylate oxygen unit"
    assert not any("H" in str(row.get("atoms", "")).split(";") for row in audit)
    assert inventory.iloc[0]["atom_names"] == "C1;O1;O2"
    assert inventory.iloc[0]["elements"] == "C;O"


def test_universal_detector_covers_the_full_s4_rule_catalogue(tmp_path: Path) -> None:
    pdb = tmp_path / "universal_rules.pdb"
    lines = []
    bonds = []

    def add_component(resname: str, resseq: int, atoms: list[tuple[str, str]], edges: list[tuple[int, int]]) -> None:
        offset = len(lines)
        serials = []
        for atom_name, element in atoms:
            serial = len(lines) + 1
            serials.append(serial)
            lines.append(_component_atom(serial, atom_name, resname, resseq, element))
        for left, right in edges:
            bonds.append((serials[left], serials[right]))

    add_component("CAR", 1, [("C1", "C"), ("O1", "O"), ("O2", "O"), ("C2", "C")], [(0, 1), (0, 2), (0, 3)])
    add_component("PHO", 2, [("P1", "P"), ("O1", "O"), ("O2", "O"), ("OB", "O"), ("P2", "P"), ("O3", "O"), ("O4", "O")], [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6)])
    add_component("SUL", 3, [("S1", "S"), ("O1", "O"), ("O2", "O"), ("O3", "O"), ("C1", "C")], [(0, 1), (0, 2), (0, 3), (0, 4)])
    add_component("SO2", 4, [("S1", "S"), ("O1", "O"), ("O2", "O"), ("C1", "C")], [(0, 1), (0, 2), (0, 3)])
    add_component("NIT", 5, [("N1", "N"), ("O1", "O"), ("O2", "O"), ("C1", "C")], [(0, 1), (0, 2), (0, 3)])
    add_component("GUA", 6, [("C1", "C"), ("N1", "N"), ("N2", "N"), ("C2", "C")], [(0, 1), (0, 2), (0, 3)])
    add_component("BEN", 7, [(f"C{i}", "C") for i in range(1, 7)], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    add_component("SUG", 8, [("O1", "O"), ("C1", "C"), ("C2", "C"), ("C3", "C"), ("C4", "C"), ("C5", "C"), ("O2", "O"), ("O3", "O")], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 6), (2, 7)])
    add_component("ALI", 9, [("C1", "C"), ("C2", "C"), ("C3", "C")], [(0, 1), (1, 2)])
    add_component("MG", 10, [("MG", "MG")], [])
    add_component("ONE", 11, [("N1", "N")], [])

    pdb.write_text(
        "".join(lines)
        + "".join(_conect(left, right) for left, right in bonds)
        + "END\n",
        encoding="utf-8",
    )

    mapping, audit = detect_universal_contact_units_from_pdbs([pdb])
    applied_rules = {str(row.get("rule")) for row in audit if row.get("decision") == "apply"}
    expected_rules = {rule for rule, _label in UNIVERSAL_CONTACT_UNIT_RULES}

    assert expected_rules <= applied_rules
    assert mapping["PHO:A:2:P1"] == "PHO phosphate 1 unit"
    assert mapping["PHO:A:2:OB"] == "PHO phosphate 1–2 bridging oxygen"
    assert mapping["BEN:A:7:C1"] == "BEN aromatic ring unit"
    assert mapping["SUG:A:8:C1"] == "SUG sugar-ring carbon unit"
    assert mapping["ALI:A:9:C1"] == "ALI aliphatic carbon unit"
    assert mapping["MG:A:10:MG"] == "Mg2+ ion"
    assert mapping["ONE:A:11:N1"] == "ONE N1 site"

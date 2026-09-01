from pathlib import Path

import yaml

from catcongraph.gui_config import DEFAULT_VALUES, build_config, config_to_values
from catcongraph.config import PUBLIC_CONFIG_SCHEMA, normalize_project_config
from catcongraph.profiles import workflow_profile
from catcongraph.qc.clash import normalize_clash_method
from catcongraph.workflow import init_project


def _write_old_config(path: Path) -> None:
    path.write_text(
        """
project:
  target_id: GAC
  output_root: results/GAC
workflow:
  profile: manuscript_legacy
inputs:
  af2_dir: data/GAC/01_af2
  template_protein_pdb: data/GAC/02_templates/template.pdb
  ligand_pdb: data/GAC/02_templates/ligand.pdb
step02:
  alignment_mode: auto
  multi_ligand_mode: sequential_dock_py
step03:
  clash:
    method: collision_py_exact
    clash_max: 10
    delta: 0.5
step06:
  node_identity:
    ligand_node_id: residue_aware
  connectivity:
    source: input_conect_then_rdkit_same_residue
step07:
  conservation_file: data/GAC/07_conservation/GAC.csv
  thresholds:
    require_graph_occurrence: false
step08:
  ged:
    executable: /tools/ged
    require_same_edge_count: true
step09:
  split_nonidentical_classes: true
  class_signature_policy: representative
  canonicalization:
    equivalence_mode: v8_exact
    recommendations:
      require_acceptance: true
      accepted_recommendation_ids: [proposal_1]
""".lstrip(),
        encoding="utf-8",
    )


def test_old_names_load_as_generic_public_names_without_losing_settings(tmp_path: Path) -> None:
    source = tmp_path / "old.yaml"
    _write_old_config(source)

    values = config_to_values(source)

    assert values["workflow_profile"] == "standard"
    assert values["step02_alignment_mode"] == "auto"
    assert values["step02_multi_ligand_mode"] == "sequential"
    assert values["equivalence_mode"] == "standard"
    assert values["class_signature_policy"] == "most_common"
    assert values["ged_require_same_edge_count"] is True
    assert values["split_nonidentical_classes"] is True
    assert values["require_graph_occurrence"] is False
    assert "proposal_1" in values["accepted_recommendation_ids_yaml"]


def test_new_gui_writes_generic_names_with_the_same_calculation_choices(tmp_path: Path) -> None:
    source = tmp_path / "old.yaml"
    _write_old_config(source)
    config = build_config(config_to_values(source))

    assert config["config_schema"] == PUBLIC_CONFIG_SCHEMA
    assert "workflow" not in config
    assert config["step01"]["global_plddt_min"] == 80.0
    assert config["step02"]["alignment_mode"] == "auto"
    assert config["step02"]["multi_ligand_mode"] == "sequential"
    assert config["step03"]["clash"]["method"] == "standard"
    assert config["step08"]["canonicalization"]["equivalence_mode"] == "standard"
    assert config["step08"]["canonicalization"]["contact_unit_mode"] == "guided_review"
    assert config["step08"]["class_signature_policy"] == "most_common"
    assert config["step06"]["thresholds"]["require_graph_occurrence"] is False
    assert "step10" not in config


def test_old_and_new_aliases_select_the_same_internal_contract() -> None:
    assert workflow_profile({"workflow": {"profile": "manuscript_legacy"}}) == workflow_profile(
        {"workflow": {"profile": "catcongraph_studio_v1"}}
    )
    assert normalize_clash_method("collision_py_exact") == normalize_clash_method("standard")


def test_explicit_editable_choices_are_not_overwritten() -> None:
    values = {
        **DEFAULT_VALUES,
        "target_id": "TEST",
        "step02_alignment_mode": "sequence_aware",
        "step02_multi_ligand_mode": "rigid_assembly",
        "plip_ligand_node_id": "residue_atom_name",
        "plip_connectivity_source": "pdb_conect",
        "ged_require_same_edge_count": False,
        "equivalence_mode": "chemistry_aware",
        "split_nonidentical_classes": False,
        "class_signature_policy": "intersection",
    }
    config = build_config(values)

    assert config["step02"]["alignment_mode"] == "sequence_aware"
    assert config["step02"]["multi_ligand_mode"] == "rigid_assembly"
    assert config["step05"]["node_identity"]["ligand_node_id"] == "residue_atom_name"
    assert config["step05"]["connectivity"]["source"] == "pdb_conect"
    assert config["step07"]["ged"]["require_same_edge_count"] is False
    assert config["step08"]["canonicalization"]["equivalence_mode"] == "chemistry_aware"
    assert config["step08"]["split_nonidentical_classes"] is False
    assert config["step08"]["class_signature_policy"] == "intersection"


def test_conservative_contact_units_disable_presets_and_proposals() -> None:
    values = {
        **DEFAULT_VALUES,
        "contact_unit_mode": "conservative_manual",
        "manual_ligand_units_yaml": "Carboxylate_O: [LIG:O1, LIG:O2]",
    }
    config = build_config(values)
    canonicalization = config["step08"]["canonicalization"]

    assert canonicalization["contact_unit_mode"] == "conservative_manual"
    assert canonicalization["use_builtin_ligand_units"] is False
    assert canonicalization["auto_detect_ligand_units"] is False
    assert canonicalization["recommendations"]["enabled"] is False
    assert canonicalization["manual_ligand_units"] == {
        "Carboxylate_O": ["LIG:O1", "LIG:O2"]
    }


def test_new_projects_default_to_automatic_universal_s4_rules() -> None:
    config = build_config(DEFAULT_VALUES)
    canonicalization = config["step08"]["canonicalization"]

    assert canonicalization["contact_unit_mode"] == "universal_rules"
    assert canonicalization["use_builtin_ligand_units"] is False
    assert canonicalization["auto_detect_ligand_units"] is True
    assert canonicalization["recommendations"]["enabled"] is True
    assert canonicalization["recommendations"]["require_acceptance"] is False


def test_generated_yaml_contains_no_deprecated_public_names() -> None:
    config = build_config(DEFAULT_VALUES)
    text = yaml.safe_dump(config, sort_keys=False)
    for deprecated in (
        "manuscript_legacy", "v8_exact", "sequential_dock_py",
        "collision_py_exact", "dock_py_exact", "representative",
    ):
        assert deprecated not in text


def test_public_config_normalizes_into_the_established_runtime_slots() -> None:
    public = build_config(DEFAULT_VALUES)
    runtime = normalize_project_config(public)

    assert runtime["_config_layout"] == PUBLIC_CONFIG_SCHEMA
    assert runtime["step05"] == public["step04"]
    assert runtime["step06"] == public["step05"]
    assert runtime["step07"] == public["step06"]
    assert runtime["step08"] == public["step07"]
    assert runtime["step09"] == public["step08"]
    assert runtime["step10"] == public["step09"]


def test_init_creates_only_config_and_uses_public_defaults(tmp_path: Path) -> None:
    project = tmp_path / "MY_ENZYME"
    config_path = init_project("MY_ENZYME", project)

    assert [path.name for path in project.iterdir()] == ["config.yaml"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["config_schema"] == PUBLIC_CONFIG_SCHEMA
    assert config["step01"]["global_plddt_min"] == 80.0
    assert config["step04"]["catalytic_region"]["distance_cutoff"] == 6.0
    assert config["step10_optional"]["enabled"] is False
    assert "step10" not in config

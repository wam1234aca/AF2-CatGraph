"""Configuration helpers for the local AF2-CatGraph graphical interface.

The GUI deliberately writes ordinary AF2-CatGraph YAML files.  The command-line
workflow is still the single implementation of the scientific method; this
module only helps users create, inspect, and validate its configuration.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from catcongraph.config import (
    PUBLIC_CONFIG_SCHEMA, load_yaml_config, normalize_project_config,
    public_output_subdir,
)


DEFAULT_VALUES: dict[str, Any] = {
    "target_id": "MY_ENZYME",
    "output_root": "results/MY_ENZYME",
    "af2_dir": "data/MY_ENZYME/01_af2",
    "template_protein_pdb": "data/MY_ENZYME/02_templates/template_protein.pdb",
    "ligand_paths": "data/MY_ENZYME/02_templates/ligand.pdb",
    "conservation_file": "data/MY_ENZYME/07_conservation/MY_ENZYME_conservation.csv",
    "consurf_mapping_enabled": True,
    "consurf_mapping_min_target_coverage": 0.90,
    "consurf_mapping_min_contact_coverage": 0.90,
    "consurf_mapping_fail_on_low_coverage": True,
    "ged_executable": "/path/to/Graph_Edit_Distance/ged",
    "ligand_resnames": "LIG",
    "conda_executable": "conda",
    "colabfold_conda_env": "",
    "workflow_conda_env": "",
    "workflow_profile": "standard",
    "global_plddt_min": 80.0,
    "plddt_filter_scope": "source_if_available",
    "af2_preprocess_manifest": "auto",
    "step02_alignment_mode": "auto",
    "step02_multi_ligand_mode": "sequential",
    # New Studio projects should consume Step01's passed table so the global
    # pLDDT decision actually controls the docking ensemble.  Existing YAMLs
    # retain their explicitly chosen af2_dir setting when they are loaded.
    "step02_input_source": "step01",
    "step02_output_naming": "short_pdb_rank",
    "step02_overwrite": True,
    "min_sequence_identity": 0.70,
    "min_alignment_coverage": 0.70,
    "min_matched_ca": 30,
    "clash_max": 10,
    "clash_delta": 0.5,
    "step03_include_resnames": "",
    "trim_enabled": False,
    "trim_timing": "before_docking",
    "trim_rules_yaml": "[]",
    "catalytic_distance_cutoff": 6.0,
    "catalytic_mean_plddt_min": 90.0,
    "catalytic_p10_plddt_min": 80.0,
    "plip_executable": "plip",
    "plip_run": True,
    "plip_save_intermediate": True,
    "plip_ligand_node_id": "residue_aware",
    "plip_connectivity_source": "auto",
    "consurf_grade_min": 8,
    "residue_ligand_distance_max": 3.0,
    "occurrence_frequency_min": 0.80,
    "require_graph_occurrence": True,
    "ged_threshold": 1.0,
    "ged_max_workers": 2,
    "ged_timeout_seconds": 300,
    "ged_require_same_edge_count": True,
    "common_frequency_min": 0.80,
    "rare_frequency_max": 0.10,
    "contact_unit_mode": "universal_rules",
    # Retained internally so historical YAML files remain readable. The public
    # GUI now describes the scientifically relevant contact-unit policy.
    "equivalence_mode": "standard",
    "manual_ligand_units_yaml": "{}",
    "recommendation_ligand_smiles_yaml": "{}",
    "accepted_recommendation_ids_yaml": "[]",
    "rejected_recommendation_ids_yaml": "[]",
    "recommendation_consensus_min_fraction": 0.80,
    "recommendation_require_acceptance": False,
    "interaction_type_map_yaml": "{}",
    "auto_detect_ligand_units": True,
    "split_nonidentical_classes": True,
    "class_signature_policy": "most_common",
    "export_candidates": True,
    "plot_formats": ["svg", "png"],
    "plot_width": 0.0,
    "plot_height": 0.0,
    "plot_dpi": 300,
    "plot_grid_cols": 3,
    "entity_order": "",
    "entity_spacing": 6.2,
    "entity_y_offsets_yaml": "[]",
    "substrate_scale": 2.75,
    "global_rotate": 0.0,
    "rotate_substrates_yaml": "{}",
    "swap_substrates": False,
    "residue_base_distance": 14.0,
    "residue_core_min_distance": 7.0,
    "residue_residue_min_distance": 8.6,
    "layout_zoom": 1.0,
    "manual_residue_offsets_yaml": "{}",
    "manual_node_offsets_yaml": "{}",
    "plot_node_size": 450,
    "plot_font_size": 6.5,
    "plot_ligand_node_scale": 0.30,
    "extra_yaml": "{}",
}


CONTACT_UNIT_GUIDED = "guided_review"
CONTACT_UNIT_UNIVERSAL = "universal_rules"
CONTACT_UNIT_CONSERVATIVE = "conservative_manual"


def normalize_contact_unit_mode(value: Any) -> str:
    """Return a supported ligand contact-unit policy."""
    raw = str(value or CONTACT_UNIT_UNIVERSAL).strip().lower()
    aliases = {
        "universal": CONTACT_UNIT_UNIVERSAL,
        "universal_rules": CONTACT_UNIT_UNIVERSAL,
        "s4": CONTACT_UNIT_UNIVERSAL,
        "s4_rules": CONTACT_UNIT_UNIVERSAL,
        "automatic": CONTACT_UNIT_UNIVERSAL,
        "auto_merge": CONTACT_UNIT_UNIVERSAL,
        "guided": CONTACT_UNIT_GUIDED,
        "recommendations": CONTACT_UNIT_GUIDED,
        "chemistry_guided": CONTACT_UNIT_GUIDED,
        "guided_review": CONTACT_UNIT_GUIDED,
        "manual": CONTACT_UNIT_CONSERVATIVE,
        "conservative": CONTACT_UNIT_CONSERVATIVE,
        "atom_level": CONTACT_UNIT_CONSERVATIVE,
        "conservative_manual": CONTACT_UNIT_CONSERVATIVE,
    }
    mode = aliases.get(raw, raw)
    if mode not in {CONTACT_UNIT_UNIVERSAL, CONTACT_UNIT_GUIDED, CONTACT_UNIT_CONSERVATIVE}:
        raise ValueError(
            "Ligand contact-unit mode must be 'universal_rules', "
            "'guided_review', or "
            f"'conservative_manual', got {value!r}."
        )
    return mode


def split_paths(value: str | list[str] | None) -> list[str]:
    """Split one-path-per-line or comma-separated user input without guessing paths."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[\n,]", str(value or "")) if part.strip()]


def split_resnames(value: str | list[str] | None) -> list[str]:
    return [item.upper() for item in split_paths(value)]


GLOBAL_RENAME_FIELDS = (
    "output_root", "af2_dir", "template_protein_pdb", "ligand_paths",
    "conservation_file", "af2_preprocess_manifest", "colabfold_fasta",
)


def global_rename_values(values: Mapping[str, Any], new_target_id: str) -> dict[str, Any]:
    """Update the project identifier and every target-derived GUI path together.

    Existing files are never renamed. The synchronized values define names and
    paths for the next saved configuration and subsequent workflow outputs.
    """
    new_target_id = str(new_target_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", new_target_id):
        raise ValueError("Target ID may contain only letters, numbers, '.', '_' and '-'.")
    old_target_id = str(values.get("target_id", "")).strip()
    updated = deepcopy(dict(values))
    updated["target_id"] = new_target_id
    if not old_target_id or old_target_id == new_target_id:
        return updated
    pattern = re.compile(re.escape(old_target_id), flags=re.IGNORECASE)
    for field in GLOBAL_RENAME_FIELDS:
        value = updated.get(field)
        if isinstance(value, str):
            updated[field] = pattern.sub(new_target_id, value)
        elif isinstance(value, list):
            updated[field] = [pattern.sub(new_target_id, str(item)) for item in value]
    return updated


def global_rename_config_path(
    current_path: str | Path,
    old_target_id: str,
    new_target_id: str,
    repository_root: Path,
) -> Path:
    """Synchronize the YAML path shown by the GUI after a project rename."""
    current_text = str(current_path).strip()
    old_target_id = str(old_target_id).strip()
    if current_text and old_target_id:
        renamed = re.sub(
            re.escape(old_target_id), str(new_target_id).strip(), current_text,
            flags=re.IGNORECASE,
        )
        if renamed != current_text:
            return Path(renamed)
    return default_config_path({"target_id": new_target_id}, repository_root)


def as_yaml_mapping(text: str, field_label: str) -> dict[str, Any]:
    """Parse an optional YAML mapping and give the GUI an actionable error."""
    try:
        value = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{field_label} is not valid YAML: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_label} must be a YAML mapping (key: value).")
    return value


def as_yaml_list(text: str, field_label: str) -> list[Any]:
    try:
        value = yaml.safe_load(text) if text.strip() else []
    except yaml.YAMLError as exc:
        raise ValueError(f"{field_label} is not valid YAML: {exc}") from exc
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_label} must be a YAML list, beginning with '-'.")
    return value


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge advanced overrides without dropping fields written by the form."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _unknown_fields(source: Mapping[str, Any], represented: Mapping[str, Any]) -> dict[str, Any]:
    """Keep config fields that the compact GUI does not expose.

    This lets a user load an expert-written configuration, adjust a visible
    field, and save it again without silently deleting unrelated advanced
    settings such as Step10 style overrides or project-specific audit options.
    """
    remainder: dict[str, Any] = {}
    for key, value in source.items():
        if str(key).startswith("_"):
            continue
        if key not in represented:
            remainder[key] = deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(represented[key], Mapping):
            nested = _unknown_fields(value, represented[key])
            if nested:
                remainder[key] = nested
    return remainder


def _float(values: Mapping[str, Any], name: str) -> float:
    try:
        return float(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _int(values: Mapping[str, Any], name: str) -> int:
    try:
        return int(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def build_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return an AF2-CatGraph configuration from validated GUI values."""
    target_id = str(values.get("target_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", target_id):
        raise ValueError("Target ID may contain only letters, numbers, '.', '_' and '-'.")
    ligand_paths = split_paths(values.get("ligand_paths"))
    if not ligand_paths:
        raise ValueError("At least one ligand/substrate/cofactor/metal PDB path is required.")
    ligand_resnames = split_resnames(values.get("ligand_resnames"))
    if not ligand_resnames:
        raise ValueError("At least one ligand residue name is required (for example LIG or ATP,MG).")

    trim_rules = as_yaml_list(
        str(values.get("trim_rules_yaml", "[]")), "Optional ligand-atom selection rules"
    )
    contact_unit_mode = normalize_contact_unit_mode(
        values.get("contact_unit_mode", CONTACT_UNIT_UNIVERSAL)
    )
    guided_contact_units = contact_unit_mode == CONTACT_UNIT_GUIDED
    automatic_contact_units = contact_unit_mode == CONTACT_UNIT_UNIVERSAL
    mapped_contact_units = automatic_contact_units or guided_contact_units
    raw_equivalence_mode = str(values.get("equivalence_mode", "standard")).strip().lower()
    equivalence_contract = (
        raw_equivalence_mode
        if raw_equivalence_mode in {"strict", "strict_chemistry", "chemistry_aware"}
        else "standard"
    )
    manual_units = as_yaml_mapping(
        str(values.get("manual_ligand_units_yaml", "{}")), "Step08 manual ligand units"
    )
    recommendation_smiles = as_yaml_mapping(
        str(values.get("recommendation_ligand_smiles_yaml", "{}")),
        "Step08 substrate SMILES annotations",
    )
    accepted_recommendations = as_yaml_list(
        str(values.get("accepted_recommendation_ids_yaml", "[]")),
        "Step08 accepted ligand-unit recommendation IDs",
    )
    rejected_recommendations = as_yaml_list(
        str(values.get("rejected_recommendation_ids_yaml", "[]")),
        "Step08 rejected ligand-unit recommendation IDs",
    )
    interaction_type_map = as_yaml_mapping(
        str(values.get("interaction_type_map_yaml", "{}")), "Step08 interaction-type map"
    )
    entity_y_offsets = as_yaml_list(str(values.get("entity_y_offsets_yaml", "[]")), "Step09 entity y offsets")
    rotate_substrates = as_yaml_mapping(
        str(values.get("rotate_substrates_yaml", "{}")), "Step09 substrate rotations"
    )
    manual_residue_offsets = as_yaml_mapping(
        str(values.get("manual_residue_offsets_yaml", "{}")), "Step09 residue offsets"
    )
    manual_node_offsets = as_yaml_mapping(
        str(values.get("manual_node_offsets_yaml", "{}")), "Step09 node offsets"
    )
    extra = as_yaml_mapping(str(values.get("extra_yaml", "{}")), "Advanced YAML override")

    inputs: dict[str, Any] = {
        "af2_dir": str(values.get("af2_dir", "")).strip(),
        "template_protein_pdb": str(values.get("template_protein_pdb", "")).strip(),
    }
    if len(ligand_paths) == 1:
        inputs["ligand_pdb"] = ligand_paths[0]
    else:
        inputs["ligand_pdbs"] = ligand_paths

    requested_connectivity = str(values.get("plip_connectivity_source", "auto")).strip()
    resolved_connectivity = (
        "input_conect_then_rdkit_same_residue"
        if requested_connectivity.lower() in {"", "auto", "default"}
        else requested_connectivity
    )

    config: dict[str, Any] = {
        "project": {
            "target_id": target_id,
            "output_root": str(values.get("output_root", "")).strip(),
        },
        "inputs": inputs,
        "step01": {
            "global_plddt_min": _float(values, "global_plddt_min"),
            "plddt_atom": "CA",
            "plddt_filter_scope": str(values.get("plddt_filter_scope", "source_if_available")),
            "af2_preprocess_manifest": str(values.get("af2_preprocess_manifest", "auto")).strip() or "auto",
        },
        "step02": {
            "input_source": str(values.get("step02_input_source", "step01")),
            "output_naming": str(values.get("step02_output_naming", "short_pdb_rank")),
            "overwrite": bool(values.get("step02_overwrite", True)),
            "multi_ligand_mode": str(values.get("step02_multi_ligand_mode", "sequential")),
            "alignment_mode": str(values.get("step02_alignment_mode", "auto")),
            "general_alignment": {
                "min_sequence_identity": _float(values, "min_sequence_identity"),
                "min_alignment_coverage": _float(values, "min_alignment_coverage"),
                "min_matched_ca": _int(values, "min_matched_ca"),
            },
        },
        "step03": {
            "ligand_selection": {
                "include_hetatm": True,
                "include_resnames": split_resnames(values.get("step03_include_resnames")) or None,
                "exclude_resnames": ["HOH", "WAT"],
            },
            "clash": {
                "method": "standard",
                "neighbor_search_cutoff": 5.0,
                "clash_max": _int(values, "clash_max"),
                "delta": _float(values, "clash_delta"),
            }
        },
        "step05": {
            "catalytic_region": {
                "ligand_resnames": ligand_resnames,
                "distance_cutoff": _float(values, "catalytic_distance_cutoff"),
                "mean_plddt_min": _float(values, "catalytic_mean_plddt_min"),
                "quantile10_plddt_min": _float(values, "catalytic_p10_plddt_min"),
            }
        },
        "step06": {
            "plip_executable": str(values.get("plip_executable", "plip")).strip(),
            "run_plip": bool(values.get("plip_run", True)),
            "clean_output": True,
            "save_intermediate_files": bool(values.get("plip_save_intermediate", True)),
            "ligand_selection": {
                "include_resnames": ligand_resnames,
                "exclude_resnames": ["HOH", "WAT"],
                "inventory_source": "input",
            },
            "node_identity": {
                "ligand_node_id": str(values.get("plip_ligand_node_id", "residue_aware")),
                "ligand_ged_label": "auto",
            },
            "connectivity": {
                "source": resolved_connectivity,
                "allow_cross_residue_rdkit_connectivity": False,
            },
        },
        "step07": {
            "conservation_file": str(values.get("conservation_file", "")).strip(),
            "conservation": {
                "mapping": {
                    "enabled": bool(values.get("consurf_mapping_enabled", True)),
                    "min_target_coverage": _float(values, "consurf_mapping_min_target_coverage"),
                    "min_contact_coverage": _float(values, "consurf_mapping_min_contact_coverage"),
                    "fail_on_low_target_coverage": bool(values.get("consurf_mapping_fail_on_low_coverage", True)),
                    "fail_on_low_contact_coverage": bool(values.get("consurf_mapping_fail_on_low_coverage", True)),
                }
            },
            "ligand": {"include_resnames": ligand_resnames, "exclude_resnames": ["HOH", "WAT"]},
            "distance": {
                "residue_ligand_distance_max": _float(values, "residue_ligand_distance_max"),
                "use_heavy_atoms_only": True,
                "protein_atom_scope": "all",
            },
            "thresholds": {
                "consurf_grade_min": _int(values, "consurf_grade_min"),
                "residue_ligand_distance_max": _float(values, "residue_ligand_distance_max"),
                "occurrence_frequency_min": _float(values, "occurrence_frequency_min"),
                "require_graph_occurrence": bool(values.get("require_graph_occurrence", True)),
            },
            "structure_filter": {
                "enabled": True,
                "require_key_residues": "all",
                "contact_definition": "graph",
            },
            "node_grouping": {
                "enabled": True,
                "contact_definition": "graph",
                "required_residue_source": "auto",
                "display_format": "one_letter_number",
                "type_order": "residue_count_desc",
                "residue_order": "sequence",
            },
        },
        "step08": {
            "use_step07_filter": True,
            "ged": {
                "executable": str(values.get("ged_executable", "")).strip(),
                "threshold": _float(values, "ged_threshold"),
                "max_workers": _int(values, "ged_max_workers"),
                "timeout_seconds": _int(values, "ged_timeout_seconds"),
                "require_same_edge_count": bool(values.get("ged_require_same_edge_count", True)),
            },
        },
        "step09": {
            "enabled": True,
            "split_nonidentical_classes": bool(values.get("split_nonidentical_classes", True)),
            "class_signature_policy": str(values.get("class_signature_policy", "most_common")),
            "thresholds": {
                "common_frequency_min": _float(values, "common_frequency_min"),
                "rare_frequency_max": _float(values, "rare_frequency_max"),
            },
            "canonicalization": {
                # Both public policies use the established class-merging
                # contract. They differ only in how ligand contact-unit rules
                # become available.
                "contact_unit_mode": contact_unit_mode,
                "equivalence_mode": equivalence_contract,
                "use_builtin_ligand_units": guided_contact_units,
                "manual_ligand_units": manual_units,
                "recommendations": {
                    "enabled": mapped_contact_units,
                    "require_acceptance": guided_contact_units,
                    "consensus_min_fraction": _float(values, "recommendation_consensus_min_fraction"),
                    "ligand_smiles": recommendation_smiles,
                    "accepted_recommendation_ids": [str(value) for value in accepted_recommendations],
                    "rejected_recommendation_ids": [str(value) for value in rejected_recommendations],
                },
                "interaction_type_map": interaction_type_map,
                "auto_detect_ligand_units": mapped_contact_units,
            },
            "export_candidates": {
                "enabled": bool(values.get("export_candidates", True)),
                "mode": "copy",
            },
        },
        "step10": {
            "plot": {
                "formats": list(values.get("plot_formats", ["svg", "png"])),
                "fig_width": _optional_positive_float(values.get("plot_width")),
                "fig_height": _optional_positive_float(values.get("plot_height")),
                "dpi": _int(values, "plot_dpi"),
                "grid_cols": _int(values, "plot_grid_cols"),
            },
            "layout": {
                # Blank is not an automatic size-based sort.  Persist the
                # declared analysis/input order so it survives either optional
                # ligand-trimming branch and every downstream writer.
                "entity_order": split_resnames(values.get("entity_order")) or ligand_resnames,
                "connectivity_source": "standard",
                "entity_spacing": _float(values, "entity_spacing"),
                "entity_y_offsets": entity_y_offsets,
                "substrate_scale": _float(values, "substrate_scale"),
                "rotate": _float(values, "global_rotate"),
                "rotate_substrates": rotate_substrates,
                "swap_substrates": bool(values.get("swap_substrates", False)),
                "residue_base_distance": _float(values, "residue_base_distance"),
                "residue_core_min_distance": _float(values, "residue_core_min_distance"),
                "residue_residue_min_distance": _float(values, "residue_residue_min_distance"),
                "layout_zoom": _float(values, "layout_zoom"),
                "manual_residue_offsets": manual_residue_offsets,
                "manual_node_offsets": manual_node_offsets,
            },
            "style": {
                "node_size": _int(values, "plot_node_size"),
                "font_size": _float(values, "plot_font_size"),
                "ligand_node_scale": _float(values, "plot_ligand_node_scale"),
            },
            "export_candidates": {
                "enabled": bool(values.get("export_candidates", True)),
                "mode": "copy",
            },
        },
        "step10_optional": {
            "enabled": bool(values.get("trim_enabled", False)),
            "timing": str(values.get("trim_timing", "before_docking")),
            "output_subdir": "10_optional_ligand_atom_trimming",
            "atom_selection": {
                "default_action": "keep",
                "drop_conect_records": False,
                "renumber_atoms": True,
                "fail_on_empty_keep_only": True,
                "exclude_resnames": ["HOH", "WAT"],
                "rules": trim_rules,
            },
        },
        "execution": {
            "conda_executable": str(values.get("conda_executable", "conda")).strip() or "conda",
            "colabfold_conda_env": str(values.get("colabfold_conda_env", "")).strip(),
            "workflow_conda_env": str(values.get("workflow_conda_env", "")).strip(),
        },
    }

    merged = deep_merge(config, extra)
    for step_name, public_default in {
        "step05": "04_catalytic_region_plddt_qc",
        "step06": "05_plip_interaction_graphs",
        "step07": "06_key_residue_screening",
        "step08": "07_ged_classification",
        "step09": "08_chemical_equivalence",
        "step10": "09_class_visualization",
    }.items():
        step_cfg = merged.get(step_name)
        if isinstance(step_cfg, dict) and step_cfg.get("output_subdir"):
            step_cfg["output_subdir"] = public_output_subdir(
                step_cfg.get("output_subdir"), public_default
            )
    return _runtime_to_public_config(merged)


def _runtime_to_public_config(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Write the same 01--09 numbering shown by the CLI and GUI."""
    runtime = deepcopy(dict(runtime))
    public: dict[str, Any] = {
        "config_schema": PUBLIC_CONFIG_SCHEMA,
        "project": runtime.get("project", {}),
        "inputs": runtime.get("inputs", {}),
        "step01": runtime.get("step01", {}),
        "step02": runtime.get("step02", {}),
        "step03": runtime.get("step03", {}),
        "step04": runtime.get("step05", {}),
        "step05": runtime.get("step06", {}),
        "step06": runtime.get("step07", {}),
        "step07": runtime.get("step08", {}),
        "step08": runtime.get("step09", {}),
        "step09": runtime.get("step10", {}),
        "step10_optional": runtime.get("step10_optional", {}),
        "execution": runtime.get("execution", {}),
    }

    # The standard scientific contract is implicit. Preserve only a genuinely
    # non-default expert profile when an older config is loaded and resaved.
    workflow = runtime.get("workflow", {}) or {}
    if str(workflow.get("profile", "")).strip().lower() in {"general", "generalized", "general_v1"}:
        public["workflow"] = {"profile": "general"}

    represented = {
        "project", "inputs", "step01", "step02", "step03", "step04", "step05",
        "step06", "step07", "step08", "step09", "step10", "step10_optional",
        "execution", "workflow", "config_schema",
    }
    for key, value in runtime.items():
        if key not in represented and not str(key).startswith("_"):
            public[key] = deepcopy(value)
    return public


def _optional_positive_float(value: Any) -> float | None:
    if value in (None, "", 0, 0.0):
        return None
    number = float(value)
    if number <= 0:
        raise ValueError("Figure width and height must be positive, or 0 for automatic sizing.")
    return number


def config_to_values(config_path: str | Path) -> dict[str, Any]:
    """Load the form fields that can be faithfully represented by the GUI."""
    cfg = load_yaml_config(config_path)
    values = deepcopy(DEFAULT_VALUES)
    project = cfg.get("project", {}) or {}
    inputs = cfg.get("inputs", {}) or {}
    step02 = cfg.get("step02", {}) or {}
    step03 = cfg.get("step03", {}) or {}
    step04 = cfg.get("step04", {}) or {}
    step10_optional = cfg.get("step10_optional", {}) or {}
    step05 = cfg.get("step05", {}) or {}
    step06 = cfg.get("step06", {}) or {}
    step07 = cfg.get("step07", {}) or {}
    step08 = cfg.get("step08", {}) or {}
    step09 = cfg.get("step09", {}) or {}
    step10 = cfg.get("step10", {}) or {}
    execution = cfg.get("execution", {}) or {}
    workflow = cfg.get("workflow", {}) or {}
    region = step05.get("catalytic_region", {}) or {}
    thresholds = step07.get("thresholds", {}) or {}
    ged = step08.get("ged", {}) or {}

    historical_trim = step04 if cfg.get("_config_layout") != PUBLIC_CONFIG_SCHEMA else {}
    values.update({
        "target_id": project.get("target_id", values["target_id"]),
        "output_root": project.get("output_root", values["output_root"]),
        "af2_dir": inputs.get("af2_dir", values["af2_dir"]),
        "template_protein_pdb": inputs.get("template_protein_pdb", values["template_protein_pdb"]),
        "ligand_paths": "\n".join(inputs.get("ligand_pdbs") or [inputs.get("ligand_pdb", "")]),
        "conservation_file": step07.get("conservation_file", values["conservation_file"]),
        "consurf_mapping_enabled": ((step07.get("conservation", {}) or {}).get("mapping", {}) or {}).get("enabled", values["consurf_mapping_enabled"]),
        "consurf_mapping_min_target_coverage": ((step07.get("conservation", {}) or {}).get("mapping", {}) or {}).get("min_target_coverage", values["consurf_mapping_min_target_coverage"]),
        "consurf_mapping_min_contact_coverage": ((step07.get("conservation", {}) or {}).get("mapping", {}) or {}).get("min_contact_coverage", values["consurf_mapping_min_contact_coverage"]),
        "consurf_mapping_fail_on_low_coverage": ((step07.get("conservation", {}) or {}).get("mapping", {}) or {}).get("fail_on_low_contact_coverage", values["consurf_mapping_fail_on_low_coverage"]),
        "ged_executable": ged.get("executable", values["ged_executable"]),
        "ligand_resnames": ", ".join(region.get("ligand_resnames", (step07.get("ligand", {}) or {}).get("include_resnames", ["LIG"])) or ["LIG"]),
        "conda_executable": execution.get("conda_executable", values["conda_executable"]),
        "colabfold_conda_env": execution.get("colabfold_conda_env", values["colabfold_conda_env"]),
        "workflow_conda_env": execution.get("workflow_conda_env", values["workflow_conda_env"]),
        "workflow_profile": (
            "standard"
            if str(workflow.get("profile", values["workflow_profile"])).strip().lower()
            in {
                "manuscript_legacy", "catcongraph_studio_v1", "standard",
                "legacy", "manuscript", "article", "article_reproduction",
                "manuscript_strict",
            }
            else workflow.get("profile", values["workflow_profile"])
        ),
        "global_plddt_min": (cfg.get("step01", {}) or {}).get("global_plddt_min", values["global_plddt_min"]),
        "plddt_filter_scope": (cfg.get("step01", {}) or {}).get("plddt_filter_scope", values["plddt_filter_scope"]),
        "af2_preprocess_manifest": (cfg.get("step01", {}) or {}).get("af2_preprocess_manifest", values["af2_preprocess_manifest"]),
        "step02_alignment_mode": (
            "coordinate_matching"
            if str(step02.get("alignment_mode", values["step02_alignment_mode"])).lower()
            in {"dock_py_exact", "legacy", "original", "coordinate_matching"}
            else step02.get("alignment_mode", values["step02_alignment_mode"])
        ),
        "step02_multi_ligand_mode": (
            "sequential"
            if str(step02.get("multi_ligand_mode", values["step02_multi_ligand_mode"])).lower()
            in {"sequential", "sequential_dock_py", "dock_py_sequential", "dockpy_sequential"}
            else step02.get("multi_ligand_mode", values["step02_multi_ligand_mode"])
        ),
        "step02_input_source": step02.get("input_source", values["step02_input_source"]),
        "step02_output_naming": step02.get("output_naming", values["step02_output_naming"]),
        "step02_overwrite": step02.get("overwrite", values["step02_overwrite"]),
        "min_sequence_identity": (step02.get("general_alignment", {}) or {}).get("min_sequence_identity", values["min_sequence_identity"]),
        "min_alignment_coverage": (step02.get("general_alignment", {}) or {}).get("min_alignment_coverage", values["min_alignment_coverage"]),
        "min_matched_ca": (step02.get("general_alignment", {}) or {}).get("min_matched_ca", values["min_matched_ca"]),
        "clash_max": (step03.get("clash", {}) or {}).get("clash_max", values["clash_max"]),
        "clash_delta": (step03.get("clash", {}) or {}).get("delta", values["clash_delta"]),
        "step03_include_resnames": ", ".join((step03.get("ligand_selection", {}) or {}).get("include_resnames") or []),
        "trim_enabled": step10_optional.get("enabled", historical_trim.get("enabled", False)),
        "trim_timing": step10_optional.get("timing", values["trim_timing"]),
        "trim_rules_yaml": yaml.safe_dump((step10_optional.get("atom_selection", historical_trim.get("atom_selection", {})) or {}).get("rules", []), sort_keys=False),
        "catalytic_distance_cutoff": region.get("distance_cutoff", values["catalytic_distance_cutoff"]),
        "catalytic_mean_plddt_min": region.get("mean_plddt_min", values["catalytic_mean_plddt_min"]),
        "catalytic_p10_plddt_min": region.get("quantile10_plddt_min", values["catalytic_p10_plddt_min"]),
        "plip_executable": step06.get("plip_executable", values["plip_executable"]),
        "plip_run": step06.get("run_plip", values["plip_run"]),
        "plip_save_intermediate": step06.get("save_intermediate_files", values["plip_save_intermediate"]),
        "plip_ligand_node_id": (step06.get("node_identity", {}) or {}).get("ligand_node_id", values["plip_ligand_node_id"]),
        "plip_connectivity_source": (
            "auto" if str((step06.get("connectivity", {}) or {}).get("source", "")).lower()
            in {"input_conect_then_rdkit_same_residue", "auto", "default", ""}
            else (step06.get("connectivity", {}) or {}).get("source", values["plip_connectivity_source"])
        ),
        "consurf_grade_min": thresholds.get("consurf_grade_min", values["consurf_grade_min"]),
        "residue_ligand_distance_max": thresholds.get("residue_ligand_distance_max", values["residue_ligand_distance_max"]),
        "occurrence_frequency_min": thresholds.get("occurrence_frequency_min", values["occurrence_frequency_min"]),
        "require_graph_occurrence": thresholds.get(
            "require_graph_occurrence", values["require_graph_occurrence"]
        ),
        "ged_threshold": ged.get("threshold", values["ged_threshold"]),
        "ged_max_workers": ged.get("max_workers", values["ged_max_workers"]),
        "ged_timeout_seconds": ged.get("timeout_seconds", values["ged_timeout_seconds"]),
        "ged_require_same_edge_count": ged.get("require_same_edge_count", values["ged_require_same_edge_count"]),
        "common_frequency_min": (step09.get("thresholds", {}) or {}).get("common_frequency_min", values["common_frequency_min"]),
        "rare_frequency_max": (step09.get("thresholds", {}) or {}).get("rare_frequency_max", values["rare_frequency_max"]),
        "contact_unit_mode": normalize_contact_unit_mode(
            (step09.get("canonicalization", {}) or {}).get(
                "contact_unit_mode",
                CONTACT_UNIT_GUIDED
                if (
                    ((step09.get("canonicalization", {}) or {}).get("recommendations", {}) or {}).get("enabled", True)
                    or (step09.get("canonicalization", {}) or {}).get("auto_detect_ligand_units", True)
                    or (step09.get("canonicalization", {}) or {}).get("use_builtin_ligand_units", True)
                )
                else CONTACT_UNIT_CONSERVATIVE,
            )
        ),
        "equivalence_mode": (
            "chemistry_aware"
            if str((step09.get("canonicalization", {}) or {}).get("equivalence_mode", values["equivalence_mode"])).lower()
            in {"strict", "strict_chemistry", "chemistry_aware"}
            else "standard"
        ),
        "manual_ligand_units_yaml": yaml.safe_dump((step09.get("canonicalization", {}) or {}).get("manual_ligand_units", {}), sort_keys=False),
        "recommendation_ligand_smiles_yaml": yaml.safe_dump(((step09.get("canonicalization", {}) or {}).get("recommendations", {}) or {}).get("ligand_smiles", {}), sort_keys=False),
        "accepted_recommendation_ids_yaml": yaml.safe_dump(((step09.get("canonicalization", {}) or {}).get("recommendations", {}) or {}).get("accepted_recommendation_ids", []), sort_keys=False),
        "rejected_recommendation_ids_yaml": yaml.safe_dump(((step09.get("canonicalization", {}) or {}).get("recommendations", {}) or {}).get("rejected_recommendation_ids", []), sort_keys=False),
        "recommendation_consensus_min_fraction": ((step09.get("canonicalization", {}) or {}).get("recommendations", {}) or {}).get("consensus_min_fraction", values["recommendation_consensus_min_fraction"]),
        "recommendation_require_acceptance": ((step09.get("canonicalization", {}) or {}).get("recommendations", {}) or {}).get("require_acceptance", values["recommendation_require_acceptance"]),
        "interaction_type_map_yaml": yaml.safe_dump((step09.get("canonicalization", {}) or {}).get("interaction_type_map", {}), sort_keys=False),
        "auto_detect_ligand_units": (step09.get("canonicalization", {}) or {}).get("auto_detect_ligand_units", values["auto_detect_ligand_units"]),
        "split_nonidentical_classes": step09.get("split_nonidentical_classes", values["split_nonidentical_classes"]),
        "class_signature_policy": (
            "most_common"
            if str(step09.get("class_signature_policy", values["class_signature_policy"])).lower()
            in {"representative", "most_common"}
            else step09.get("class_signature_policy", values["class_signature_policy"])
        ),
        "export_candidates": (step09.get("export_candidates", {}) or {}).get("enabled", values["export_candidates"]),
        "plot_formats": (step10.get("plot", {}) or {}).get("formats", values["plot_formats"]),
        "plot_width": (step10.get("plot", {}) or {}).get("fig_width", 0.0) or 0.0,
        "plot_height": (step10.get("plot", {}) or {}).get("fig_height", 0.0) or 0.0,
        "plot_dpi": (step10.get("plot", {}) or {}).get("dpi", values["plot_dpi"]),
        "plot_grid_cols": (step10.get("plot", {}) or {}).get("grid_cols", values["plot_grid_cols"]),
        "entity_order": ", ".join((step10.get("layout", {}) or {}).get("entity_order", []) or []),
        "entity_spacing": (step10.get("layout", {}) or {}).get("entity_spacing", values["entity_spacing"]),
        "entity_y_offsets_yaml": yaml.safe_dump((step10.get("layout", {}) or {}).get("entity_y_offsets", []), sort_keys=False),
        "substrate_scale": (step10.get("layout", {}) or {}).get("substrate_scale", values["substrate_scale"]),
        "global_rotate": (step10.get("layout", {}) or {}).get("rotate", values["global_rotate"]),
        "rotate_substrates_yaml": yaml.safe_dump((step10.get("layout", {}) or {}).get("rotate_substrates", {}), sort_keys=False),
        "swap_substrates": (step10.get("layout", {}) or {}).get("swap_substrates", values["swap_substrates"]),
        "residue_base_distance": (step10.get("layout", {}) or {}).get("residue_base_distance", values["residue_base_distance"]),
        "residue_core_min_distance": (step10.get("layout", {}) or {}).get("residue_core_min_distance", values["residue_core_min_distance"]),
        "residue_residue_min_distance": (step10.get("layout", {}) or {}).get("residue_residue_min_distance", values["residue_residue_min_distance"]),
        "layout_zoom": (step10.get("layout", {}) or {}).get("layout_zoom", values["layout_zoom"]),
        "manual_residue_offsets_yaml": yaml.safe_dump((step10.get("layout", {}) or {}).get("manual_residue_offsets", {}), sort_keys=False),
        "manual_node_offsets_yaml": yaml.safe_dump((step10.get("layout", {}) or {}).get("manual_node_offsets", {}), sort_keys=False),
        "plot_node_size": (step10.get("style", {}) or {}).get("node_size", values["plot_node_size"]),
        "plot_font_size": (step10.get("style", {}) or {}).get("font_size", values["plot_font_size"]),
        "plot_ligand_node_scale": (step10.get("style", {}) or {}).get("ligand_node_scale", values["plot_ligand_node_scale"]),
    })
    # Preserve fields outside the compact form as an explicit advanced overlay.
    # Known fields are intentionally not placed in the overlay, so an edited
    # widget remains authoritative when the config is saved again.
    represented = normalize_project_config(build_config({**values, "extra_yaml": "{}"}))
    values["extra_yaml"] = yaml.safe_dump(_unknown_fields(cfg, represented), sort_keys=False, allow_unicode=True)
    return values


def default_config_path(values: Mapping[str, Any], repository_root: Path) -> Path:
    target_id = str(values.get("target_id", "MY_ENZYME")).strip() or "MY_ENZYME"
    return repository_root / "projects" / target_id / "config.gui.yaml"


def write_config(path: str | Path, config: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Generated by AF2-CatGraph GUI. This is a standard AF2-CatGraph config.\n"
        "# Keep this file with your results to make the run reproducible.\n\n"
        + yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output

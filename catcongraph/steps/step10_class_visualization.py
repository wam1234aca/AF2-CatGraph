#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class visualization after ligand contact-unit grouping.

This module follows the established visualization scripts
(`visualize_classes.py` / `visualize_classes_layout_fan_fix_v5.py`) instead of
the simplified canonical-layout prototype.

The contact-unit grouping output is used only to select merged classes and representatives.
    - The actual drawing uses the original Step06 atom-level
      `interaction_output.txt` and `substrate_connectivity.txt` whenever available.
    - This restores the article-like substrate geometry and residue fan layout.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from catcongraph.visualization.legacy_class_visualizer import (
    EnzymeSubstrateVisualizer,
    INTERACTION_COLORS,
    ATOM_COLORS,
)
from catcongraph.config import load_yaml_config, public_output_subdir


# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------

def _mini_yaml_scalar(value: str):
    text = value.strip()
    if text in {"", "null", "None", "~"}:
        return None
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_mini_yaml_scalar(x.strip()) for x in inner.split(",")]
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", text, flags=re.I):
            return float(text)
    except Exception:
        pass
    return text


def _mini_yaml_load(text: str) -> dict:
    root: dict = {}
    stack: List[Tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _mini_yaml_scalar(value)
    return root


def load_config_dict(config_path: Path) -> dict:
    return load_yaml_config(config_path)


def get_project_root(config_path: Path) -> Path:
    if config_path.name in {"config.yaml", "config.yml"} and config_path.parent.parent.name == "projects":
        return config_path.parent.parent.parent.resolve()
    return Path.cwd().resolve()


def get_target_id(config: dict, config_path: Path) -> str:
    project_cfg = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
    for key in ["target_id", "target", "project_id"]:
        if config.get(key):
            return str(config[key])
    for key in ["target_id", "id", "name"]:
        if project_cfg.get(key):
            return str(project_cfg[key])
    return config_path.parent.name


def get_output_root(config: dict, root: Path, target_id: str) -> Path:
    project_cfg = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
    for value in [config.get("output_root"), project_cfg.get("output_root")]:
        if value:
            p = Path(str(value))
            return p.resolve() if p.is_absolute() else (root / p).resolve()
    return (root / "results" / target_id).resolve()


def resolve_path(value, root: Path) -> Optional[Path]:
    if value is None or str(value).lower() == "auto":
        return None
    p = Path(str(value))
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv_table(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        if not fieldnames:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def natural_key(text: str):
    parts = re.split(r"(\d+)", str(text))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def sequential_merged_class_sort_key(value: object):
    """Sort definitive Step09 labels as class_1, class_2, ..., class_N."""
    text = str(value or "").strip()
    match = re.fullmatch(r"class_(\d+)", text, flags=re.I)
    if match:
        return (0, int(match.group(1)), text.lower())
    return (1, natural_key(text), text.lower())


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def split_ids(value: str) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    return [x for x in re.split(r"[;,]\s*|\s+", text) if x]


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def find_step09_dir(output_root: Path, cfg: dict, root: Path, step09_cfg: Optional[dict] = None) -> Path:
    explicit = resolve_path(cfg.get("input_step09_dir"), root)
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"Configured step10.input_step09_dir does not exist: {explicit}")
        return explicit
    step09_cfg = step09_cfg or {}
    configured_subdir = public_output_subdir(
        step09_cfg.get("output_subdir"), "08_chemical_equivalence"
    )
    candidates = [
        output_root / configured_subdir,
        output_root / "08_chemical_equivalence",
        output_root / "09_chemical_equivalence",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in sorted(output_root.glob("*chemical_equivalence*")):
        if p.is_dir():
            return p

    # Step09's report stores the exact output directory and table paths. This
    # supports custom output_subdir values and older directory names.
    if output_root.exists():
        report_paths = list(output_root.rglob("*_step08_report.json"))
        report_paths += list(output_root.rglob("*_step09_report.json"))
        for report_path in sorted(report_paths):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            reported_dir = report.get("output_dir")
            if reported_dir:
                p = Path(str(reported_dir)).expanduser()
                if not p.is_absolute():
                    p = (root / p).resolve()
                if p.exists():
                    return p

        # A valid merged summary is sufficient to infer the Step09 root.
        tables = list(output_root.rglob("*_step08_merged_class_summary.csv"))
        tables += list(output_root.rglob("*_step09_merged_class_summary.csv"))
        for table in sorted(tables):
            if table.parent.name == "tables":
                return table.parent.parent
            return table.parent
    raise FileNotFoundError(
        "Cannot find Step09 output directory. Step10 depends on Step09. "
        "Run `python scripts/09_chemical_equivalence.py --config <config> --clean` first, "
        "or set step10.input_step09_dir."
    )


def find_step06_dir(output_root: Path, cfg: dict, root: Path) -> Optional[Path]:
    explicit = resolve_path(cfg.get("input_step06_dir"), root)
    if explicit:
        return explicit
    for p in [output_root / "05_plip_interaction_graphs", output_root / "06_plip_interaction_graphs"]:
        if p.exists():
            return p
    matches = sorted(output_root.glob("*plip_interaction_graphs*"))
    return matches[0] if matches else None


def find_step08_dir(output_root: Path, cfg: dict, root: Path, step08_cfg: Optional[dict] = None) -> Optional[Path]:
    explicit = resolve_path(cfg.get("input_step08_dir"), root)
    if explicit:
        return explicit
    step08_cfg = step08_cfg or {}
    configured_subdir = public_output_subdir(
        step08_cfg.get("output_subdir"), "07_ged_classification"
    )
    for p in [output_root / configured_subdir, output_root / "07_ged_classification", output_root / "08_ged_classification"]:
        if p.exists():
            return p
    if output_root.exists():
        report_paths = list(output_root.rglob("*_step07_report.json"))
        report_paths += list(output_root.rglob("*_step08_report.json"))
        for report_path in sorted(report_paths):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            reported_dir = report.get("output_dir")
            if reported_dir:
                p = Path(str(reported_dir)).expanduser()
                if not p.is_absolute():
                    p = (root / p).resolve()
                if p.exists():
                    return p
    return None


def find_table(step09_dir: Path, target_id: str, name: str) -> Path:
    candidates = [
        step09_dir / "tables" / f"{target_id}_step08_{name}.csv",
        step09_dir / f"{target_id}_step08_{name}.csv",
        step09_dir / "tables" / f"{target_id}_step09_{name}.csv",
        step09_dir / f"{target_id}_step09_{name}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    report_paths = list(step09_dir.rglob("*_step08_report.json"))
    report_paths += list(step09_dir.rglob("*_step09_report.json"))
    for report_path in sorted(report_paths):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        value = (report.get("tables", {}) or {}).get(name)
        if value:
            p = Path(str(value)).expanduser()
            if not p.is_absolute():
                p = (step09_dir / p).resolve()
            if p.exists():
                return p
    matches = sorted(step09_dir.rglob(f"*step08*{name}*.csv"))
    matches += sorted(step09_dir.rglob(f"*step09*{name}*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find Step09 table '{name}' in {step09_dir}")


def find_plip_folder(step06_dir: Optional[Path], sid: str) -> Optional[Path]:
    if not step06_dir or not step06_dir.exists():
        return None
    candidates: List[Path] = []
    search_roots = [
        step06_dir / "plip_outputs",
        step06_dir / "plip_output",
        step06_dir,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            if sid not in p.name and sid not in str(p):
                continue
            if (p / "interaction_output.txt").exists() and (p / "substrate_connectivity.txt").exists():
                candidates.append(p)
    if not candidates:
        return None
    # Prefer paths under plip_outputs and shortest names.
    return sorted(set(candidates), key=lambda p: ("plip_outputs" not in str(p), len(str(p)), str(p)))[0]


def find_graph_file(step06_dir: Optional[Path], step08_dir: Optional[Path], sid: str) -> Optional[Path]:
    roots = []
    if step08_dir:
        roots.extend([step08_dir / "work" / "filtered_graphs", step08_dir])
    if step06_dir:
        roots.extend([step06_dir / "ged_input_files", step06_dir])
    candidates: List[Path] = []
    for root in roots:
        if not root or not root.exists():
            continue
        candidates.extend(root.rglob(f"*{sid}*final.txt"))
        candidates.extend(root.rglob(f"*{sid}*.txt"))
    candidates = [p for p in candidates if p.is_file()]
    return sorted(set(candidates), key=lambda p: (0 if "final" in p.name else 1, len(str(p)), str(p)))[0] if candidates else None


# ---------------------------------------------------------------------------
# Fallback folder reconstruction
# ---------------------------------------------------------------------------

def short_ligand_label(label: str) -> str:
    s = str(label or "").strip()
    m = re.fullmatch(r"[A-Za-z0-9]+_[A-Za-z]\d+_([A-Za-z0-9]+)", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"[A-Za-z0-9]+:[A-Za-z]:\d+:([A-Za-z0-9]+)", s)
    if m:
        return m.group(1)
    return s


def is_protein_like(label: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}\d+[A-Za-z]?", str(label)))


def parse_ged_substrate_connectivity(graph_file: Path) -> Dict[str, Set[str]]:
    nodes: Dict[int, str] = {}
    edges: List[Tuple[int, int]] = []
    with graph_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    try:
                        nodes[int(parts[1])] = parts[2]
                    except Exception:
                        pass
            elif line.startswith("e "):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        edges.append((int(parts[1]), int(parts[2])))
                    except Exception:
                        pass
    # Ligand nodes are those that do not look like protein residue labels and are not pure integers.
    lig_ids = {i for i, lab in nodes.items() if not is_protein_like(lab) and not str(lab).isdigit()}
    conn: Dict[str, Set[str]] = {}
    for i in lig_ids:
        conn.setdefault(short_ligand_label(nodes[i]), set())
    for a, b in edges:
        if a in lig_ids and b in lig_ids:
            la, lb = short_ligand_label(nodes[a]), short_ligand_label(nodes[b])
            if la != lb:
                conn.setdefault(la, set()).add(lb)
                conn.setdefault(lb, set()).add(la)
    return conn


def create_fallback_plot_folder(
    work_dir: Path,
    class_name: str,
    sid: str,
    raw_rows: List[dict],
    graph_file: Optional[Path],
) -> Path:
    folder = work_dir / "plot_inputs" / sanitize_filename(class_name)
    folder.mkdir(parents=True, exist_ok=True)
    # Representative marker file, kept for traceability and old-style workflows.
    (folder / f"{sid}_final.txt").write_text(f"t # {sid}\n", encoding="utf-8")

    # interaction_output.txt: preserve original PLIP type/ligand/protein.
    with (folder / "interaction_output.txt").open("w", encoding="utf-8") as handle:
        for r in raw_rows:
            raw_type = r.get("raw_type", "")
            ligand = short_ligand_label(r.get("ligand_atom", ""))
            protein = r.get("protein_raw") or r.get("protein_canonical") or ""
            if raw_type and ligand and protein:
                handle.write(f"Type: {raw_type}\tLigand: {ligand}\tProtein: {protein}\n")

    # substrate_connectivity.txt: prefer GED graph ligand-ligand edges.
    conn: Dict[str, Set[str]] = {}
    if graph_file and graph_file.exists():
        try:
            conn = parse_ged_substrate_connectivity(graph_file)
        except Exception:
            conn = {}
    if not conn:
        for r in raw_rows:
            ligand = short_ligand_label(r.get("ligand_atom", ""))
            if ligand:
                conn.setdefault(ligand, set())
    with (folder / "substrate_connectivity.txt").open("w", encoding="utf-8") as handle:
        for atom in sorted(conn, key=natural_key):
            neighbors = ",".join(sorted(conn[atom], key=natural_key))
            handle.write(f"{atom} -> {neighbors}\n")
    return folder


# ---------------------------------------------------------------------------
# Plotting wrappers using the legacy visualizer
# ---------------------------------------------------------------------------

def _as_offset_map(value) -> dict:
    """Normalize optional visualization-only manual offsets.

    The keys may be internal node IDs (e.g. PHE182A), manuscript-style residue
    labels (F182), or atom labels (O4). Values are two-number [dx, dy] lists in
    the pre-autofit layout coordinate system. These offsets only affect Step10
    figures; no classification tables are changed.
    """
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, val in value.items():
        if isinstance(val, str):
            parts = [x for x in re.split(r"[,;\s]+", val.strip()) if x]
        elif isinstance(val, (list, tuple)):
            parts = list(val)
        else:
            continue
        if len(parts) < 2:
            continue
        try:
            out[str(key).strip()] = (float(parts[0]), float(parts[1]))
        except Exception:
            continue
    return out


def make_legacy_args(cfg: dict, full_config: Optional[dict] = None):
    plot = cfg.get("plot", {}) or {}
    style = cfg.get("style", {}) or {}
    # Keep backward compatibility: both legacy_layout and layout are accepted.
    layout = {}
    layout.update(cfg.get("legacy_layout", {}) or {})
    layout.update(cfg.get("layout", {}) or {})

    can = ((full_config or {}).get("step09", {}) or {}).get("canonicalization", {}) or {}
    chemistry_mode = str(can.get("equivalence_mode", "standard") or "standard").strip().lower()
    default_connectivity_source = (
        "standard" if chemistry_mode in {"standard", "catcongraph_studio_v1", "v8", "v8_exact", "legacy", "compat"}
        else "virtual_preferred"
    )

    # The entity arrangement is provenance: it must follow the user-declared
    # template-ligand order even if a PDB writer later reorders atoms.  An
    # explicit Step10 order wins; otherwise inherit the analysis order already
    # used by Step05/06/07.
    entity_order = layout.get("entity_order", [])
    if not entity_order and isinstance(full_config, dict):
        entity_order = (
            ((full_config.get("step05", {}) or {}).get("catalytic_region", {}) or {}).get("ligand_resnames")
            or ((full_config.get("step06", {}) or {}).get("ligand_selection", {}) or {}).get("include_resnames")
            or ((full_config.get("step07", {}) or {}).get("ligand", {}) or {}).get("include_resnames")
            or []
        )
    if isinstance(entity_order, str):
        entity_order = [item.strip().upper() for item in re.split(r"[,;\\s]+", entity_order) if item.strip()]

    rotate_substrates = layout.get("rotate_substrates", {}) or {}
    if isinstance(rotate_substrates, dict):
        rotate_substrates = {
            int(k): float(v)
            for k, v in rotate_substrates.items()
            if str(k).strip().lstrip("-").isdigit()
        }

    def positive_float(value, default):
        """Accept GUI ``null``/0 as automatic figure sizing.

        The graphical form deliberately writes ``null`` when a user chooses
        automatic figure dimensions.  ``float(None)`` previously made Step10
        fail before any plot was created.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        return number if number > 0 else float(default)

    return argparse.Namespace(
        # Figure size. Prefer the names used in visualize_classes.py.
        fig_width=positive_float(plot.get("fig_width"), plot.get("panel_width") or 65.643 / 25.4),
        fig_height=positive_float(plot.get("fig_height"), plot.get("panel_height") or 49.488 / 25.4),
        dpi=int(plot.get("dpi", 300)),

        # Node / label styling.
        node_size=int(style.get("node_size", 450)),
        ligand_node_scale=float(style.get("ligand_node_scale", 0.30)),
        residue_node_scale=float(style.get("residue_node_scale", 1.0)),
        ion_node_scale=float(style.get("ion_node_scale", 1.0)),
        ligand_node_alpha=float(style.get("ligand_node_alpha", 0.82)),
        protein_node_alpha=float(style.get("protein_node_alpha", 0.62)),
        ion_node_alpha=float(style.get("ion_node_alpha", 0.85)),
        ligand_node_color=style.get("ligand_node_color", "skyblue"),
        protein_node_color=style.get("protein_node_color", "#E0E0E0"),
        node_border_color=style.get("node_border_color", "black"),
        show_atom_labels=as_bool(style.get("show_atom_labels", True)),
        atom_label_color=style.get("atom_label_color", "white"),
        atom_font_delta=float(style.get("atom_font_delta", 2.0)),

        # Edge styling.
        covalent_edge_width=float(style.get("covalent_edge_width", 0.8)),
        interaction_edge_width=float(style.get("interaction_edge_width", 1.0)),
        edge_alpha=float(style.get("edge_alpha", 0.82)),
        hbond_edge_color=style.get("hbond_edge_color", INTERACTION_COLORS.get("hydrogen_bonds", "#1f77b4")),
        saltbridge_edge_color=style.get("saltbridge_edge_color", INTERACTION_COLORS.get("salt_bridges", "#DC143C")),
        hydrophobic_edge_color=style.get("hydrophobic_edge_color", INTERACTION_COLORS.get("hydrophobic_interactions", "#2E8B57")),
        pistack_edge_color=style.get("pistack_edge_color", INTERACTION_COLORS.get("pi_stacks", "#DAA520")),
        pication_edge_color=style.get("pication_edge_color", INTERACTION_COLORS.get("pi_cation_interactions", "#FF8C00")),
        metal_edge_color=style.get("metal_edge_color", INTERACTION_COLORS.get("metal_complexes", "#6A5ACD")),
        halogen_edge_color=style.get("halogen_edge_color", INTERACTION_COLORS.get("halogen_bonds", "#A0522D")),

        # Font styling.
        font_size=float(style.get("font_size", 6.5)),
        font_color=style.get("font_color", "black"),
        font_weight=style.get("font_weight", "bold"),
        class_font_size=float(style.get("class_font_size", max(float(style.get("font_size", 6.5)) + 1.0, 7.5))),
        small_font_size=float(style.get("small_font_size", max(float(style.get("font_size", 6.5)), 6.5))),
        show_class_size=as_bool(style.get("show_class_size", True)),
        show_representative_title=as_bool(style.get("show_representative_title", False)),

        # Layout parameters copied from / aligned with visualize_classes.py, now configurable.
        swap_substrates=as_bool(layout.get("swap_substrates", False)),
        rotate_substrates=rotate_substrates,
        rotate=float(layout.get("rotate", 0.0)),
        entity_order=entity_order,
        # The compatibility default is v8's structured-CSV-first reader.
        # New virtual-topology behavior remains an opt-in layout setting.
        connectivity_source=str(layout.get("connectivity_source", default_connectivity_source)),
        merge_text_connectivity_with_csv=as_bool(layout.get("merge_text_connectivity_with_csv", False)),
        substrate_scale=float(layout.get("substrate_scale", 2.75)),
        substrate_fallback_scale=float(layout.get("substrate_fallback_scale", 4.0)),
        entity_spacing=float(layout.get("entity_spacing", 5.0)),
        entity_y_offsets=layout.get("entity_y_offsets", []),
        entity_bbox_padding=float(layout.get("entity_bbox_padding", 1.2)),
        entity_repel_iterations=int(layout.get("entity_repel_iterations", 400)),
        entity_repel_step=float(layout.get("entity_repel_step", 0.1)),
        ion_spacing=float(layout.get("ion_spacing", 5.0)),
        ion_y_offset=float(layout.get("ion_y_offset", 0.0)),

        residue_base_distance=float(layout.get("residue_base_distance", layout.get("residue_distance", 14.0))),
        residue_angle_bin_deg=float(layout.get("residue_angle_bin_deg", 15.0)),
        residue_fan_base_deg=float(layout.get("residue_fan_base_deg", 10.0)),
        residue_fan_step_deg=float(layout.get("residue_fan_step_deg", 6.0)),
        residue_fan_max_deg=float(layout.get("residue_fan_max_deg", 55.0)),
        residue_layer_step=float(layout.get("residue_layer_step", 1.6)),
        residue_residue_min_distance=float(layout.get("residue_residue_min_distance", 8.6)),
        residue_core_min_distance=float(layout.get("residue_core_min_distance", 7.0)),
        residue_repel_iterations=int(layout.get("residue_repel_iterations", 450)),
        residue_repel_step=float(layout.get("residue_repel_step", 0.25)),
        residue_core_repel_step=float(layout.get("residue_core_repel_step", 0.3)),
        autofit_fill=float(layout.get("autofit_fill", 0.92)),
        layout_zoom=float(layout.get("layout_zoom", 1.0)),
        manual_node_offsets=_as_offset_map(layout.get("manual_node_offsets", {})),
        manual_residue_offsets=_as_offset_map(layout.get("manual_residue_offsets", {})),
    )


def _residue_one_letter_label(vis: EnzymeSubstrateVisualizer, node: str) -> str:
    return vis._clean_residue_label(str(node))


def _apply_manual_layout_offsets(vis: EnzymeSubstrateVisualizer, G, pos: dict, args) -> dict:
    """Apply optional user-defined visual offsets before autofit.

    This is deliberately limited to Step10 layout polishing. It does not change
    interaction graphs, GED classes, candidate selection, or any CSV table.
    """
    node_offsets = getattr(args, "manual_node_offsets", {}) or {}
    residue_offsets = getattr(args, "manual_residue_offsets", {}) or {}
    if not node_offsets and not residue_offsets:
        return pos

    adjusted = {k: v.copy() if hasattr(v, "copy") else v for k, v in pos.items()}
    for node, data in G.nodes(data=True):
        if node not in adjusted:
            continue
        candidates = [str(node)]
        display_label = str(data.get("display_label", "") or "")
        if display_label:
            candidates.append(display_label)
        if data.get("type") == "residue":
            candidates.append(_residue_one_letter_label(vis, str(node)))
            offsets = residue_offsets
        else:
            offsets = node_offsets

        dxdy = None
        for key in candidates:
            if key in offsets:
                dxdy = offsets[key]
                break
        if dxdy is None:
            continue
        dx, dy = dxdy
        adjusted[node] = adjusted[node] + np.array([dx, dy], dtype=float)
    return adjusted


def _apply_layout_zoom(final_pos: dict, zoom: float) -> dict:
    if not final_pos or abs(float(zoom) - 1.0) < 1e-9:
        return final_pos
    center = np.mean([np.array(p, dtype=float) for p in final_pos.values()], axis=0)
    return {node: tuple(((np.array(p, dtype=float) - center) * float(zoom)) + center) for node, p in final_pos.items()}


def draw_legacy_panel(
    vis: EnzymeSubstrateVisualizer,
    ax,
    folder: Path,
    class_label: str,
    class_size: int,
    is_candidate: bool,
    args,
    use_template: bool,
    candidate_color: str = "#ff9900",
    normal_color: str = "#BDBDBD",
):
    # Pass all tunable layout/style parameters into the legacy engine.
    vis.runtime_args = args
    G = vis.build_interaction_graph(str(folder))
    if not G.nodes:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{class_label}\n(no graph)", ha="center", va="center", transform=ax.transAxes)
        return False

    frame_width = 20.0
    frame_height = 20.0 * (float(args.fig_height) / float(args.fig_width))
    if use_template and vis.layout_template is not None:
        pos = vis._apply_template_layout(G, getattr(args, "rotate", 0.0))
    else:
        pos = vis._main_layout_engine(G, getattr(args, "swap_substrates", False), getattr(args, "rotate_substrates", {}))

    # Keep the reusable layout template unmodified, then apply optional
    # visualization-only nudges consistently to every panel.
    adjusted_pos = _apply_manual_layout_offsets(vis, G, pos, args)
    final_pos = vis._get_autofit_layout(
        G,
        adjusted_pos,
        (float(args.fig_width), float(args.fig_height)),
        (frame_width, frame_height),
        getattr(args, "rotate", 0.0),
        use_template=(use_template and vis.layout_template is not None),
        args=args,
    )
    final_pos = _apply_layout_zoom(final_pos, getattr(args, "layout_zoom", 1.0))

    if vis.layout_template is None:
        # Save the first valid class as the layout template, exactly like the legacy workflow.
        vis.layout_template = pos.copy()
        vis.template_residue_map = {
            vis._get_residue_id_num(n): n
            for n, d in G.nodes(data=True)
            if d.get("type") == "residue" and vis._get_residue_id_num(n) is not None
        }

    ax.set_xlim(-frame_width / 2, frame_width / 2)
    ax.set_ylim(-frame_height / 2, frame_height / 2)
    vis._draw_graph_to_ax(ax, G, final_pos, args)

    # Update legend inventory.
    drawn_interaction_types = {d.get("type") for _, _, d in G.edges(data=True) if d.get("type") != "bond"}
    elements = {d.get("element") for _, d in G.nodes(data=True) if d.get("type") in {"atom", "ion"} and d.get("element") != "H"}
    vis.global_interaction_types.update(x for x in drawn_interaction_types if x)
    vis.global_atom_elements.update(x for x in elements if x)

    # Article-style frame and class label.
    ax.set_aspect("equal", anchor="C")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = Rectangle(
        (0.01, 0.01), 0.98, 0.98,
        transform=ax.transAxes,
        fill=False,
        edgecolor=candidate_color if is_candidate else normal_color,
        linewidth=1.2 if is_candidate else 0.65,
        linestyle="-" if is_candidate else (0, (5, 5)),
        alpha=1.0 if is_candidate else 0.85,
        zorder=20,
    )
    ax.add_patch(rect)
    ax.text(0.045, 0.085, class_label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=float(getattr(args, "class_font_size", max(float(args.font_size) + 1.0, 7.5))),
            fontweight="bold", color="black")
    if getattr(args, "show_class_size", True):
        ax.text(0.05, 0.91, f"n={class_size}", transform=ax.transAxes, ha="left", va="top",
                fontsize=float(getattr(args, "small_font_size", max(float(args.font_size), 6.5))),
                color="#444444")
    if getattr(args, "show_representative_title", False):
        ax.text(0.50, 0.965, folder.name, transform=ax.transAxes, ha="center", va="top",
                fontsize=float(getattr(args, "small_font_size", max(float(args.font_size), 6.5))),
                color="black", fontweight="bold")
    if is_candidate:
        ax.text(0.96, 0.91, "candidate", transform=ax.transAxes, ha="right", va="top",
                fontsize=float(getattr(args, "small_font_size", max(float(args.font_size), 6.5))),
                color=candidate_color, fontweight="bold")
    return True


def add_summary_legend(fig, vis: EnzymeSubstrateVisualizer, args, bottom: float = 0.015):
    label_map = {
        "hydrophobic_interactions": "Hydrophobic",
        "hydrogen_bonds": "H-bond",
        "pi_stacks": "Pi-stacking",
        "salt_bridges": "Salt bridge",
        "pi_cation_interactions": "Pi-cation",
        "metal_complexes": "Metal complex",
        "halogen_bonds": "Halogen bond",
    }
    handles = []
    for itype in sorted(vis.global_interaction_types):
        linestyle = "-" if itype == "hydrophobic_interactions" else "--"
        handles.append(Line2D([0], [0], color=INTERACTION_COLORS.get(itype, "gray"),
                              lw=args.interaction_edge_width, linestyle=linestyle,
                              label=label_map.get(itype, itype)))
    for elem in sorted(vis.global_atom_elements):
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=ATOM_COLORS.get(elem.upper(), "black"),
                              linestyle="None", label=elem.upper(), markersize=5.5))
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, bottom),
                   ncol=3 if len(handles) > 4 else len(handles), fontsize=7.0,
                   frameon=True, framealpha=0.92, facecolor="#F0F0F0", edgecolor="gray")


def inspect_topology_integrity(folder: Path, args) -> dict:
    """Audit the exact Step06 topology that will be passed to the plotter.

    A figure can look plausible even when every PLIP atom was introduced as an
    isolated node.  This check therefore works on the same legacy visualizer
    graph as the drawing code and reports disconnected covalent components
    before a user interprets the layout as molecular connectivity.
    """
    vis = EnzymeSubstrateVisualizer()
    vis.runtime_args = args
    graph = vis.build_interaction_graph(str(folder))
    ligand_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("type") in {"atom", "ion"}
    ]
    bond_edges = [
        (left, right) for left, right, data in graph.edges(data=True)
        if data.get("type") == "bond"
    ]
    interaction_ligand_nodes = {
        node
        for left, right, data in graph.edges(data=True)
        if data.get("type") != "bond"
        for node in (left, right)
        if node in ligand_nodes
    }
    bond_degree = {node: 0 for node in ligand_nodes}
    for left, right in bond_edges:
        if left in bond_degree:
            bond_degree[left] += 1
        if right in bond_degree:
            bond_degree[right] += 1

    # Check connectivity within each physical ligand entity, not across
    # separate cofactor/metal/substrate entities that are intentionally drawn
    # as separate fragments.
    by_entity: Dict[object, List[str]] = {}
    for node in ligand_nodes:
        by_entity.setdefault(graph.nodes[node].get("entity_key", graph.nodes[node].get("entity_id", "")), []).append(node)
    disconnected_entities: List[str] = []
    for entity, nodes in by_entity.items():
        non_ions = [node for node in nodes if graph.nodes[node].get("type") != "ion"]
        if len(non_ions) < 2:
            continue
        pending = set(non_ions)
        components = 0
        while pending:
            components += 1
            todo = [pending.pop()]
            while todo:
                current = todo.pop()
                for neighbor in graph.neighbors(current):
                    if neighbor in pending and graph.edges[current, neighbor].get("type") == "bond":
                        pending.remove(neighbor)
                        todo.append(neighbor)
        if components > 1:
            disconnected_entities.append(f"{entity} ({components} components)")

    unattached = sorted(
        node for node in interaction_ligand_nodes
        if len(by_entity.get(graph.nodes[node].get("entity_key", graph.nodes[node].get("entity_id", "")), [])) > 1
        and bond_degree.get(node, 0) == 0
    )
    issues: List[str] = []
    if not ligand_nodes:
        issues.append("no_ligand_nodes")
    if disconnected_entities:
        issues.append("disconnected_ligand_entities=" + "; ".join(disconnected_entities))
    if unattached:
        issues.append("unattached_PLIP_ligand_nodes=" + "; ".join(unattached))
    return {
        "topology_status": "ok" if not issues else "warning",
        "n_ligand_nodes": len(ligand_nodes),
        "n_ligand_bond_edges": len(bond_edges),
        "n_interaction_ligand_nodes": len(interaction_ligand_nodes),
        "n_unattached_plip_ligand_nodes": len(unattached),
        "unattached_plip_ligand_nodes": ";".join(unattached),
        "n_disconnected_ligand_entities": len(disconnected_entities),
        "disconnected_ligand_entities": "; ".join(disconnected_entities),
        "topology_message": "; ".join(issues) if issues else "All PLIP ligand endpoints are attached to their within-entity covalent topology.",
    }


# ---------------------------------------------------------------------------
# Candidate export
# ---------------------------------------------------------------------------

def copy_or_symlink(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copy"


def find_pdb(output_root: Path, step06_dir: Optional[Path], sid: str) -> Optional[Path]:
    roots = [step06_dir, output_root]
    candidates: List[Path] = []
    for root in roots:
        if root and root.exists():
            candidates.extend(root.rglob(f"*{sid}*.pdb"))
    return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))[0] if candidates else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_step10(config_path: str | Path, clean_output: bool = False) -> dict:
    config_path = Path(config_path).resolve()
    root = get_project_root(config_path)
    config = load_config_dict(config_path)
    target_id = get_target_id(config, config_path)
    output_root = get_output_root(config, root, target_id)
    cfg = config.get("step10", {}) or {}

    if cfg.get("enabled", True) is False:
        return {"target_id": target_id, "status": "skipped", "message": "step10.enabled is false."}

    out_dir = output_root / public_output_subdir(
        cfg.get("output_subdir"), "09_class_visualization"
    )
    figures_dir = out_dir / "figures"
    panels_dir = figures_dir / "class_panels"
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    work_dir = out_dir / "work"
    candidates_dir = out_dir / "candidate_classes"

    if clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    for d in [figures_dir, panels_dir, tables_dir, reports_dir, work_dir, candidates_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(reports_dir / f"{target_id}_step09_visualization.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    step09_dir = find_step09_dir(output_root, cfg, root, config.get("step09", {}) or {})
    step06_dir = find_step06_dir(output_root, cfg, root)
    step08_dir = find_step08_dir(output_root, cfg, root, config.get("step08", {}) or {})
    merged_table = find_table(step09_dir, target_id, "merged_class_summary")
    raw_map_table = find_table(step09_dir, target_id, "raw_to_canonical_mapping")

    merged_rows = read_csv_table(merged_table)
    raw_rows = read_csv_table(raw_map_table)

    raw_by_structure: Dict[str, List[dict]] = {}
    for r in raw_rows:
        sid = r.get("structure_id", "")
        if sid:
            raw_by_structure.setdefault(sid, []).append(r)

    plot_cfg = cfg.get("plot", {}) or {}
    style_cfg = cfg.get("style", {}) or {}
    export_cfg = cfg.get("export_candidates", {}) or {}
    formats = plot_cfg.get("formats", ["svg", "png"])
    if isinstance(formats, str):
        formats = [formats]
    grid_cols = int(plot_cfg.get("grid_cols", 3))
    args = make_legacy_args(cfg, full_config=config)

    class_records = []
    # Step09 has already assigned definitive contiguous merged labels.
    # Step10 sorts and displays those labels, but never renumbers them.
    for row in sorted(
        merged_rows,
        key=lambda r: sequential_merged_class_sort_key(r.get("merged_class", "")),
    ):
        cls = row.get("merged_class", "")
        sids = split_ids(row.get("structure_ids", ""))
        rep = row.get("representative_structure_id") or (sids[0] if sids else "")
        is_candidate = as_bool(row.get("is_catalytic_candidate", False))
        class_size = int(row.get("n_structures") or len(sids) or 0)

        plip_folder = find_plip_folder(step06_dir, rep)
        source_kind = "step06_plip_folder" if plip_folder else "fallback_reconstructed"
        graph_file = find_graph_file(step06_dir, step08_dir, rep)
        representative_pdb = find_pdb(output_root, step06_dir, rep)
        if not plip_folder:
            plip_folder = create_fallback_plot_folder(
                work_dir,
                cls,
                rep,
                raw_by_structure.get(rep, []),
                graph_file,
            )

        class_records.append({
            "merged_class": cls,
            "label": (
                row.get("merged_class_display")
                or cls.replace("class_", "Class ")
            ),
            "representative_structure_id": rep,
            "structure_ids": ";".join(sids),
            "n_structures": class_size,
            "is_catalytic_candidate": is_candidate,
            "input_folder": str(plip_folder),
            "input_source": source_kind,
            "graph_file": str(graph_file or ""),
            # Store a direct, absolute PDB pointer in the summary so the GUI
            # never has to guess its way back through moved Step04/05 tables.
            "representative_pdb_path": str(representative_pdb or ""),
            "original_classes": row.get("original_classes", ""),
            "n_common_edges_missing": row.get("n_common_edges_missing", ""),
            "missing_common_edges": row.get("missing_common_edges", ""),
            "n_rare_edges_present": row.get("n_rare_edges_present", ""),
            "rare_edges_present": row.get("rare_edges_present", ""),
        })

    # Draw individual panels with one shared template.
    vis = EnzymeSubstrateVisualizer()
    summary_rows: List[dict] = []
    topology_audit_rows: List[dict] = []
    candidate_color = style_cfg.get("candidate_border_color", "#ff9900")
    normal_color = style_cfg.get("normal_border_color", "#BDBDBD")

    for i, rec in enumerate(class_records):
        folder = Path(rec["input_folder"])
        audit_row = {
            "merged_class": rec["merged_class"],
            "representative_structure_id": rec["representative_structure_id"],
            "input_folder": str(folder),
            "input_source": rec["input_source"],
            **inspect_topology_integrity(folder, args),
        }
        topology_audit_rows.append(audit_row)
        for fmt in formats:
            fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
            draw_legacy_panel(
                vis, ax, folder, rec["label"], int(rec["n_structures"]),
                bool(rec["is_catalytic_candidate"]), args,
                use_template=(i > 0), candidate_color=candidate_color, normal_color=normal_color,
            )
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            out_path = panels_dir / f"{target_id}_{sanitize_filename(rec['merged_class'])}.{fmt}"
            fig.savefig(out_path, dpi=args.dpi, transparent=False, facecolor="white", pad_inches=0)
            plt.close(fig)
            rec[f"panel_{fmt}"] = str(out_path)
        summary_rows.append({**dict(rec), **audit_row})

    topology_audit_table = tables_dir / f"{target_id}_step09_topology_integrity_audit.csv"
    write_csv(topology_audit_table, topology_audit_rows)
    if as_bool((cfg.get("layout", {}) or {}).get("fail_on_topology_integrity_error", False)):
        failed = [row for row in topology_audit_rows if row["topology_status"] != "ok"]
        if failed:
            raise RuntimeError(
                "Step10 topology-integrity audit found disconnected ligand nodes. "
                f"Inspect {topology_audit_table} and rerun Step06 with "
                "connectivity.source=input_conect_then_rdkit_same_residue."
            )

    # Draw grid figure using same template state.
    n = len(class_records)
    grid_paths: Dict[str, str] = {}
    if n:
        cols = max(1, grid_cols)
        rows = math.ceil(n / cols)
        for fmt in formats:
            fig, axes = plt.subplots(
                rows, cols,
                figsize=(cols * float(args.fig_width), rows * float(args.fig_height) + 0.62),
                dpi=args.dpi,
            )
            try:
                axes_list = list(axes.ravel())
            except Exception:
                axes_list = [axes]
            for ax in axes_list:
                ax.axis("off")
            # New visualizer for grid so first class is template inside grid too.
            grid_vis = EnzymeSubstrateVisualizer()
            for i, (ax, rec) in enumerate(zip(axes_list, class_records)):
                draw_legacy_panel(
                    grid_vis, ax, Path(rec["input_folder"]), rec["label"], int(rec["n_structures"]),
                    bool(rec["is_catalytic_candidate"]), args,
                    use_template=(i > 0), candidate_color=candidate_color, normal_color=normal_color,
                )
            fig.suptitle(f"{target_id} class interaction networks after Step09", fontsize=10, fontweight="bold", y=0.995)
            add_summary_legend(fig, grid_vis, args, bottom=0.012)
            fig.subplots_adjust(left=0.008, right=0.992, top=0.972, bottom=0.055, wspace=0.035, hspace=0.065)
            grid_path = figures_dir / f"{target_id}_step09_class_grid.{fmt}"
            # Keep a fixed canvas. Tight-cropping makes the SVG proportions unstable and can flatten panels.
            fig.savefig(grid_path, dpi=args.dpi, transparent=False, facecolor="white", pad_inches=0)
            plt.close(fig)
            grid_paths[fmt] = str(grid_path)

    # Candidate export.
    export_rows: List[dict] = []
    if as_bool(export_cfg.get("enabled", True)):
        mode = str(export_cfg.get("mode", "copy")).lower()
        if mode not in {"copy", "symlink"}:
            mode = "copy"
        for rec in class_records:
            if not rec["is_catalytic_candidate"]:
                continue
            cls_dir = candidates_dir / sanitize_filename(rec["merged_class"])
            for sub in ["structures", "ged_graphs", "plip_outputs"]:
                (cls_dir / sub).mkdir(parents=True, exist_ok=True)
            for sid in split_ids(rec.get("structure_ids", "")):
                plip = find_plip_folder(step06_dir, sid)
                pdb = find_pdb(output_root, step06_dir, sid)
                graph = find_graph_file(step06_dir, step08_dir, sid)
                for file_type, src, subdir in [
                    ("plip_output_dir", plip, "plip_outputs"),
                    ("structure_pdb", pdb, "structures"),
                    ("ged_graph", graph, "ged_graphs"),
                ]:
                    if src:
                        action = copy_or_symlink(src, cls_dir / subdir / src.name, mode)
                    else:
                        action = "not_found"
                    export_rows.append({
                        "merged_class": rec["merged_class"],
                        "structure_id": sid,
                        "file_type": file_type,
                        "source": str(src or ""),
                        "action": action,
                    })

    summary_table = tables_dir / f"{target_id}_step09_class_visual_summary.csv"
    export_table = tables_dir / f"{target_id}_step09_candidate_export_summary.csv"
    write_csv(summary_table, summary_rows)
    write_csv(export_table, export_rows, fieldnames=["merged_class", "structure_id", "file_type", "source", "action"])

    report = {
        "target_id": target_id,
        "status": "completed",
        "output_dir": str(out_dir),
        "input_step09_dir": str(step09_dir),
        "input_step06_dir": str(step06_dir or ""),
        "n_classes": len(class_records),
        "class_numbering_source": "Step09 sequential merged_class labels",
        "n_candidate_classes": sum(1 for r in class_records if r["is_catalytic_candidate"]),
        "n_fallback_reconstructed_inputs": sum(1 for r in class_records if r["input_source"] == "fallback_reconstructed"),
        "figures": {
            "class_panels_dir": str(panels_dir),
            "class_grid_svg": grid_paths.get("svg", ""),
            "class_grid_png": grid_paths.get("png", ""),
        },
        "tables": {
            "class_visual_summary": str(summary_table),
            "topology_integrity_audit": str(topology_audit_table),
            "candidate_export_summary": str(export_table),
        },
    }
    report_path = reports_dir / f"{target_id}_step09_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step10 legacy-style class visualization after Step09.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    parser.add_argument("--clean", action="store_true", help="Clean Step10 output first.")
    args = parser.parse_args(argv)
    report = run_step10(args.config, clean_output=args.clean)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

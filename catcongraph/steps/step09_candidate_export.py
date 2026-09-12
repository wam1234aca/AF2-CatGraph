#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step09 candidate structure exporter.

This helper is intentionally separated from the Step08 ligand contact-unit grouping logic.
It does not change candidate definitions.  It only copies or symlinks files for
manual inspection after Step09 has produced candidate_catalytic_classes.csv and
candidate_structures.csv.

Default behavior:
  - if no candidate structures exist, create empty fixed-header index tables;
  - if candidate structures exist, export available source PDB, GED graph, PLIP
    interaction_output and PLIP report files into a Step09 subfolder.

This module is used by the replacement scripts/09_chemical_equivalence.py wrapper
in this patch, and can also be run directly via scripts/09_export_candidate_structures.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from catcongraph.config import load_yaml_config, public_output_subdir


# ---------------------------------------------------------------------------
# Minimal config/table helpers
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
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
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


def load_config(config_path: Path) -> dict:
    return load_yaml_config(config_path)


def get_project_root(config_path: Path) -> Path:
    if config_path.name in {"config.yaml", "config.yml"} and config_path.parent.parent.name == "projects":
        return config_path.parent.parent.parent.resolve()
    return Path.cwd().resolve()


def get_target_id(config: dict, config_path: Path) -> str:
    project_cfg = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
    for obj in (config, project_cfg):
        for key in ("target_id", "id", "name", "target"):
            if isinstance(obj, dict) and obj.get(key):
                return str(obj[key])
    return config_path.parent.name


def get_output_root(config: dict, root: Path, target_id: str) -> Path:
    project_cfg = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
    for obj in (config, project_cfg):
        if isinstance(obj, dict) and obj.get("output_root"):
            p = Path(str(obj["output_root"]))
            return p if p.is_absolute() else (root / p)
    return root / "results" / target_id


def read_csv(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def structure_id_from_name(name: str) -> str:
    base = Path(str(name)).stem
    for suffix in [
        "_final",
        "_trimmed_complex",
        "_complex_complex_complex",
        "_complex_complex",
        "_complex",
        "_protonated",
        "_xcbrefx",
    ]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def find_step09_dir(output_root: Path, target_id: str, step_cfg: dict, root: Path) -> Path:
    if step_cfg.get("output_dir"):
        p = Path(str(step_cfg["output_dir"]))
        return p if p.is_absolute() else (root / p)
    sub = public_output_subdir(step_cfg.get("output_subdir"), "08_chemical_equivalence")
    return output_root / str(sub)


def candidate_structures_table(step09_dir: Path, target_id: str, step_cfg: dict, root: Path) -> Path:
    explicit = step_cfg.get("candidate_structures_table") or step_cfg.get("input_candidate_structures_table")
    if explicit:
        p = Path(str(explicit))
        return p if p.is_absolute() else (root / p)
    return step09_dir / "tables" / f"{target_id}_step08_candidate_structures.csv"


def build_structure_file_index(output_root: Path, target_id: str, step_cfg: dict, root: Path) -> Dict[str, List[Path]]:
    """Find source structure PDB files with broad but deterministic priority.

    This avoids changing Step09 logic.  If Step08/Step09 tables do not carry PDB
    paths, we search under the target output folder and choose the most likely
    complex for each structure id.
    """
    search_roots = []
    for key in ("source_structure_dir", "source_complex_dir", "input_complex_dir"):
        if step_cfg.get(key):
            p = Path(str(step_cfg[key]))
            search_roots.append(p if p.is_absolute() else (root / p))
    search_roots.append(output_root)
    search_roots = [p for p in search_roots if p.exists()]

    priority_tokens = [
        "04_catalytic_region_plddt_qc",
        "05_catalytic_region_plddt_qc",
        "05_pocket_plddt_qc",
        "04_ligand_atom_trimming",
        "03_clash_filter",
        "02_template_guided_docking",
        "05_plip_interaction_graphs",
        "06_plip_interaction_graphs",
    ]
    index: Dict[str, List[Path]] = {}
    for sr in search_roots:
        for p in sr.rglob("*.pdb"):
            if p.name.startswith("."):
                continue
            sid = structure_id_from_name(p.name)
            if not sid:
                continue
            index.setdefault(sid, []).append(p)

    def priority(path: Path) -> tuple:
        text = str(path)
        score = 999
        for i, tok in enumerate(priority_tokens):
            if tok in text:
                score = i
                break
        # Prefer final user-facing complexes over PLIP-generated protonated/xcbrefx files.
        name = path.name.lower()
        penalty = 0
        if "protonated" in name or "xcbrefx" in name or "plipfixed" in name:
            penalty += 10
        if "trimmed_complex" in name:
            penalty -= 1
        if name.endswith("_complex.pdb"):
            penalty -= 1
        return (score + penalty, len(str(path)), str(path))

    for sid in list(index):
        index[sid] = sorted(set(index[sid]), key=priority)
    return index


def build_aux_file_index(output_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    out: Dict[str, Dict[str, List[Path]]] = {}
    patterns = [
        ("graph", "*_final.txt"),
        ("interaction", "*interaction_output*.txt"),
        ("plip_report_xml", "report.xml"),
        ("plip_report_txt", "report.txt"),
    ]
    for kind, pattern in patterns:
        for p in output_root.rglob(pattern):
            if p.name.startswith("."):
                continue
            if p.name in {"report.xml", "report.txt", "interaction_output.txt"}:
                sid = structure_id_from_name(p.parent.name)
            else:
                sid = structure_id_from_name(p.name)
            out.setdefault(sid, {}).setdefault(kind, []).append(p)
    for sid in out:
        for kind in out[sid]:
            out[sid][kind] = sorted(set(out[sid][kind]), key=lambda x: (len(str(x)), str(x)))
    return out


def safe_link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    try:
        os.symlink(src, dst)
        return "symlinked"
    except Exception:
        shutil.copy2(src, dst)
        return "copied_fallback"


# ---------------------------------------------------------------------------
# Main exporter
# ---------------------------------------------------------------------------


def export_step09_candidates(config_path: str | Path, step09_report: Optional[dict] = None) -> dict:
    config_path = Path(config_path).resolve()
    root = get_project_root(config_path)
    config = load_config(config_path)
    target_id = get_target_id(config, config_path)
    output_root = get_output_root(config, root, target_id)

    step09_cfg = config.get("step09", {}) or {}
    export_cfg = step09_cfg.get("export_candidates", {}) or {}
    enabled = export_cfg.get("enabled", True)
    mode = str(export_cfg.get("mode", "symlink")).lower()
    if mode not in {"copy", "symlink"}:
        mode = "symlink"

    step09_dir = find_step09_dir(output_root, target_id, step09_cfg, root)
    table = candidate_structures_table(step09_dir, target_id, step09_cfg, root)

    export_dir = step09_dir / str(export_cfg.get("output_subdir", "candidate_exports"))
    complexes_dir = export_dir / "complexes"
    graphs_dir = export_dir / "ged_input_files"
    interactions_dir = export_dir / "interaction_outputs"
    plip_reports_dir = export_dir / "plip_reports"
    tables_dir = export_dir / "tables"
    reports_dir = export_dir / "reports"
    for d in [complexes_dir, graphs_dir, interactions_dir, plip_reports_dir, tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fixed_headers = [
        "structure_id",
        "original_class",
        "merged_class",
        "source_complex_path",
        "exported_complex_path",
        "graph_path",
        "exported_graph_path",
        "interaction_output_path",
        "exported_interaction_output_path",
        "status",
        "notes",
    ]
    out_index = tables_dir / f"{target_id}_step08_candidate_export_index.csv"

    if not enabled:
        write_csv(out_index, [], fixed_headers)
        return {
            "target_id": target_id,
            "status": "skipped",
            "reason": "step09.export_candidates.enabled is false",
            "candidate_structures_table": str(table),
            "export_index": str(out_index),
        }

    rows = read_csv(table)
    if not rows:
        write_csv(out_index, [], fixed_headers)
        report = {
            "target_id": target_id,
            "status": "completed",
            "n_candidate_structures": 0,
            "message": "No candidate structures were listed by Step09.",
            "candidate_structures_table": str(table),
            "export_dir": str(export_dir),
            "export_index": str(out_index),
        }
        (reports_dir / f"{target_id}_step08_candidate_export_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report

    pdb_index = build_structure_file_index(output_root, target_id, step09_cfg, root)
    aux_index = build_aux_file_index(output_root)

    export_rows: List[dict] = []
    for r in rows:
        sid = r.get("structure_id") or structure_id_from_name(r.get("graph_file", ""))
        original_class = r.get("original_class") or r.get("class") or ""
        merged_class = r.get("merged_class") or ""
        cls_dir = merged_class or "unclassified"

        notes = []
        status = "exported"

        source_complex = None
        for key in ("complex_path", "source_complex_path", "pdb_path", "structure_path"):
            if r.get(key):
                p = Path(r[key])
                if not p.is_absolute():
                    p = root / p
                if p.exists():
                    source_complex = p
                    break
        if source_complex is None:
            candidates = pdb_index.get(sid, [])
            source_complex = candidates[0] if candidates else None

        graph = None
        inter = None
        report_file = None
        if sid in aux_index:
            graph = (aux_index[sid].get("graph") or [None])[0]
            inter = (aux_index[sid].get("interaction") or [None])[0]
            report_file = (aux_index[sid].get("plip_report_xml") or aux_index[sid].get("plip_report_txt") or [None])[0]

        exported_complex = ""
        exported_graph = ""
        exported_inter = ""
        exported_report = ""

        if source_complex:
            dst = complexes_dir / cls_dir / source_complex.name
            safe_link_or_copy(source_complex, dst, mode)
            exported_complex = str(dst)
        else:
            status = "partial"
            notes.append("source_complex_not_found")

        if graph:
            dst = graphs_dir / cls_dir / graph.name
            safe_link_or_copy(graph, dst, mode)
            exported_graph = str(dst)
        else:
            notes.append("graph_not_found")

        if inter:
            dst = interactions_dir / cls_dir / f"{sid}_interaction_output.txt"
            safe_link_or_copy(inter, dst, mode)
            exported_inter = str(dst)
        else:
            notes.append("interaction_output_not_found")

        if report_file:
            dst = plip_reports_dir / cls_dir / f"{sid}_{report_file.name}"
            safe_link_or_copy(report_file, dst, mode)
            exported_report = str(dst)

        export_rows.append(
            {
                "structure_id": sid,
                "original_class": original_class,
                "merged_class": merged_class,
                "source_complex_path": str(source_complex or ""),
                "exported_complex_path": exported_complex,
                "graph_path": str(graph or ""),
                "exported_graph_path": exported_graph,
                "interaction_output_path": str(inter or ""),
                "exported_interaction_output_path": exported_inter,
                "plip_report_path": str(report_file or ""),
                "exported_plip_report_path": exported_report,
                "status": status,
                "notes": ";".join(notes),
            }
        )

    headers = list(dict.fromkeys(fixed_headers + ["plip_report_path", "exported_plip_report_path"]))
    write_csv(out_index, export_rows, headers)

    report = {
        "target_id": target_id,
        "status": "completed",
        "n_candidate_structures": len(rows),
        "n_export_rows": len(export_rows),
        "export_mode": mode,
        "candidate_structures_table": str(table),
        "export_dir": str(export_dir),
        "export_index": str(out_index),
    }
    (reports_dir / f"{target_id}_step08_candidate_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export Step09 candidate structures for manual inspection.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    args = parser.parse_args(argv)
    report = export_step09_candidates(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

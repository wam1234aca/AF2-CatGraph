"""Fast result discovery and offline 3D class viewing for AF2-CatGraph.

This module is intentionally independent of Streamlit so it can be tested as
ordinary Python code.  The browser view uses the vendored 3Dmol.js library and
therefore does not require a connection to an external JavaScript CDN.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import html
import json
from math import dist
from pathlib import Path
import re
from typing import Any, Iterable


PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
WATER_RESNAMES = {"HOH", "WAT", "DOD"}
INTERACTION_COLORS = {
    "hydrogen_bonds": "#2574c5",
    "salt_bridges": "#d14b5a",
    "hydrophobic_interactions": "#3c9b68",
    "pi_stacks": "#c48816",
    "pi_cation_interactions": "#ef8f19",
    "metal_complexes": "#7255b8",
    "halogen_bonds": "#a36341",
}


def result_files(root: str | Path, include_intermediates: bool = False) -> list[Path]:
    """Return browser-relevant files without eagerly indexing every PLIP detail.

    A Step06 run can contain thousands of per-structure files.  Those files are
    still available when requested, but scanning them on every browser click is
    a common cause of slow remote Streamlit sessions.
    """
    base = Path(root)
    if not base.exists():
        return []
    if include_intermediates:
        return sorted((path for path in base.rglob("*") if path.is_file()), key=lambda path: str(path).lower())

    # Do *not* start with base.rglob here. A large Step06 run contains a deep
    # PLIP tree, and walking it after every Streamlit click is enough to make a
    # remote interface feel frozen. The user-facing final products have stable
    # directory names, so scan those directly. "Show intermediates" above is
    # still available for complete audit access.
    preferred_dirs = {"tables", "reports", "figures", "candidate_exports", "candidate_classes"}
    files: list[Path] = [path for path in base.iterdir() if path.is_file()]
    for step_dir in base.iterdir():
        if not step_dir.is_dir():
            continue
        for child in step_dir.iterdir():
            if child.is_dir() and child.name.lower() in preferred_dirs:
                files.extend(path for path in child.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: str(path).lower())


def read_delimited(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = []
        for index, row in enumerate(reader):
            rows.append({str(key): str(value or "") for key, value in row.items()})
            if limit is not None and index + 1 >= limit:
                break
        return rows


def find_class_summary(output_root: str | Path) -> Path | None:
    root = Path(output_root)
    direct = list(root.glob("09_class_visualization/tables/*_step09_class_visual_summary.csv"))
    candidates = direct or list(root.rglob("*_step09_class_visual_summary.csv"))
    if not candidates:
        legacy = list(root.glob("09_class_visualization/tables/*_step10_class_visual_summary.csv"))
        legacy += list(root.glob("10_class_visualization/tables/*_step10_class_visual_summary.csv"))
        candidates = legacy or list(root.rglob("*_step10_class_visual_summary.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def class_rows(output_root: str | Path) -> list[dict[str, str]]:
    summary = find_class_summary(output_root)
    if not summary:
        return []
    return read_delimited(summary)


def _resolve_existing_path(value: str, roots: Iterable[Path]) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "auto"}:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in roots:
        p = root / candidate
        if p.exists():
            return p
    return None


def _choose_column(columns: Iterable[str], keywords: Iterable[str]) -> str | None:
    normalized = {str(column).lower(): str(column) for column in columns}
    for word in keywords:
        for lower, original in normalized.items():
            if word in lower:
                return original
    return None


def _upstream_index_tables(output_root: Path) -> list[Path]:
    step_names = [
        "04_catalytic_region_plddt_qc", "05_catalytic_region_plddt_qc", "04_ligand_atom_trimming", "03_clash_filter",
        "02_template_guided_docking", "01_af2_prepare",
    ]
    tables = []
    for name in step_names:
        table_dir = output_root / name / "tables"
        if table_dir.exists():
            tables.extend(table_dir.glob("*.csv"))
    return sorted(tables)


def representative_pdb(output_root: str | Path, row: dict[str, str], repository_root: str | Path | None = None) -> Path | None:
    """Resolve a class representative PDB from Step10/Step05/Step03 indices."""
    output_root = Path(output_root)
    repository_root = Path(repository_root) if repository_root else output_root.parent.parent
    sid = str(row.get("representative_structure_id", "")).strip()
    for key in ("representative_pdb", "representative_pdb_path", "structure_pdb"):
        direct = _resolve_existing_path(row.get(key, ""), [repository_root, output_root])
        if direct and direct.suffix.lower() == ".pdb":
            return direct
    direct_folder = _resolve_existing_path(row.get("input_folder", ""), [repository_root, output_root])
    if direct_folder and direct_folder.is_dir():
        pdbs = sorted(direct_folder.glob("*.pdb"))
        if pdbs:
            return pdbs[0]

    for table in _upstream_index_tables(output_root):
        try:
            rows = read_delimited(table)
        except (OSError, csv.Error):
            continue
        if not rows:
            continue
        id_col = _choose_column(rows[0].keys(), ["structure_id", "complex_id", "id"])
        path_col = _choose_column(rows[0].keys(), ["complex_pdb", "output_pdb", "pdb_path", "pdb"])
        if not path_col:
            continue
        for item in rows:
            record_id = str(item.get(id_col, "")) if id_col else ""
            if sid and record_id and record_id != sid:
                continue
            resolved = _resolve_existing_path(item.get(path_col, ""), [repository_root, output_root, table.parent.parent])
            if resolved and resolved.suffix.lower() == ".pdb":
                return resolved

    # Candidate exports are a useful final fallback, especially when Step10 was
    # configured to copy them but an upstream index was moved.
    candidate_roots = [
        output_root / "09_class_visualization" / "candidate_classes",
        output_root / "08_chemical_equivalence" / "candidate_exports" / "pdbs",
        output_root / "10_class_visualization" / "candidate_classes",
        output_root / "09_chemical_equivalence" / "candidate_exports" / "pdbs",
    ]
    for candidate_root in candidate_roots:
        if candidate_root.exists() and sid:
            matches = sorted(candidate_root.rglob(f"*{sid}*.pdb"))
            if matches:
                return matches[0]
    return None


def class_interactions(
    row: dict[str, str], output_root: str | Path | None = None, repository_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Read a representative class's PLIP records, with Step06 fallbacks."""
    folder = Path(str(row.get("input_folder", ""))).expanduser()
    if not folder.exists() and not folder.is_absolute():
        roots = [Path(value) for value in (output_root, repository_root) if value]
        resolved = _resolve_existing_path(str(folder), roots)
        if resolved:
            folder = resolved
    candidates: list[Path] = []
    if folder.exists():
        candidates.extend([folder / "interaction_output.csv", folder / "interaction_output.txt"])

    # Step10 can reconstruct a small drawing folder when old Step09 outputs do
    # not retain the original PLIP directory.  In that case recover the source
    # records from Step06 using the representative structure ID.
    sid = str(row.get("representative_structure_id", "")).strip()
    if output_root and sid:
        root = Path(output_root)
        for step06 in sorted(root.glob("*plip_interaction_graphs*")):
            candidates.extend([
                step06 / "plip_outputs" / sid / "interaction_output.csv",
                step06 / "plip_outputs" / sid / "interaction_output.txt",
            ])
            candidates.extend(step06.glob(f"plip_outputs/**/*{sid}*/interaction_output.csv"))
            candidates.extend(step06.glob(f"plip_outputs/**/*{sid}*/interaction_output.txt"))

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            return read_delimited(path)
        rows: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields: dict[str, str] = {}
            for field in line.split("\t"):
                if ":" in field:
                    key, value = field.split(":", 1)
                    fields[key.strip().lower().replace(" ", "_")] = value.strip()
            if not fields:
                continue
            rows.append({
                "type": fields.get("type", ""),
                "ligand_node": fields.get("ligand", ""),
                "protein_node": fields.get("protein", ""),
                "ligand_atom_serial": fields.get("ligandserial", ""),
                "ligand_node_key": fields.get("ligandnodekey", ""),
            })
        if rows:
            return rows
    return []


def _pdb_atoms(pdb_text: str) -> list[dict[str, Any]]:
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        try:
            atoms.append({
                "serial": int(line[6:11]),
                "record": line[:6].strip(),
                "atom": line[12:16].strip(),
                "resn": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resi": int(line[22:26]),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": (line[76:78].strip() or line[12:14].strip()).upper(),
            })
        except ValueError:
            continue
    return atoms


def _protein_identity(value: str) -> tuple[str, int, str] | None:
    match = re.match(r"([A-Za-z]{3})(-?\d+)([A-Za-z0-9]?)", str(value or "").strip())
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2)), match.group(3).strip()


def _ligand_identity(value: str, fallback_atom: str) -> tuple[str, str, int, str] | None:
    parts = str(value or "").split(":")
    if len(parts) >= 4:
        try:
            return parts[0].upper(), parts[1].strip(), int(parts[2]), parts[3].strip()
        except ValueError:
            return None
    return None


def _interaction_geometry(pdb_text: str, interactions: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    atoms = _pdb_atoms(pdb_text)
    by_serial = {atom["serial"]: atom for atom in atoms}
    lines: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    residue_keys: set[tuple[str, int]] = set()
    ligand_resnames = sorted({atom["resn"] for atom in atoms if atom["record"] == "HETATM" and atom["resn"] not in WATER_RESNAMES})

    for interaction in interactions:
        ligand = None
        serial = str(interaction.get("ligand_atom_serial", "")).strip()
        if serial.isdigit():
            ligand = by_serial.get(int(serial))
        if ligand is None:
            identity = _ligand_identity(interaction.get("ligand_node_key", ""), interaction.get("ligand_node", ""))
            if identity:
                resn, chain, resi, atom_name = identity
                ligand = next((atom for atom in atoms if atom["resn"] == resn and atom["chain"] == chain and atom["resi"] == resi and atom["atom"] == atom_name), None)
        if ligand is None:
            continue
        protein_id = _protein_identity(interaction.get("protein_node", ""))
        if not protein_id:
            continue
        resn, resi, chain = protein_id
        residue_atoms = [
            atom for atom in atoms
            if atom["record"] == "ATOM" and atom["resn"] == resn and atom["resi"] == resi
            and (not chain or atom["chain"] == chain) and atom["element"] != "H"
        ]
        if not residue_atoms:
            continue
        protein = min(residue_atoms, key=lambda atom: dist(
            (atom["x"], atom["y"], atom["z"]), (ligand["x"], ligand["y"], ligand["z"])
        ))
        itype = str(interaction.get("type", "")).strip().lower()
        lines.append({
            "start": {key: protein[key] for key in ("x", "y", "z")},
            "end": {key: ligand[key] for key in ("x", "y", "z")},
            "color": INTERACTION_COLORS.get(itype, "#606c76"),
            "type": itype,
            "protein_atom": protein,
            "ligand_atom": ligand,
        })
        residue_key = (protein["chain"], protein["resi"])
        if residue_key not in residue_keys:
            residue_keys.add(residue_key)
            labels.append({
                "text": f"{resn}{resi}",
                "position": {key: protein[key] for key in ("x", "y", "z")},
                "chain": protein["chain"], "resi": protein["resi"], "resn": resn,
            })
    return lines, labels, ligand_resnames


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


@lru_cache(maxsize=1)
def _3dmol_source() -> str:
    return (Path(__file__).resolve().parent / "static" / "3Dmol-min.js").read_text(encoding="utf-8")


def viewer_html(pdb_text: str, interactions: list[dict[str, str]], title: str = "") -> tuple[str, str]:
    """Return offline HTML plus a short status message for the class viewer."""
    lines, labels, ligand_resnames = _interaction_geometry(pdb_text, interactions)
    if not ligand_resnames:
        return "", "No non-water HETATM ligand was found in this representative PDB."
    contact_residues = [{"chain": item["chain"], "resi": item["resi"]} for item in labels]
    payload = {
        "pdb": pdb_text,
        "lines": lines,
        "labels": labels,
        "ligands": ligand_resnames,
        "contacts": contact_residues,
    }
    source = _3dmol_source()
    page = f"""<!DOCTYPE html>
<html><head><meta charset=\"utf-8\"><style>
html, body, #viewer {{width:100%; height:100%; margin:0; overflow:hidden; background:#ffffff;}}
#note {{position:absolute; top:9px; left:11px; z-index:2; color:#27414a; font:12px Arial; background:rgba(255,255,255,.78); padding:5px 7px; border-radius:5px;}}
</style></head><body><div id=\"note\">{html.escape(title)} · drag to rotate · scroll to zoom</div><div id=\"viewer\"></div>
<script>{source}</script><script>
const payload = {_json(payload)};
const viewer = $3Dmol.createViewer('viewer', {{backgroundColor: '#ffffff', antialias: true}});
viewer.addModel(payload.pdb, 'pdb');
viewer.setStyle({{hetflag:false}}, {{cartoon: {{color: 'spectrum', opacity: 0.88}}}});
viewer.setStyle({{resn: payload.ligands}}, {{stick: {{radius: 0.20, colorscheme: 'Jmol'}}, sphere: {{scale: 0.22, colorscheme: 'Jmol'}}}});
payload.contacts.forEach(item => viewer.addStyle({{chain:item.chain, resi:item.resi}}, {{stick: {{radius:0.18, color:'#e05263'}}}}));
payload.lines.forEach(item => viewer.addCylinder({{start:item.start, end:item.end, radius:0.055, color:item.color, dashed:true, fromCap:1, toCap:1}}));
payload.labels.forEach(item => viewer.addLabel(item.text, {{position:item.position, fontColor:'#222222', backgroundColor:'#ffffff', backgroundOpacity:0.68, borderColor:'#bbbbbb', borderThickness:1, fontSize:12}}));
viewer.zoomTo({{resn: payload.ligands}}); viewer.zoom(1.45); viewer.render();
window.addEventListener('resize', () => {{viewer.resize(); viewer.render();}});
</script></body></html>"""
    status = f"Showing {len(ligand_resnames)} ligand entity type(s), {len(labels)} contacting residue(s), and {len(lines)} PLIP-derived dashed interaction line(s)."
    return page, status


def pymol_script(pdb_filename: str, interactions: list[dict[str, str]], ligand_resnames: Iterable[str]) -> str:
    """Create a transparent PyMOL script for the same representative structure."""
    residues = []
    distance_commands = []
    for index, interaction in enumerate(interactions, start=1):
        protein_id = _protein_identity(interaction.get("protein_node", ""))
        ligand_id = _ligand_identity(interaction.get("ligand_node_key", ""), interaction.get("ligand_node", ""))
        if not protein_id or not ligand_id:
            continue
        resn, resi, chain = protein_id
        lig_resn, lig_chain, lig_resi, lig_atom = ligand_id
        residue = f"(chain {chain} and resi {resi})" if chain else f"(resi {resi})"
        if residue not in residues:
            residues.append(residue)
        protein_selection = f"(chain {chain} and resi {resi})" if chain else f"(resi {resi})"
        ligand_selection = f"(resn {lig_resn} and chain {lig_chain} and resi {lig_resi} and name {lig_atom})"
        distance_commands.append(
            f"distance interaction_{index}, {protein_selection}, {ligand_selection}\nset dash_color, gray70, interaction_{index}\nset dash_width, 2.0, interaction_{index}"
        )
    ligands = "+".join(sorted(set(str(name) for name in ligand_resnames))) or "LIG"
    contact_selection = " or ".join(residues) if residues else "none"
    distance_block = "\n".join(distance_commands)
    return f"""# PyMOL view generated by AF2-CatGraph
load {pdb_filename}, catcongraph_model
hide everything
show cartoon, polymer.protein
color lightblue, polymer.protein
select ligand, resn {ligands}
show sticks, ligand
util.cbag ligand
select contact_residues, {contact_selection}
show sticks, contact_residues
color salmon, contact_residues
label (contact_residues and name CA), \"%s%s\" % (resn, resi)
set label_size, 18
set dash_gap, 0.35
set dash_length, 0.18
{distance_block}
zoom (ligand or contact_residues), 8
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping


def _conect_edges(path: Path) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("CONECT"):
                continue
            values = [int(x) for x in re.findall(r"\d+", line[6:])]
            if len(values) < 2:
                continue
            for other in values[1:]:
                if other != values[0]:
                    edges.add(tuple(sorted((values[0], other))))
    return edges


def audit_trimmed_connectivity(
    input_pdb: str | Path,
    trace_rows: Iterable[Mapping[str, Any]],
    *,
    structure_id: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in trace_rows]
    ligand_rows = [row for row in rows if bool(row.get("is_ligand_candidate"))]
    kept = {int(row["atom_serial"]) for row in ligand_rows if bool(row.get("keep_atom"))}
    removed = {int(row["atom_serial"]) for row in ligand_rows if not bool(row.get("keep_atom"))}
    names = {int(row["atom_serial"]): str(row.get("atom_name_original", "")) for row in ligand_rows}
    edges = _conect_edges(Path(input_pdb))
    ligand_edges = {edge for edge in edges if edge[0] in kept | removed and edge[1] in kept | removed}
    kept_edges = {edge for edge in ligand_edges if edge[0] in kept and edge[1] in kept}
    cut_edges = {edge for edge in ligand_edges if (edge[0] in kept) != (edge[1] in kept)}

    adjacency = {serial: set() for serial in kept}
    for a, b in kept_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = 0
    unseen = set(kept)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            new = adjacency.get(current, set()) & unseen
            unseen.difference_update(new)
            stack.extend(new)

    return {
        "standard_structure_id": structure_id,
        "input_complex_pdb_path": str(input_pdb),
        "n_ligand_atoms_before": len(ligand_rows),
        "n_ligand_atoms_kept": len(kept),
        "n_ligand_atoms_removed": len(removed),
        "n_explicit_ligand_bonds": len(ligand_edges),
        "n_explicit_bonds_retained": len(kept_edges),
        "n_trim_boundary_bonds": len(cut_edges),
        "trim_boundary_bonds": ";".join(
            f"{names.get(a, a)}({a})-{names.get(b, b)}({b})" for a, b in sorted(cut_edges)
        ),
        "n_kept_components_from_explicit_bonds": components if ligand_edges else None,
        "connectivity_audit_status": "explicit_conect_audited" if ligand_edges else "no_input_conect_downstream_inference_required",
    }


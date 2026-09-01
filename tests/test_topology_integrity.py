from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from catcongraph.steps.step10_class_visualization import inspect_topology_integrity


def _folder(tmp_path: Path, with_bond: bool) -> Path:
    folder = tmp_path / ("connected" if with_bond else "disconnected")
    folder.mkdir()
    if with_bond:
        connectivity = (
            "O1[LIG:A:1:O1;serial=1] -> C1[LIG:A:1:C1;serial=2]\n"
            "C1[LIG:A:1:C1;serial=2] -> O1[LIG:A:1:O1;serial=1]\n"
        )
    else:
        connectivity = (
            "O1[LIG:A:1:O1;serial=1] -> \n"
            "C1[LIG:A:1:C1;serial=2] -> \n"
        )
    (folder / "substrate_connectivity.txt").write_text(connectivity, encoding="utf-8")
    (folder / "interaction_output.txt").write_text(
        "Type: hydrogen_bonds\tLigand: O1\tProtein: ASP9A\tLigandNodeKey: LIG:A:1:O1\n",
        encoding="utf-8",
    )
    return folder


def test_topology_integrity_accepts_connected_plip_endpoint(tmp_path: Path) -> None:
    audit = inspect_topology_integrity(_folder(tmp_path, True), SimpleNamespace(connectivity_source="v8_exact"))
    assert audit["topology_status"] == "ok"
    assert audit["n_ligand_bond_edges"] == 1
    assert audit["n_unattached_plip_ligand_nodes"] == 0


def test_topology_integrity_flags_scattered_ligand_endpoint(tmp_path: Path) -> None:
    audit = inspect_topology_integrity(_folder(tmp_path, False), SimpleNamespace(connectivity_source="v8_exact"))
    assert audit["topology_status"] == "warning"
    assert audit["n_ligand_bond_edges"] == 0
    assert audit["n_unattached_plip_ligand_nodes"] == 1

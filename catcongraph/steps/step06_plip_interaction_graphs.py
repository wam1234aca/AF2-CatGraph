#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PLIP interaction profiling and GED graph export.

This module integrates the user's previous standalone interaction_profiler.py
workflow into AF2-CatGraph while keeping the classification-critical PLIP
interaction extraction rules fixed by default:
- ligand atom labels in GED are residue-aware by default;
- ligand connectivity now uses input-PDB CONECT when available, otherwise RDKit
  bond perception is run on the input complex PDB first so the resulting bonds
  use the same atom serials/residue identities as the GED ligand inventory.
  PLIP-processed RDKit connectivity is available only as an explicit opt-in
  fallback because PLIP/OpenBabel intermediate files can renumber/reorder atoms.

PLIP interaction semantics intentionally remain unchanged:
- PLIP is called with ``-x --breakcomposite``.
- Non-water HETATM atoms, including metal ions, are treated as ligand/cofactor
  atoms unless the config explicitly restricts residue names.
- PLIP interaction types are kept as raw XML group tags.
- Protein nodes are ``RESTYPE + RESNR + RESCHAIN``.
- Ligand connectivity is obtained from input-PDB CONECT records first, with a
  same-residue RDKit fallback when input CONECT is unavailable.
- GED files contain the global vertex blueprint and local binary edges.
- Connectivity debug files are written from the same serial-aware ligand graph
  that is exported to GED.
- Batch runs now clean stale Step06 outputs by default, write per-structure
  inspection files immediately after each successful complex, and write
  failure_error.txt/failure_traceback.txt for failed complexes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import argparse
import json
import logging
import re
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET

import networkx as nx
import pandas as pd
from catcongraph.config import load_yaml_config, public_output_subdir

try:  # Step06 deliberately preserves the legacy RDKit connectivity behavior.
    from rdkit import Chem
except Exception as exc:  # pragma: no cover - runtime path when RDKit is absent
    Chem = None  # type: ignore[assignment]
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None


logger = logging.getLogger(__name__)
WATER_RESNAMES = {"HOH", "WAT"}
LEGACY_CLASSIFICATION_MODE = "interaction_profiler_v3_4"
DEFAULT_LIGAND_NODE_ID_MODE = "residue_atom_name"
DEFAULT_LIGAND_GED_LABEL_MODE = "residue_atom_name"
DEFAULT_CONNECTIVITY_SOURCE = "input_conect_then_rdkit_same_residue"
DEFAULT_ALLOW_CROSS_RESIDUE_RDKIT_CONNECTIVITY = False


@dataclass
class LigandAtomRecord:
    standard_structure_id: str
    complex_pdb_path: str
    source_pdb_path: str
    atom_serial: str
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: str
    insertion_code: str
    element: str
    node_key: str
    ged_label: str


@dataclass
class ConnectivityRecord:
    standard_structure_id: str
    complex_pdb_path: str
    source_pdb_path: str
    ligand_node_key_1: str
    ligand_node_key_2: str
    ligand_node_label_1: str
    ligand_node_label_2: str
    ligand_atom_serial_1: str
    ligand_atom_serial_2: str
    edge_type: str
    source: str


@dataclass
class InteractionRecord:
    standard_structure_id: str
    complex_pdb_path: str
    plip_report_path: str
    interaction_type: str
    ligand_node: str
    protein_node: str
    ligand_atom_serial: str
    source_xml_group: str
    source_xml_tag: str
    ligand_node_key: str = ""
    ligand_node_label: str = ""
    ligand_residue_name: str = ""
    ligand_chain_id: str = ""
    ligand_residue_number: str = ""


@dataclass
class GraphSummary:
    standard_structure_id: str
    complex_pdb_path: str
    plip_work_dir: str
    plip_report_path: str
    ged_file_path: str
    n_ligand_atoms: int
    n_ligand_nodes: int
    n_protein_nodes: int
    n_total_nodes: int
    n_interaction_records: int
    n_ligand_connectivity_edges: int
    n_total_edges: int
    status: str
    error_message: str = ""


class SingleComplexProcessor:
    """Process one complex with legacy PLIP semantics and corrected atom identity."""

    def __init__(
        self,
        pdb_path: Path,
        output_dir: Path,
        structure_id: Optional[str] = None,
        plip_path: str = "plip",
        run_plip: bool = True,
        debug: bool = False,
        save_intermediate_files: bool = True,
        include_resnames: Optional[Sequence[str]] = None,
        exclude_resnames: Optional[Sequence[str]] = None,
        ligand_node_id_mode: str = DEFAULT_LIGAND_NODE_ID_MODE,
        ligand_ged_label_mode: str = DEFAULT_LIGAND_GED_LABEL_MODE,
        ligand_inventory_source: str = "input",
        connectivity_source: str = DEFAULT_CONNECTIVITY_SOURCE,
        allow_cross_residue_rdkit_connectivity: bool = DEFAULT_ALLOW_CROSS_RESIDUE_RDKIT_CONNECTIVITY,
    ) -> None:
        self.pdb_path = Path(pdb_path)
        self.base_name = structure_id or self.pdb_path.stem
        self.work_dir = Path(output_dir) / self.base_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.plip_path = plip_path
        self.run_plip_enabled = bool(run_plip)
        self.debug = bool(debug)
        self.save_intermediate_files = bool(save_intermediate_files)
        self.report_file = self.work_dir / "report.xml"
        self.input_pdb_copy = self.work_dir / self.pdb_path.name
        self.processed_pdb_path: Optional[Path] = None
        self.atom_mapping: Dict[str, str] = {}
        self.hetatm_serials: set[str] = set()
        self.interactions: List[Dict[str, str]] = []
        self.graph = nx.Graph()
        self.include_resnames = _norm_set(include_resnames)
        self.exclude_resnames = _norm_set(exclude_resnames or WATER_RESNAMES)
        self.ligand_node_id_mode = str(ligand_node_id_mode or DEFAULT_LIGAND_NODE_ID_MODE)
        self.ligand_ged_label_mode = str(ligand_ged_label_mode or DEFAULT_LIGAND_GED_LABEL_MODE)
        self.ligand_inventory_source = str(ligand_inventory_source or "input")
        requested_connectivity_source = str(connectivity_source or DEFAULT_CONNECTIVITY_SOURCE).strip().lower()
        # GUI configurations historically wrote ``auto`` although the worker
        # only recognised explicit strings containing ``conect`` or ``rdkit``.
        # The consequence was an empty ligand bond graph and a scattered Step10
        # substrate drawing.  Treat auto/default as the documented, original
        # serial-preserving strategy instead of silently disabling connectivity.
        self.connectivity_source_requested = requested_connectivity_source or "auto"
        self.connectivity_source = (
            DEFAULT_CONNECTIVITY_SOURCE
            if self.connectivity_source_requested in {"", "auto", "default"}
            else self.connectivity_source_requested
        )
        self.allow_cross_residue_rdkit_connectivity = bool(allow_cross_residue_rdkit_connectivity)
        self.plip_stdout = ""
        self.plip_stderr = ""
        self.ligand_atoms: Dict[str, Dict[str, str]] = {}
        self.serial_to_node: Dict[str, str] = {}
        self.node_to_serial: Dict[str, str] = {}
        self.connectivity_edges: List[Dict[str, str]] = []
        self._connectivity_edge_keys: set[Tuple[str, str]] = set()

    def run_plip(self) -> None:
        if self.pdb_path.resolve() != self.input_pdb_copy.resolve():
            shutil.copy2(self.pdb_path, self.input_pdb_copy)

        if not self.run_plip_enabled:
            if not self.report_file.exists():
                raise FileNotFoundError(
                    f"step06.run_plip is false, but report.xml was not found: {self.report_file}"
                )
            return

        # Legacy command shape from interaction_profiler.py v3.4.
        cmd = [self.plip_path, "-f", self.pdb_path.name, "-x", "--breakcomposite"]
        try:
            logger.info("Running PLIP for %s", self.pdb_path.name)
            completed = subprocess.run(
                cmd,
                check=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
            )
            self.plip_stdout = completed.stdout or ""
            self.plip_stderr = completed.stderr or ""
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"PLIP executable not found: {self.plip_path}. Install PLIP or set step06.plip_executable."
            ) from exc
        except subprocess.CalledProcessError as exc:
            self.plip_stdout = exc.stdout or ""
            self.plip_stderr = exc.stderr or ""
            if self.debug:
                (self.work_dir / "plip_stdout.txt").write_text(self.plip_stdout, encoding="utf-8")
                (self.work_dir / "plip_stderr.txt").write_text(self.plip_stderr, encoding="utf-8")
            raise RuntimeError(
                f"PLIP failed for {self.pdb_path.name}. STDERR:\n{self.plip_stderr[:4000]}"
            ) from exc

        if not self.report_file.exists():
            candidates = sorted(self.work_dir.glob("**/report.xml"))
            if candidates:
                self.report_file = candidates[0]
            else:
                raise FileNotFoundError(f"PLIP finished but report.xml was not found in {self.work_dir}")

        if self.debug:
            (self.work_dir / "plip_stdout.txt").write_text(self.plip_stdout, encoding="utf-8")
            (self.work_dir / "plip_stderr.txt").write_text(self.plip_stderr, encoding="utf-8")

    def parse_plip_output(self) -> None:
        tree = ET.parse(self.report_file)
        root = tree.getroot()
        pdbfile_elem = root.find(".//pdbfile")
        if pdbfile_elem is not None and pdbfile_elem.text:
            candidate = self.work_dir / Path(pdbfile_elem.text.strip()).name
            self.processed_pdb_path = candidate if candidate.exists() else self.input_pdb_copy
        else:
            self.processed_pdb_path = self.input_pdb_copy

        if not self.input_pdb_copy.exists():
            raise FileNotFoundError(f"Input PDB copy not found at {self.input_pdb_copy}")
        if self.processed_pdb_path is None or not self.processed_pdb_path.exists():
            raise FileNotFoundError(f"PLIP-processed PDB not found at {self.processed_pdb_path}")

        input_atoms = _parse_pdb_atom_records(self.input_pdb_copy)
        processed_atoms = _parse_pdb_atom_records(self.processed_pdb_path)

        # Keep an atom-name mapping for all atoms PLIP may refer to. Prefer the
        # input PDB for ligand identity, then fill any PLIP-only serials.
        all_atoms = dict(input_atoms)
        for serial, atom in processed_atoms.items():
            all_atoms.setdefault(serial, atom)
        self.atom_mapping = {serial: atom["atom_name"] for serial, atom in all_atoms.items()}

        source = self.ligand_inventory_source.lower()
        if source == "processed":
            inventory_atoms = processed_atoms
            inventory_path = self.processed_pdb_path
        elif source == "union":
            inventory_atoms = dict(input_atoms)
            for serial, atom in processed_atoms.items():
                inventory_atoms.setdefault(serial, atom)
            inventory_path = self.input_pdb_copy
        else:
            inventory_atoms = input_atoms
            inventory_path = self.input_pdb_copy

        selected = {
            serial: atom
            for serial, atom in inventory_atoms.items()
            if _is_selected_ligand_atom(atom, self.include_resnames, self.exclude_resnames)
        }
        if not selected and source != "processed":
            # Conservative fallback for pre-existing PLIP outputs without the
            # copied input PDB content.
            selected = {
                serial: atom
                for serial, atom in processed_atoms.items()
                if _is_selected_ligand_atom(atom, self.include_resnames, self.exclude_resnames)
            }
            inventory_path = self.processed_pdb_path

        self._register_ligand_atoms(selected, inventory_path)
        logger.info(
            "Found %d selected non-water HETATM ligand/cofactor serials for %s.",
            len(self.hetatm_serials),
            self.base_name,
        )

    def _register_ligand_atoms(self, selected: Mapping[str, Mapping[str, str]], source_path: Path) -> None:
        used_node_keys: set[str] = set()
        atom_name_counts: Dict[str, int] = {}
        for _serial, atom in selected.items():
            atom_name = str(atom.get("atom_name", "UNK")) or "UNK"
            atom_name_counts[atom_name] = atom_name_counts.get(atom_name, 0) + 1

        for serial, atom in sorted(selected.items(), key=lambda item: _atom_sort_key(item[1])):
            atom_dict = dict(atom)
            atom_dict["source_pdb_path"] = str(source_path)
            node_key = _make_ligand_node_key(atom_dict, self.ligand_node_id_mode)
            if node_key in used_node_keys:
                # Rare altloc/duplicate safeguard. The GED label can still stay
                # unchanged; only the internal key is made unique.
                node_key = f"{node_key}@serial:{serial}"
            used_node_keys.add(node_key)
            ged_label = _make_ligand_ged_label(
                atom_dict,
                self.ligand_ged_label_mode,
                duplicate_atom_name=(atom_name_counts.get(str(atom_dict.get("atom_name", "UNK")) or "UNK", 0) > 1),
            )
            atom_dict["node_key"] = node_key
            atom_dict["ged_label"] = ged_label
            self.ligand_atoms[serial] = atom_dict
            self.serial_to_node[serial] = node_key
            self.node_to_serial[node_key] = serial
            self.hetatm_serials.add(serial)
            self.atom_mapping[serial] = atom_dict.get("atom_name", "UNK")

    def build_graph_and_interactions(self) -> None:
        if self.processed_pdb_path is None:
            raise RuntimeError("parse_plip_output must be called before build_graph_and_interactions")

        # Add every selected ligand/cofactor atom directly from the PDB inventory.
        # This prevents RDKit or PLIP-processed files from silently dropping atoms,
        # and prevents duplicate atom names from collapsing separate substrates.
        for serial, atom in sorted(self.ligand_atoms.items(), key=lambda item: _atom_sort_key(item[1])):
            node_key = self.serial_to_node[serial]
            self.graph.add_node(
                node_key,
                type="ligand",
                serial=serial,
                ged_label=atom.get("ged_label", atom.get("atom_name", node_key)),
                atom_name=atom.get("atom_name", ""),
                resname=atom.get("resname", ""),
                chain_id=atom.get("chain_id", ""),
                resseq=atom.get("resseq", ""),
                icode=atom.get("icode", ""),
                element=atom.get("element", ""),
            )

        self._build_ligand_connectivity()

        # Legacy PLIP XML interaction parsing with robust ligand-atom resolution.
        tree = ET.parse(self.report_file)
        root = tree.getroot()
        all_binding_sites = root.findall(".//bindingsite")
        if not all_binding_sites:
            logger.warning("No <bindingsite> found in report for %s", self.base_name)
            return

        for site in all_binding_sites:
            interactions_xml = site.find("interactions")
            if interactions_xml is None:
                continue
            for group in interactions_xml:
                for inter in group:
                    try:
                        protein_node = self._protein_node_from_interaction(inter)
                        if not protein_node:
                            continue
                        lig_atom_serial = self._resolve_ligand_serial_from_plip_interaction(inter, site, group.tag)

                        if lig_atom_serial and lig_atom_serial in self.hetatm_serials:
                            ligand_node_key = self.serial_to_node.get(lig_atom_serial)
                            if not ligand_node_key:
                                continue
                            ligand_atom = self.ligand_atoms.get(lig_atom_serial, {})
                            ligand_label = ligand_atom.get("ged_label") or self.atom_mapping.get(lig_atom_serial, "UNK")
                            self.graph.add_node(protein_node, type="protein", ged_label=protein_node)
                            if not self.graph.has_node(ligand_node_key):
                                self.graph.add_node(ligand_node_key, type="ligand", ged_label=ligand_label)
                            self.graph.add_edge(ligand_node_key, protein_node, type=group.tag, edge_kind="interaction")
                            self.interactions.append(
                                {
                                    "type": group.tag,
                                    "ligand_node": ligand_label,
                                    "ligand_node_key": ligand_node_key,
                                    "ligand_node_label": ligand_label,
                                    "protein_node": protein_node,
                                    "ligand_atom_serial": lig_atom_serial,
                                    "source_xml_group": group.tag,
                                    "source_xml_tag": inter.tag,
                                    "ligand_residue_name": ligand_atom.get("resname", ""),
                                    "ligand_chain_id": ligand_atom.get("chain_id", ""),
                                    "ligand_residue_number": ligand_atom.get("resseq", ""),
                                }
                            )
                    except (AttributeError, KeyError, ValueError):
                        continue

    def _protein_node_from_interaction(self, inter: ET.Element) -> str:
        restype = _xml_text(inter, "restype")
        resnr = _xml_text(inter, "resnr")
        reschain = _xml_text(inter, "reschain")
        if not (restype and resnr and reschain):
            return ""
        return f"{restype}{resnr}{reschain}"

    def _resolve_ligand_serial_from_plip_interaction(
        self, inter: ET.Element, site: ET.Element, group_tag: str
    ) -> Optional[str]:
        # First keep the legacy atom-index preference. For donor/acceptor tags,
        # only the tag whose value is one of our selected ligand serials is used.
        candidate_tags = [
            "ligcarbonidx",
            "lig_idx",
            "ligand_idx",
            "lig_atom_idx",
            "donoridx",
            "acceptoridx",
            "metal_idx",
        ]
        candidate_values: List[str] = []
        for tag in candidate_tags:
            value = _xml_text(inter, tag)
            if value:
                candidate_values.append(value)
                if value in self.hetatm_serials:
                    return value

        for idx_elem in inter.findall(".//lig_idx_list/idx"):
            if idx_elem.text and idx_elem.text.strip():
                value = idx_elem.text.strip()
                candidate_values.append(value)
                if value in self.hetatm_serials:
                    return value

        # PLIP metal_complexes can report atom indices that differ from the copied
        # PDB serials by PLIP processing/renumbering. For metals, the ligand
        # residue normally contains one atom, so resolving by ligand residue is
        # safer than silently dropping the interaction.
        resolved = self._resolve_ligand_serial_by_xml_metadata(inter, site, group_tag)
        if resolved:
            return resolved

        # Last conservative fallback: if PLIP gives a coordinate for the ligand
        # atom, match it to the nearest selected ligand atom in the same residue.
        coord = _xml_coord(inter, ["metalcoo", "ligcoo"])
        if coord is not None:
            return self._resolve_ligand_serial_by_coordinate(inter, site, group_tag, coord)
        return None

    def _resolve_ligand_serial_by_xml_metadata(
        self, inter: ET.Element, site: ET.Element, group_tag: str
    ) -> Optional[str]:
        resname, chain_id, resseq = _xml_ligand_residue_identity(inter, site)
        metal_type = (_xml_text(inter, "metal_type") or "").upper()
        group_is_metal = "metal" in group_tag.lower() or inter.tag.lower().startswith("metal")
        matches: List[Tuple[str, Mapping[str, str]]] = []
        for serial, atom in self.ligand_atoms.items():
            if resname and atom.get("resname", "").upper() != resname.upper():
                continue
            if chain_id and atom.get("chain_id", "") != chain_id:
                continue
            if resseq and atom.get("resseq", "") != resseq:
                continue
            matches.append((serial, atom))

        if group_is_metal and matches:
            metal_matches = []
            for serial, atom in matches:
                atom_name = atom.get("atom_name", "").upper().replace(" ", "")
                element = atom.get("element", "").upper()
                if metal_type and (atom_name == metal_type or element == metal_type):
                    metal_matches.append((serial, atom))
                elif element in {"MG", "ZN", "FE", "MN", "CA", "NA", "CU", "CO", "NI"}:
                    metal_matches.append((serial, atom))
            if len(metal_matches) == 1:
                return metal_matches[0][0]

        if len(matches) == 1:
            return matches[0][0]
        return None

    def _resolve_ligand_serial_by_coordinate(
        self, inter: ET.Element, site: ET.Element, group_tag: str, coord: Tuple[float, float, float]
    ) -> Optional[str]:
        resname, chain_id, resseq = _xml_ligand_residue_identity(inter, site)
        best: Tuple[float, str] | None = None
        for serial, atom in self.ligand_atoms.items():
            if resname and atom.get("resname", "").upper() != resname.upper():
                continue
            if chain_id and atom.get("chain_id", "") != chain_id:
                continue
            if resseq and atom.get("resseq", "") != resseq:
                continue
            xyz = _atom_coord(atom)
            if xyz is None:
                continue
            dist2 = sum((xyz[i] - coord[i]) ** 2 for i in range(3))
            if best is None or dist2 < best[0]:
                best = (dist2, serial)
        if best is not None and best[0] <= 0.50 ** 2:
            return best[1]
        return None

    def _build_ligand_connectivity(self) -> None:
        source = self.connectivity_source
        if source in {"none", "off", "disabled"}:
            return

        added_from_conect = False
        # Explicit connectivity means CONECT records in the user/input
        # complex PDB only. PLIP-processed PDB files can contain tool-guessed
        # CONECT records, so they are not treated as authoritative unless the
        # user explicitly asks for processed_conect in config.
        if "conect" in source or source.startswith(("pdb", "input")):
            before = len(self.connectivity_edges)
            self._add_pdb_conect_connectivity(self.input_pdb_copy, "input_pdb_conect", authoritative=True)
            added_from_conect = len(self.connectivity_edges) > before

            if not added_from_conect and "processed_conect" in source:
                for source_label, source_path in _unique_existing_paths(
                    [("processed_pdb_conect", self.processed_pdb_path)]
                ):
                    before = len(self.connectivity_edges)
                    self._add_pdb_conect_connectivity(source_path, source_label, authoritative=False)
                    added_from_conect = added_from_conect or len(self.connectivity_edges) > before

        should_run_rdkit = "rdkit" in source and (not added_from_conect or "always" in source or source.startswith("rdkit"))
        if should_run_rdkit:
            # Use RDKit bond perception on the *input* complex first.
            # The ligand atom inventory and GED nodes are registered from the input
            # PDB by default, so input RDKit bonds preserve the same atom serials.
            # PLIP/OpenBabel processed PDBs may contain chemically useful bond
            # guesses, but they can renumber/reorder atoms; using their serials
            # against input-derived nodes caused shifted phosphate bonds such as
            # O3-O4 instead of P1-O4. Therefore processed RDKit connectivity is
            # only used when explicitly requested or when the source string contains
            # "processed_rdkit".
            before_rdkit = len(self.connectivity_edges)
            rdkit_sources: List[Tuple[str, Optional[Path]]] = [("rdkit_input_pdb", self.input_pdb_copy)]
            if "processed_rdkit" in source or "processed" in source:
                rdkit_sources.append(("rdkit_processed_pdb", self.processed_pdb_path))
            for source_label, source_path in _unique_existing_paths(rdkit_sources):
                before = len(self.connectivity_edges)
                self._add_rdkit_connectivity(source_path, source_label)
                if len(self.connectivity_edges) > before and "always" not in source:
                    break
            if self.debug and len(self.connectivity_edges) == before_rdkit:
                logger.warning("No ligand connectivity edges recovered by RDKit fallback for %s", self.base_name)

    def _add_pdb_conect_connectivity(self, pdb_path: Path, source_label: str, authoritative: bool = True) -> None:
        edge_type = "input_pdb_conect_bond" if authoritative else "processed_pdb_conect_bond"
        for s1, s2 in _parse_pdb_conect_edges(pdb_path):
            if s1 in self.hetatm_serials and s2 in self.hetatm_serials:
                self._record_ligand_connectivity_edge(s1, s2, edge_type, source_label, pdb_path)

    def _record_ligand_connectivity_edge(
        self, s1: str, s2: str, edge_type: str, source_label: str, source_path: Path
    ) -> None:
        n1 = self.serial_to_node.get(s1)
        n2 = self.serial_to_node.get(s2)
        if not (n1 and n2) or n1 == n2:
            return
        edge_key = tuple(sorted((n1, n2)))
        if edge_key in self._connectivity_edge_keys:
            return
        self._connectivity_edge_keys.add(edge_key)
        self.graph.add_edge(n1, n2, type=edge_type, edge_kind="ligand_connectivity")
        a1 = self.ligand_atoms.get(s1, {})
        a2 = self.ligand_atoms.get(s2, {})
        self.connectivity_edges.append(
            {
                "ligand_node_key_1": n1,
                "ligand_node_key_2": n2,
                "ligand_node_label_1": a1.get("ged_label", self.atom_mapping.get(s1, "UNK")),
                "ligand_node_label_2": a2.get("ged_label", self.atom_mapping.get(s2, "UNK")),
                "ligand_atom_serial_1": s1,
                "ligand_atom_serial_2": s2,
                "edge_type": edge_type,
                "source": source_label,
                "source_pdb_path": str(source_path),
            }
        )

    def _add_rdkit_connectivity(self, pdb_path: Path, source_label: str) -> None:
        if Chem is None:
            raise RuntimeError(
                "RDKit is required only when Step06 connectivity_source uses an RDKit fallback. "
                f"Original import error: {_RDKIT_IMPORT_ERROR}"
            )
        ligand_mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=False)
        if not ligand_mol:
            return
        for bond in ligand_mol.GetBonds():
            s1_info = bond.GetBeginAtom().GetPDBResidueInfo()
            s2_info = bond.GetEndAtom().GetPDBResidueInfo()
            if not (s1_info and s2_info):
                continue
            s1, s2 = str(s1_info.GetSerialNumber()), str(s2_info.GetSerialNumber())
            if s1 not in self.hetatm_serials or s2 not in self.hetatm_serials:
                continue
            if not self.allow_cross_residue_rdkit_connectivity:
                a1 = self.ligand_atoms.get(s1, {})
                a2 = self.ligand_atoms.get(s2, {})
                if not _same_ligand_residue(a1, a2):
                    # RDKit can guess chemically impossible cross-substrate or
                    # metal-coordination bonds from distance alone. Do not let
                    # those guessed bonds change GED classification by default.
                    continue
            self._record_ligand_connectivity_edge(s1, s2, "rdkit_same_residue_bond", source_label, pdb_path)

    def ligand_atom_inventory_rows(self, standard_structure_id: str, complex_pdb_path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for serial, atom in sorted(self.ligand_atoms.items(), key=lambda item: _atom_sort_key(item[1])):
            rows.append(
                asdict(
                    LigandAtomRecord(
                        standard_structure_id=standard_structure_id,
                        complex_pdb_path=complex_pdb_path,
                        source_pdb_path=atom.get("source_pdb_path", ""),
                        atom_serial=serial,
                        atom_name=atom.get("atom_name", ""),
                        residue_name=atom.get("resname", ""),
                        chain_id=atom.get("chain_id", ""),
                        residue_number=atom.get("resseq", ""),
                        insertion_code=atom.get("icode", ""),
                        element=atom.get("element", ""),
                        node_key=atom.get("node_key", ""),
                        ged_label=atom.get("ged_label", ""),
                    )
                )
            )
        return rows

    def connectivity_inventory_rows(self, standard_structure_id: str, complex_pdb_path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for edge in self.connectivity_edges:
            rows.append(
                asdict(
                    ConnectivityRecord(
                        standard_structure_id=standard_structure_id,
                        complex_pdb_path=complex_pdb_path,
                        source_pdb_path=edge.get("source_pdb_path", ""),
                        ligand_node_key_1=edge.get("ligand_node_key_1", ""),
                        ligand_node_key_2=edge.get("ligand_node_key_2", ""),
                        ligand_node_label_1=edge.get("ligand_node_label_1", ""),
                        ligand_node_label_2=edge.get("ligand_node_label_2", ""),
                        ligand_atom_serial_1=edge.get("ligand_atom_serial_1", ""),
                        ligand_atom_serial_2=edge.get("ligand_atom_serial_2", ""),
                        edge_type=edge.get("edge_type", ""),
                        source=edge.get("source", ""),
                    )
                )
            )
        return rows

    def write_debug_files(
        self,
        global_mapping: Optional[Mapping[str, int]] = None,
        global_label_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not self.save_intermediate_files and not self.debug:
            return
        logger.info("Writing Step06 intermediate files for %s", self.base_name)

        with (self.work_dir / "interaction_output.txt").open("w", encoding="utf-8") as handle:
            for inter in self.interactions:
                handle.write(
                    f"Type: {inter['type']}\tLigand: {inter['ligand_node']}\tProtein: {inter['protein_node']}"
                    f"\tLigandSerial: {inter.get('ligand_atom_serial', '')}"
                    f"\tLigandNodeKey: {inter.get('ligand_node_key', '')}\n"
                )

        pd.DataFrame(self.interactions).to_csv(self.work_dir / "interaction_output.csv", index=False)

        ligand_nodes = _sorted_ligand_nodes(self.graph)
        with (self.work_dir / "substrate_connectivity.txt").open("w", encoding="utf-8") as handle:
            for node in ligand_nodes:
                data = self.graph.nodes[node]
                label = data.get("ged_label", node)
                serial = data.get("serial", "")
                neighbors = []
                for neighbor in self.graph.neighbors(node):
                    ndata = self.graph.nodes[neighbor]
                    if ndata.get("type") == "ligand":
                        neighbors.append(
                            f"{ndata.get('ged_label', neighbor)}[{neighbor};serial={ndata.get('serial', '')}]"
                        )
                handle.write(f"{label}[{node};serial={serial}] -> {', '.join(sorted(neighbors))}\n")

        pd.DataFrame(self.ligand_atom_inventory_rows(self.base_name, str(self.pdb_path))).to_csv(
            self.work_dir / "ligand_atom_inventory.csv", index=False
        )
        pd.DataFrame(self.connectivity_inventory_rows(self.base_name, str(self.pdb_path))).to_csv(
            self.work_dir / "substrate_connectivity_edges.csv", index=False
        )

        if global_mapping:
            protein_nodes = _sorted_protein_nodes(self.graph)
            nodes = ligand_nodes + protein_nodes
            node_map = {node: i for i, node in enumerate(nodes)}

            with (self.work_dir / "final_nodes.txt").open("w", encoding="utf-8") as handle:
                handle.write("LocalIndex\tNodeKey\tGED_Label\tType\tSerial\tResidue\n")
                for node in nodes:
                    data = self.graph.nodes[node]
                    residue = ""
                    if data.get("type") == "ligand":
                        residue = f"{data.get('resname', '')}:{data.get('chain_id', '')}:{data.get('resseq', '')}{data.get('icode', '')}"
                    handle.write(
                        f"{node_map[node]}\t{node}\t{data.get('ged_label', node)}\t{data.get('type', '')}"
                        f"\t{data.get('serial', '')}\t{residue}\n"
                    )

            with (self.work_dir / "final_edges.txt").open("w", encoding="utf-8") as handle:
                handle.write("LocalIndex1\tLocalIndex2\tType\tNodeKey1\tNodeKey2\n")
                for u, v, data in sorted(
                    self.graph.edges(data=True), key=lambda item: (node_map.get(item[0], 10**9), node_map.get(item[1], 10**9))
                ):
                    handle.write(
                        f"{node_map.get(u, -1)}\t{node_map.get(v, -1)}\t{data.get('type', 'unknown')}\t{u}\t{v}\n"
                    )

            with (self.work_dir / "global_final_nodes.txt").open("w", encoding="utf-8") as handle:
                handle.write("GlobalNodeIndex\tNodeKey\tGED_Label\tPresentInThisGraph\n")
                for node, index in sorted(global_mapping.items(), key=lambda item: item[1]):
                    # global_mapping is the union blueprint across all successful
                    # structures. Therefore many global nodes may be absent from the
                    # current per-structure graph. Do not index self.graph.nodes[node]
                    # unless the node is actually present.
                    present = node in self.graph
                    if global_label_map and node in global_label_map:
                        label = global_label_map[node]
                    elif present:
                        label = self.graph.nodes[node].get("ged_label", node)
                    else:
                        label = node
                    handle.write(f"{index}\t{node}\t{label}\t{int(present)}\n")

            with (self.work_dir / "global_final_edges.txt").open("w", encoding="utf-8") as handle:
                handle.write("GlobalIndex1\tGlobalIndex2\tType\tNodeKey1\tNodeKey2\n")
                for u, v, data in self.graph.edges(data=True):
                    if u in global_mapping and v in global_mapping:
                        handle.write(
                            f"{global_mapping[u]}\t{global_mapping[v]}\t{data.get('type', 'unknown')}\t{u}\t{v}\n"
                        )
        logger.info("Finished writing Step06 intermediate files for %s", self.base_name)

    def process(self) -> nx.Graph:
        self.run_plip()
        self.parse_plip_output()
        self.build_graph_and_interactions()
        logger.info(
            "Built graph for %s with %d nodes and %d edges.",
            self.base_name,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph


def create_global_node_mapping(graphs: Sequence[nx.Graph]) -> Tuple[Dict[str, int], Dict[str, str], List[str]]:
    """Create the global blueprint: mapping, labels, and complete node list."""
    all_nodes = set().union(*(g.nodes for g in graphs))
    node_attrs: Dict[str, Dict[str, Any]] = {}
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            if node not in node_attrs:
                node_attrs[node] = dict(data)
            else:
                for key, value in data.items():
                    if key not in node_attrs[node] or node_attrs[node][key] in (None, ""):
                        node_attrs[node][key] = value

    protein_nodes = sorted(
        [n for n in all_nodes if _node_is_protein(n, node_attrs.get(n, {}))],
        key=lambda x: _protein_sort_key(x),
    )
    ligand_nodes = sorted(
        [n for n in all_nodes if n not in protein_nodes],
        key=lambda x: _ligand_global_sort_key(x, node_attrs.get(x, {})),
    )

    global_nodes_list = ligand_nodes + protein_nodes
    global_mapping = {node: i for i, node in enumerate(global_nodes_list)}

    protein_label_map = {node: str(i + 1) for i, node in enumerate(protein_nodes)}
    global_label_map = {
        node: str(node_attrs.get(node, {}).get("ged_label") or node_attrs.get(node, {}).get("atom_name") or node)
        for node in ligand_nodes
    }
    global_label_map.update(protein_label_map)
    return global_mapping, global_label_map, global_nodes_list


def convert_graph_to_ged_format(
    graph: nx.Graph,
    all_global_nodes: Sequence[str],
    global_mapping: Mapping[str, int],
    global_label_map: Mapping[str, str],
    graph_name: str,
) -> str:
    """Generate a GED file string using the global blueprint format."""
    lines = [f"t # {graph_name}"]
    node_lines = [f"v {global_mapping[node]} {global_label_map[node]}" for node in all_global_nodes]
    lines.extend(node_lines)

    unique_edges = set()
    for u, v in graph.edges:
        if u in global_mapping and v in global_mapping:
            u_idx, v_idx = sorted([global_mapping[u], global_mapping[v]])
            unique_edges.add(f"e {u_idx} {v_idx} 1")
    edge_lines = sorted(list(unique_edges), key=lambda x: (int(x.split()[1]), int(x.split()[2])))
    lines.extend(edge_lines)
    return "\n".join(lines) + "\n"


def load_config(config_path: Path | str) -> dict:
    return load_yaml_config(config_path)


def _target_id(config: Mapping[str, Any]) -> str:
    return str(config.get("project", {}).get("target_id", "target"))


def _output_root(config: Mapping[str, Any]) -> Path:
    project = config.get("project", {}) or {}
    return Path(project.get("output_root", f"results/{_target_id(config)}"))


def _norm_set(value: Optional[Iterable[Any]]) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    return {str(x).strip().upper() for x in value if str(x).strip()}


def _safe_int(text: Any, default: int = 10**9) -> int:
    try:
        return int(str(text).strip())
    except Exception:
        return default


def _infer_element(atom_name: str, element: str = "") -> str:
    if element.strip():
        return element.strip().upper()
    cleaned = re.sub(r"[^A-Za-z]", "", atom_name.strip())
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and cleaned[:2].upper() in {"MG", "ZN", "FE", "MN", "CA", "NA", "CL", "CU", "CO", "NI"}:
        return cleaned[:2].upper()
    return cleaned[0].upper()


def _parse_pdb_atom_records(pdb_path: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    if not pdb_path.exists():
        return records
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record_name = line[0:6].strip().upper()
            if record_name not in {"ATOM", "HETATM"}:
                continue
            serial = line[6:11].strip()
            if not serial:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            element = _infer_element(atom_name, line[76:78].strip() if len(line) >= 78 else "")
            x = line[30:38].strip() if len(line) >= 38 else ""
            y = line[38:46].strip() if len(line) >= 46 else ""
            z = line[46:54].strip() if len(line) >= 54 else ""
            records[serial] = {
                "record_name": record_name,
                "serial": serial,
                "atom_name": atom_name,
                "resname": resname,
                "chain_id": chain_id,
                "resseq": resseq,
                "icode": icode,
                "element": element,
                "x": x,
                "y": y,
                "z": z,
            }
    return records


def _parse_pdb_conect_edges(pdb_path: Path) -> List[Tuple[str, str]]:
    edges: set[Tuple[str, str]] = set()
    if not pdb_path.exists():
        return []
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("CONECT"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            source = parts[1]
            for target in parts[2:]:
                if source == target:
                    continue
                edges.add(tuple(sorted((source, target), key=lambda x: _safe_int(x))))
    return sorted(edges, key=lambda pair: (_safe_int(pair[0]), _safe_int(pair[1])))


def _same_ligand_residue(a1: Mapping[str, str], a2: Mapping[str, str]) -> bool:
    return (
        str(a1.get("resname", "")) == str(a2.get("resname", ""))
        and str(a1.get("chain_id", "")) == str(a2.get("chain_id", ""))
        and str(a1.get("resseq", "")) == str(a2.get("resseq", ""))
        and str(a1.get("icode", "")) == str(a2.get("icode", ""))
    )


def _xml_text(root: ET.Element, path: str) -> str:
    elem = root.find(path)
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _first_xml_text(root: ET.Element, paths: Sequence[str]) -> str:
    for path in paths:
        value = _xml_text(root, path)
        if value:
            return value
    return ""


def _xml_ligand_residue_identity(inter: ET.Element, site: ET.Element) -> Tuple[str, str, str]:
    resname = _first_xml_text(inter, ["restype_lig", "ligtype", "ligand_type", "hetid"])
    chain_id = _first_xml_text(inter, ["reschain_lig", "ligchain", "chain_lig"])
    resseq = _first_xml_text(inter, ["resnr_lig", "ligposition", "position_lig"])
    if not resname:
        resname = _first_xml_text(site, [".//identifiers/hetid", ".//hetid"])
    if not chain_id:
        chain_id = _first_xml_text(site, [".//identifiers/chain", ".//chain"])
    if not resseq:
        resseq = _first_xml_text(site, [".//identifiers/position", ".//position"])
    return resname.upper(), chain_id, resseq


def _xml_coord(inter: ET.Element, tags: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    for tag in tags:
        elem = inter.find(tag)
        if elem is None:
            continue
        # PLIP XML versions differ: some use text like "x, y, z", some use
        # nested <x>/<y>/<z> fields. Support both without changing rules.
        if elem.text and elem.text.strip():
            numbers = re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", elem.text)
            if len(numbers) >= 3:
                return (float(numbers[0]), float(numbers[1]), float(numbers[2]))
        nested = [_xml_text(elem, key) for key in ["x", "y", "z"]]
        if all(nested):
            try:
                return (float(nested[0]), float(nested[1]), float(nested[2]))
            except ValueError:
                pass
    return None


def _atom_coord(atom: Mapping[str, str]) -> Optional[Tuple[float, float, float]]:
    try:
        return (float(atom.get("x", "")), float(atom.get("y", "")), float(atom.get("z", "")))
    except Exception:
        return None


def _is_selected_ligand_atom(atom: Mapping[str, str], include_resnames: set[str], exclude_resnames: set[str]) -> bool:
    if str(atom.get("record_name", "")).upper() != "HETATM":
        return False
    resname = str(atom.get("resname", "")).upper()
    if resname in exclude_resnames:
        return False
    if include_resnames and resname not in include_resnames:
        return False
    return True


def _atom_sort_key(atom: Mapping[str, str]) -> Tuple[str, str, int, str, str, int]:
    return (
        str(atom.get("resname", "")),
        str(atom.get("chain_id", "")),
        _safe_int(atom.get("resseq", "")),
        str(atom.get("icode", "")),
        str(atom.get("atom_name", "")),
        _safe_int(atom.get("serial", "")),
    )


def _make_ligand_node_key(atom: Mapping[str, str], mode: str) -> str:
    atom_name = str(atom.get("atom_name", "UNK")) or "UNK"
    serial = str(atom.get("serial", ""))
    resname = str(atom.get("resname", "UNK")) or "UNK"
    chain = str(atom.get("chain_id", "_")) or "_"
    resseq = str(atom.get("resseq", "")) or "?"
    icode = str(atom.get("icode", ""))
    mode = mode.lower()
    if mode in {"atom_name", "legacy_atom_name"}:
        return atom_name
    if mode == "serial_atom_name":
        return f"{atom_name}@serial:{serial}"
    if mode == "residue_atom_name_serial":
        return f"{resname}:{chain}:{resseq}{icode}:{atom_name}@serial:{serial}"
    # Default: stable per-residue atom identity. This fixes multi-ligand atom-name collisions.
    return f"{resname}:{chain}:{resseq}{icode}:{atom_name}"


def _sanitize_ged_token(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "_"
    # GED labels are whitespace-delimited. Keep labels compact and parser-safe.
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.:-]", "_", text)
    return text


def _residue_atom_label(atom: Mapping[str, str]) -> str:
    atom_name = _sanitize_ged_token(atom.get("atom_name", "UNK"))
    resname = _sanitize_ged_token(atom.get("resname", "UNK"))
    chain = _sanitize_ged_token(atom.get("chain_id", "_"))
    resseq = _sanitize_ged_token(atom.get("resseq", "?"))
    icode = _sanitize_ged_token(atom.get("icode", ""))
    if icode and icode != "_":
        resseq = f"{resseq}{icode}"
    return f"{resname}_{chain}{resseq}_{atom_name}"


def _make_ligand_ged_label(atom: Mapping[str, str], mode: str, duplicate_atom_name: bool = False) -> str:
    atom_name = str(atom.get("atom_name", "UNK")) or "UNK"
    mode = mode.lower()
    if mode == "node_key":
        return _sanitize_ged_token(_make_ligand_node_key(atom, DEFAULT_LIGAND_NODE_ID_MODE))
    if mode in {"residue_atom_name", "unique_residue_atom_name"}:
        return _residue_atom_label(atom)
    if mode in {"atom_name_with_residue_for_duplicates", "duplicate_aware_atom_name"}:
        return _residue_atom_label(atom) if duplicate_atom_name else _sanitize_ged_token(atom_name)
    if mode == "element":
        return _sanitize_ged_token(str(atom.get("element", "")) or atom_name)
    # Legacy mode: exactly the readable PDB atom name. This can produce duplicate
    # GED labels for two substrates that both contain atoms named C8/O2/P1/etc.
    return _sanitize_ged_token(atom_name)


def _unique_existing_paths(items: Sequence[Tuple[str, Optional[Path]]]) -> List[Tuple[str, Path]]:
    seen: set[Path] = set()
    output: List[Tuple[str, Path]] = []
    for label, path in items:
        if path is None or not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append((label, path))
    return output


def _node_is_protein(node: str, attrs: Mapping[str, Any]) -> bool:
    if attrs.get("type") == "protein":
        return True
    return bool(re.match(r"^[A-Z]{3}\d+[A-Za-z]$", str(node)))


def _protein_sort_key(node: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", str(node))
    return (_safe_int(match.group(1) if match else ""), str(node))


def _ligand_global_sort_key(node: str, attrs: Mapping[str, Any]) -> Tuple[str, str, str, int, str, str, int, str]:
    return (
        str(attrs.get("ged_label", attrs.get("atom_name", node))),
        str(attrs.get("resname", "")),
        str(attrs.get("chain_id", "")),
        _safe_int(attrs.get("resseq", "")),
        str(attrs.get("icode", "")),
        str(attrs.get("atom_name", "")),
        _safe_int(attrs.get("serial", "")),
        str(node),
    )


def _sorted_ligand_nodes(graph: nx.Graph) -> List[str]:
    return sorted(
        [n for n, d in graph.nodes(data=True) if d.get("type") == "ligand"],
        key=lambda n: _ligand_global_sort_key(n, graph.nodes[n]),
    )


def _sorted_protein_nodes(graph: nx.Graph) -> List[str]:
    return sorted(
        [n for n, d in graph.nodes(data=True) if d.get("type") == "protein"],
        key=lambda n: _protein_sort_key(n),
    )


def resolve_step06_input_table(config: Mapping[str, Any]) -> Path:
    target_id = _target_id(config)
    step06 = config.get("step06", {}) or {}
    requested = step06.get("input_complex_table", "auto")
    if requested and requested != "auto":
        return Path(requested)

    root = _output_root(config)
    candidates = [
        # Step05 is the last structural QC step. Auto mode must consume only
        # complexes that passed catalytic-region/pocket pLDDT QC when that table
        # exists.  Older builds missed the current directory name and fell back
        # to Step03, so Step05-failed structures were still sent to PLIP.
        root / "04_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step04_passed_complex_index.csv",
        root / "04_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        # Historical output locations remain readable.
        root / "05_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        # Backward-compatible legacy Step05 directory names.
        root / "05_pocket_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        root / "05_pocket_plddt_filter" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        root / "04_ligand_atom_trimming" / "tables" / f"{target_id}_step04_trimmed_complex_index.csv",
        root / "03_clash_filter" / "tables" / f"{target_id}_step03_passed_complex_index.csv",
        root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_complex_index.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find Step06 input complex table. Run Step05/Step04/Step03 first or set "
        "step06.input_complex_table in config.yaml. Checked: "
        + ", ".join(str(x) for x in candidates)
    )


def _infer_complex_path_column(df: pd.DataFrame, configured: str = "auto") -> str:
    if configured and configured != "auto":
        if configured not in df.columns:
            raise KeyError(f"Configured complex path column '{configured}' not found in input table.")
        return configured
    candidates = [
        "complex_pdb_path",
        "trimmed_complex_pdb_path",
        "output_complex_pdb",
        "complex_path",
        "pdb_path",
        "path",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"Cannot infer complex PDB path column. Available columns: {list(df.columns)}")


def _infer_structure_id(row: pd.Series, pdb_path: Path) -> str:
    for column in ["standard_structure_id", "structure_id", "id", "model_id"]:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    stem = pdb_path.stem
    for suffix in ["_trimmed_complex", "_complex", "_trimmed"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _resolve_existing_path(path_text: str, config_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [path, config_path.parent / path, config_path.parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def run_step06(
    config_path: Path | str,
    max_structures: Optional[int] = None,
    clean_output: Optional[bool] = None,
) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    target_id = _target_id(config)
    root = _output_root(config)
    step06 = config.get("step06", {}) or {}

    if step06.get("enabled", True) is False:
        return {
            "target_id": target_id,
            "status": "skipped",
            "message": "step06.enabled is false. PLIP interaction profiling was not run.",
        }

    mode = str(step06.get("reproducibility_mode", LEGACY_CLASSIFICATION_MODE))
    if mode != LEGACY_CLASSIFICATION_MODE:
        raise ValueError(
            f"Unsupported step06.reproducibility_mode={mode!r}. "
            f"This version intentionally locks Step06 to {LEGACY_CLASSIFICATION_MODE!r}."
        )

    output_dir = root / public_output_subdir(
        step06.get("output_subdir"), "05_plip_interaction_graphs"
    )

    # Reproducibility safeguard:
    # Step06 GED files are generated from a global node blueprint. If old files
    # from a previous --max-structures test are left in the output directory,
    # users can easily mistake stale GED/debug files for current batch results.
    # Therefore, when PLIP is being run normally, Step06 starts from a clean
    # output directory by default. Set step06.clean_output: false or pass
    # --no-clean to keep existing outputs for manual/resume workflows.
    default_clean_output = bool(step06.get("run_plip", True))
    clean_requested = default_clean_output if clean_output is None else bool(clean_output)
    clean_requested = bool(step06.get("clean_output", clean_requested)) if clean_output is None else clean_requested
    if clean_requested and output_dir.exists():
        shutil.rmtree(output_dir)

    plip_dir = output_dir / "plip_outputs"
    ged_dir = output_dir / "ged_input_files"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    debug_dir = output_dir / "debug"
    for directory in [plip_dir, ged_dir, tables_dir, reports_dir, debug_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    input_table = resolve_step06_input_table(config)
    df = pd.read_csv(input_table)
    if max_structures is not None and max_structures > 0:
        df = df.head(max_structures)
    path_col = _infer_complex_path_column(df, step06.get("input_complex_column", "auto"))

    ligand_cfg = step06.get("ligand_selection", {}) or {}
    node_cfg = step06.get("node_identity", {}) or {}
    connectivity_cfg = step06.get("connectivity", {}) or {}
    requested_connectivity_source = str(
        connectivity_cfg.get("source", DEFAULT_CONNECTIVITY_SOURCE)
    ).strip().lower() or "auto"
    resolved_connectivity_source = (
        DEFAULT_CONNECTIVITY_SOURCE
        if requested_connectivity_source in {"auto", "default"}
        else requested_connectivity_source
    )
    processors: List[SingleComplexProcessor] = []
    graph_summaries: List[GraphSummary] = []
    interaction_rows: List[Dict[str, Any]] = []
    ligand_atom_rows: List[Dict[str, Any]] = []
    connectivity_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        pdb_path = _resolve_existing_path(str(row[path_col]), config_path)
        sid = _infer_structure_id(row, pdb_path)
        ged_path = ged_dir / f"{sid}_final.txt"
        try:
            if not pdb_path.exists():
                raise FileNotFoundError(f"Input complex PDB not found: {pdb_path}")
            processor = SingleComplexProcessor(
                pdb_path=pdb_path,
                output_dir=plip_dir,
                structure_id=sid,
                plip_path=str(step06.get("plip_executable", step06.get("plip_path", "plip"))),
                run_plip=bool(step06.get("run_plip", True)),
                debug=bool(step06.get("debug", False)),
                save_intermediate_files=bool(step06.get("save_intermediate_files", True)),
                include_resnames=ligand_cfg.get("include_resnames", None),
                exclude_resnames=ligand_cfg.get("exclude_resnames", ["HOH", "WAT"]),
                ligand_node_id_mode=str(node_cfg.get("ligand_node_id", DEFAULT_LIGAND_NODE_ID_MODE)),
                ligand_ged_label_mode=str(node_cfg.get("ligand_ged_label", DEFAULT_LIGAND_GED_LABEL_MODE)),
                ligand_inventory_source=str(ligand_cfg.get("inventory_source", "input")),
                connectivity_source=resolved_connectivity_source,
                allow_cross_residue_rdkit_connectivity=bool(
                    connectivity_cfg.get(
                        "allow_cross_residue_rdkit_connectivity",
                        DEFAULT_ALLOW_CROSS_RESIDUE_RDKIT_CONNECTIVITY,
                    )
                ),
            )
            processor.process()
            processors.append(processor)

            # Write per-structure inspection files immediately after each
            # successful complex is processed. Final GED/global mapping files are
            # still written after all successful structures are known, but these
            # files make long batch runs inspectable while they are still running:
            # interaction_output.txt, interaction_output.csv,
            # substrate_connectivity.txt, substrate_connectivity_edges.csv, and
            # ligand_atom_inventory.csv.
            processor.write_debug_files(global_mapping=None, global_label_map=None)

            for inter in processor.interactions:
                interaction_rows.append(
                    asdict(
                        InteractionRecord(
                            standard_structure_id=sid,
                            complex_pdb_path=str(pdb_path),
                            plip_report_path=str(processor.report_file),
                            interaction_type=inter.get("type", ""),
                            ligand_node=inter.get("ligand_node", ""),
                            protein_node=inter.get("protein_node", ""),
                            ligand_atom_serial=inter.get("ligand_atom_serial", ""),
                            source_xml_group=inter.get("source_xml_group", ""),
                            source_xml_tag=inter.get("source_xml_tag", ""),
                            ligand_node_key=inter.get("ligand_node_key", ""),
                            ligand_node_label=inter.get("ligand_node_label", ""),
                            ligand_residue_name=inter.get("ligand_residue_name", ""),
                            ligand_chain_id=inter.get("ligand_chain_id", ""),
                            ligand_residue_number=inter.get("ligand_residue_number", ""),
                        )
                    )
                )

            ligand_atom_rows.extend(processor.ligand_atom_inventory_rows(sid, str(pdb_path)))
            connectivity_rows.extend(processor.connectivity_inventory_rows(sid, str(pdb_path)))

            graph_summaries.append(
                GraphSummary(
                    standard_structure_id=sid,
                    complex_pdb_path=str(pdb_path),
                    plip_work_dir=str(processor.work_dir),
                    plip_report_path=str(processor.report_file),
                    ged_file_path=str(ged_path),
                    n_ligand_atoms=len(processor.ligand_atoms),
                    n_ligand_nodes=sum(1 for _n, d in processor.graph.nodes(data=True) if d.get("type") == "ligand"),
                    n_protein_nodes=sum(1 for _n, d in processor.graph.nodes(data=True) if d.get("type") == "protein"),
                    n_total_nodes=processor.graph.number_of_nodes(),
                    n_interaction_records=len(processor.interactions),
                    n_ligand_connectivity_edges=len(processor.connectivity_edges),
                    n_total_edges=processor.graph.number_of_edges(),
                    status="success",
                )
            )
        except Exception as exc:
            # A failed structure may still have useful PLIP files in its work
            # directory. Always write a human-readable failure marker there so a
            # batch run never leaves a silent, half-populated directory.
            failure_work_dir = plip_dir / sid
            failure_work_dir.mkdir(parents=True, exist_ok=True)
            (failure_work_dir / "failure_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            (failure_work_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")

            base = row.to_dict()
            base.update(
                {
                    "standard_structure_id": sid,
                    "complex_pdb_path": str(pdb_path),
                    "plip_work_dir": str(failure_work_dir),
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            failed_rows.append(base)
            graph_summaries.append(
                GraphSummary(
                    standard_structure_id=sid,
                    complex_pdb_path=str(pdb_path),
                    plip_work_dir=str(plip_dir / sid),
                    plip_report_path="",
                    ged_file_path="",
                    n_ligand_atoms=0,
                    n_ligand_nodes=0,
                    n_protein_nodes=0,
                    n_total_nodes=0,
                    n_interaction_records=0,
                    n_ligand_connectivity_edges=0,
                    n_total_edges=0,
                    status="failed",
                    error_message=str(exc),
                )
            )

    if processors:
        graphs = [p.graph for p in processors]
        global_mapping, global_label_map, global_nodes_list = create_global_node_mapping(graphs)
        for processor in processors:
            ged_content = convert_graph_to_ged_format(
                processor.graph,
                global_nodes_list,
                global_mapping,
                global_label_map,
                processor.base_name,
            )
            (ged_dir / f"{processor.base_name}_final.txt").write_text(ged_content, encoding="utf-8")
            processor.write_debug_files(global_mapping=global_mapping, global_label_map=global_label_map)
    else:
        global_mapping, global_label_map, global_nodes_list = {}, {}, []

    summary_df = pd.DataFrame([asdict(x) for x in graph_summaries])
    interactions_df = pd.DataFrame(interaction_rows)
    ligand_atoms_df = pd.DataFrame(ligand_atom_rows)
    connectivity_df = pd.DataFrame(connectivity_rows)
    failed_df = pd.DataFrame(failed_rows)
    global_nodes_df = pd.DataFrame(
        [
            {
                "global_node_index": global_mapping[node],
                "node_key": node,
                "ged_label": global_label_map.get(node, ""),
            }
            for node in global_nodes_list
        ]
    )

    graph_summary_path = tables_dir / f"{target_id}_step05_graph_summary.csv"
    interaction_edges_path = tables_dir / f"{target_id}_step05_interaction_edges.csv"
    ligand_atoms_path = tables_dir / f"{target_id}_step05_ligand_atom_inventory.csv"
    connectivity_path = tables_dir / f"{target_id}_step05_ligand_connectivity_edges.csv"
    global_nodes_path = tables_dir / f"{target_id}_step05_global_node_mapping.csv"
    graph_index_path = tables_dir / f"{target_id}_step05_graph_index.csv"
    failed_path = tables_dir / f"{target_id}_step05_failed_structures.csv"

    summary_df.to_csv(graph_summary_path, index=False)
    interactions_df.to_csv(interaction_edges_path, index=False)
    ligand_atoms_df.to_csv(ligand_atoms_path, index=False)
    connectivity_df.to_csv(connectivity_path, index=False)
    global_nodes_df.to_csv(global_nodes_path, index=False)
    summary_df[summary_df["status"] == "success"].to_csv(graph_index_path, index=False)
    failed_df.to_csv(failed_path, index=False)

    report = {
        "target_id": target_id,
        "status": "success",
        "config_path": str(config_path),
        "reproducibility_mode": mode,
        "input_complex_table": str(input_table),
        "input_complex_column": path_col,
        "output_dir": str(output_dir),
        "n_input_complexes": int(len(df)),
        "n_success": int(sum(1 for x in graph_summaries if x.status == "success")),
        "n_failed": int(sum(1 for x in graph_summaries if x.status != "success")),
        "n_interaction_records": int(len(interactions_df)),
        "n_interaction_edges": int(len(interactions_df)),
        "n_ligand_atoms": int(len(ligand_atoms_df)),
        "n_ligand_connectivity_edges": int(len(connectivity_df)),
        "n_global_nodes": int(len(global_nodes_list)),
        "plip_executable": str(step06.get("plip_executable", step06.get("plip_path", "plip"))),
        "run_plip": bool(step06.get("run_plip", True)),
        "clean_output": clean_requested,
        "ligand_node_id_mode": str(node_cfg.get("ligand_node_id", DEFAULT_LIGAND_NODE_ID_MODE)),
        "ligand_ged_label_mode": str(node_cfg.get("ligand_ged_label", DEFAULT_LIGAND_GED_LABEL_MODE)),
        "connectivity_source_requested": requested_connectivity_source,
        "connectivity_source": resolved_connectivity_source,
        "allow_cross_residue_rdkit_connectivity": bool(
            connectivity_cfg.get(
                "allow_cross_residue_rdkit_connectivity",
                DEFAULT_ALLOW_CROSS_RESIDUE_RDKIT_CONNECTIVITY,
            )
        ),
        "files": {
            "graph_summary": str(graph_summary_path),
            "interaction_edges": str(interaction_edges_path),
            "ligand_atom_inventory": str(ligand_atoms_path),
            "ligand_connectivity_edges": str(connectivity_path),
            "global_node_mapping": str(global_nodes_path),
            "graph_index": str(graph_index_path),
            "failed": str(failed_path),
            "ged_input_dir": str(ged_dir),
            "plip_output_dir": str(plip_dir),
        },
    }

    (reports_dir / f"{target_id}_step05_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# Step06 PLIP interaction graph summary: {target_id}",
        "",
        f"- Reproducibility mode: `{mode}`",
        f"- Input complexes: {len(df)}",
        f"- Successfully profiled: {report['n_success']}",
        f"- Failed: {report['n_failed']}",
        f"- Interaction records: {report['n_interaction_records']}",
        f"- Ligand atoms inventoried: {report['n_ligand_atoms']}",
        f"- Ligand connectivity edges: {report['n_ligand_connectivity_edges']}",
        f"- Global graph nodes: {report['n_global_nodes']}",
        f"- Ligand node ID mode: `{report['ligand_node_id_mode']}`",
        f"- Connectivity source: `{report['connectivity_source']}`",
        "",
        "## Main outputs",
        "",
        f"- `{graph_summary_path}`",
        f"- `{interaction_edges_path}`",
        f"- `{ligand_atoms_path}`",
        f"- `{connectivity_path}`",
        f"- `{global_nodes_path}`",
        f"- `{graph_index_path}`",
        f"- `{ged_dir}`",
        "",
        "Each per-structure PLIP work directory also contains `interaction_output.txt`, `interaction_output.csv`, `substrate_connectivity.txt`, `substrate_connectivity_edges.csv`, and `ligand_atom_inventory.csv` for manual inspection.",
        "",
        "Step06 does not modify or rewrite complex structures. It reads complex paths from the input index table and exports PLIP/graph-derived analysis files.",
        "",
    ]
    (reports_dir / f"{target_id}_step05_summary.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run PLIP and build interaction-graph/GED files.")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    parser.add_argument("--max-structures", type=int, default=0, help="Optional number of structures to process; 0 means all.")
    clean_group = parser.add_mutually_exclusive_group()
    clean_group.add_argument("--clean", action="store_true", help="Remove the existing Step06 output directory before running.")
    clean_group.add_argument("--no-clean", action="store_true", help="Keep existing Step06 output files.")
    args = parser.parse_args(argv)
    max_structures = None if args.max_structures == 0 else args.max_structures
    clean_output = True if args.clean else False if args.no_clean else None
    report = run_step06(args.config, max_structures=max_structures, clean_output=clean_output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class-level interaction-graph visualization.

This script is the culmination of the visualization workflow, integrating
the powerful, user-perfected visualization engine (from new-graphic-1.py)
with a fully automated pipeline that processes GED classification results.

It retains all advanced visualization and parameterization capabilities
while significantly improving user feedback through enhanced logging,
progress bars, and a comprehensive suite of command-line options for
fine-tuned adjustments. The default settings are calibrated to produce
the standard, user-approved visual style.
"""

from __future__ import annotations
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print("提示: tqdm 库未安装。推荐使用 'pip install tqdm' 来获得进度条体验。")
        return iterable

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdCoordGen
except ImportError:
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    rdCoordGen = None  # type: ignore[assignment]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.transforms import Bbox

# =============================================================================
# CONSTANTS AND CONFIG (PRESERVED FROM YOUR ORIGINAL SCRIPT)
# =============================================================================
AA_3_TO_1 = {"ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V"}
INTERACTION_COLORS: Dict[str, str] = {
    "hydrophobic_interactions":"#2E8B57", "hydrogen_bonds":"#1f77b4", "pi_stacks":"#DAA520",
    "salt_bridges":"#DC143C", "pi_cation_interactions":"#FF8C00", "halogen_bonds":"#A0522D", "metal_complexes":"#6A5ACD"
}
ATOM_COLORS: Dict[str, str] = {
    "C":"#333333", "N":"#3050F8", "O":"#FF2000", "P":"#FF8000", "S":"#FFFF30", "H":"#FFFFFF",
    "F":"#90E050", "CL":"#1FF01F", "BR":"#A62929", "I":"#940094"
}
ION_COLOR = "#228B22"
ION_STYLES: Dict[str, Dict] = {
    "MG":{"color":ION_COLOR,"size":110}, "ZN":{"color":ION_COLOR,"size":110}, "MN":{"color":ION_COLOR,"size":110},
    "FE":{"color":ION_COLOR,"size":110}, "CA":{"color":ION_COLOR,"size":110}, "NA":{"color":ION_COLOR,"size":110},
    "K":{"color":ION_COLOR,"size":110}, "CU":{"color":ION_COLOR,"size":110}
}
METAL_IONS = set(ION_STYLES.keys())
ATOM_COLORS.update({ion.upper(): style["color"] for ion, style in ION_STYLES.items()})

# =============================================================================
# VISUALIZATION ENGINE CLASS (YOUR PRESERVED CORE LOGIC)
# =============================================================================
class EnzymeSubstrateVisualizer:
    def __init__(self) -> None:
        self.global_interaction_types: set = set()
        self.global_atom_elements: set = set()
        self.MM_PER_INCH = 25.4
        self.FRAME_WIDTH_MM = 65.643
        self.FRAME_HEIGHT_MM = 49.488
        self.KNOWN_MULTI_CHAR_ELEMENTS = {el for el in ATOM_COLORS.keys() if len(el) > 1}
        self.layout_template: Optional[Dict[str, np.ndarray]] = None
        self.template_residue_map: Optional[Dict[str, str]] = None
        self.template_scale_ratio: Optional[float] = None
        self.runtime_args = argparse.Namespace()

    def _arg(self, name: str, default):
        return getattr(getattr(self, 'runtime_args', argparse.Namespace()), name, default)

    def _parse_file_lines(self, path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"文件解析失败 {path}: {e}")
            return []

    def _get_atom_element(self, atom_label: str) -> str:
        label_upper = atom_label.strip().upper()
        match = re.match(r"([A-Z]+)", label_upper)
        if not match:
            return "C"
        letters = match.group(1)
        if len(letters) >= 2 and letters[:2] in self.KNOWN_MULTI_CHAR_ELEMENTS: return letters[:2]
        return letters[0]

    def _get_residue_id_num(self, res_label: str) -> Optional[str]:
        match = re.match(r"[A-Z]{3}(\d+)", res_label.upper())
        if match: return match.group(1)
        return None

    def _clean_residue_label(self, res: str) -> str:
        match = re.match(r"([A-Z]{3})(\d+)", res.upper())
        if match:
            res_name, res_num = match.groups()
            return f"{AA_3_TO_1.get(res_name, '?')}{res_num}"
        return res

    def _is_valid_ligand_token(self, token: str) -> bool:
        """Reject non-atom tokens that can accidentally appear in Step06 text files.

        This specifically prevents structure IDs / file paths such as
        `2PBR_AF2_0025` or `results/...trimmed_complex.pdb` from becoming
        fake ligand atoms with labels like `2PBR...` or element `R`.
        """
        text = str(token or "").strip().strip(",")
        if not text:
            return False
        lower = text.lower()

        bad_substrings = [
            "/", "\\", ".pdb", ".cif", ".mol2", ".sdf", ".txt", ".csv",
            "results", "complex", "trimmed", "protein", "report.xml",
        ]
        if any(x in lower for x in bad_substrings):
            return False
        if re.search(r"\b[A-Za-z0-9]+_AF2_\d+", text):
            return False
        if text.startswith("#") or text.startswith("t #") or re.match(r"^[tev]\s+\d+\b", text):
            return False
        if any(ch.isspace() for ch in text):
            return False

        if re.match(r"^[^\[\]\s,]+\[[^;\]]+(?:;[^\]]*)?\]$", text):
            return True
        if re.match(r"^[A-Za-z0-9]+_[A-Za-z]\d+_[A-Za-z]{1,3}\d*[A-Za-z0-9]*$", text):
            return True
        if len(text.split(":")) >= 4:
            return True
        if re.match(r"^(?:[A-Z]{1,2}\d{0,3}|[A-Z]{1,2})$", text, flags=re.I):
            return True
        return False

    def _parse_ligand_token(self, token: str) -> Optional[Tuple[str, str, str, str]]:
        """Return (node_id, atom_name, entity_key, element) for Step06 ligand tokens.

        Supported examples:
          C8[ATP:L:1:C8;serial=1579]
          ATP_L1_C8
          ATP:L:1:C8
          MG[MG:L:3:MG;serial=1599]
          O7
        """
        text = str(token or "").strip().strip(',')
        if not text or not self._is_valid_ligand_token(text):
            return None
        # Strip optional list brackets generated by Step06 substrate_connectivity.txt.
        m = re.match(r"^(?P<atom>[^\[\]\s,]+)\[(?P<key>[^;\]]+)(?:;[^\]]*)?\]$", text)
        if m:
            atom = m.group('atom').strip()
            key = m.group('key').strip()
            parts = key.split(':')
            if len(parts) >= 4:
                res, chain, resseq, atom_from_key = parts[0], parts[1], parts[2], parts[3]
                atom = atom_from_key or atom
                node_id = f"{res}_{chain}{resseq}_{atom}"
                entity_key = f"{res}:{chain}:{resseq}"
                return node_id, atom, entity_key, self._get_atom_element(atom)
            return key, atom, key.rsplit(':', 1)[0], self._get_atom_element(atom)

        # Step06 unique ligand label, e.g. ATP_L1_O7.
        m = re.match(r"^(?P<res>[A-Za-z0-9]+)_(?P<chain>[A-Za-z])(?P<resseq>\d+)_(?P<atom>[A-Za-z0-9]+)$", text)
        if m:
            res, chain, resseq, atom = m.group('res'), m.group('chain'), m.group('resseq'), m.group('atom')
            return text, atom, f"{res}:{chain}:{resseq}", self._get_atom_element(atom)

        # PDB-like key from interaction_output: ATP:L:1:O7.
        parts = text.split(':')
        if len(parts) >= 4:
            res, chain, resseq, atom = parts[0], parts[1], parts[2], parts[3]
            node_id = f"{res}_{chain}{resseq}_{atom}"
            return node_id, atom, f"{res}:{chain}:{resseq}", self._get_atom_element(atom)

        # Plain atom fallback. This is only unambiguous for single-substrate / unique atoms.
        return text, text, "UNK:_:0", self._get_atom_element(text)

    def _parse_kv_interaction_line(self, line: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        parts = re.split(r"\t+", line.strip())
        if len(parts) == 1:
            parts = re.split(r"\s{2,}", line.strip())
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                fields[k.strip().lower().replace(' ', '_')] = v.strip()
        return fields

    def _add_ligand_node(self, G: nx.Graph, node_id: str, atom: str, entity_key: str, entity_id_map: Dict[str, int]) -> None:
        if entity_key not in entity_id_map:
            entity_id_map[entity_key] = len(entity_id_map)
        element = self._get_atom_element(atom)
        node_type = 'ion' if element.upper() in METAL_IONS else 'atom'
        if node_id not in G:
            G.add_node(node_id, type=node_type, element=element, entity_id=entity_id_map[entity_key], display_label=atom, entity_key=entity_key)

    def _read_substrate_connectivity(self, folder: str, G: nx.Graph) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
        """Read Step06 substrate_connectivity.txt into graph.

        Returns:
          key_to_node: maps PDB node keys like ATP:L:1:O7 to internal unique node ids.
          atom_to_nodes: maps display atom names like O7 to possible node ids.
          entity_id_map: maps ligand residue keys to entity ids.
        """
        key_to_node: Dict[str, str] = {}
        atom_to_nodes: Dict[str, List[str]] = defaultdict(list)
        entity_id_map: Dict[str, int] = {}

        def register(token: str) -> Optional[str]:
            parsed = self._parse_ligand_token(token)
            if not parsed:
                return None
            node_id, atom, entity_key, _element = parsed
            self._add_ligand_node(G, node_id, atom, entity_key, entity_id_map)
            atom_to_nodes.setdefault(atom, [])
            if node_id not in atom_to_nodes[atom]:
                atom_to_nodes[atom].append(node_id)
            # If token has explicit PDB key, store it.
            bracket = re.match(r"^[^\[]+\[([^;\]]+)", token.strip())
            if bracket:
                key_to_node[bracket.group(1)] = node_id
            parts = str(token).strip().split(':')
            if len(parts) >= 4:
                key_to_node[':'.join(parts[:4])] = node_id
            # Also map Step06 unique label to itself.
            key_to_node[node_id] = node_id
            return node_id

        def register_inventory_row(row: Dict[str, str]) -> Optional[str]:
            """Register an inventory atom when the virtual topology is absent.

            ``substrate_connectivity.txt`` is the virtual connectivity file
            produced by Step06 and is the historical Step10 drawing contract.
            The structured inventory is deliberately a *fallback*: registering
            it first changes the atom universe and consequently the layout of
            established class figures.  It remains useful for a genuinely
            structured-only Step06 export.
            """
            raw_node_key = str(row.get("node_key", "")).strip()
            atom = str(row.get("atom_name", "")).strip()
            resname = str(row.get("residue_name", "")).strip() or "UNK"
            chain = str(row.get("chain_id", "")).strip() or "_"
            resseq = str(row.get("residue_number", "")).strip() or "0"
            if not raw_node_key or not atom:
                return None

            # Keep the historical Step10 node identity (ATP_L1_O7) while
            # accepting the modern structured Step06 key (ATP:L:1:O7).  This
            # restores the original RDKit/template layout without losing the
            # repaired covalent-edge attachment.
            parsed = self._parse_ligand_token(raw_node_key)
            if parsed:
                node_id, parsed_atom, entity_key, _element = parsed
                atom = parsed_atom or atom
            else:
                node_id = raw_node_key
                entity_key = f"{resname}:{chain}:{resseq}"
            self._add_ligand_node(G, node_id, atom, entity_key, entity_id_map)
            atom_to_nodes.setdefault(atom, [])
            if node_id not in atom_to_nodes[atom]:
                atom_to_nodes[atom].append(node_id)
            key_to_node[raw_node_key] = node_id
            key_to_node[f"{resname}:{chain}:{resseq}:{atom}"] = node_id
            key_to_node[raw_node_key.upper()] = node_id
            key_to_node[f"{resname}:{chain}:{resseq}:{atom}".upper()] = node_id
            key_to_node[node_id] = node_id
            key_to_node[node_id.upper()] = node_id
            return node_id

        def resolve_or_register(token: str) -> Optional[str]:
            """Use Step06's inventory node identity before parsing a token.

            Step06 uses colon-delimited `ligand_node_key` values in its
            structured connectivity CSV, whereas the historical text parser
            converted those keys to underscore labels.  Resolving the inventory
            key first keeps the covalent edge attached to the same node that
            carries the PLIP interaction records, instead of creating a second,
            disconnected copy of every substrate atom.
            """
            text = str(token or "").strip()
            return key_to_node.get(text) or key_to_node.get(text.upper()) or register(text)

        def read_virtual_connectivity() -> Tuple[int, int]:
            """Read the Step06 virtual topology without adding any extra atoms."""
            n_nodes_before = G.number_of_nodes()
            n_edges_before = G.number_of_edges()
            conn_path = os.path.join(folder, "substrate_connectivity.txt")
            conn_lines = self._parse_file_lines(conn_path)
            for line in conn_lines:
                if '->' not in line:
                    continue
                left, right = line.split('->', 1)
                left_raw = left.strip()
                if not self._is_valid_ligand_token(left_raw):
                    continue
                left_id = resolve_or_register(left_raw)
                if not left_id:
                    continue
                for raw_nb in right.split(','):
                    nb = raw_nb.strip()
                    if not nb or not self._is_valid_ligand_token(nb):
                        continue
                    nb_id = resolve_or_register(nb)
                    if nb_id and nb_id != left_id:
                        G.add_edge(left_id, nb_id, type='bond')
            return G.number_of_nodes() - n_nodes_before, G.number_of_edges() - n_edges_before

        def read_structured_connectivity() -> Tuple[int, int]:
            """Fallback reader for structured-only Step06 exports."""
            n_nodes_before = G.number_of_nodes()
            n_edges_before = G.number_of_edges()
            inventory_path = os.path.join(folder, "ligand_atom_inventory.csv")
            if os.path.exists(inventory_path):
                try:
                    with open(inventory_path, "r", encoding="utf-8-sig", newline="") as handle:
                        for row in csv.DictReader(handle):
                            register_inventory_row(row)
                except Exception as e:
                    logging.warning(f"Failed to parse ligand_atom_inventory.csv in {folder}: {e}")

            csv_path = os.path.join(folder, "substrate_connectivity_edges.csv")
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        columns = [c for c in (reader.fieldnames or [])]
                        lower = {c.lower(): c for c in columns}
                        src_col = dst_col = None
                        # Exact Step06 keys must win over provenance columns
                        # such as ``source`` / ``connectivity_source``.
                        candidate_pairs = [
                            ("ligand_node_key_1", "ligand_node_key_2"),
                            ("ligand_node_label_1", "ligand_node_label_2"),
                            ("source", "target"), ("src", "dst"), ("from", "to"),
                            ("atom1", "atom2"), ("atom_1", "atom_2"),
                            ("node1", "node2"), ("node_1", "node_2"),
                            ("u", "v"), ("source_label", "target_label"),
                            ("source_node", "target_node"), ("source_node_key", "target_node_key"),
                            ("src_label", "dst_label"), ("ligand_atom_1", "ligand_atom_2"),
                            ("atom_name_1", "atom_name_2"),
                        ]
                        for a, b in candidate_pairs:
                            if a in lower and b in lower:
                                src_col, dst_col = lower[a], lower[b]
                                break
                        if not (src_col and dst_col):
                            logging.warning(
                                f"Cannot identify source/target columns in {csv_path}; "
                                "structured connectivity was ignored."
                            )
                        else:
                            for row in reader:
                                left_raw = str(row.get(src_col, "")).strip()
                                right_raw = str(row.get(dst_col, "")).strip()
                                if not (self._is_valid_ligand_token(left_raw) and self._is_valid_ligand_token(right_raw)):
                                    continue
                                left_id = resolve_or_register(left_raw)
                                right_id = resolve_or_register(right_raw)
                                if left_id and right_id and left_id != right_id:
                                    G.add_edge(left_id, right_id, type='bond')
                except Exception as e:
                    logging.warning(f"Failed to parse substrate_connectivity_edges.csv in {folder}: {e}")
            return G.number_of_nodes() - n_nodes_before, G.number_of_edges() - n_edges_before

        def read_standard_connectivity() -> Tuple[int, int]:
            """Read the established topology source in its reproducible order.

            A valid structured edge CSV is used first; the virtual text file is
            read only when that CSV supplies no edges. This preserves the 2-D
            topology for existing projects.
            """
            n_nodes_before = G.number_of_nodes()
            n_edges_before = G.number_of_edges()
            csv_path = os.path.join(folder, "substrate_connectivity_edges.csv")
            csv_edges_added = 0
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        columns = [column for column in (reader.fieldnames or [])]
                        lower = {column.lower(): column for column in columns}
                        src_col = dst_col = None
                        candidate_pairs = [
                            ("source", "target"), ("src", "dst"), ("from", "to"),
                            ("atom1", "atom2"), ("atom_1", "atom_2"),
                            ("node1", "node2"), ("node_1", "node_2"),
                            ("u", "v"), ("source_label", "target_label"),
                            ("source_node", "target_node"), ("source_node_key", "target_node_key"),
                            ("src_label", "dst_label"), ("ligand_atom_1", "ligand_atom_2"),
                            ("atom_name_1", "atom_name_2"),
                        ]
                        for left_name, right_name in candidate_pairs:
                            if left_name in lower and right_name in lower:
                                src_col, dst_col = lower[left_name], lower[right_name]
                                break
                        if src_col and dst_col:
                            for row in reader:
                                left_raw = str(row.get(src_col, "")).strip()
                                right_raw = str(row.get(dst_col, "")).strip()
                                if not (
                                    self._is_valid_ligand_token(left_raw)
                                    and self._is_valid_ligand_token(right_raw)
                                ):
                                    continue
                                left_id = register(left_raw)
                                right_id = register(right_raw)
                                if left_id and right_id and left_id != right_id:
                                    G.add_edge(left_id, right_id, type="bond")
                                    csv_edges_added += 1
                        else:
                            logging.warning(
                                f"Cannot identify source/target columns in {csv_path}; "
                                "falling back to substrate_connectivity.txt."
                            )
                except Exception as exc:
                    logging.warning(f"Failed to parse substrate_connectivity_edges.csv in {folder}: {exc}")

            if csv_edges_added == 0:
                conn_path = os.path.join(folder, "substrate_connectivity.txt")
                for line in self._parse_file_lines(conn_path):
                    if "->" not in line:
                        continue
                    left, right = line.split("->", 1)
                    left_raw = left.strip()
                    if not self._is_valid_ligand_token(left_raw):
                        continue
                    left_id = register(left_raw)
                    if not left_id:
                        continue
                    for raw_nb in right.split(","):
                        neighbor = raw_nb.strip()
                        if not neighbor or not self._is_valid_ligand_token(neighbor):
                            continue
                        neighbor_id = register(neighbor)
                        if neighbor_id and neighbor_id != left_id:
                            G.add_edge(left_id, neighbor_id, type="bond")
            return G.number_of_nodes() - n_nodes_before, G.number_of_edges() - n_edges_before

        # The virtual file is emitted by Step06 specifically for Step10 and
        # encodes the intended substrate topology and node labels.  It is now
        # the default again.  This restores old figures while retaining a
        # robust structured fallback for exports that have no virtual file.
        source_mode = str(self._arg('connectivity_source', 'standard')).strip().lower()
        merge_sources = bool(self._arg('merge_text_connectivity_with_csv', False)) or source_mode in {'merge', 'both'}
        if source_mode in {'standard', 'catcongraph_studio_v1', 'v8', 'v8_exact', 'legacy', 'legacy_v8'}:
            read_standard_connectivity()
        elif source_mode in {'structured', 'structured_preferred', 'csv'}:
            structured_nodes, _structured_edges = read_structured_connectivity()
            if structured_nodes == 0:
                read_virtual_connectivity()
        else:
            virtual_nodes, _virtual_edges = read_virtual_connectivity()
            if virtual_nodes == 0 or merge_sources:
                read_structured_connectivity()

        return key_to_node, atom_to_nodes, entity_id_map

    def _find_ligand_node_for_interaction(self, G: nx.Graph, key_to_node: Dict[str, str], atom_to_nodes: Dict[str, List[str]], ligand: str, ligand_key: str, entity_id_map: Dict[str, int]) -> Optional[str]:
        # Prefer exact PDB node key; this is critical when ATP and TMP share atom names such as O3/C8/P1.
        if ligand_key:
            parsed = self._parse_ligand_token(ligand_key)
            if parsed:
                node_id, atom, entity_key, _el = parsed
                if node_id not in G:
                    self._add_ligand_node(G, node_id, atom, entity_key, entity_id_map)
                return node_id
            if ligand_key in key_to_node:
                return key_to_node[ligand_key]
        # Then try exact unique ligand label.
        parsed_lig = self._parse_ligand_token(ligand)
        if parsed_lig:
            node_id, atom, entity_key, _el = parsed_lig
            if node_id in G:
                return node_id
            if ligand in key_to_node:
                return key_to_node[ligand]
            # Plain atom names are unsafe when duplicated; only use unique mapping.
            if ':' not in ligand and '_' not in ligand:
                candidates = atom_to_nodes.get(ligand, [])
                if len(candidates) == 1:
                    return candidates[0]
            # Last resort: add it as an isolated ligand node so the interaction is not lost.
            self._add_ligand_node(G, node_id, atom, entity_key, entity_id_map)
            return node_id
        return None

    def build_interaction_graph(self, folder: str) -> nx.Graph:
        """Build an atom-level interaction graph from Step06 PLIP output.

        This is intentionally Step06-native. It uses:
          - substrate_connectivity.txt for ligand covalent topology
          - interaction_output.txt for protein-ligand interactions
          - LigandNodeKey when present, so duplicate atom names in different substrates are not confused
        """
        G = nx.Graph()
        key_to_node, atom_to_nodes, entity_id_map = self._read_substrate_connectivity(folder, G)

        inter_path = os.path.join(folder, "interaction_output.txt")
        inter_lines = self._parse_file_lines(inter_path)

        pat_en = re.compile(r"Type:\s*(?P<type>\S+)\s+Ligand:\s*(?P<ligand>\S+)\s+Protein:\s*(?P<protein>\S+)")
        pat_cn = re.compile(r"相互作用类型:\s*(?P<type>\S+).*?底物原子:\s*(?P<ligand>\S+).*?残基:\s*(?P<protein>\S+)")

        type_map = {
            "hydrogen_bond": "hydrogen_bonds", "hydrogen_bonds": "hydrogen_bonds", "hbond": "hydrogen_bonds",
            "hydrophobic_interaction": "hydrophobic_interactions", "hydrophobic_interactions": "hydrophobic_interactions",
            "salt_bridge": "salt_bridges", "salt_bridges": "salt_bridges", "ionic": "salt_bridges",
            "pi_stack": "pi_stacks", "pi_stacks": "pi_stacks", "pi-stacking": "pi_stacks",
            "pi_cation_interaction": "pi_cation_interactions", "pi_cation_interactions": "pi_cation_interactions",
            "halogen_bond": "halogen_bonds", "halogen_bonds": "halogen_bonds",
            "metal_complex": "metal_complexes", "metal_complexes": "metal_complexes", "metal": "metal_complexes",
            "water_bridge": "water_bridges", "water_bridges": "water_bridges",
        }

        for line in inter_lines:
            fields = self._parse_kv_interaction_line(line)
            raw_type = fields.get('type') or fields.get('interaction_type') or ""
            ligand = fields.get('ligand') or fields.get('ligand_atom') or ""
            protein = fields.get('protein') or fields.get('protein_raw') or fields.get('residue') or ""
            ligand_key = fields.get('ligandnodekey') or fields.get('ligand_node_key') or ""
            if not (raw_type and ligand and protein):
                mm = pat_en.match(line) or pat_cn.match(line)
                if not mm:
                    continue
                data = mm.groupdict()
                raw_type, ligand, protein = data.get('type', ''), data.get('ligand', ''), data.get('protein', '')
            int_type = type_map.get(str(raw_type).strip().lower().replace('-', '_'), str(raw_type).strip().lower().replace('-', '_'))
            ligand_node = self._find_ligand_node_for_interaction(G, key_to_node, atom_to_nodes, ligand, ligand_key, entity_id_map)
            if ligand_node:
                G.add_node(protein, type='residue')
                G.add_edge(ligand_node, protein, type=int_type)
        return G

    def _generate_entity_layout_rdkit(self, G_entity: nx.Graph) -> Optional[Dict[str, np.ndarray]]:
        """Generate a 2D ligand layout.

        RDKit is preferred for chemically sensible layouts. If the inferred
        connectivity has impossible valences (which can happen with old AF2/PDB
        outputs or partially inferred connectivity), fall back to a deterministic
        NetworkX spring layout rather than dropping the entity. This prevents
        blank/flat plots while keeping the workflow robust.
        """
        if not G_entity.nodes:
            return None

        def fallback_layout() -> Dict[str, np.ndarray]:
            if len(G_entity.nodes) == 1:
                return {next(iter(G_entity.nodes)): np.array([0.0, 0.0])}
            pos0 = nx.spring_layout(G_entity, seed=7, k=1.35, iterations=200)
            arr = np.array(list(pos0.values()))
            center = np.mean(arr, axis=0)
            scale = float(self._arg('substrate_fallback_scale', 4.0 if len(G_entity.nodes) > 6 else 2.6))
            return {node: (np.array(p) - center) * scale for node, p in pos0.items()}

        if Chem is None or AllChem is None or rdCoordGen is None:
            return fallback_layout()

        mol = Chem.RWMol()
        atom_map = {}
        try:
            for node in G_entity.nodes():
                atom_map[node] = mol.AddAtom(Chem.Atom(G_entity.nodes[node].get('element', 'C')))
            for u, v in G_entity.edges():
                if u in atom_map and v in atom_map:
                    mol.AddBond(atom_map[u], atom_map[v], Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return fallback_layout()
            scale_factor = float(self._arg('substrate_scale', 2.75))
            try:
                rdCoordGen.AddCoords(mol)
            except Exception:
                AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            pos = {node: np.array([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]) for node, i in atom_map.items()}
            center = np.mean(list(pos.values()), axis=0)
            return {node: (p - center) * scale_factor for node, p in pos.items()}
        except Exception:
            return fallback_layout()

    def _repel_core_entities(self, pos: Dict, entities: Dict[int, Set[str]]) -> Dict:
        entity_centers = {eid: np.mean([pos[n] for n in nodes if n in pos], axis=0) for eid, nodes in entities.items()}
        entity_bboxes_templates = {}
        for eid, nodes in entities.items():
            nodes_with_pos = [n for n in nodes if n in pos]
            if not nodes_with_pos: continue
            coords = np.array([pos[n] for n in nodes_with_pos])
            min_c, max_c = np.min(coords, axis=0), np.max(coords, axis=0)
            pad = float(self._arg('entity_bbox_padding', 1.2))
            width, height = (max_c[0] - min_c[0]) + pad, (max_c[1] - min_c[1]) + pad
            entity_bboxes_templates[eid] = {'width': width, 'height': height}
        for _ in range(int(self._arg('entity_repel_iterations', 400))):
            moved = False
            eids = list(entity_centers.keys())
            for i in range(len(eids)):
                for j in range(i + 1, len(eids)):
                    eid1, eid2 = eids[i], eids[j]
                    t1, t2 = entity_bboxes_templates.get(eid1), entity_bboxes_templates.get(eid2)
                    if not t1 or not t2: continue
                    center1, center2 = entity_centers[eid1], entity_centers[eid2]
                    bbox1 = Bbox.from_bounds(center1[0] - t1['width']/2, center1[1] - t1['height']/2, t1['width'], t1['height'])
                    bbox2 = Bbox.from_bounds(center2[0] - t2['width']/2, center2[1] - t2['height']/2, t2['width'], t2['height'])
                    if bbox1.overlaps(bbox2):
                        moved = True
                        delta = center1 - center2
                        dist = np.linalg.norm(delta) + 1e-6
                        move_vec = (delta / dist) * float(self._arg('entity_repel_step', 0.1))
                        entity_centers[eid1] += move_vec
                        entity_centers[eid2] -= move_vec
            if not moved: break
        final_pos = pos.copy()
        for eid, new_center in entity_centers.items():
            nodes_with_pos = [n for n in entities[eid] if n in pos]
            if not nodes_with_pos: continue
            old_center = np.mean([pos[n] for n in nodes_with_pos], axis=0)
            for node in nodes_with_pos: final_pos[node] = pos[node] + (new_center - old_center)
        return final_pos

    def _place_and_repel_residues(self, pos: Dict, G: nx.Graph, nodes_to_place: Set[str], fixed_nodes: Dict[str, np.ndarray]) -> Dict:
        if not nodes_to_place: return pos
        final_pos = pos.copy()
        core_nodes_pos = list(fixed_nodes.values())
        if not core_nodes_pos: core_nodes_pos = [np.array([0,0])]
        core_center = np.mean(core_nodes_pos, axis=0)
        
        residue_info = {}
        # Group residues by their "anchor direction" so residues pointing to the same region fan out instead of stacking.
        # This is the real reason your plots become unreadable: many residues share the same anchor atom(s),
        # so they get the same angle and then overlap.
        group_bins = {}  # key -> list[res]
        base_angle = {}
        base_dist = float(self._arg('residue_base_distance', 14.0))
        angle_bin = np.deg2rad(float(self._arg('residue_angle_bin_deg', 15.0)))

        for res in nodes_to_place:
            anchor_atoms = [n for n in G.neighbors(res) if n in fixed_nodes]
            if not anchor_atoms:
                ang = np.random.uniform(0, 2 * np.pi)
                base_angle[res] = ang
                key = int(round(ang / angle_bin))
                group_bins.setdefault(key, []).append(res)
                continue

            anchor_pos = np.mean([fixed_nodes[a] for a in anchor_atoms], axis=0)
            direction_vec = anchor_pos - core_center
            ang = np.arctan2(direction_vec[1], direction_vec[0])
            base_angle[res] = ang
            key = int(round(ang / angle_bin))
            group_bins.setdefault(key, []).append(res)

        # Assign angles within each bin by fanning out; assign distance in layers if needed.
        for key, residues_in_bin in group_bins.items():
            residues_in_bin = sorted(residues_in_bin, key=lambda r: self._clean_residue_label(r))
            k = len(residues_in_bin)
            ang0 = (key * angle_bin)
            # Fan width grows with k, but is fully configurable.
            fan_base = float(self._arg('residue_fan_base_deg', 10.0))
            fan_step = float(self._arg('residue_fan_step_deg', 6.0))
            fan_max = float(self._arg('residue_fan_max_deg', 55.0))
            fan = min(np.deg2rad(fan_base + fan_step * max(0, k - 1)), np.deg2rad(fan_max))
            if k == 1:
                residue_info[residues_in_bin[0]] = {'angle': ang0, 'dist': base_dist}
            else:
                angles = np.linspace(ang0 - fan, ang0 + fan, k)
                mid = (k - 1) / 2.0
                for i, res in enumerate(residues_in_bin):
                    # If many residues share an angle region, push outer ones slightly further out.
                    layer = abs(i - mid)
                    dist = base_dist + float(self._arg('residue_layer_step', 1.6)) * layer
                    residue_info[res] = {'angle': float(angles[i]), 'dist': float(dist)}

        # Convert polar placement to XY
        for res, info in residue_info.items():
            angle, dist = info['angle'], info['dist']
            final_pos[res] = core_center + np.array([dist * np.cos(angle), dist * np.sin(angle)])
        residue_radius = float(self._arg('residue_residue_min_distance', 8.6))
        core_radius = float(self._arg('residue_core_min_distance', 7.0))
        residue_radius_sq, core_radius_sq = residue_radius**2, core_radius**2
        all_fixed_nodes = list(fixed_nodes.keys())
        for _ in range(int(self._arg('residue_repel_iterations', 450))):
            for r1 in nodes_to_place:
                for r2 in nodes_to_place:
                    if r1 >= r2: continue
                    delta, dist_sq = final_pos[r1] - final_pos[r2], np.sum((final_pos[r1] - final_pos[r2])**2)
                    if dist_sq < residue_radius_sq:
                        move_vec = (delta / (np.sqrt(dist_sq) + 1e-6)) * float(self._arg('residue_repel_step', 0.25))
                        final_pos[r1] += move_vec; final_pos[r2] -= move_vec
                for c_node in all_fixed_nodes:
                    delta, dist_sq = final_pos[r1] - final_pos[c_node], np.sum((final_pos[r1] - final_pos[c_node])**2)
                    if dist_sq < core_radius_sq:
                         move_vec = (delta / (np.sqrt(dist_sq) + 1e-6)) * float(self._arg('residue_core_repel_step', 0.3))
                         final_pos[r1] += move_vec
        return final_pos

    def _main_layout_engine(self, G: nx.Graph, swap_substrates: bool, rotate_substrates: Dict[int, float]) -> Dict[str, np.ndarray]:
        core_nodes = {n for n, d in G.nodes(data=True) if d.get('type') in {'atom', 'ion'}}
        residues = {n for n, d in G.nodes(data=True) if d.get('type') == 'residue'}
        entities = defaultdict(set)
        for node in core_nodes:
            entities[G.nodes[node]['entity_id']].add(node)

        def entity_name(eid, nodes):
            # Entity key is usually ATP:L:1, TMP:L:2, MG:L:3.
            for n in nodes:
                key = str(G.nodes[n].get('entity_key', ''))
                if key:
                    return key.split(':')[0].upper()
            elems = {str(G.nodes[n].get('element', '')).upper() for n in nodes}
            return sorted(elems)[0] if elems else str(eid)

        entity_order_cfg = self._arg('entity_order', [])
        entity_order = [str(x).upper() for x in entity_order_cfg] if isinstance(entity_order_cfg, (list, tuple)) else []
        entity_items = list(entities.items())

        if entity_order:
            order_index = {name: i for i, name in enumerate(entity_order)}
            # Allow ions to be placed between substrates, e.g. ATP, MG, TMP.
            ordered_entities = sorted(
                entity_items,
                key=lambda item: (
                    order_index.get(entity_name(item[0], item[1]), 10_000),
                    0 if G.nodes[list(item[1])[0]].get('type') != 'ion' else 1,
                    -len(item[1]),
                    entity_name(item[0], item[1]),
                )
            )
        else:
            non_ions = sorted(
                [(eid, nodes) for eid, nodes in entity_items if G.nodes[list(nodes)[0]]['type'] != 'ion'],
                key=lambda item: len(item[1]),
                reverse=True,
            )
            ions = sorted(
                [(eid, nodes) for eid, nodes in entity_items if G.nodes[list(nodes)[0]]['type'] == 'ion'],
                key=lambda item: entity_name(item[0], item[1]),
            )
            ordered_entities = non_ions + ions

        final_core_pos, offset_x = {}, 0.0
        non_ion_counter = 0

        for display_i, (eid, nodes) in enumerate(ordered_entities):
            first = list(nodes)[0]
            is_ion = G.nodes[first].get('type') == 'ion'

            y_offsets = self._arg('entity_y_offsets', None)
            if isinstance(y_offsets, (list, tuple)) and len(y_offsets):
                y_offset = float(y_offsets[display_i % len(y_offsets)])
            else:
                y_offset = 0.0

            if is_ion:
                node = first
                final_core_pos[node] = np.array([offset_x, y_offset + float(self._arg('ion_y_offset', 0.0))])
                offset_x += float(self._arg('ion_spacing', self._arg('entity_spacing', 5.0)))
                continue

            ideal_layout = self._generate_entity_layout_rdkit(G.subgraph(nodes))
            if not ideal_layout:
                continue

            non_ion_counter += 1
            if non_ion_counter in rotate_substrates:
                angle_deg = rotate_substrates[non_ion_counter]
                angle_rad = np.deg2rad(angle_deg)
                logging.info(f"    - Applying rotation of {angle_deg} degrees to substrate {non_ion_counter} (Entity ID: {eid}).")
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                for node, pos_val in ideal_layout.items():
                    ideal_layout[node] = pos_val @ rot_matrix.T

            max_x, min_x = (max(p[0] for p in ideal_layout.values()), min(p[0] for p in ideal_layout.values())) if ideal_layout else (0, 0)
            for node, p in ideal_layout.items():
                final_core_pos[node] = p + np.array([offset_x - min_x, y_offset])
            offset_x += (max_x - min_x) + float(self._arg('entity_spacing', 5.0))

        # Keep the old optional swap behavior only when no explicit entity_order was supplied.
        if (not entity_order) and swap_substrates:
            non_ion_eids = [
                eid for eid, nodes in ordered_entities
                if G.nodes[list(nodes)[0]].get('type') != 'ion'
            ]
            if len(non_ion_eids) >= 2:
                eid1, eid2 = non_ion_eids[0], non_ion_eids[1]
                logging.info("    - Swapping substrate positions.")
                nodes1, nodes2 = entities[eid1], entities[eid2]
                center1 = np.mean([final_core_pos[n] for n in nodes1 if n in final_core_pos], axis=0)
                center2 = np.mean([final_core_pos[n] for n in nodes2 if n in final_core_pos], axis=0)
                delta = center2 - center1
                for n in nodes1:
                    if n in final_core_pos:
                        final_core_pos[n] += delta
                for n in nodes2:
                    if n in final_core_pos:
                        final_core_pos[n] -= delta

        return self._place_and_repel_residues(final_core_pos, G, residues, final_core_pos.copy())


    def _apply_template_layout(self, G: nx.Graph, rotate_angle: float) -> Dict[str, np.ndarray]:
        if self.layout_template is None or self.template_residue_map is None: raise ValueError("Layout template not found.")
        pos, unmapped_residues = {}, set()
        for node, data in G.nodes(data=True):
            if data.get('type') in {'atom', 'ion'}:
                if node in self.layout_template: pos[node] = self.layout_template[node]
            elif data.get('type') == 'residue':
                res_id = self._get_residue_id_num(node)
                if res_id and res_id in self.template_residue_map and self.template_residue_map[res_id] in self.layout_template:
                    pos[node] = self.layout_template[self.template_residue_map[res_id]]
                else: unmapped_residues.add(node)
        if unmapped_residues:
            logging.warning(f"    - New residues found: {', '.join(unmapped_residues)}. Placing them around the core.")
            pos = self._place_and_repel_residues(pos, G, unmapped_residues, pos.copy())
        if rotate_angle != 0.0:
            logging.info(f"    - Applying global rotation: {rotate_angle} degrees.")
            angle_rad = np.deg2rad(rotate_angle)
            rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
            center = np.mean(list({k: np.array(v) for k,v in pos.items()}.values()), axis=0)
            for node, p in {k: np.array(v) for k,v in pos.items()}.items(): pos[node] = (p - center) @ rot_matrix.T + center
        # --- Final gentle collision resolve (data-units) ---
        # Motivation: node sizes are in points^2 while coordinates are in data units; a small mismatch can leave
        # a few residue pairs still overlapping (e.g., V413/F247). This pass uses coordinate span to estimate
        # a safe minimum spacing in data units, without changing overall style.
        residue_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "residue"]
        if residue_nodes:
            xs = [pos[n][0] for n in residue_nodes]
            ys = [pos[n][1] for n in residue_nodes]
            span_x = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
            span_y = (max(ys) - min(ys)) if len(ys) > 1 else 1.0
            span = max(span_x, span_y, 1.0)

            # Tuned to be subtle: ~3.2% of span as residue radius.
            r_res = span * 0.048
            min_d2 = (2.0 * r_res) ** 2

            for _ in range(420):
                moved = False
                for i in range(len(residue_nodes)):
                    a = residue_nodes[i]
                    xa, ya = pos[a]
                    for j in range(i + 1, len(residue_nodes)):
                        b = residue_nodes[j]
                        xb, yb = pos[b]
                        dx, dy = xb - xa, yb - ya
                        d2 = dx * dx + dy * dy
                        if d2 < 1e-12:
                            dx, dy = 1e-6, 0.0
                            d2 = dx * dx + dy * dy
                        if d2 < min_d2:
                            d = math.sqrt(d2)
                            # push each half the overlap
                            push = (2.0 * r_res - d) * 0.55
                            ux, uy = dx / d, dy / d
                            pos[a] = (xa - ux * push, ya - uy * push)
                            pos[b] = (xb + ux * push, yb + uy * push)
                            xa, ya = pos[a]
                            moved = True
                if not moved:
                    break

        # --- Targeted micro-nudge for a known stubborn overlap (V413 vs F247) ---
        # Only activates if both residues exist and are still too close after the general resolve.
        if ("V413" in pos) and ("F247" in pos):
            xa, ya = pos["V413"]
            xb, yb = pos["F247"]
            dx, dy = xb - xa, yb - ya
            d2 = dx*dx + dy*dy
            if d2 < min_d2:
                d = math.sqrt(max(d2, 1e-12))
                ux, uy = dx / d, dy / d
                # push them apart along their connecting line, plus a tiny perpendicular component
                push = (2.0 * r_res - d) * 0.9
                px, py = -uy, ux
                pos["V413"] = (xa - ux*push - px*(0.15*r_res), ya - uy*push - py*(0.15*r_res))
                pos["F247"] = (xb + ux*push + px*(0.15*r_res), yb + uy*push + py*(0.15*r_res))


        return pos


    def _draw_hydrophobic_fan(self, ax: Axes, atom_pos, residue_pos, color: str, width: float, alpha: float):
        """Draw hydrophobic interaction as a green fan/arc glyph, matching the article-style figures.

        Hydrophobic contacts do not represent one precise atom-atom bond, so a fan glyph is
        more faithful than a straight line. The glyph is placed between the ligand atom and
        the residue, closer to the residue side.
        """
        from matplotlib.patches import Arc
        import numpy as _np
        a = _np.array(atom_pos, dtype=float)
        r = _np.array(residue_pos, dtype=float)
        vec = a - r
        norm = _np.linalg.norm(vec) + 1e-9
        u = vec / norm
        n = _np.array([-u[1], u[0]])
        center = r + vec * 0.42
        angle = math.degrees(math.atan2(u[1], u[0]))
        # Multiple arcs imitate the green semicircular "hydrophobic comb" in the reference figures.
        for i, offset in enumerate(_np.linspace(-0.28, 0.28, 6)):
            c = center + n * offset
            arc = Arc(
                c,
                width=0.62 + 0.035 * i,
                height=0.28 + 0.018 * i,
                angle=angle,
                theta1=60,
                theta2=150,
                lw=width,
                color=color,
                alpha=alpha,
                zorder=3,
            )
            ax.add_patch(arc)


    def _draw_hydrophobic_fan(self, ax: Axes, start_pos, end_pos, args: argparse.Namespace):
        """Draw a fan glyph for hydrophobic contacts, matching the article-style diagrams."""
        sx, sy = start_pos
        ex, ey = end_pos
        vx, vy = ex - sx, ey - sy
        norm = math.hypot(vx, vy) or 1.0
        vx, vy = vx / norm, vy / norm
        nx, ny = -vy, vx
        # Put the fan closer to the residue side but not on the label itself.
        cx, cy = sx * 0.58 + ex * 0.42, sy * 0.58 + ey * 0.42
        angle = math.degrees(math.atan2(vy, vx))
        for k in range(-3, 4):
            ox, oy = nx * k * 0.11, ny * k * 0.11
            arc = Arc((cx + ox, cy + oy), width=0.95 + 0.05 * abs(k), height=0.34 + 0.02 * abs(k),
                      angle=angle, theta1=62, theta2=132,
                      color=args.hydrophobic_edge_color, lw=max(args.interaction_edge_width, 1.0),
                      alpha=args.edge_alpha, zorder=2)
            ax.add_patch(arc)

    def _draw_graph_to_ax(self, ax: Axes, G: nx.Graph, pos: Dict[str, Tuple[float, float]], args: argparse.Namespace):
        for u, v, data in G.edges(data=True):
            if u in pos and v in pos:
                start_pos, end_pos = pos[u], pos[v]
                edge_type = data.get('type')
                edge_styles = {
                    'bond': ('gray', 'solid', args.covalent_edge_width),
                    'hydrogen_bonds': (args.hbond_edge_color, 'dashed', args.interaction_edge_width),
                    'salt_bridges': (args.saltbridge_edge_color, 'dashed', args.interaction_edge_width),
                    'hydrophobic_interactions': (args.hydrophobic_edge_color, 'fan', args.interaction_edge_width),
                    'pi_stacks': (args.pistack_edge_color, 'dashdot', args.interaction_edge_width),
                    'pi_cation_interactions': (getattr(args, 'pication_edge_color', INTERACTION_COLORS.get('pi_cation_interactions', '#FF8C00')), 'dashdot', args.interaction_edge_width),
                    'metal_complexes': (getattr(args, 'metal_edge_color', INTERACTION_COLORS.get('metal_complexes', '#6A5ACD')), 'dashed', args.interaction_edge_width),
                    'halogen_bonds': (getattr(args, 'halogen_edge_color', INTERACTION_COLORS.get('halogen_bonds', '#A0522D')), 'dotted', args.interaction_edge_width),
                    'water_bridges': ('#17becf', 'dotted', args.interaction_edge_width),
                }
                color, style, width = edge_styles.get(edge_type, ('grey', 'solid', 1.0))

                if edge_type == 'bond':
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], linestyle=style, color=color, lw=width, zorder=1)
                elif style == 'fan':
                    # Make sure the fan is drawn from residue to ligand for a consistent orientation.
                    if G.nodes[u].get('type') == 'residue':
                        self._draw_hydrophobic_fan(ax, start_pos, end_pos, args)
                    else:
                        self._draw_hydrophobic_fan(ax, end_pos, start_pos, args)
                else:
                    ax.add_patch(FancyArrowPatch(start_pos, end_pos, arrowstyle='-', color=color, linestyle=style, linewidth=width, connectionstyle="arc3,rad=0.08", zorder=2, alpha=args.edge_alpha))

        for node, data in G.nodes(data=True):
            if node in pos:
                x, y = pos[node]
                node_type = data.get('type')
                if node_type == 'atom':
                    ax.scatter(x, y, s=args.node_size*float(getattr(args, 'ligand_node_scale', 0.30)), color=ATOM_COLORS.get(data.get('element', 'C').upper(), "black"), alpha=float(getattr(args, 'ligand_node_alpha', 0.82)), zorder=4, edgecolors='none')
                    if getattr(args, 'show_atom_labels', True):
                        label = data.get('display_label') or ''.join(re.findall(r'[A-Z]?[0-9]+', node)) or node
                        ax.text(x, y, label, color=getattr(args, 'atom_label_color', 'white'), fontsize=max(args.font_size-float(getattr(args, 'atom_font_delta', 2.0)), 1.0), ha="center", va="center", weight='bold', zorder=5)
                elif node_type == 'residue':
                    ax.scatter(x, y, s=args.node_size*float(getattr(args, 'residue_node_scale', 1.0)), color=args.protein_node_color, alpha=float(getattr(args, 'protein_node_alpha', 0.62)), zorder=3, edgecolors='none')
                    ax.text(x, y, self._clean_residue_label(node), fontsize=args.font_size, color='black', ha="center", va="center", weight='bold', zorder=5)
                elif node_type == 'ion':
                    style = ION_STYLES.get(data.get('element'), {"color": "green", "size": 110})
                    ax.scatter(x, y, s=style["size"]*float(getattr(args, 'ion_node_scale', 1.0)), color=style["color"], alpha=float(getattr(args, 'ion_node_alpha', 0.85)), zorder=4, edgecolors='none')
                    if getattr(args, 'show_atom_labels', True):
                        ax.text(x, y, data.get('display_label') or data.get('element'), fontsize=args.font_size, color=getattr(args, 'atom_label_color', 'white'), ha="center", va="center", weight='bold', zorder=5)

    def _get_autofit_layout(self, G: nx.Graph, pos: Dict, fig_dims: Tuple[float, float], frame_dims: Tuple[float, float], rotate_angle: float, use_template: bool, args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
        pos_np = {k: np.array(v) for k, v in pos.items() if v is not None}
        if not pos_np: return {}
        if rotate_angle != 0 and not use_template:
            angle_rad = np.deg2rad(rotate_angle)
            rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
            center = np.mean(list(pos_np.values()), axis=0)
            for node, p in pos_np.items(): pos_np[node] = (p - center) @ rot_matrix.T + center
        
        visual_center = np.mean(list(pos_np.values()), axis=0)
        
        if use_template and self.template_scale_ratio is not None:
            scale_ratio = self.template_scale_ratio
            logging.debug(f"    - Applying template scale ratio: {scale_ratio:.4f}")
        else:
            temp_fig = Figure(figsize=fig_dims)
            temp_ax = temp_fig.add_subplot(111)
            self._draw_graph_to_ax(temp_ax, G, {n: p - visual_center for n, p in pos_np.items()}, args)
            canvas = FigureCanvasAgg(temp_fig)
            try: centered_bbox = temp_ax.get_tightbbox(canvas.get_renderer()).transformed(temp_ax.transData.inverted())
            except (ValueError, AttributeError): centered_bbox = Bbox.from_extents(-1, -1, 1, 1)
            plt.close(temp_fig)
            content_dims = np.array([centered_bbox.width, centered_bbox.height])
            content_dims[content_dims < 1e-6] = 1.0
            autofit_fill = float(getattr(args, 'autofit_fill', 0.92))
            scale_ratio = min((np.array(frame_dims) * autofit_fill)[0] / content_dims[0], (np.array(frame_dims) * autofit_fill)[1] / content_dims[1])
            logging.debug(f"    - Calculated new scale ratio: {scale_ratio:.4f}")
            if self.template_scale_ratio is None:
                self.template_scale_ratio = scale_ratio
                logging.info(f"    - Scale ratio {scale_ratio:.4f} saved as template for consistency.")
        
        return {node: tuple((p - visual_center) * scale_ratio) for node, p in pos_np.items()}

    def _draw_legend_on_ax(self, ax: Axes, interaction_types: set, atom_elements: set, args: argparse.Namespace, is_summary=False):
        label_map = {"hydrophobic_interactions": "Hydrophobic", "hydrogen_bonds": "H-bond", "pi_stacks": "Pi-stacking", "salt_bridges": "Salt bridge", "pi_cation_interactions": "Pi-cation", "metal_complexes": "Metal complex", "halogen_bonds": "Halogen bond"}
        interaction_handles = []
        for itype in sorted(list(interaction_types)):
            linestyle = "--"
            if itype == "hydrophobic_interactions":
                linestyle = "-"
            interaction_handles.append(Line2D([0], [0], color=INTERACTION_COLORS.get(itype, 'gray'), lw=args.interaction_edge_width, linestyle=linestyle, label=label_map.get(itype, itype.replace("_", "-").capitalize())))
        atom_handles = [Line2D([0], [0], marker="o", color='w', markerfacecolor=ATOM_COLORS.get(elem.upper(), 'black'), linestyle="None", label=elem.upper(), markersize=8) for elem in sorted(list(atom_elements))]
        all_handles = interaction_handles + atom_handles
        if is_summary:
            ax.legend(handles=all_handles, loc='center', fontsize=12, frameon=True, framealpha=0.9, facecolor='#F0F0F0', edgecolor='gray', ncol=2 if len(all_handles) > 5 else 1)
            ax.axis('off')
        else:
            ax.legend(handles=all_handles, loc='best', fontsize=args.font_size - 2, frameon=False)
    
    def create_summary_legend(self, out_dir: str, fmt: str = "svg", args: argparse.Namespace = None):
        if not self.global_interaction_types and not self.global_atom_elements: return
        fig, ax = plt.subplots(figsize=(5, 3.5))
        self._draw_legend_on_ax(ax, self.global_interaction_types, self.global_atom_elements, args, is_summary=True)
        out_path = os.path.join(out_dir, f"legend.{fmt}")
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1, transparent=True)
        plt.close(fig)
        logging.info(f"Summary legend saved to: {out_path}")

    def visualize_interactions(self, folder: str, out_file: Optional[str] = None, args: argparse.Namespace = None, draw_legend: bool = True, use_template: bool = False):
        logging.debug(f"Visualizing interactions for: {os.path.basename(folder)}")
        G = self.build_interaction_graph(folder)
        if not G.nodes:
            logging.warning(f"Could not build a valid graph for {folder}. Skipping plot generation.")
            return

        fig_width_in, fig_height_in = args.fig_width, args.fig_height
        frame_width, frame_height = 20.0, 20.0 * (fig_height_in / fig_width_in)
        args = args or argparse.Namespace()
        swap, sub_rotate, global_rotate = getattr(args, 'swap_substrates', False), getattr(args, 'rotate_substrates', {}), getattr(args, 'rotate', 0.0)
        
        pos = self._apply_template_layout(G, global_rotate) if use_template else self._main_layout_engine(G, swap, sub_rotate)
        final_pos = self._get_autofit_layout(G, pos, (fig_width_in, fig_height_in), (frame_width, frame_height), global_rotate, use_template, args)
        
        if not use_template and self.layout_template is None:
            logging.info("    - First plot layout saved as template for subsequent plots.")
            self.layout_template = pos.copy()
            self.template_residue_map = { self._get_residue_id_num(n): n for n, d in G.nodes(data=True) if d.get('type') == 'residue' and self._get_residue_id_num(n) is not None }
        
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
        ax.set_xlim(-frame_width/2, frame_width/2); ax.set_ylim(-frame_height/2, frame_height/2)
        self._draw_graph_to_ax(ax, G, final_pos, args)
        
        drawn_interaction_types = {d.get('type') for _,_,d in G.edges(data=True) if d.get('type') != 'bond'}
        elements = {d['element'] for _,d in G.nodes(data=True) if d.get('type') in ['atom','ion'] and d.get('element') != 'H'}
        self.global_interaction_types.update(drawn_interaction_types)
        self.global_atom_elements.update(elements)
        if draw_legend: self._draw_legend_on_ax(ax, drawn_interaction_types, elements, args)
        
        ax.set_aspect("equal", anchor='C'); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_linestyle((0, (5, 5))); spine.set_color('gray'); spine.set_alpha(0.7); spine.set_linewidth(0.5)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        if out_file:
            plt.savefig(out_file, dpi=args.dpi, transparent=True, pad_inches=0)
            logging.info(f"Plot saved successfully to: {out_file}")
        else:
            plt.show()
        plt.close(fig)

def main_workflow(args: argparse.Namespace):
    args.output.mkdir(exist_ok=True)
    
    log_file = args.output / 'visualization.log'
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8')]
    )
    if args.verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(console_handler)

    logging.info("======================================================")
    logging.info("=== Starting Interaction Visualization for Classes ===")
    logging.info("======================================================")

    class_folders = sorted([d for d in args.input.iterdir() if d.is_dir() and d.name.startswith('class_')])
    if not class_folders:
        logging.error(f"No 'class_*' folders found in {args.input}. Aborting.")
        return

    logging.info(f"Found {len(class_folders)} classes to visualize.")
    vis_proc = EnzymeSubstrateVisualizer()
    
    class_iterator = tqdm(enumerate(class_folders), total=len(class_folders), desc="Processing Classes")
    for i, class_path in class_iterator:
        class_name = class_path.name
        class_iterator.set_postfix_str(class_name)
        logging.info(f"--- Processing {class_name} ---")
        
        representative_file = next(class_path.glob("*_final.txt"), None)
        if not representative_file:
            logging.warning(f"No representative file in {class_name}. Skipping."); continue
        
        base_name = representative_file.stem.replace('_final', '')
        logging.info(f"Representative conformation: {base_name}")
        plip_subfolder = args.plip_dir / base_name
        
        plot_output_dir = args.output / class_name
        plot_output_dir.mkdir(exist_ok=True)
        out_path = plot_output_dir / f"{class_name}_interactions.{args.format}"
        
        use_template_flag = i > 0
        current_args = args
        if use_template_flag:
            temp_args = argparse.Namespace(**vars(args))
            temp_args.swap_substrates, temp_args.rotate_substrates = False, {}
            current_args = temp_args

        try:
            vis_proc.visualize_interactions(str(plip_subfolder), str(out_path), args=current_args, draw_legend=False, use_template=use_template_flag)
        except Exception as e:
            logging.error(f"Error processing {class_name}: {e}", exc_info=True)
            
    logging.info("--- Generating Summary Legend ---")
    vis_proc.create_summary_legend(str(args.output), args.format, args)
    logging.info("========================================")
    logging.info("=== Visualization process complete! ===")
    logging.info("========================================")
    print(f"\n✅ All plots generated in: {args.output}")
    print(f"📄 Detailed log saved to: {log_file}")

def main():
    class RotateSubstrateAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, 'rotate_substrates'): setattr(namespace, 'rotate_substrates', {})
            getattr(namespace, 'rotate_substrates')[int(values[0])] = values[1]
            
    parser = argparse.ArgumentParser(description="为 GED 分类结果生成代表性相互作用图", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    io_group = parser.add_argument_group('Input/Output Files')
    io_group.add_argument("-i", "--input", required=True, type=Path, help="[输入] GED分类结果目录 (包含 'class_*' 子文件夹)。")
    io_group.add_argument("-p", "--plip_dir", required=True, type=Path, help="[输入] PLIP原始数据目录 (例如 'surviving_plip_outputs')。")
    io_group.add_argument("-o", "--output", required=True, type=Path, help="[输出] 保存最终图像的新目录。")
    io_group.add_argument("-f", "--format", default="svg", choices=["svg", "png", "pdf"], help="输出图像格式。")

    micromanagement_group = parser.add_argument_group('Micro-management (for first class only)')
    micromanagement_group.add_argument("--swap_substrates", action='store_true', help="[位置] 交换最大的两个底物的位置。")
    micromanagement_group.add_argument("--rotate_substrate", nargs=2, type=float, action=RotateSubstrateAction, metavar=('<N>', '<ANGLE>'), help="[朝向] 独立旋转底物 N 指定度数 ANGLE。")
    micromanagement_group.add_argument("--rotate", type=float, default=0.0, help="[全局] 对所有图像进行全局旋转。")

    fig_group = parser.add_argument_group('Figure Styling')
    fig_group.add_argument('--fig_width', type=float, default=65.643/25.4, help="图像宽度 (英寸)。")
    fig_group.add_argument('--fig_height', type=float, default=49.488/25.4, help="图像高度 (英寸)。")
    fig_group.add_argument('--dpi', type=int, default=300, help="图像分辨率 (DPI)。")

    node_group = parser.add_argument_group('Node Styling')
    node_group.add_argument('--node_size', type=int, default=450, help="残基节点大小。")
    node_group.add_argument('--ligand_node_color', type=str, default='skyblue', help="配体原子节点颜色。")
    node_group.add_argument('--protein_node_color', type=str, default='#E0E0E0', help="蛋白质残基节点颜色。")
    node_group.add_argument('--node_border_color', type=str, default='black', help="节点边框颜色。")

    edge_group = parser.add_argument_group('Edge Styling')
    edge_group.add_argument('--covalent_edge_width', type=float, default=0.8, help="共价键线条宽度。")
    edge_group.add_argument('--interaction_edge_width', type=float, default=1.0, help="相互作用线条宽度。")
    edge_group.add_argument('--edge_alpha', type=float, default=0.8, help="线条透明度。")
    edge_group.add_argument('--hbond_edge_color', type=str, default=INTERACTION_COLORS['hydrogen_bonds'], help="氢键颜色。")
    edge_group.add_argument('--saltbridge_edge_color', type=str, default=INTERACTION_COLORS['salt_bridges'], help="盐桥颜色。")
    edge_group.add_argument('--hydrophobic_edge_color', type=str, default=INTERACTION_COLORS['hydrophobic_interactions'], help="疏水相互作用颜色。")
    edge_group.add_argument('--pistack_edge_color', type=str, default=INTERACTION_COLORS['pi_stacks'], help="Pi-stacking颜色。")

    font_group = parser.add_argument_group('Font Styling')
    font_group.add_argument('--font_size', type=int, default=6.5, help="标签字体大小。")
    font_group.add_argument('--font_color', type=str, default='black', help="标签字体颜色。")
    font_group.add_argument('--font_weight', type=str, default='bold', help="标签字体粗细 ('normal', 'bold')。")
    
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument("-v", "--verbose", action="store_true", help="启用详细调试模式，输出更详尽的日志到控制台。")

    args = parser.parse_args()
    main_workflow(args)

if __name__ == "__main__":
    main()

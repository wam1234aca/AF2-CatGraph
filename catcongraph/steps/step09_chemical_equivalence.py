#!/usr/bin/env python3
"""Post-GED ligand contact-unit grouping and merged-class analysis.

Step09 is intentionally post-GED.  It reads the original Step08 GED classes and
Step06 PLIP interaction outputs, applies ligand contact-unit and
interaction-label grouping, merges original GED classes with identical grouped
signatures, computes common/rare interaction frequencies, and exports candidate
catalytic structures.

The implementation is designed to be reusable across enzymes/substrates:
  * interaction labels are grouped into four robust contact types;
  * ligand atom endpoints can be mapped automatically with the universal
    Supplementary Table S4 contact-unit rules, or kept at atom level under the
    conservative policy;
  * unknown or ambiguous atoms are kept as residue-aware atom labels rather than
    being force-merged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
try:  # Only the explicitly selected strict-chemistry mode needs NetworkX here.
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - guarded at strict-mode entry
    nx = None  # type: ignore[assignment]
from catcongraph.profiles import GENERAL, validated_contact_unit_target, workflow_profile
from catcongraph.config import load_yaml_config, public_output_subdir

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Ligand contact-unit defaults
# ---------------------------------------------------------------------------

ION_ATOMS = {"MG", "ZN", "MN", "FE", "CA", "NA", "K", "CU", "CO", "NI", "CD", "HG"}
CONTACT_UNIT_GUIDED = "guided_review"
CONTACT_UNIT_UNIVERSAL = "universal_rules"
CONTACT_UNIT_CONSERVATIVE = "conservative_manual"
METAL_ELEMENTS = {"MG", "ZN", "MN", "FE", "CA", "NA", "K", "CU", "CO", "NI", "CD", "HG"}
LIGAND_DISPLAY_ALIASES = {
    # PDB residue names are limited to three characters. These aliases are
    # ligand-wide display conventions, not enzyme- or target-specific rules.
    "TMP": "dTMP",
    "DTM": "dTMP",
}

# Auditable rule catalogue used by the universal workflow.  The detector below
# applies these rules within each ligand component only.  Unknown atoms are not
# discarded: rule 12 retains them as ligand- and atom-specific sites.
UNIVERSAL_CONTACT_UNIT_RULES: Tuple[Tuple[str, str], ...] = (
    ("carboxylate_oxygen", "Carboxylate oxygen"),
    ("phosphate_or_phosphonate", "Phosphate/phosphonate non-bridging oxygen"),
    ("phosphate_bridging_oxygen", "Phosphate-bridging oxygen"),
    ("sulfate_or_sulfonate_terminal_oxygen", "Sulfate/sulfonate oxygen"),
    ("nitro_oxygen", "Nitro oxygen"),
    ("sulfonyl_oxygen", "Sulfonyl oxygen"),
    ("delocalized_terminal_nitrogen", "Delocalized nitrogen"),
    ("aromatic_carbon_fragment", "Aromatic ring fragment"),
    ("non_aromatic_carbon_fragment", "Alkyl fragment"),
    ("sugar_ring_carbon_fragment", "Sugar-ring carbon fragment"),
    ("metal_ion", "Metal ion"),
    ("individual_atom", "Individual atom"),
)

# Built-in contact-unit suggestions for the supported example targets.
BUILTIN_LIGAND_UNITS: Dict[str, Dict[str, List[str]]] = {
    "GAC": {
        "N1 site": ["N1"],
        "C1 site": ["C1"],
        "O1 site": ["O1"],
        "Aliphatic carbon unit": ["C2", "C3", "C4", "C5"],
        "N2 site": ["N2"],
        "Carboxylate oxygen unit": ["O2", "O3"],
    },
    "AqTMK": {
        "ATP phosphate 1 unit": ["P1", "O2", "O3", "O4"],
        "ATP phosphate 2 unit": ["P2", "O6", "O7"],
        "ATP phosphate 3 unit": ["P3", "O9", "O10", "O11"],
        "dTMP phosphate unit": ["P4", "O12", "O13", "O14", "O15"],
        "ATP phosphate 1–2 bridging oxygen": ["O5"],
        "ATP phosphate 2–3 bridging oxygen": ["O8"],
        "Mg2+ ion": ["MG"],
    },
    "2PBR": {
        "ATP phosphate 1 unit": ["P1", "O2", "O3", "O4"],
        "ATP phosphate 2 unit": ["P2", "O6", "O7"],
        "ATP phosphate 3 unit": ["P3", "O9", "O10", "O11"],
        "dTMP phosphate unit": ["P4", "O12", "O13", "O14", "O15"],
        "ATP phosphate 1–2 bridging oxygen": ["O5"],
        "ATP phosphate 2–3 bridging oxygen": ["O8"],
        "Mg2+ ion": ["MG"],
    },
    "PTP1B": {
        "Aliphatic carbon unit": ["C1", "C2"],
        "Carboxylate oxygen unit": ["O5", "O6"],
        "Phosphate unit": ["O1", "O2", "O3", "O4", "P1"],
        "N1 site": ["N1"],
        "Aromatic ring unit": ["C3", "C4", "C5", "C6", "C7", "C8"],
    },
    "6B90": {
        "Aliphatic carbon unit": ["C1", "C2"],
        "Carboxylate oxygen unit": ["O5", "O6"],
        "Phosphate unit": ["O1", "O2", "O3", "O4", "P1"],
        "N1 site": ["N1"],
        "Aromatic ring unit": ["C3", "C4", "C5", "C6", "C7", "C8"],
    },
}


def equivalence_mode(cfg: Mapping[str, object]) -> str:
    """Return the backward-compatible contact-unit grouping contract.

    ``standard`` is the default. With identical upstream inputs it follows the
    established mapping precedence, built-in definitions, automatic unit
    detector, and class-merging semantics. Strict chemistry inference remains
    an explicit opt-in.
    """
    can = ((cfg.get("step09", {}) or {}).get("canonicalization", {}) or {})  # type: ignore[union-attr]
    raw = str(can.get("equivalence_mode", "standard") or "standard").strip().lower()
    aliases = {
        "standard": "standard", "catcongraph_studio_v1": "standard",
        "v8": "standard", "v8_exact": "standard", "legacy": "standard", "compat": "standard",
        "chemistry_aware": "strict_chemistry", "strict": "strict_chemistry",
        "strict_chemistry": "strict_chemistry",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"standard", "strict_chemistry"}:
        raise ValueError(
            "contact-unit equivalence_mode must be 'standard' or "
            f"'strict_chemistry', got {raw!r}."
        )
    return mode


def contact_unit_mode(cfg: Mapping[str, object]) -> str:
    """Return the public policy for obtaining ligand contact-unit mappings.

    Historical configs did not carry this field. They retain their previous
    behaviour by defaulting to guided review. Only an explicit conservative
    selection disables built-ins and proposal generation.
    """
    can = ((cfg.get("step09", {}) or {}).get("canonicalization", {}) or {})  # type: ignore[union-attr]
    raw = str(can.get("contact_unit_mode", CONTACT_UNIT_GUIDED) or CONTACT_UNIT_GUIDED).strip().lower()
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
            "contact_unit_mode must be 'universal_rules', 'guided_review', "
            "or 'conservative_manual', "
            f"got {raw!r}."
        )
    return mode

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

DEFAULT_TYPE_MAP = {
    "hydrogen_bonds": "Polar contact",
    "hydrogen_bond": "Polar contact",
    "hydrogen bond": "Polar contact",
    "hbond": "Polar contact",
    "hbonds": "Polar contact",
    "salt_bridges": "Polar contact",
    "salt_bridge": "Polar contact",
    "salt bridge": "Polar contact",
    "saltbridge": "Polar contact",
    "ionic": "Polar contact",
    "water_bridges": "Polar contact",
    "water_bridge": "Polar contact",
    "water bridge": "Polar contact",
    "halogen_bonds": "Polar contact",
    "halogen_bond": "Polar contact",
    "halogen bond": "Polar contact",
    "hydrophobic_interactions": "Hydrophobic contact",
    "hydrophobic_interaction": "Hydrophobic contact",
    "hydrophobic interaction": "Hydrophobic contact",
    "hydrophobic": "Hydrophobic contact",
    "pi_stacks": "π contact",
    "pi_stack": "π contact",
    "pi-stacking": "π contact",
    "pi stacking": "π contact",
    "pistacking": "π contact",
    "pi_cation_interactions": "π contact",
    "pi_cation": "π contact",
    "pi-cation": "π contact",
    "pication": "π contact",
    "metal_complexes": "Metal contact",
    "metal_complex": "Metal contact",
    "metal coordination": "Metal contact",
    "metal": "Metal contact",
}

COVALENT_CUTOFFS = {
    tuple(sorted(("P", "O"))): 1.95,
    tuple(sorted(("P", "N"))): 1.95,
    tuple(sorted(("S", "O"))): 1.85,
    tuple(sorted(("N", "O"))): 1.60,
    tuple(sorted(("C", "O"))): 1.70,
    tuple(sorted(("C", "N"))): 1.75,
    tuple(sorted(("C", "C"))): 1.85,
    tuple(sorted(("C", "S"))): 1.90,
    tuple(sorted(("C", "P"))): 2.00,
}


@dataclass(frozen=True)
class LigAtom:
    serial: int
    atom_name: str
    resname: str
    chain: str
    resseq: str
    element: str
    x: float
    y: float
    z: float

    @property
    def identity(self) -> Tuple[str, str, str, str]:
        return (self.resname.upper(), self.chain or "_", str(self.resseq), self.atom_name.upper())


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def natural_key(s: object) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def class_num(c: object) -> int:
    m = re.search(r"(\d+)$", str(c))
    return int(m.group(1)) if m else 10**9


def standard_class_name(value: object) -> str:
    s = str(value).strip()
    if not s:
        return s
    # Preserve manuscript labels such as Type1/Family1 if they are already used.
    if re.search(r"^(type|family)[_\s-]*\d+$", s, flags=re.I):
        m = re.search(r"^(type|family)[_\s-]*(\d+)$", s, flags=re.I)
        assert m is not None
        prefix = m.group(1)
        return f"{prefix}{int(m.group(2))}"
    if s.lower().startswith("class_"):
        return "class_" + str(class_num(s))
    if s.isdigit():
        return f"class_{int(s)}"
    m = re.search(r"class[_\s-]*(\d+)", s, flags=re.I)
    if m:
        return f"class_{int(m.group(1))}"
    return s


def class_display_name(value: object, prefix: str = "Class") -> str:
    text = str(value)
    chem = re.search(r"__chem(\d+)", text, flags=re.I)
    n = class_num(value)
    if n < 10**9:
        base = f"{prefix} {n}"
        return f"{base} (chem {int(chem.group(1))})" if chem else base
    return text


def class_display_list(values: object, prefix: str = "Class") -> str:
    parts = [p for p in str(values or "").split(";") if p]
    return "; ".join(class_display_name(p, prefix=prefix) for p in parts)


def protein_display_label(label: object, preserve_chain: bool = False) -> str:
    """Return manuscript-style protein labels such as D9, K13, or Y46."""
    s = str(label or "").strip()
    # Strip a single chain suffix from Step06 labels when present.
    m = re.fullmatch(r"([A-Z]{3})(-?\d+)([A-Za-z]?)", s)
    if m:
        aa = AA3_TO_AA1.get(m.group(1).upper(), m.group(1)[:1].upper())
        chain = m.group(3)
        suffix = f"[{chain}]" if preserve_chain and chain else ""
        return f"{aa}{int(m.group(2))}{suffix}"
    return s


def _looks_like_class_dir(name: str) -> bool:
    return bool(re.search(r"^(class|type|family)[_\s-]*\d+$", str(name), flags=re.I))


def _class_sort_key(value: object) -> str:
    text = str(value)
    original = re.search(r"(?:class|type|family)[_\s-]*(\d+)", text, flags=re.I)
    chem = re.search(r"__chem(\d+)", text, flags=re.I)
    original_num = int(original.group(1)) if original else class_num(value)
    chem_num = int(chem.group(1)) if chem else 0
    return f"{original_num:06d}_{chem_num:04d}_{text}"


def build_sequential_merged_class_tables(
    class_signature_df: pd.DataFrame,
    structure_class_df: pd.DataFrame,
) -> Tuple[List[dict], List[dict]]:
    """Merge identical canonical signatures and assign contiguous labels.

    Ligand contact-unit grouping is unchanged. The only changed behavior is
    the final merged-class name:

      * groups are ordered by their smallest original Step08 class;
      * the ordered groups become class_1, class_2, ..., class_N;
      * all original Step08 labels remain in ``original_classes`` and the
        original-to-merged mapping table.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for _, row in class_signature_df.iterrows():
        signature = str(row.get("canonical_signature", ""))
        original_class = str(row.get("class", ""))
        if original_class:
            groups[signature].append(original_class)

    def group_order_key(item: Tuple[str, List[str]]):
        _signature, classes = item
        ordered = tuple(sorted(set(classes), key=_class_sort_key))
        anchor = _class_sort_key(ordered[0]) if ordered else _class_sort_key("")
        return anchor, tuple(_class_sort_key(value) for value in ordered)

    ordered_groups = sorted(groups.items(), key=group_order_key)

    merged_rows: List[dict] = []
    original_to_merged_rows: List[dict] = []

    for sequential_index, (signature, classes) in enumerate(ordered_groups, start=1):
        classes_sorted = sorted(set(classes), key=_class_sort_key)
        merged_class = f"class_{sequential_index}"

        subset = structure_class_df[
            structure_class_df["class"].astype(str).isin(classes_sorted)
        ]
        structures = sorted(
            set(subset["structure_id"].astype(str)),
            key=natural_key,
        )

        for original_class in classes_sorted:
            original_to_merged_rows.append(
                {
                    "original_class": original_class,
                    "merged_class": merged_class,
                    "canonical_signature": signature,
                }
            )

        merged_rows.append(
            {
                "merged_class": merged_class,
                "n_original_classes": len(classes_sorted),
                "original_classes": ";".join(classes_sorted),
                "n_structures": len(structures),
                "structure_ids": ";".join(structures),
                "canonical_signature_edge_count": len(
                    [value for value in signature.split(";") if value]
                ),
                "canonical_signature": signature,
            }
        )

    return merged_rows, original_to_merged_rows


def strip_suffixes(name: object) -> str:
    s = str(name).strip()
    s = Path(s).name
    for suffix in [".txt", ".pdb", ".csv"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    changed = True
    suffixes = [
        "_interaction_output",
        "_interactions",
        "_final",
        "_graph",
        "_trimmed_complex",
        "_docked_complex",
        "_complex_complex_complex",
        "_complex_complex",
        "_complex",
        "_model",
    ]
    while changed:
        changed = False
        for suffix in suffixes:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                changed = True
    return s


def normalize_structure_id(value: object) -> str:
    return strip_suffixes(value)


def read_bool(x: object, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    text = str(x).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_yaml(path: Path) -> dict:
    return load_yaml_config(path)


def resolve_path(value: object, config_path: Path, cwd: Optional[Path] = None) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    cwd = cwd or Path.cwd()
    if (cwd / p).exists() or str(value).startswith(("results/", "projects/", "data/")):
        return (cwd / p).resolve()
    if config_path.parent.parent.name == "projects":
        root = config_path.parent.parent.parent
        if (root / p).exists() or str(value).startswith(("results/", "data/")):
            return (root / p).resolve()
    return (config_path.parent / p).resolve()


def get_project(cfg: dict) -> Tuple[str, Path]:
    project = cfg.get("project", {}) or {}
    if not isinstance(project, dict):
        project = {}
    target_id = str(project.get("target_id") or cfg.get("target_id") or "target").strip()
    out = project.get("output_root") or project.get("output_dir") or cfg.get("output_root") or f"results/{target_id}"
    return target_id, Path(str(out))


def write_csv(path: Path, df_or_rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_rows, pd.DataFrame):
        df_or_rows.to_csv(path, index=False)
        return
    pd.DataFrame(df_or_rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def _is_probable_step08_class_table(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return False
    name = path.name.lower()
    if "graph_classes" in name or "class_assign" in name:
        return True
    try:
        columns = {str(c).strip().lower() for c in pd.read_csv(path, nrows=0).columns}
    except Exception:
        return False
    class_like = bool(columns & {"class", "class_id", "graph_class", "type", "family"})
    structure_like = bool(columns & {"structure_id", "graph_id", "graph_file", "filename", "file"})
    return class_like and structure_like


def _class_directory_if_valid(path: Path) -> Optional[Path]:
    if not path.exists() or not path.is_dir():
        return None
    try:
        if any(x.is_dir() and _looks_like_class_dir(x.name) for x in path.iterdir()):
            return path
    except OSError:
        return None
    return None


def _step08_paths_from_report(report_path: Path, config_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    table_value = (report.get("tables", {}) or {}).get("graph_classes")
    class_value = (report.get("directories", {}) or {}).get("classes")
    table_path = resolve_path(table_value, config_path) if table_value else None
    class_path = resolve_path(class_value, config_path) if class_value else None
    if table_path and _is_probable_step08_class_table(table_path):
        return table_path, None
    valid_class_dir = _class_directory_if_valid(class_path) if class_path else None
    if valid_class_dir:
        return None, valid_class_dir
    return None, None


def _inspect_step08_root(root: Path, target_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    if not root.exists():
        return None, None
    exact_candidates = [
        root / "tables" / f"{target_id}_step07_graph_classes.csv",
        root / "tables" / f"{target_id}_step08_graph_classes.csv",
        root / "tables" / f"{target_id}_graph_classes.csv",
        root / "tables" / "graph_classes.csv",
        root / f"{target_id}_step08_graph_classes.csv",
        root / "graph_classes.csv",
    ]
    for path in exact_candidates:
        if _is_probable_step08_class_table(path):
            return path, None

    for pattern in ("*step07*graph_classes*.csv", "*step08*graph_classes*.csv", "*graph_classes*.csv", "*class_assign*.csv"):
        for path in sorted(root.rglob(pattern)):
            if _is_probable_step08_class_table(path):
                return path, None

    for path in [root / "classes", root]:
        valid = _class_directory_if_valid(path)
        if valid:
            return None, valid
    for path in sorted(root.rglob("classes")):
        valid = _class_directory_if_valid(path)
        if valid:
            return None, valid

    result_files = sorted(root.rglob("ged_classification_results.txt"))
    if result_files:
        return result_files[0], None
    return None, None


def find_step08_class_table(cfg: dict, config_path: Path, output_root: Path, target_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    step09 = cfg.get("step09", {}) or {}
    step08 = cfg.get("step08", {}) or {}
    searched: List[Path] = []

    explicit_table = step09.get("input_step08_class_table")
    if explicit_table and str(explicit_table).strip().lower() not in {"auto", "none", "null"}:
        path = resolve_path(explicit_table, config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configured step09.input_step08_class_table does not exist: {path}"
            )
        if path.is_dir():
            table_path, class_dir = _inspect_step08_root(path, target_id)
            if table_path or class_dir:
                return table_path, class_dir
            raise FileNotFoundError(
                f"Configured Step08 directory contains no graph class table or class folders: {path}"
            )
        return path, None

    roots: List[Path] = []
    explicit_dir = step09.get("input_step08_dir")
    if explicit_dir and str(explicit_dir).strip().lower() not in {"auto", "none", "null"}:
        roots.append(resolve_path(explicit_dir, config_path))
    output_subdir = public_output_subdir(
        step08.get("output_subdir"), "07_ged_classification"
    )
    roots.extend(
        [
            output_root / output_subdir,
            output_root / "07_ged_classification",
            output_root / "08_ged_classification",
            output_root / "08_graph_classification",
            output_root / "08_graph_edit_distance",
        ]
    )

    deduped_roots: List[Path] = []
    seen = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if str(root) not in seen:
            deduped_roots.append(root)
            seen.add(str(root))

    for root in deduped_roots:
        searched.append(root)
        table_path, class_dir = _inspect_step08_root(root, target_id)
        if table_path or class_dir:
            return table_path, class_dir

    report_candidates: List[Path] = []
    for root in deduped_roots:
        report_candidates.extend(sorted(root.glob("reports/*_step07_report.json")))
        report_candidates.extend(sorted(root.glob("reports/*_step08_report.json")))
    if output_root.exists():
        report_candidates.extend(sorted(output_root.rglob("*_step07_report.json")))
        report_candidates.extend(sorted(output_root.rglob("*_step08_report.json")))
    report_seen = set()
    for report_path in report_candidates:
        if str(report_path) in report_seen:
            continue
        report_seen.add(str(report_path))
        table_path, class_dir = _step08_paths_from_report(report_path, config_path)
        if table_path or class_dir:
            return table_path, class_dir

    # Last-resort compatibility search for older/custom Step08 directory names.
    if output_root.exists():
        for path in sorted(output_root.rglob("*.csv")):
            lower = str(path).lower()
            if ("07" in lower or "08" in lower) and _is_probable_step08_class_table(path):
                return path, None
        for path in sorted(output_root.rglob("classes")):
            if "07" in str(path).lower() or "08" in str(path).lower():
                valid = _class_directory_if_valid(path)
                if valid:
                    return None, valid

    searched_text = "\n  - ".join(str(path) for path in searched) or str(output_root)
    raise FileNotFoundError(
        "Cannot find Step08 graph classification output. Step09 requires the Step08 class table or class folders.\n"
        f"Searched:\n  - {searched_text}\n"
        f"Run Step08 first:\n  python scripts/08_ged_classification.py --config {config_path} --clean\n"
        "Then rerun Step09, or set step09.input_step08_class_table / step09.input_step08_dir to the actual Step08 output."
    )


def resolve_step06_dir(cfg: dict, config_path: Path, output_root: Path) -> Path:
    step09 = cfg.get("step09", {}) or {}
    explicit = step09.get("input_step06_dir")
    if explicit and str(explicit).lower() != "auto":
        p = resolve_path(explicit, config_path)
        if p.exists():
            return p
    candidates = [
        output_root / "05_plip_interaction_graphs",
        output_root / "05_plip_interaction_graphs" / "plip_outputs",
        output_root / "06_plip_interaction_graphs",
        output_root / "06_plip_interaction_graphs" / "plip_outputs",
        output_root / "plip_outputs",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find Step06 PLIP output directory. Set step09.input_step06_dir.")


def find_step07_passed_table(
    cfg: dict,
    config_path: Path,
    output_root: Path,
    target_id: str,
) -> Optional[Path]:
    """Locate the Step07-passed structure table used for frequency reporting.

    This table is deliberately resolved independently of Step08.  The resulting
    interaction-frequency denominator therefore remains the complete population
    that passed Step07, even if a graph is later missing or GED classification
    fails for one of those structures.
    """
    step09 = cfg.get("step09", {}) or {}
    frequency_cfg = step09.get("step07_interaction_frequency", {}) or {}
    explicit = frequency_cfg.get("input_passed_table") or step09.get("input_step07_passed_table")
    if explicit and str(explicit).strip().lower() not in {"auto", "none", "null"}:
        path = resolve_path(explicit, config_path)
        if not path.exists():
            raise FileNotFoundError(f"Step07 passed structure table not found: {path}")
        return path

    candidates = [
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step06_passed_complex_index.csv",
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step06_structure_key_residue_filter.csv",
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step07_passed_complex_index.csv",
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step07_structure_key_residue_filter.csv",
        output_root / "07_key_residue_screening" / "tables" / f"{target_id}_step07_passed_complex_index.csv",
        output_root / "07_key_residue_screening" / "tables" / f"{target_id}_step07_structure_key_residue_filter.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def read_step07_passed_structures(table_path: Path) -> List[dict]:
    """Return one normalized record per structure that passed Step07."""
    df = pd.read_csv(table_path)
    if "pass_key_residue_filter" in df.columns:
        mask = df["pass_key_residue_filter"].map(lambda value: read_bool(value, False))
        df = df[mask].copy()

    rows: List[dict] = []
    seen: Set[str] = set()
    for _, row in df.iterrows():
        aliases: List[str] = []
        for column in [
            "structure_id", "normalized_structure_id", "pdb_path", "complex_path",
            "trimmed_complex_path", "passed_complex_path",
        ]:
            if column not in df.columns or pd.isna(row.get(column)):
                continue
            value = str(row.get(column)).strip()
            if not value:
                continue
            aliases.extend([value, Path(value).stem, normalize_structure_id(value)])
        aliases = list(dict.fromkeys(alias for alias in aliases if alias))
        if not aliases:
            continue
        raw_structure_id = row.get("structure_id") if "structure_id" in df.columns else None
        if raw_structure_id is None or pd.isna(raw_structure_id) or not str(raw_structure_id).strip():
            raw_structure_id = aliases[0]
        structure_id = normalize_structure_id(raw_structure_id) or normalize_structure_id(aliases[0])
        if structure_id in seen:
            continue
        seen.add(structure_id)
        rows.append({"structure_id": structure_id, "aliases": aliases})
    return rows


def collect_step07_passed_interactions(
    passed_structures: Sequence[Mapping[str, object]],
    plip_index: Mapping[str, Path],
    pdb_index: Mapping[str, Path],
) -> Tuple[List[dict], List[dict], List[Path]]:
    """Parse every PLIP interaction for the complete Step07-passed population."""
    raw_records: List[dict] = []
    missing: List[dict] = []
    ligand_pdbs: List[Path] = []
    for item in passed_structures:
        structure_id = str(item.get("structure_id") or "")
        aliases = [structure_id] + [str(x) for x in (item.get("aliases") or [])]
        interaction_path = next(
            (path for alias in aliases if (path := find_interaction_file(alias, plip_index)) is not None),
            None,
        )
        if interaction_path is None:
            missing.append({"structure_id": structure_id, "reason": "missing_interaction_output"})
            continue
        raw_records.extend(parse_interaction_output(interaction_path, structure_id, "Step07_passed"))

        pdb_path = next(
            (path for alias in aliases if (path := find_pdb_for_structure(alias, pdb_index)) is not None),
            None,
        )
        if pdb_path is not None:
            ligand_pdbs.append(pdb_path)
    return raw_records, missing, ligand_pdbs


def build_step07_interaction_frequency_tables(
    canonical_records: pd.DataFrame,
    passed_structure_ids: Iterable[str],
    *,
    high_frequency_min: float = 0.80,
    low_frequency_max: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build detailed and publication-style interaction-frequency tables.

    Each canonical interaction is counted at most once per structure.  Frequency
    is divided by all structures that passed Step07, including a passed structure
    whose PLIP output is missing; missing files are reported separately.
    """
    structure_ids = {normalize_structure_id(value) for value in passed_structure_ids if normalize_structure_id(value)}
    n_total = len(structure_ids)
    edge_to_structures: Dict[str, Set[str]] = defaultdict(set)
    if not canonical_records.empty:
        for _, row in canonical_records.iterrows():
            edge = str(row.get("canonical_edge") or "").strip()
            structure_id = normalize_structure_id(row.get("structure_id"))
            if edge and structure_id in structure_ids:
                edge_to_structures[edge].add(structure_id)

    rows: List[dict] = []
    for edge, members in edge_to_structures.items():
        frequency = len(members) / n_total if n_total else 0.0
        if frequency >= high_frequency_min:
            category = "High-frequency"
        elif frequency < low_frequency_max:
            category = "Low-frequency"
        else:
            category = "Other"
        rows.append(
            {
                "Interaction": edge,
                "Structures with interaction": len(members),
                "Step07-passed structures": n_total,
                "Frequency": frequency,
                "Category": category,
                "Structure IDs": ";".join(sorted(members, key=natural_key)),
            }
        )
    detailed = pd.DataFrame(
        rows,
        columns=[
            "Interaction", "Structures with interaction", "Step07-passed structures",
            "Frequency", "Category", "Structure IDs",
        ],
    )
    if not detailed.empty:
        detailed = detailed.sort_values(
            ["Frequency", "Interaction"], ascending=[False, True]
        ).reset_index(drop=True)
    publication = detailed[["Interaction", "Frequency", "Category"]].copy()
    return detailed, publication


def build_all_collected_interaction_frequency_tables(
    canonical_records: pd.DataFrame,
    population_structure_ids: Iterable[str],
    *,
    population_label: str,
    high_frequency_min: float = 0.80,
    low_frequency_max: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete Step09 interaction-frequency table.

    This output is intentionally separate from the candidate-class tables.  It
    reports every canonical interaction collected from the selected population,
    counts an interaction at most once per structure, and labels it as:

      * High-frequency: frequency >= ``high_frequency_min``
      * Low-frequency:  frequency < ``low_frequency_max``
      * Other:          all remaining interactions

    The publication table has exactly the three columns requested for the
    manuscript: Interaction, Frequency, and Category.  The detailed table adds
    residue/type/ligand-unit fields, numerator, denominator, thresholds, source
    population, and supporting structure IDs.
    """
    structure_ids = {
        normalize_structure_id(value)
        for value in population_structure_ids
        if normalize_structure_id(value)
    }
    n_total = len(structure_ids)

    edge_to_structures: Dict[str, Set[str]] = defaultdict(set)
    edge_metadata: Dict[str, Tuple[str, str, str]] = {}

    if not canonical_records.empty:
        for _, row in canonical_records.iterrows():
            edge = str(row.get("canonical_edge") or "").strip()
            structure_id = normalize_structure_id(row.get("structure_id"))
            if not edge or structure_id not in structure_ids:
                continue
            edge_to_structures[edge].add(structure_id)
            edge_metadata[edge] = (
                str(row.get("protein_canonical") or "").strip(),
                str(row.get("interaction_supertype") or "").strip(),
                str(row.get("canonical_ligand_unit") or "").strip(),
            )

    rows: List[dict] = []
    for edge, members in edge_to_structures.items():
        frequency = len(members) / n_total if n_total else 0.0
        if frequency >= high_frequency_min:
            category = "High-frequency"
        elif frequency < low_frequency_max:
            category = "Low-frequency"
        else:
            category = "Other"

        protein, interaction_type, ligand_unit = edge_metadata.get(edge, ("", "", ""))
        rows.append(
            {
                "Population": population_label,
                "Interaction": edge,
                "Protein residue": protein,
                "Interaction type": interaction_type,
                "Ligand contact unit": ligand_unit,
                "Structures with interaction": len(members),
                "Total structures": n_total,
                "Frequency": frequency,
                "Category": category,
                "High-frequency threshold": high_frequency_min,
                "Low-frequency threshold": low_frequency_max,
                "Structure IDs": ";".join(sorted(members, key=natural_key)),
            }
        )

    detailed = pd.DataFrame(
        rows,
        columns=[
            "Population",
            "Interaction",
            "Protein residue",
            "Interaction type",
            "Ligand contact unit",
            "Structures with interaction",
            "Total structures",
            "Frequency",
            "Category",
            "High-frequency threshold",
            "Low-frequency threshold",
            "Structure IDs",
        ],
    )
    if not detailed.empty:
        detailed = detailed.sort_values(
            ["Frequency", "Interaction"],
            ascending=[False, True],
        ).reset_index(drop=True)

    publication = detailed[["Interaction", "Frequency", "Category"]].copy()
    return detailed, publication


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    lower = {c.lower(): c for c in df.columns}
    class_candidates = ["class", "ged_class", "class_id", "original_class", "original_ged_class", "cluster", "family"]
    struct_candidates = ["structure_id", "structure", "structure_name", "graph_file", "graph_name", "filename", "file", "member", "pdb_id"]
    path_candidates = ["graph_path", "path", "file_path"]
    class_col = next((lower[c] for c in class_candidates if c in lower), None)
    struct_col = next((lower[c] for c in struct_candidates if c in lower), None)
    graph_col = next((lower[c] for c in ["graph_file", "graph_name", "filename", "file"] if c in lower), None)
    graph_path_col = next((lower[c] for c in path_candidates if c in lower), None)
    if class_col is None:
        for c in df.columns:
            if "class" in c.lower():
                class_col = c
                break
    if struct_col is None:
        for c in df.columns:
            if any(k in c.lower() for k in ["structure", "graph", "file", "name"]):
                struct_col = c
                break
    if class_col is None or struct_col is None:
        raise ValueError(f"Cannot infer class/structure columns from: {df.columns.tolist()}")
    return class_col, struct_col, graph_col, graph_path_col


def read_class_table(table_path: Path) -> pd.DataFrame:
    if table_path.suffix.lower() == ".csv":
        df = pd.read_csv(table_path)
        class_col, struct_col, graph_col, graph_path_col = infer_columns(df)
        out = pd.DataFrame()
        out["class"] = df[class_col].map(standard_class_name)
        out["structure_id"] = df[struct_col].map(normalize_structure_id)
        out["graph_file"] = df[graph_col].astype(str) if graph_col else out["structure_id"] + "_final.txt"
        out["graph_path"] = df[graph_path_col].astype(str) if graph_path_col else ""
        return out.drop_duplicates().sort_values(["class", "structure_id"], key=lambda s: s.map(str)).reset_index(drop=True)

    records = []
    current = None
    for line in table_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"class[_\s-]*(\d+)", line, flags=re.I)
        if m:
            current = f"class_{int(m.group(1))}"
        if current:
            for token in re.findall(r"[\w.\-]+_final\.txt|[\w.\-]+\.pdb|[\w.\-]+_complex", line):
                records.append({"class": current, "structure_id": normalize_structure_id(token), "graph_file": token, "graph_path": ""})
    if not records:
        raise ValueError(f"Cannot parse Step08 class file: {table_path}")
    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


def reconstruct_class_table(class_dir: Path) -> pd.DataFrame:
    records = []
    for d in sorted(class_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if not d.is_dir() or not _looks_like_class_dir(d.name):
            continue
        cname = standard_class_name(d.name)
        for fp in sorted(d.glob("*_final.txt"), key=lambda p: natural_key(p.name)):
            records.append({"class": cname, "structure_id": normalize_structure_id(fp.name), "graph_file": fp.name, "graph_path": str(fp)})
    if not records:
        raise ValueError(f"No class_*/ *_final.txt files found under {class_dir}")
    return pd.DataFrame(records)


def _pdb_has_hetatm(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return any(line.startswith("HETATM") for line in fh)
    except Exception:
        return False


def build_plip_index(step06_dir: Path) -> Dict[str, Path]:
    roots = []
    if (step06_dir / "plip_outputs").exists():
        roots.append(step06_dir / "plip_outputs")
    roots.append(step06_dir)
    index: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("interaction_output.txt"):
            folder = p.parent
            aliases = {folder.name, normalize_structure_id(folder.name)}
            for pdb in folder.glob("*.pdb"):
                aliases.add(pdb.stem)
                aliases.add(normalize_structure_id(pdb.stem))
                if pdb.stem.endswith("_complex"):
                    aliases.add(pdb.stem[:-8])
            for a in aliases:
                if a:
                    index.setdefault(a, p)
    return index


def build_plip_pdb_index(step06_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    roots = []
    if (step06_dir / "plip_outputs").exists():
        roots.append(step06_dir / "plip_outputs")
    roots.append(step06_dir)
    for root in roots:
        if not root.exists():
            continue
        for pdb in root.rglob("*.pdb"):
            if not _pdb_has_hetatm(pdb):
                continue
            aliases = {pdb.parent.name, pdb.stem, normalize_structure_id(pdb.parent.name), normalize_structure_id(pdb.stem)}
            if pdb.stem.endswith("_complex"):
                aliases.add(pdb.stem[:-8])
            for a in aliases:
                if a:
                    index.setdefault(a, pdb)
    return index


def find_interaction_file(structure_id: str, plip_index: Mapping[str, Path]) -> Optional[Path]:
    sid = normalize_structure_id(structure_id)
    candidates = [structure_id, sid]
    if sid.endswith("_complex"):
        candidates.append(sid[:-8])
    if sid.endswith("_final"):
        candidates.append(sid[:-6])
    for c in candidates:
        if c in plip_index:
            return plip_index[c]
    matches = [(k, v) for k, v in plip_index.items() if k.endswith(sid) or sid.endswith(k)]
    if matches:
        return sorted(matches, key=lambda kv: (len(kv[0]), kv[0]))[0][1]
    return None


# ---------------------------------------------------------------------------
# PDB chemistry detection
# ---------------------------------------------------------------------------


def _guess_element(atom_name: str, element: str = "") -> str:
    e = str(element or "").strip().upper()
    if e:
        return e[:2] if len(e) > 1 and e[:2] in ION_ATOMS else e[0]
    name = str(atom_name).strip().upper().lstrip("0123456789")
    if len(name) >= 2 and name[:2] in ION_ATOMS:
        return name[:2]
    return re.sub(r"[^A-Z]", "", name)[:1] or ""


def parse_ligand_atoms_from_pdb(path: Path) -> Tuple[List[LigAtom], Dict[int, Set[int]]]:
    """Read the ligand heavy-atom graph used by Step08.

    PLIP/Open Babel may append ligand hydrogens to ``*_complex_protonated.pdb``
    files even when the project ligand template contains heavy atoms only.
    Hydrogen and deuterium are not ligand contact units in CatConGraph, so they
    must not enter the ensemble audit or alter heavy-atom terminality tests.
    Filtering them before the adjacency graph is built also removes their
    ``CONECT`` edges from every automatic contact-unit policy.
    """
    atoms: List[LigAtom] = []
    by_serial: Dict[int, LigAtom] = {}
    conect: Dict[int, Set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("HETATM"):
                try:
                    serial = int(line[6:11])
                    atom_name = line[12:16].strip().upper()
                    resname = line[17:20].strip().upper()
                    chain = line[21:22].strip() or "_"
                    resseq = line[22:26].strip()
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    element = _guess_element(atom_name, line[76:78] if len(line) >= 78 else "")
                except Exception:
                    continue
                if element.upper() in {"H", "D"}:
                    continue
                atom = LigAtom(serial, atom_name, resname, chain, resseq, element, x, y, z)
                atoms.append(atom)
                by_serial[serial] = atom
            elif line.startswith("CONECT"):
                nums = [int(x) for x in re.findall(r"\d+", line[6:])]
                if len(nums) >= 2:
                    src = nums[0]
                    for dst in nums[1:]:
                        conect[src].add(dst)
                        conect[dst].add(src)
    ligand_serials = set(by_serial)
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for a, neighs in conect.items():
        if a not in ligand_serials:
            continue
        for b in neighs:
            if b in ligand_serials:
                adjacency[a].add(b)
                adjacency[b].add(a)
    # Conservative distance fallback is evaluated per ligand component.  This
    # handles a multi-ligand PDB in which one component has CONECT records but
    # another component does not, without ever inferring bonds across residues.
    by_component: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
    for atom in atoms:
        by_component[(atom.resname, atom.chain, atom.resseq)].append(atom)
    for component_atoms in by_component.values():
        component_serials = {atom.serial for atom in component_atoms}
        if any(adjacency.get(serial, set()) & component_serials for serial in component_serials):
            continue
        for index, a in enumerate(component_atoms):
            for b in component_atoms[index + 1:]:
                key = tuple(sorted((a.element, b.element)))
                cutoff = COVALENT_CUTOFFS.get(key)
                if not cutoff:
                    continue
                distance = math.sqrt(
                    (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
                )
                if distance <= cutoff:
                    adjacency[a.serial].add(b.serial)
                    adjacency[b.serial].add(a.serial)
    return atoms, adjacency


def _identity_keys(atom: LigAtom) -> List[str]:
    resname, chain, resseq, name = atom.identity
    return [
        f"{resname}:{chain}:{resseq}:{name}",
        f"{resname}:{resseq}:{name}",
        f"{resname}:{name}",
        name,
    ]


def _unit_name(base_label: str, center: str, counter: int) -> str:
    """Human-readable generic ligand-unit label.

    The base labels follow Supplementary Table S4, while the optional center
    token keeps separate functional groups from being over-merged in unfamiliar
    ligands. Manuscript examples use explicit/built-in units and therefore keep
    the exact Table S6/S10/S14 names.
    """
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(center or "")).strip("_").upper()
    return f"{base_label} ({safe})" if safe else f"{base_label} {counter}"


def detect_ligand_units_from_pdbs(
    pdb_paths: Iterable[Path], *, strict_ion_identity: bool = False
) -> Tuple[Dict[str, str], List[dict]]:
    """Detect conservative functional-group atom equivalences from ligand PDBs.

    Detection is deliberately limited to common, high-confidence equivalence
    groups. Unknown atoms are not merged.
    """
    mapping: Dict[str, str] = {}
    audit: List[dict] = []
    seen_paths = []
    for path in sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p)):
        try:
            atoms, adj = parse_ligand_atoms_from_pdb(path)
        except Exception as exc:
            audit.append({"source_pdb": str(path), "rule": "parse_failed", "detail": str(exc), "unit": "", "atoms": ""})
            continue
        if not atoms:
            continue
        seen_paths.append(str(path))
        by_serial = {a.serial: a for a in atoms}
        by_residue: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
        for atom in atoms:
            by_residue[(atom.resname, atom.chain, atom.resseq)].append(atom)

        # Metals/ions. General mode requires a true metal element in a
        # single-atom residue. Legacy mode keeps the original atom-name behavior
        # so manuscript results remain unchanged.
        for atom in atoms:
            residue_atoms = by_residue[(atom.resname, atom.chain, atom.resseq)]
            is_ion = (
                atom.element.upper() in METAL_ELEMENTS and len(residue_atoms) == 1
                if strict_ion_identity
                else atom.element.upper() in ION_ATOMS or atom.atom_name.upper() in ION_ATOMS
            )
            if is_ion:
                unit = "Mg2+ ion" if atom.atom_name.upper() == "MG" else f"{atom.atom_name.upper()} ion"
                for k in _identity_keys(atom):
                    mapping.setdefault(k, unit)
                audit.append({
                    "source_pdb": str(path), "rule": "metal_ion", "unit": unit,
                    "atoms": atom.atom_name, "residue_name": atom.resname,
                    "connectivity_evidence": "single_atom_metal_residue" if strict_ion_identity else "element_or_atom_name",
                })

        for _resid, group_atoms in by_residue.items():
            group_serials = {a.serial for a in group_atoms}
            # Phosphate centers and bridge oxygens.
            p_atoms = [a for a in group_atoms if a.element == "P" or a.atom_name.upper().startswith("P")]
            o_atoms = [a for a in group_atoms if a.element == "O" or a.atom_name.upper().startswith("O")]
            o_to_p = {o.serial: [by_serial[n] for n in adj.get(o.serial, set()) if n in group_serials and by_serial[n].element == "P"] for o in o_atoms}
            for p_atom in p_atoms:
                neigh_o = [by_serial[n] for n in adj.get(p_atom.serial, set()) if n in group_serials and by_serial[n].element == "O"]
                if not neigh_o:
                    continue
                unit = _unit_name("Phosphate O site", p_atom.atom_name, 1)
                for atom in [p_atom] + [o for o in neigh_o if len(o_to_p.get(o.serial, [])) <= 1]:
                    for k in _identity_keys(atom):
                        mapping.setdefault(k, unit)
                audit.append({
                    "source_pdb": str(path), "rule": "phosphate_center", "unit": unit,
                    "atoms": ";".join(a.atom_name for a in [p_atom] + neigh_o),
                    "residue_name": _resid[0],
                    "connectivity_evidence": "pdb_conect_or_local_geometry",
                })
            for o, ps in o_to_p.items():
                if len(ps) >= 2:
                    atom = by_serial[o]
                    unit = _unit_name("Phosphate bridging O site", "-".join(sorted(p.atom_name.upper() for p in ps[:2])), 1)
                    for k in _identity_keys(atom):
                        mapping.setdefault(k, unit)
                    audit.append({
                        "source_pdb": str(path), "rule": "phosphate_bridge_oxygen", "unit": unit,
                        "atoms": atom.atom_name, "residue_name": _resid[0],
                        "connectivity_evidence": "pdb_conect_or_local_geometry",
                    })

            # Carboxylate / carboxyl-like C bonded to two O atoms.
            counter = 1
            for c_atom in [a for a in group_atoms if a.element == "C"]:
                neigh_o = [by_serial[n] for n in adj.get(c_atom.serial, set()) if n in group_serials and by_serial[n].element == "O"]
                if len(neigh_o) >= 2:
                    unit = _unit_name("Carboxylate O site", c_atom.atom_name, counter)
                    counter += 1
                    for atom in neigh_o:
                        for k in _identity_keys(atom):
                            mapping.setdefault(k, unit)
                    audit.append({
                        "source_pdb": str(path), "rule": "carboxylate_or_carbonyl_oxygen_pair", "unit": unit,
                        "atoms": ";".join(o.atom_name for o in neigh_o), "residue_name": _resid[0],
                        "connectivity_evidence": "pdb_conect_or_local_geometry",
                    })

            # Sulfate/sulfonate/sulfonyl.
            counter = 1
            for s_atom in [a for a in group_atoms if a.element == "S"]:
                neigh_o = [by_serial[n] for n in adj.get(s_atom.serial, set()) if n in group_serials and by_serial[n].element == "O"]
                if len(neigh_o) >= 2:
                    prefix = "Sulfate/sulfonate O site" if len(neigh_o) >= 4 else "Sulfonyl O site"
                    unit = _unit_name(prefix, s_atom.atom_name, counter)
                    counter += 1
                    for atom in neigh_o:
                        for k in _identity_keys(atom):
                            mapping.setdefault(k, unit)
                    audit.append({
                        "source_pdb": str(path), "rule": "sulfur_oxygen_group", "unit": unit,
                        "atoms": ";".join(o.atom_name for o in neigh_o), "residue_name": _resid[0],
                        "connectivity_evidence": "pdb_conect_or_local_geometry",
                    })

            # Nitro: N bonded to at least two O atoms.
            counter = 1
            for n_atom in [a for a in group_atoms if a.element == "N"]:
                neigh_o = [by_serial[n] for n in adj.get(n_atom.serial, set()) if n in group_serials and by_serial[n].element == "O"]
                if len(neigh_o) >= 2:
                    unit = _unit_name("Nitro O site", n_atom.atom_name, counter)
                    counter += 1
                    for atom in neigh_o:
                        for k in _identity_keys(atom):
                            mapping.setdefault(k, unit)
                    audit.append({
                        "source_pdb": str(path), "rule": "nitro", "unit": unit,
                        "atoms": ";".join(o.atom_name for o in neigh_o), "residue_name": _resid[0],
                        "connectivity_evidence": "pdb_conect_or_local_geometry",
                    })

            # Guanidinium/amidinium-like C bonded to multiple N atoms.
            counter = 1
            for c_atom in [a for a in group_atoms if a.element == "C"]:
                neigh_n = [by_serial[n] for n in adj.get(c_atom.serial, set()) if n in group_serials and by_serial[n].element == "N"]
                if len(neigh_n) >= 2:
                    unit = _unit_name("Multi-N site", c_atom.atom_name, counter)
                    counter += 1
                    for atom in neigh_n:
                        for k in _identity_keys(atom):
                            mapping.setdefault(k, unit)
                    audit.append({
                        "source_pdb": str(path), "rule": "guanidinium_or_amidinium", "unit": unit,
                        "atoms": ";".join(n.atom_name for n in neigh_n), "residue_name": _resid[0],
                        "connectivity_evidence": "pdb_conect_or_local_geometry",
                    })
    if not seen_paths:
        audit.append({"source_pdb": "", "rule": "auto_detection_skipped", "unit": "", "atoms": "", "detail": "No ligand PDB found."})
    return mapping, audit


def _metal_contact_unit(element: str) -> str:
    """Return a readable ion label without inventing an unknown oxidation state."""
    labels = {
        "MG": "Mg2+ ion", "ZN": "Zn2+ ion", "CA": "Ca2+ ion",
        "NA": "Na+ ion", "K": "K+ ion", "MN": "Mn2+ ion",
        "CU": "Cu ion", "CO": "Co ion", "NI": "Ni ion",
        "FE": "Fe ion", "CD": "Cd2+ ion", "HG": "Hg ion",
    }
    token = str(element or "").upper()
    return labels.get(token, f"{token.title()} ion" if token else "Metal ion")


def _merge_overlapping_atom_sets(groups: Iterable[Set[int]]) -> List[Set[int]]:
    """Merge fused/spiro ring descriptions into stable ring systems."""
    merged: List[Set[int]] = []
    for group in [set(value) for value in groups if value]:
        hits = [existing for existing in merged if existing & group]
        if not hits:
            merged.append(group)
            continue
        combined = set(group)
        for existing in hits:
            combined.update(existing)
            merged.remove(existing)
        merged.append(combined)
    return sorted(merged, key=lambda value: (min(value), len(value)))


def _automatic_unit_name(
    ligand_label: str, descriptor: str, ordinal: int, total: int,
    *, include_ligand: bool,
) -> str:
    qualified = bool(include_ligand or total > 1)
    prefix = f"{ligand_label} " if qualified else ""
    readable_descriptor = descriptor if qualified else descriptor[:1].upper() + descriptor[1:]
    position = f" {ordinal}" if total > 1 else ""
    return f"{prefix}{readable_descriptor}{position} unit"


def _universal_candidates_for_pdb(path: Path) -> Tuple[List[dict], List[dict]]:
    """Apply the twelve S4 rules to one ligand-bearing PDB.

    The result contains component-aware mapping keys so two ligand components
    are never merged solely because their atom names happen to match.
    """
    atoms, adjacency = parse_ligand_atoms_from_pdb(path)
    atoms = [a for a in atoms if a.resname.upper() not in {"HOH", "WAT", "DOD"}]
    by_serial = {atom.serial: atom for atom in atoms}
    by_component: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
    for atom in atoms:
        by_component[(atom.resname.upper(), atom.chain or "_", str(atom.resseq))].append(atom)
    include_ligand_context = len(by_component) > 1
    resname_counts: Dict[str, int] = defaultdict(int)
    for resname, _chain, _resseq in by_component:
        resname_counts[resname] += 1

    candidates: List[dict] = []
    parse_audit: List[dict] = []

    for component, component_atoms in sorted(by_component.items()):
        resname, chain, resseq = component
        serials = {atom.serial for atom in component_atoms}
        duplicate_component = resname_counts[resname] > 1
        display_resname = LIGAND_DISPLAY_ALIASES.get(resname, resname)
        ligand_label = (
            display_resname
            if not duplicate_component
            else f"{display_resname} {chain}:{resseq}"
        )
        claimed: Set[int] = set()

        def component_keys(atom: LigAtom) -> List[str]:
            keys = [
                f"{resname}:{chain}:{resseq}:{atom.atom_name}",
                f"{resname}:{resseq}:{atom.atom_name}",
            ]
            if not duplicate_component:
                keys.append(f"{resname}:{atom.atom_name}")
            return [key.upper() for key in keys]

        def add_candidate(rule: str, unit: str, members: Iterable[LigAtom], center: str = "") -> None:
            selected = sorted({member.serial: member for member in members}.values(), key=lambda a: natural_key(a.atom_name))
            if not selected:
                return
            claimed.update(member.serial for member in selected)
            candidates.append({
                "source_pdb": str(path), "component": f"{resname}:{chain}:{resseq}",
                "resname": resname, "chain": chain, "resseq": resseq,
                "rule": rule, "unit": unit, "center_atom": center,
                "atoms": tuple(member.atom_name.upper() for member in selected),
                "mapping_keys": tuple(key for member in selected for key in component_keys(member)),
                "connectivity_evidence": "pdb_conect_or_local_geometry",
            })

        # Rule 11: every metal is a separate unit.
        for atom in component_atoms:
            if atom.element.upper() in METAL_ELEMENTS and len(component_atoms) == 1:
                add_candidate("metal_ion", _metal_contact_unit(atom.element), [atom], atom.atom_name)

        # Rules 2 and 3: phosphate/phosphonate groups and P--O--P bridges.
        p_atoms = sorted(
            [atom for atom in component_atoms if atom.element.upper() == "P"],
            key=lambda atom: natural_key(atom.atom_name),
        )
        p_position = {atom.serial: index for index, atom in enumerate(p_atoms, start=1)}
        oxygen_atoms = [atom for atom in component_atoms if atom.element.upper() == "O"]
        oxygen_to_p = {
            oxygen.serial: [
                by_serial[n] for n in adjacency.get(oxygen.serial, set())
                if n in serials and by_serial[n].element.upper() == "P"
            ]
            for oxygen in oxygen_atoms
        }
        for index, p_atom in enumerate(p_atoms, start=1):
            neighbours = [by_serial[n] for n in adjacency.get(p_atom.serial, set()) if n in serials]
            oxygens = [atom for atom in neighbours if atom.element.upper() == "O"]
            if not oxygens:
                continue
            descriptor = "phosphonate" if any(atom.element.upper() == "C" for atom in neighbours) else "phosphate"
            unit = _automatic_unit_name(
                ligand_label, descriptor, index, len(p_atoms),
                include_ligand=include_ligand_context,
            )
            nonbridging = [oxygen for oxygen in oxygens if len(oxygen_to_p.get(oxygen.serial, [])) < 2]
            add_candidate("phosphate_or_phosphonate", unit, [p_atom, *nonbridging], p_atom.atom_name)
        for oxygen, centres in sorted(oxygen_to_p.items()):
            if len(centres) < 2:
                continue
            atom = by_serial[oxygen]
            positions = sorted(p_position[centre.serial] for centre in centres[:2])
            span = "–".join(str(value) for value in positions)
            bridge_prefix = f"{ligand_label} " if include_ligand_context or len(p_atoms) > 1 else ""
            bridge_descriptor = "phosphate" if bridge_prefix else "Phosphate"
            unit = f"{bridge_prefix}{bridge_descriptor} {span} bridging oxygen"
            add_candidate("phosphate_bridging_oxygen", unit, [atom], atom.atom_name)

        # Rule 1: oxygens bonded to the same carboxylate/carboxyl centre.
        carboxylate_groups: List[Tuple[LigAtom, List[LigAtom]]] = []
        for carbon in [atom for atom in component_atoms if atom.element.upper() == "C"]:
            oxygens = [
                by_serial[n] for n in adjacency.get(carbon.serial, set())
                if n in serials and by_serial[n].element.upper() == "O" and n not in claimed
            ]
            terminal = [oxygen for oxygen in oxygens if len(adjacency.get(oxygen.serial, set()) & serials) == 1]
            if len(terminal) >= 2:
                carboxylate_groups.append((carbon, terminal))
        for index, (carbon, oxygens) in enumerate(carboxylate_groups, start=1):
            unit = _automatic_unit_name(
                ligand_label, "carboxylate oxygen", index, len(carboxylate_groups),
                include_ligand=include_ligand_context,
            )
            add_candidate("carboxylate_oxygen", unit, oxygens, carbon.atom_name)

        # Rules 4 and 6: terminal oxygen groups around the same sulfur centre.
        sulfur_groups: List[Tuple[LigAtom, List[LigAtom], str, str]] = []
        for sulfur in [atom for atom in component_atoms if atom.element.upper() == "S"]:
            oxygens = [
                by_serial[n] for n in adjacency.get(sulfur.serial, set())
                if n in serials and by_serial[n].element.upper() == "O" and n not in claimed
            ]
            terminal = [oxygen for oxygen in oxygens if len(adjacency.get(oxygen.serial, set()) & serials) == 1]
            if len(terminal) >= 3:
                sulfur_groups.append((sulfur, terminal, "sulfate/sulfonate oxygen", "sulfate_or_sulfonate_terminal_oxygen"))
            elif len(terminal) == 2:
                sulfur_groups.append((sulfur, terminal, "sulfonyl oxygen", "sulfonyl_oxygen"))
        totals_by_sulfur_rule: Dict[str, int] = defaultdict(int)
        for _center, _members, _descriptor, rule in sulfur_groups:
            totals_by_sulfur_rule[rule] += 1
        sulfur_seen: Dict[str, int] = defaultdict(int)
        for sulfur, oxygens, descriptor, rule in sulfur_groups:
            sulfur_seen[rule] += 1
            unit = _automatic_unit_name(
                ligand_label, descriptor, sulfur_seen[rule], totals_by_sulfur_rule[rule],
                include_ligand=include_ligand_context,
            )
            add_candidate(rule, unit, oxygens, sulfur.atom_name)

        # Rule 5: terminal oxygens in a nitro group.
        nitro_groups: List[Tuple[LigAtom, List[LigAtom]]] = []
        for nitrogen in [atom for atom in component_atoms if atom.element.upper() == "N"]:
            oxygens = [
                by_serial[n] for n in adjacency.get(nitrogen.serial, set())
                if n in serials and by_serial[n].element.upper() == "O" and n not in claimed
            ]
            terminal = [oxygen for oxygen in oxygens if len(adjacency.get(oxygen.serial, set()) & serials) == 1]
            if len(terminal) >= 2:
                nitro_groups.append((nitrogen, terminal))
        for index, (nitrogen, oxygens) in enumerate(nitro_groups, start=1):
            unit = _automatic_unit_name(
                ligand_label, "nitro oxygen", index, len(nitro_groups),
                include_ligand=include_ligand_context,
            )
            add_candidate("nitro_oxygen", unit, oxygens, nitrogen.atom_name)

        # Rule 7: reviewed-style terminal N atoms; ring nitrogens are excluded
        # because a terminal atom must have only one neighbour in the component.
        nitrogen_groups: List[Tuple[LigAtom, List[LigAtom]]] = []
        for carbon in [atom for atom in component_atoms if atom.element.upper() == "C"]:
            nitrogens = [
                by_serial[n] for n in adjacency.get(carbon.serial, set())
                if n in serials and by_serial[n].element.upper() == "N" and n not in claimed
            ]
            terminal = [nitrogen for nitrogen in nitrogens if len(adjacency.get(nitrogen.serial, set()) & serials) == 1]
            if len(terminal) >= 2:
                nitrogen_groups.append((carbon, terminal))
        for index, (carbon, nitrogens) in enumerate(nitrogen_groups, start=1):
            unit = _automatic_unit_name(
                ligand_label, "delocalized nitrogen", index, len(nitrogen_groups),
                include_ligand=include_ligand_context,
            )
            add_candidate("delocalized_terminal_nitrogen", unit, nitrogens, carbon.atom_name)

        # Rules 8 and 10: ring-system carbon fragments.  PDB files generally do
        # not encode aromatic bond orders, so graph rings are used.  A 5/6 ring
        # with one ring oxygen and at least two exocyclic oxygens is classified
        # as a sugar ring; other common hetero/all-carbon 5/6 rings are treated
        # as aromatic-like contact fragments and only their carbons are grouped.
        graph = nx.Graph() if nx is not None else None
        cycles: List[List[int]] = []
        if graph is not None:
            graph.add_nodes_from(serials)
            graph.add_edges_from(
                (left, right)
                for left in serials for right in adjacency.get(left, set())
                if right in serials and left < right
            )
            cycles = [cycle for cycle in nx.cycle_basis(graph) if 5 <= len(cycle) <= 6]
        sugar_cycles: List[Set[int]] = []
        aromatic_cycles: List[Set[int]] = []
        for cycle in cycles:
            elements = [by_serial[serial].element.upper() for serial in cycle]
            carbon_count = elements.count("C")
            oxygen_count = elements.count("O")
            exocyclic_oxygen_count = sum(
                1
                for serial in cycle if by_serial[serial].element.upper() == "C"
                for neighbour in adjacency.get(serial, set())
                if neighbour in serials and neighbour not in cycle and by_serial[neighbour].element.upper() == "O"
            )
            if oxygen_count == 1 and carbon_count == len(cycle) - 1 and exocyclic_oxygen_count >= 2:
                sugar_cycles.append(set(cycle))
            elif carbon_count >= 2 and set(elements) <= {"C", "N", "O", "S"}:
                aromatic_cycles.append(set(cycle))
        sugar_systems = _merge_overlapping_atom_sets(sugar_cycles)
        aromatic_systems = _merge_overlapping_atom_sets(aromatic_cycles)
        ring_serials = set().union(*sugar_systems, *aromatic_systems) if (sugar_systems or aromatic_systems) else set()
        for index, system in enumerate(sugar_systems, start=1):
            carbons = [by_serial[serial] for serial in system if by_serial[serial].element.upper() == "C" and serial not in claimed]
            unit = _automatic_unit_name(
                ligand_label, "sugar-ring carbon", index, len(sugar_systems),
                include_ligand=include_ligand_context,
            )
            add_candidate("sugar_ring_carbon_fragment", unit, carbons)
        for index, system in enumerate(aromatic_systems, start=1):
            carbons = [by_serial[serial] for serial in system if by_serial[serial].element.upper() == "C" and serial not in claimed]
            unit = _automatic_unit_name(
                ligand_label, "aromatic ring", index, len(aromatic_systems),
                include_ligand=include_ligand_context,
            )
            add_candidate("aromatic_carbon_fragment", unit, carbons)

        # Rule 9: contiguous non-ring, non-functional-centre carbon fragments.
        available_carbons = {
            atom.serial for atom in component_atoms
            if atom.element.upper() == "C" and atom.serial not in claimed
            and atom.serial not in ring_serials
        }
        carbon_components: List[Set[int]] = []
        pending = set(available_carbons)
        while pending:
            seed = min(pending)
            stack = [seed]
            component_set: Set[int] = set()
            while stack:
                current = stack.pop()
                if current not in pending:
                    continue
                pending.remove(current)
                component_set.add(current)
                stack.extend(
                    neighbour for neighbour in adjacency.get(current, set())
                    if neighbour in pending and neighbour in available_carbons
                )
            if len(component_set) >= 2:
                carbon_components.append(component_set)
        carbon_components.sort(key=lambda value: min(value))
        for index, component_set in enumerate(carbon_components, start=1):
            unit = _automatic_unit_name(
                ligand_label, "aliphatic carbon", index, len(carbon_components),
                include_ligand=include_ligand_context,
            )
            add_candidate(
                "non_aromatic_carbon_fragment", unit,
                [by_serial[serial] for serial in component_set],
            )

        # Rule 12: every remaining atom is retained as a component-aware site.
        for atom in sorted(component_atoms, key=lambda value: natural_key(value.atom_name)):
            if atom.serial in claimed:
                continue
            unit = (
                f"{ligand_label} {atom.atom_name.upper()} site"
                if include_ligand_context
                else f"{atom.atom_name.upper()} site"
            )
            add_candidate("individual_atom", unit, [atom], atom.atom_name)

    if not candidates:
        parse_audit.append({
            "source_pdb": str(path), "rule": "universal_rules_skipped",
            "decision": "keep_atom_level", "detail": "No non-water ligand atoms found.",
        })
    return candidates, parse_audit


def detect_universal_contact_units_from_pdbs(
    pdb_paths: Iterable[Path], *, consensus_min_fraction: float = 0.80
) -> Tuple[Dict[str, str], List[dict]]:
    """Apply the S4 contact-unit rules automatically across an ensemble."""
    if not 0 < float(consensus_min_fraction) <= 1:
        raise ValueError("universal contact-unit consensus_min_fraction must be in (0, 1].")
    paths = sorted({path.resolve() for path in pdb_paths if path and path.exists()}, key=lambda path: str(path))
    all_candidates: List[dict] = []
    audit: List[dict] = []
    component_sources: Dict[str, Set[str]] = defaultdict(set)
    for path in paths:
        try:
            candidates, path_audit = _universal_candidates_for_pdb(path)
        except Exception as exc:
            audit.append({
                "source_pdb": str(path), "rule": "parse_failed", "decision": "keep_atom_level",
                "detail": str(exc), "unit": "", "atoms": "",
            })
            continue
        audit.extend(path_audit)
        all_candidates.extend(candidates)
        for candidate in candidates:
            component_sources[str(candidate["component"])].add(str(path))

    grouped: Dict[Tuple[str, str, str, Tuple[str, ...]], List[dict]] = defaultdict(list)
    for candidate in all_candidates:
        signature = (
            str(candidate["component"]), str(candidate["rule"]),
            str(candidate["unit"]), tuple(candidate["atoms"]),
        )
        grouped[signature].append(candidate)

    mapping: Dict[str, str] = {}
    for signature, rows in sorted(grouped.items()):
        component, rule, unit, atoms = signature
        sources = {str(row["source_pdb"]) for row in rows}
        available = max(1, len(component_sources.get(component, set())))
        required = max(1, math.ceil(float(consensus_min_fraction) * available))
        accepted = len(sources) >= required
        if accepted:
            for row in rows:
                for key in row["mapping_keys"]:
                    mapping.setdefault(str(key).upper(), unit)
        first = rows[0]
        audit.append({
            "source_pdb": ";".join(sorted(sources)),
            "residue_name": first["resname"], "chain": first["chain"], "resseq": first["resseq"],
            "rule": rule, "unit": unit, "center_atom": first.get("center_atom", ""),
            "atoms": ";".join(atoms), "connectivity_evidence": first.get("connectivity_evidence", ""),
            "ensemble_observed_structures": len(sources),
            "ensemble_available_structures": available,
            "ensemble_min_fraction": float(consensus_min_fraction),
            "decision": "apply" if accepted else "keep_atom_level",
            "confidence": "high" if accepted else "insufficient_recurrence",
            "detail": "Universal S4 contact-unit rule applied within one ligand component."
            if accepted else "Rule was not reproduced in enough ligand-bearing structures.",
        })
    if not paths:
        audit.append({
            "source_pdb": "", "rule": "universal_rules_skipped", "unit": "", "atoms": "",
            "decision": "keep_atom_level", "confidence": "none", "detail": "No ligand PDB found.",
        })
    return mapping, audit


def detect_functional_group_consensus_units_from_pdbs(
    pdb_paths: Iterable[Path], *, consensus_min_fraction: float = 0.80
) -> Tuple[Dict[str, str], List[dict]]:
    """Infer only recurring, chemistry-local equivalence units.

    This is the automatic general-workflow rule.  It deliberately does not
    learn enzyme-specific atom names and it never maps a bare atom name across
    residues.  A group must be a known local chemical motif (phosphate,
    carboxylate/carbonyl oxygen pair, sulfur--oxygen group, nitro, multi-N site
    or a single-atom metal) and must recur for the same ligand residue name in
    a configurable fraction of the observed complexes.  PDB ``CONECT`` is
    used when present; otherwise a local covalent-radius geometry check is used
    solely to establish that motif, and the evidence type is retained in the
    audit table.

    Explicit YAML/registry rules still take precedence.  This consensus filter
    is what makes automatic PTP1B-like phosphate/carboxylate merging work for
    docking outputs which often have no usable ``CONECT`` records, without
    silently applying the three manuscript enzymes' hand-written presets.
    """
    paths = sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p))
    raw_mapping, raw_audit = detect_ligand_units_from_pdbs(paths, strict_ion_identity=True)
    if not paths:
        return {}, raw_audit

    # Count the number of observed complex files in which each ligand residue
    # name occurs.  This makes the threshold substrate-specific rather than
    # depending on the total number of protein conformers in a run.
    residue_observed_in: Dict[str, Set[str]] = defaultdict(set)
    for path in paths:
        try:
            atoms, _adj = parse_ligand_atoms_from_pdb(path)
        except Exception:
            continue
        for atom in atoms:
            residue_observed_in[atom.resname.upper()].add(str(path))

    motif_sources: Dict[Tuple[str, str, str, str], Set[str]] = defaultdict(set)
    for row in raw_audit:
        rule = str(row.get("rule", ""))
        residue = str(row.get("residue_name", "")).upper()
        unit = str(row.get("unit", ""))
        atoms = ";".join(sorted(x for x in str(row.get("atoms", "")).split(";") if x))
        source = str(row.get("source_pdb", ""))
        if residue and unit and atoms and source and rule != "auto_detection_skipped":
            motif_sources[(residue, rule, unit, atoms)].add(source)

    accepted_units: Set[Tuple[str, str]] = set()
    reviewed_audit: List[dict] = []
    accepted_by_motif: Dict[Tuple[str, str, str, str], bool] = {}
    for motif, source_paths in sorted(motif_sources.items()):
        residue, _rule, unit, _atoms = motif
        denominator = max(1, len(residue_observed_in.get(residue, set())))
        observed = len(source_paths)
        required = max(1, math.ceil(float(consensus_min_fraction) * denominator))
        accepted = observed >= required
        accepted_by_motif[motif] = accepted
        if accepted:
            accepted_units.add((residue, unit))

    for row in raw_audit:
        out = dict(row)
        residue = str(out.get("residue_name", "")).upper()
        unit = str(out.get("unit", ""))
        rule = str(out.get("rule", ""))
        atoms = ";".join(sorted(x for x in str(out.get("atoms", "")).split(";") if x))
        motif = (residue, rule, unit, atoms)
        if motif in accepted_by_motif:
            denominator = max(1, len(residue_observed_in.get(residue, set())))
            observed = len(motif_sources[motif])
            accepted = accepted_by_motif[motif]
            out.update({
                "ensemble_observed_structures": observed,
                "ensemble_available_structures": denominator,
                "ensemble_min_fraction": consensus_min_fraction,
                "decision": "merge" if accepted else "keep_atom_level",
                "confidence": "high" if accepted and out.get("connectivity_evidence") == "single_atom_metal_residue" else ("moderate" if accepted else "insufficient_recurrence"),
                "detail": "Recurring local functional group across ligand-bearing complexes." if accepted else "Functional group was not reproduced in enough ligand-bearing complexes.",
            })
        reviewed_audit.append(out)

    # The old detector also adds bare keys (e.g. ``O3``).  Those are convenient
    # for one-off manuscript data but unsafe for unseen substrates containing
    # multiple heteroligands.  Automatic mapping is residue-aware only.
    mapping: Dict[str, str] = {}
    for key, unit in raw_mapping.items():
        parts = str(key).split(":")
        if len(parts) != 2:
            continue
        residue = parts[0].upper()
        if (residue, str(unit)) in accepted_units:
            mapping[str(key).upper()] = str(unit)

    if not reviewed_audit:
        reviewed_audit.append({
            "source_pdb": "", "rule": "functional_group_consensus_skipped",
            "unit": "", "atoms": "", "decision": "keep_atom_level",
            "confidence": "none", "detail": "No ligand functional group could be established.",
        })
    return mapping, reviewed_audit


def detect_strict_symmetry_units_from_pdbs(pdb_paths: Iterable[Path]) -> Tuple[Dict[str, str], List[dict]]:
    """Return only ligand atom groups proven equivalent by PDB graph symmetry.

    A PDB normally lacks bond orders, formal charges, protonation, and tautomer
    state.  Therefore this routine intentionally does *not* infer broad
    functional groups from distances.  It requires explicit ligand--ligand
    ``CONECT`` records and uses element-coloured graph automorphisms within each
    ligand residue.  Any molecule lacking that evidence remains atom-level and
    is recorded in the audit table for manual review.
    """
    if nx is None:
        raise RuntimeError(
            "strict_chemistry equivalence mode requires the 'networkx' package. "
            "Install the bundled requirements.txt, then rerun."
        )
    observations: Dict[str, List[Set[frozenset[str]]]] = defaultdict(list)
    audit: List[dict] = []

    for path in sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not any(line.startswith("CONECT") for line in text.splitlines()):
            audit.append({
                "source_pdb": str(path), "rule": "strict_symmetry_skipped_missing_conect",
                "unit": "", "atoms": "", "decision": "keep_atom_level",
                "confidence": "insufficient_connectivity",
                "detail": "No explicit ligand CONECT records; automatic merging was disabled.",
            })
            continue
        try:
            atoms, _adjacency = parse_ligand_atoms_from_pdb(path)
        except Exception as exc:
            audit.append({"source_pdb": str(path), "rule": "strict_symmetry_parse_failed", "unit": "", "atoms": "", "decision": "keep_atom_level", "confidence": "none", "detail": str(exc)})
            continue
        by_serial = {atom.serial: atom for atom in atoms}
        # Do not accept ``parse_ligand_atoms_from_pdb``'s distance fallback in
        # this strict mode.  Rebuild only explicit PDB CONECT links.
        explicit_adjacency: Dict[int, Set[int]] = defaultdict(set)
        for line in text.splitlines():
            if not line.startswith("CONECT"):
                continue
            serials = [int(value) for value in re.findall(r"\d+", line[6:])]
            if len(serials) < 2:
                continue
            source = serials[0]
            for neighbor in serials[1:]:
                if source in by_serial and neighbor in by_serial:
                    explicit_adjacency[source].add(neighbor)
                    explicit_adjacency[neighbor].add(source)
        by_residue: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
        for atom in atoms:
            by_residue[(atom.resname, atom.chain, atom.resseq)].append(atom)

        for (resname, _chain, _resseq), group_atoms in by_residue.items():
            names = [atom.atom_name for atom in group_atoms]
            if len(names) != len(set(names)):
                audit.append({"source_pdb": str(path), "rule": "strict_symmetry_skipped_duplicate_atom_names", "unit": "", "atoms": ";".join(names), "decision": "keep_atom_level", "confidence": "ambiguous_identity", "detail": f"{resname} contains duplicate atom names."})
                continue
            graph = nx.Graph()
            group_serials = {atom.serial for atom in group_atoms}
            for atom in group_atoms:
                graph.add_node(atom.serial, element=atom.element.upper())
            for serial in group_serials:
                for neighbor in explicit_adjacency.get(serial, set()):
                    if neighbor in group_serials:
                        graph.add_edge(serial, neighbor)
            if graph.number_of_edges() == 0:
                audit.append({"source_pdb": str(path), "rule": "strict_symmetry_skipped_no_ligand_edges", "unit": "", "atoms": ";".join(names), "decision": "keep_atom_level", "confidence": "insufficient_connectivity", "detail": f"{resname} has no explicit intra-residue ligand bonds."})
                continue

            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph, graph, node_match=nx.algorithms.isomorphism.categorical_node_match("element", "")
            )
            orbit: Dict[int, Set[int]] = {serial: {serial} for serial in graph.nodes}
            n_automorphisms = 0
            complete = True
            for mapping in matcher.isomorphisms_iter():
                n_automorphisms += 1
                if n_automorphisms > 4096:
                    complete = False
                    break
                for source, target in mapping.items():
                    orbit[source].add(target)
            if not complete:
                audit.append({"source_pdb": str(path), "rule": "strict_symmetry_skipped_automorphism_cap", "unit": "", "atoms": ";".join(names), "decision": "keep_atom_level", "confidence": "ambiguous_large_symmetry", "detail": "Automorphism enumeration exceeded the safe audit cap."})
                continue

            classes: Set[frozenset[str]] = set()
            for members in orbit.values():
                if len(members) < 2:
                    continue
                atom_names = frozenset(by_serial[serial].atom_name.upper() for serial in members)
                elements = {by_serial[serial].element.upper() for serial in members}
                if len(atom_names) >= 2 and len(elements) == 1:
                    classes.add(atom_names)
            observations[resname.upper()].append(classes)
            for atom in group_atoms:
                if atom.element.upper() in METAL_ELEMENTS and len(group_atoms) == 1:
                    unit = "Mg2+ ion" if atom.element.upper() == "MG" else f"{atom.element.upper()} ion"
                    audit.append({"source_pdb": str(path), "rule": "strict_metal_identity", "unit": unit, "atoms": atom.atom_name, "decision": "merge_by_element_identity", "confidence": "high", "detail": "Single-atom metal/ion residue."})

    mapping: Dict[str, str] = {}
    for resname, class_sets in observations.items():
        if not class_sets:
            continue
        # A symmetry class must be reproduced in every observed conformation of
        # the same ligand residue before it becomes an automatic merge rule.
        shared = set.intersection(*class_sets)
        for atom_names in sorted(shared, key=lambda names: tuple(sorted(names))):
            unit = f"Symmetry unit {resname} ({','.join(sorted(atom_names))})"
            for atom_name in atom_names:
                mapping[f"{resname}:{atom_name}"] = unit
            audit.append({"source_pdb": "ensemble_consensus", "rule": "strict_graph_automorphism", "unit": unit, "atoms": ";".join(sorted(atom_names)), "decision": "merge", "confidence": "high", "detail": "Element-coloured graph automorphism reproduced across every observed conformation."})

    # Add safe single-ion labels after the consensus pass.  They are an identity
    # mapping, not an organic functional-group inference.
    for path in sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p)):
        try:
            atoms, _ = parse_ligand_atoms_from_pdb(path)
        except Exception:
            continue
        residues: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
        for atom in atoms:
            residues[(atom.resname, atom.chain, atom.resseq)].append(atom)
        for (resname, _chain, _resseq), group in residues.items():
            if len(group) == 1 and group[0].element.upper() in METAL_ELEMENTS:
                atom = group[0]
                mapping.setdefault(f"{resname.upper()}:{atom.atom_name.upper()}", "Mg2+ ion" if atom.element.upper() == "MG" else f"{atom.element.upper()} ion")
    if not audit:
        audit.append({"source_pdb": "", "rule": "strict_symmetry_skipped", "unit": "", "atoms": "", "decision": "keep_atom_level", "confidence": "none", "detail": "No ligand PDB was available."})
    return mapping, audit


# ---------------------------------------------------------------------------
# Substrate inventory and reviewable chemistry recommendations
# ---------------------------------------------------------------------------


def _as_string_list(value: object) -> List[str]:
    """Read a list-like YAML value without treating a molecule name as characters."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;\\n]+", value) if item.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _recommendation_settings(cfg: Mapping[str, object]) -> Mapping[str, object]:
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) if isinstance(step09, Mapping) else {}
    can = can or {}
    recommendations = can.get("recommendations", {}) if isinstance(can, Mapping) else {}
    return recommendations if isinstance(recommendations, Mapping) else {}


def _recommendations_require_acceptance(cfg: Mapping[str, object]) -> bool:
    """Return whether new-substrate proposals must be explicitly accepted.

    Existing strict-mode YAML files that predate the review workflow keep their
    historical automatic behavior.  Studio-generated/new configurations carry
    the explicit ``recommendations`` block and therefore default to review
    before application.
    """
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) if isinstance(step09, Mapping) else {}
    can = can or {}
    if not isinstance(can, Mapping) or "recommendations" not in can:
        return False
    settings = _recommendation_settings(cfg)
    return read_bool(settings.get("require_acceptance"), True)


def _configured_ligand_smiles(cfg: Mapping[str, object]) -> Dict[str, str]:
    """Return optional residue-name -> SMILES annotations for the audit output.

    PDB atom names are still the only identifiers used to apply a contact-unit
    rule.  SMILES are an independent chemical-identity check; the code never
    guesses a PDB atom-name mapping from a SMILES string.
    """
    settings = _recommendation_settings(cfg)
    raw = settings.get("ligand_smiles", settings.get("substrate_smiles", {}))
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(resname).strip().upper(): str(smiles).strip()
        for resname, smiles in raw.items()
        if str(resname).strip() and str(smiles).strip()
    }


def _smiles_summary(smiles: str) -> dict:
    """Summarize an optional SMILES without making RDKit a runtime requirement.

    The full Studio environment ships RDKit.  A lightweight installation can
    still run Step09 and records that SMILES validation was unavailable rather
    than silently discarding the supplied substrate identity.
    """
    if not smiles:
        return {
            "smiles": "", "smiles_status": "not_supplied", "smiles_heavy_atom_count": "",
            "smiles_formal_charge": "", "smiles_formula": "",
        }
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdMolDescriptors  # type: ignore
    except ModuleNotFoundError:
        return {
            "smiles": smiles, "smiles_status": "not_validated_rdkit_unavailable",
            "smiles_heavy_atom_count": "", "smiles_formal_charge": "", "smiles_formula": "",
        }
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            "smiles": smiles, "smiles_status": "invalid_smiles", "smiles_heavy_atom_count": "",
            "smiles_formal_charge": "", "smiles_formula": "",
        }
    return {
        "smiles": smiles,
        "smiles_status": "validated_rdkit",
        "smiles_heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "smiles_formal_charge": int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
        "smiles_formula": rdMolDescriptors.CalcMolFormula(molecule),
    }


def build_substrate_inventory(
    cfg: Mapping[str, object], pdb_paths: Iterable[Path]
) -> pd.DataFrame:
    """Extract a residue-aware inventory of the substrate/cofactor atoms seen by Step09."""
    smiles_by_resname = _configured_ligand_smiles(cfg)
    rows: Dict[Tuple[str, str, str], dict] = {}
    for path in sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p)):
        try:
            atoms, _adjacency = parse_ligand_atoms_from_pdb(path)
        except Exception as exc:
            key = ("", "", str(path))
            rows[key] = {
                "resname": "", "chain": "", "resseq": "", "n_atom_records": 0,
                "atom_names": "", "elements": "", "n_source_pdbs": 0,
                "source_pdbs": str(path), "inventory_status": f"parse_failed: {exc}",
                **_smiles_summary(""),
            }
            continue
        grouped: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
        for atom in atoms:
            # Crystallographic waters are not a substrate identity and should
            # never appear as a proposed ligand contact unit.
            if atom.resname.upper() in {"HOH", "WAT", "DOD"}:
                continue
            grouped[(atom.resname.upper(), atom.chain, atom.resseq)].append(atom)
        for (resname, chain, resseq), group in grouped.items():
            key = (resname, chain, resseq)
            entry = rows.setdefault(key, {
                "resname": resname, "chain": chain, "resseq": resseq,
                "n_atom_records": len(group),
                "atom_names": ";".join(sorted({atom.atom_name for atom in group})),
                "elements": ";".join(sorted({atom.element for atom in group})),
                "n_source_pdbs": 0, "source_pdbs": [], "inventory_status": "observed_in_pdb",
                **_smiles_summary(smiles_by_resname.get(resname, "")),
            })
            entry["n_source_pdbs"] = int(entry["n_source_pdbs"]) + 1
            entry["source_pdbs"].append(str(path))
    output = []
    for entry in rows.values():
        row = dict(entry)
        if isinstance(row.get("source_pdbs"), list):
            row["source_pdbs"] = ";".join(sorted(set(row["source_pdbs"])))
        output.append(row)
    columns = [
        "resname", "chain", "resseq", "n_atom_records", "atom_names", "elements",
        "n_source_pdbs", "source_pdbs", "inventory_status", "smiles", "smiles_status",
        "smiles_heavy_atom_count", "smiles_formal_charge", "smiles_formula",
    ]
    return pd.DataFrame(output, columns=columns).sort_values(
        ["resname", "chain", "resseq"], kind="stable"
    ).reset_index(drop=True) if output else pd.DataFrame(columns=columns)


def _recommendation_id(resname: str, rule: str, atoms: Sequence[str]) -> str:
    tokens = [resname.upper(), rule.upper(), *sorted(atom.upper() for atom in atoms)]
    return re.sub(r"[^A-Z0-9]+", "_", "__".join(tokens)).strip("_")


def _explicit_ligand_adjacency(path: Path, atoms: Sequence[LigAtom]) -> Dict[int, Set[int]]:
    """Return only intra-ligand PDB ``CONECT`` edges.

    A PDB coordinate file does not carry bond order, formal charge or
    protonation.  The previous recommender therefore used local distances to
    create a whole "functional group" and could accidentally include bridge
    oxygens, ring oxygens, centre atoms and unrelated nitrogens.  Suggestions
    for a new substrate must have stronger evidence, so this helper *never*
    falls back to distance-based bonds.
    """
    ligand_serials = {atom.serial for atom in atoms}
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return adjacency
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        serials = [int(value) for value in re.findall(r"\d+", line[6:])]
        if len(serials) < 2 or serials[0] not in ligand_serials:
            continue
        for neighbor in serials[1:]:
            if neighbor in ligand_serials:
                adjacency[serials[0]].add(neighbor)
                adjacency[neighbor].add(serials[0])
    return adjacency


def _reviewable_terminal_group_candidates(pdb_paths: Iterable[Path]) -> List[dict]:
    """Find narrow, non-overlapping terminal-atom equivalence candidates.

    The output intentionally models "potentially interchangeable terminal
    atoms", not broad functional fragments.  Centre P/C/S atoms, bridge atoms
    (degree > 1), ring atoms and multi-N motifs are never proposed.  Each
    candidate consequently has a disjoint atom list that can safely be
    accepted as one manual rule after the enzyme-specific contact pattern is
    reviewed.
    """
    rows: List[dict] = []
    for path in sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p)):
        try:
            atoms, geometry_adjacency = parse_ligand_atoms_from_pdb(path)
        except Exception:
            continue
        explicit_adjacency = _explicit_ligand_adjacency(path, atoms)
        if any(explicit_adjacency.values()):
            adjacency = explicit_adjacency
            connectivity_evidence = "explicit_pdb_conect_terminal_atoms"
        elif any(geometry_adjacency.values()):
            # Template-ligand PDB files commonly omit CONECT records even when
            # their coordinates provide an unambiguous local covalent graph.
            # This fallback is deliberately confined to *preview proposals*:
            # it never changes a class until the user accepts the stable ID.
            # It is also narrower than the normal Step06 topology fallback,
            # because only terminal oxygen groups around P/C/S centres can be
            # proposed below.
            adjacency = geometry_adjacency
            connectivity_evidence = "local_geometry_covalent_cutoff"
        else:
            adjacency = {}
            connectivity_evidence = "no_local_connectivity"
        if not any(adjacency.values()):
            continue
        by_serial = {atom.serial: atom for atom in atoms}
        by_residue: Dict[Tuple[str, str, str], List[LigAtom]] = defaultdict(list)
        for atom in atoms:
            if atom.resname.upper() not in {"HOH", "WAT", "DOD"}:
                by_residue[(atom.resname.upper(), atom.chain, atom.resseq)].append(atom)

        for (resname, _chain, _resseq), group in by_residue.items():
            group_serials = {atom.serial for atom in group}
            for center in group:
                element = center.element.upper()
                if element not in {"P", "C", "S"}:
                    continue
                neighbors = [
                    by_serial[serial]
                    for serial in adjacency.get(center.serial, set())
                    if serial in group_serials and serial in by_serial
                ]
                # A terminal O has exactly one explicit bond in this ligand
                # residue.  This rejects phosphate bridges, glycosidic/ester
                # O atoms and every ring O without needing to guess a bond
                # order from a coordinate-only PDB.
                terminal_oxygens = sorted(
                    [
                        atom for atom in neighbors
                        if atom.element.upper() == "O"
                        and len(adjacency.get(atom.serial, set()) & group_serials) == 1
                    ],
                    key=lambda atom: natural_key(atom.atom_name),
                )
                if len(terminal_oxygens) < 2:
                    continue
                if element == "P":
                    rule = "phosphate_terminal_oxygen_group"
                    unit = f"Terminal phosphate O site ({center.atom_name})"
                elif element == "C":
                    # PDB CONECT has no bond orders, so this remains a
                    # carboxyl-like review candidate, never an assertion that
                    # the two O atoms are chemically identical.
                    rule = "carboxylate_or_carbonyl_oxygen_pair"
                    unit = f"Carboxyl-like terminal O site ({center.atom_name})"
                else:
                    rule = "sulfur_terminal_oxygen_group"
                    unit = f"Terminal sulfur O site ({center.atom_name})"
                rows.append({
                    "resname": resname,
                    "rule": rule,
                    "center_atom": center.atom_name.upper(),
                    "proposed_unit": unit,
                    "atoms": tuple(atom.atom_name.upper() for atom in terminal_oxygens),
                    "source_pdb": str(path),
                    "connectivity_evidence": connectivity_evidence,
                })
    return rows


def _enzyme_contact_assessment(
    candidate_atoms: Sequence[str],
    resname: str,
    raw_records: Iterable[Mapping[str, object]] | None,
) -> tuple[str, str]:
    """Flag a group whose observed enzyme contacts are atom-specific.

    This does not decide whether a group is equivalent; it tells the reviewer
    when the current enzyme already distinguishes its individual atoms.
    """
    if raw_records is None:
        return "not_assessed_no_interaction_records", "Manual review is required before applying any proposal."
    signatures: Dict[str, Set[str]] = {atom.upper(): set() for atom in candidate_atoms}
    for record in raw_records:
        ligand_atom = str(record.get("ligand_atom", ""))
        ligand_key = str(record.get("ligand_node_key", ""))
        observed_resname, _chain, _resseq, atom = extract_ligand_identity(ligand_atom, ligand_key)
        if observed_resname.upper() != resname.upper() or atom.upper() not in signatures:
            continue
        protein = protein_canonical(str(record.get("protein_raw", "")), True)
        interaction = normalize_type_name(str(record.get("raw_type", "")))
        if protein or interaction:
            signatures[atom.upper()].add(f"{protein}|{interaction}")
    observed = [values for values in signatures.values() if values]
    if not observed:
        return "no_contacts_for_candidate_atoms", "No PLIP contact was recorded for these atoms in the analysed ensemble."
    if len({tuple(sorted(values)) for values in observed}) > 1 or len(observed) != len(signatures):
        return "enzyme_contact_divergent", "The analysed enzyme contacts these candidate atoms differently; retain atom-level labels unless a mechanism-based rationale supports merging."
    return "enzyme_contact_indistinguishable", "Observed PLIP contact signatures were identical across the candidate atoms; mechanism, metal coordination and stereochemistry still require review."


def build_ligand_unit_recommendations(
    cfg: Mapping[str, object], pdb_paths: Iterable[Path],
    raw_records: Iterable[Mapping[str, object]] | None = None,
    *, target_id: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str], dict]:
    """Produce user-reviewable ligand contact-unit proposals.

    A proposal is intentionally *not* an active mapping until its stable ID is
    listed in ``accepted_recommendation_ids``.  This separates a chemically
    plausible local fragment from the enzyme-specific decision that its atoms
    should be collapsed for interaction-graph comparison.
    """
    settings = _recommendation_settings(cfg)
    enabled = read_bool(settings.get("enabled"), True)
    columns = [
        "recommendation_id", "status", "resname", "rule", "proposed_unit", "atoms",
        "recommendation_kind", "confidence", "connectivity_evidence", "center_atom",
        "ensemble_observed_structures", "ensemble_available_structures",
        "ensemble_min_fraction", "enzyme_contact_assessment", "rationale", "source_pdbs",
    ]
    resolved_target_id = str(
        target_id
        if target_id is not None
        else ((cfg.get("project", {}) or {}).get("target_id", ""))  # type: ignore[union-attr]
    ).strip()
    validated_case = validated_contact_unit_target(resolved_target_id)
    if contact_unit_mode(cfg) == CONTACT_UNIT_CONSERVATIVE:
        return pd.DataFrame(columns=columns), {}, {
            "contact_unit_mode": CONTACT_UNIT_CONSERVATIVE,
            "recommendations_enabled": False,
            "recommendations_disabled_reason": "conservative_manual_policy",
            "n_ligand_unit_recommendations": 0,
            "n_accepted_ligand_unit_recommendations": 0,
        }
    try:
        fraction = float(settings.get("consensus_min_fraction", 0.80))
    except (TypeError, ValueError) as exc:
        raise ValueError("step09.canonicalization.recommendations.consensus_min_fraction must be numeric.") from exc
    if not 0 < fraction <= 1:
        raise ValueError("step09.canonicalization.recommendations.consensus_min_fraction must be in (0, 1].")

    # Universal mode is an active, auditable rule application rather than an
    # acceptance queue. Every target, including the three manuscript cases,
    # goes through the same structure-derived S4 detector.
    if contact_unit_mode(cfg) == CONTACT_UNIT_UNIVERSAL:
        mapping, universal_audit = detect_universal_contact_units_from_pdbs(
            pdb_paths, consensus_min_fraction=fraction
        )
        rows = []
        for item in universal_audit:
            rule = str(item.get("rule", ""))
            atoms = tuple(value for value in str(item.get("atoms", "")).split(";") if value)
            resname = str(item.get("residue_name", ""))
            component_id = "_".join(
                value for value in (resname, str(item.get("chain", "")), str(item.get("resseq", "")))
                if value
            )
            rows.append({
                "recommendation_id": _recommendation_id(component_id or "LIGAND", rule or "unassigned", atoms),
                "status": "applied" if item.get("decision") == "apply" else "not_applied",
                "resname": resname,
                "rule": rule,
                "proposed_unit": str(item.get("unit", "")),
                "atoms": ";".join(atoms),
                "recommendation_kind": "universal_s4_rule",
                "confidence": str(item.get("confidence", "")),
                "connectivity_evidence": str(item.get("connectivity_evidence", "")),
                "center_atom": str(item.get("center_atom", "")),
                "ensemble_observed_structures": item.get("ensemble_observed_structures", ""),
                "ensemble_available_structures": item.get("ensemble_available_structures", ""),
                "ensemble_min_fraction": item.get("ensemble_min_fraction", fraction),
                "enzyme_contact_assessment": "automatic_rule_application",
                "rationale": str(item.get("detail", "")),
                "source_pdbs": str(item.get("source_pdb", "")),
            })
        table = pd.DataFrame(rows, columns=columns)
        applied = int(sum(str(row.get("status")) == "applied" for row in rows))
        return table, mapping, {
            "contact_unit_mode": CONTACT_UNIT_UNIVERSAL,
            "recommendations_enabled": True,
            "recommendation_method": "universal_s4_rules_automatic",
            "recommendation_consensus_min_fraction": fraction,
            "n_ligand_unit_recommendations": len(rows),
            "n_accepted_ligand_unit_recommendations": applied,
            "accepted_recommendation_mapping_keys": sorted(mapping),
            "universal_rule_catalogue": [rule for rule, _label in UNIVERSAL_CONTACT_UNIT_RULES],
        }

    if not enabled:
        return pd.DataFrame(columns=columns), {}, {
            "recommendations_enabled": False, "n_ligand_unit_recommendations": 0,
            "n_accepted_ligand_unit_recommendations": 0,
        }

    accepted_ids = {value.upper() for value in _as_string_list(settings.get("accepted_recommendation_ids", []))}
    rejected_ids = {value.upper() for value in _as_string_list(settings.get("rejected_recommendation_ids", []))}
    if accepted_ids & rejected_ids:
        overlap = ", ".join(sorted(accepted_ids & rejected_ids))
        raise ValueError(f"A ligand-unit recommendation cannot be both accepted and rejected: {overlap}")

    if validated_case:
        unit_definitions = BUILTIN_LIGAND_UNITS.get(validated_case, {})
        rows = []
        accepted_mapping: Dict[str, str] = {}
        generated_ids: Set[str] = set()
        for unit, atoms in unit_definitions.items():
            recommendation_id = _recommendation_id(
                validated_case, "contact_unit", tuple(sorted(atoms))
            )
            configured_id = recommendation_id.upper()
            generated_ids.add(configured_id)
            status = (
                "accepted" if configured_id in accepted_ids
                else "rejected" if configured_id in rejected_ids
                else "recommended"
            )
            rows.append({
                "recommendation_id": recommendation_id,
                "status": status,
                "resname": validated_case,
                "rule": "contact_unit",
                "proposed_unit": unit,
                "atoms": ";".join(atoms),
                "recommendation_kind": "built_in_suggestion",
                "confidence": "high",
                "connectivity_evidence": "configured_definition",
                "center_atom": "",
                "ensemble_observed_structures": "",
                "ensemble_available_structures": "",
                "ensemble_min_fraction": "",
                "enzyme_contact_assessment": "",
                "rationale": "Suggested contact unit.",
                "source_pdbs": "",
            })
            if status == "accepted":
                for atom_name in atoms:
                    accepted_mapping[atom_name.upper()] = unit
        unmatched_accepted_ids = sorted(accepted_ids - generated_ids)
        unmatched_rejected_ids = sorted(rejected_ids - generated_ids)
        table = pd.DataFrame(rows, columns=columns)
        return table, accepted_mapping, {
            "contact_unit_mode": CONTACT_UNIT_GUIDED,
            "recommendations_enabled": True,
            "recommendation_method": "configured_contact_unit_suggestions",
            "n_ligand_unit_recommendations": len(rows),
            "n_accepted_ligand_unit_recommendations": sum(
                row["status"] == "accepted" for row in rows
            ),
            "accepted_recommendation_ids": sorted(accepted_ids),
            "rejected_recommendation_ids": sorted(rejected_ids),
            "matched_accepted_recommendation_ids": sorted(accepted_ids & generated_ids),
            "unmatched_accepted_recommendation_ids": unmatched_accepted_ids,
            "unmatched_rejected_recommendation_ids": unmatched_rejected_ids,
            "accepted_recommendation_mapping_keys": sorted(accepted_mapping),
        }

    paths = sorted({p.resolve() for p in pdb_paths if p and p.exists()}, key=lambda p: str(p))
    residue_observed_in: Dict[str, Set[str]] = defaultdict(set)
    for path in paths:
        try:
            atoms, _unused = parse_ligand_atoms_from_pdb(path)
        except Exception:
            continue
        for atom in atoms:
            if atom.resname.upper() not in {"HOH", "WAT", "DOD"}:
                residue_observed_in[atom.resname.upper()].add(str(path))

    candidates = _reviewable_terminal_group_candidates(paths)
    candidate_sources: Dict[Tuple[str, str, str, Tuple[str, ...]], Set[str]] = defaultdict(set)
    candidate_metadata: Dict[Tuple[str, str, str, Tuple[str, ...]], dict] = {}
    for candidate in candidates:
        key = (
            str(candidate["resname"]), str(candidate["rule"]), str(candidate["center_atom"]),
            tuple(candidate["atoms"]),
        )
        candidate_sources[key].add(str(candidate["source_pdb"]))
        candidate_metadata[key] = candidate

    rows_by_id: Dict[str, dict] = {}
    for key, source_paths in sorted(candidate_sources.items()):
        resname, rule, center_atom, atoms = key
        available = max(1, len(residue_observed_in.get(resname, set())))
        observed = len(source_paths)
        if observed < max(1, math.ceil(fraction * available)):
            continue
        candidate = candidate_metadata[key]
        recommendation_id = _recommendation_id(resname, rule, atoms)
        enzyme_assessment, contact_rationale = _enzyme_contact_assessment(atoms, resname, raw_records)
        rows_by_id[recommendation_id] = {
            "recommendation_id": recommendation_id,
            "status": "", "resname": resname, "rule": rule,
            "proposed_unit": str(candidate["proposed_unit"]), "atoms": ";".join(atoms),
            "recommendation_kind": "review_required",
            "confidence": (
                "high" if str(candidate["connectivity_evidence"]) == "explicit_pdb_conect_terminal_atoms"
                else "moderate_review_required"
            ),
            "connectivity_evidence": str(candidate["connectivity_evidence"]),
            "center_atom": center_atom,
            "ensemble_observed_structures": observed,
            "ensemble_available_structures": available,
            "ensemble_min_fraction": fraction,
            "enzyme_contact_assessment": enzyme_assessment,
            "rationale": (
                (
                    "Only terminal atoms with explicit PDB CONECT evidence are proposed. "
                    if str(candidate["connectivity_evidence"]) == "explicit_pdb_conect_terminal_atoms"
                    else "The PDB lacks explicit CONECT records; a conservative local covalent-geometry check supports this preview only. "
                )
                + "Centre and bridge atoms are excluded. " + contact_rationale
            ),
            "source_pdbs": ";".join(sorted(source_paths)),
        }

    accepted_mapping: Dict[str, str] = {}
    rows = []
    for recommendation_id, row in sorted(rows_by_id.items()):
        configured_id = recommendation_id.upper()
        status = "accepted" if configured_id in accepted_ids else ("rejected" if configured_id in rejected_ids else "proposed")
        output = dict(row)
        output["status"] = status
        rows.append(output)
        if status == "accepted":
            for atom_name in str(output["atoms"]).split(";"):
                accepted_mapping[f"{output['resname']}:{atom_name}"] = str(output["proposed_unit"])

    # Explicitly listed IDs that were not generated should be visible in the
    # output instead of silently doing nothing (for example after ligand atom
    # names changed between template and docked complex).
    generated_ids = {str(row["recommendation_id"]).upper() for row in rows}
    unmatched_accepted_ids = sorted(accepted_ids - generated_ids)
    unmatched_rejected_ids = sorted(rejected_ids - generated_ids)
    for stale_id in sorted((accepted_ids | rejected_ids) - generated_ids):
        rows.append({
            "recommendation_id": stale_id, "status": "unmatched_configured_id", "resname": "",
            "rule": "", "proposed_unit": "", "atoms": "", "recommendation_kind": "",
            "confidence": "none", "connectivity_evidence": "", "center_atom": "",
            "ensemble_observed_structures": "", "ensemble_available_structures": "",
            "ensemble_min_fraction": fraction,
            "enzyme_contact_assessment": "",
            "rationale": "Configured ID was not generated from the current ligand-bearing PDB ensemble.",
            "source_pdbs": "",
        })
    table = pd.DataFrame(rows, columns=columns)
    return table, accepted_mapping, {
        "recommendations_enabled": True,
        "recommendation_method": "explicit_conect_or_conservative_local_geometry_terminal_atoms_with_optional_enzyme_contact_audit",
        "recommendation_consensus_min_fraction": fraction,
        "n_ligand_unit_recommendations": int(len(rows_by_id)),
        "n_accepted_ligand_unit_recommendations": int(sum(row["status"] == "accepted" for row in rows)),
        "accepted_recommendation_ids": sorted(accepted_ids),
        "rejected_recommendation_ids": sorted(rejected_ids),
        "matched_accepted_recommendation_ids": sorted(accepted_ids & generated_ids),
        "unmatched_accepted_recommendation_ids": unmatched_accepted_ids,
        "unmatched_rejected_recommendation_ids": unmatched_rejected_ids,
        "accepted_recommendation_mapping_keys": sorted(accepted_mapping),
    }


# ---------------------------------------------------------------------------
# Interaction parsing and grouping
# ---------------------------------------------------------------------------


def parse_interaction_output(path: Path, structure_id: str, cls: str, graph_file: str = "", graph_path: str = "") -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = {}
            for field in raw.split("\t"):
                if ":" in field:
                    k, v = field.split(":", 1)
                    parts[k.strip().lower()] = v.strip()
            if not parts:
                continue
            records.append({
                "line_no": i,
                "raw_line": raw,
                "raw_type": parts.get("type", ""),
                "ligand_atom": parts.get("ligand", ""),
                "protein_raw": parts.get("protein", ""),
                "ligand_serial": parts.get("ligandserial", ""),
                "ligand_node_key": parts.get("ligandnodekey", ""),
                "structure_id": structure_id,
                "class": cls,
                "graph_file": graph_file,
                "graph_path": graph_path,
                "interaction_output": str(path),
            })
    return records


def protein_canonical(label: str, strip_single_chain_suffix: bool = True) -> str:
    s = str(label or "").strip()
    if strip_single_chain_suffix and re.match(r"^[A-Z]{3}-?\d+[A-Za-z]$", s):
        return s[:-1]
    return s


def extract_ligand_identity(ligand_atom: str, ligand_node_key: str = "") -> Tuple[str, str, str, str]:
    """Return resname, chain, resseq, atom name from Step06 labels when possible."""
    key = str(ligand_node_key or "").strip()
    if ":" in key:
        toks = key.split(":")
        if len(toks) >= 4:
            return toks[0].upper(), toks[1] or "_", toks[2], toks[-1].upper()
    s = str(ligand_atom or "").strip()
    if "_" in s:
        parts = s.split("_")
        atom = parts[-1].upper()
        res = parts[0].upper() if parts else "LIG"
        return res, "_", "", atom
    return "", "_", "", s.upper()


def ligand_lookup_keys(ligand_atom: str, ligand_node_key: str = "") -> List[str]:
    resname, chain, resseq, atom = extract_ligand_identity(ligand_atom, ligand_node_key)
    keys = []
    if resname and resseq:
        keys.extend([f"{resname}:{chain}:{resseq}:{atom}", f"{resname}:{resseq}:{atom}"])
    if resname:
        keys.append(f"{resname}:{atom}")
    keys.append(atom)
    # Also include the raw node key for users who define fully custom labels.
    if ligand_node_key:
        keys.insert(0, ligand_node_key)
    return [k.upper() for k in keys if k]


def default_keep_ligand_unit(
    ligand_atom: str,
    ligand_node_key: str = "",
    *,
    infer_ion_from_atom_name: bool = True,
) -> str:
    resname, chain, resseq, atom = extract_ligand_identity(ligand_atom, ligand_node_key)
    if infer_ion_from_atom_name and atom.upper() in ION_ATOMS:
        if atom.upper() == "MG":
            return "Mg2+ ion"
        return f"{atom.upper()} ion"
    if resname and resseq and atom:
        return f"{resname}_{resseq}_{atom}".replace(":", "_")
    return str(ligand_atom or atom).strip()


def normalize_type_name(raw: str) -> str:
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def collect_manual_units(cfg: dict, target_id: str) -> Tuple[Dict[str, str], str]:
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) or {}
    unit_defs = None
    source = "none"
    for key in ["manual_ligand_units", "ligand_units", "canonical_ligand_units", "target_ligand_units"]:
        if key in can and can[key]:
            unit_defs = can[key]
            source = f"config.step09.canonicalization.{key}"
            break
    if isinstance(unit_defs, dict) and target_id in unit_defs and isinstance(unit_defs[target_id], dict):
        unit_defs = unit_defs[target_id]
        source += f".{target_id}"

    atom_to_unit: Dict[str, str] = {}
    if isinstance(unit_defs, dict):
        for unit, atoms in unit_defs.items():
            if isinstance(atoms, str):
                # atom_key: unit style
                atom_to_unit[str(unit).upper()] = atoms
            else:
                for atom in atoms or []:
                    atom_to_unit[str(atom).upper()] = str(unit)
    elif isinstance(unit_defs, list):
        for item in unit_defs:
            if isinstance(item, dict):
                unit = item.get("unit") or item.get("name") or item.get("canonical_unit")
                atoms = item.get("atoms") or item.get("atom_names") or []
                for atom in atoms:
                    atom_to_unit[str(atom).upper()] = str(unit)
    return atom_to_unit, source


def collect_builtin_units(target_id: str, cfg: dict) -> Tuple[Dict[str, str], str]:
    """Return built-in contact-unit suggestions when enabled."""
    selected_mode = contact_unit_mode(cfg)
    if selected_mode == CONTACT_UNIT_UNIVERSAL:
        return {}, "disabled_in_universal_rules"
    if selected_mode == CONTACT_UNIT_CONSERVATIVE:
        return {}, "disabled_by_conservative_manual_policy"
    if equivalence_mode(cfg) != "standard":
        return {}, "disabled_in_strict_chemistry_mode"
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) or {}
    if not read_bool(can.get("use_builtin_ligand_units"), True):
        return {}, "disabled"
    if (
        validated_contact_unit_target(target_id)
        and contact_unit_mode(cfg) == CONTACT_UNIT_GUIDED
        and _recommendations_require_acceptance(cfg)
    ):
        return {}, "applied_after_review_acceptance"
    rejected_ids = {
        value.upper()
        for value in _as_string_list(_recommendation_settings(cfg).get("rejected_recommendation_ids", []))
    }
    requested_target = str(target_id).strip().casefold()
    for key, unit_definitions in BUILTIN_LIGAND_UNITS.items():
        if key.casefold() == requested_target:
            mapping: Dict[str, str] = {}
            for unit, atoms in unit_definitions.items():
                recommendation_id = _recommendation_id(
                    validated_contact_unit_target(target_id) or key,
                    "contact_unit",
                    tuple(sorted(atoms)),
                ).upper()
                if recommendation_id in rejected_ids:
                    continue
                for atom in atoms:
                    mapping[str(atom).upper()] = unit
            return mapping, f"builtin.{key}"
    return {}, "none"


def collect_ligand_units(cfg: dict, target_id: str, ligand_pdbs: Iterable[Path]) -> Tuple[Dict[str, str], dict, pd.DataFrame]:
    manual, manual_source = collect_manual_units(cfg, target_id)
    builtin, builtin_source = collect_builtin_units(target_id, cfg)
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) or {}
    public_contact_mode = contact_unit_mode(cfg)
    auto_enabled = (
        public_contact_mode in {CONTACT_UNIT_UNIVERSAL, CONTACT_UNIT_GUIDED}
        and read_bool(can.get("auto_detect_ligand_units"), True)
    )
    mode = equivalence_mode(cfg)
    require_acceptance = _recommendations_require_acceptance(cfg)
    _recommendation_df, accepted_recommendations, recommendation_report = build_ligand_unit_recommendations(
        cfg, ligand_pdbs, target_id=target_id
    )
    if public_contact_mode == CONTACT_UNIT_UNIVERSAL:
        fraction = float(_recommendation_settings(cfg).get("consensus_min_fraction", 0.80))
        universal_mapping, auto_audit = detect_universal_contact_units_from_pdbs(
            ligand_pdbs, consensus_min_fraction=fraction
        )
        combined = dict(universal_mapping)
        combined.update({key.upper(): value for key, value in manual.items()})
        report = {
            "contact_unit_mode": CONTACT_UNIT_UNIVERSAL,
            "manual_unit_source": manual_source,
            "builtin_unit_source": builtin_source,
            "auto_detect_ligand_units": True,
            "n_manual_atom_mappings": len(manual),
            "n_builtin_atom_mappings": 0,
            "n_auto_atom_mappings": len(universal_mapping),
            "n_accepted_recommendation_atom_mappings": 0,
            "n_total_atom_mappings": len(combined),
            "unknown_ligand_atoms_policy": "ligand_and_atom_name_specific_site",
            "strict_ion_identity": True,
            "automatic_rule": "Supplementary_Table_S4_universal_contact_units",
            "equivalence_mode": mode,
            **recommendation_report,
        }
        return combined, report, pd.DataFrame(auto_audit)

    if mode == "standard":
        # Existing v8 projects (which predate the review block) retain their
        # original automatic detector.  New Studio projects explicitly ask for
        # acceptance first; in that case generic geometry/PDB-name guesses are
        # audit-only, while validated built-ins and manual rules still apply.
        profile = workflow_profile(cfg)
        auto, auto_audit = (
            detect_ligand_units_from_pdbs(
                ligand_pdbs, strict_ion_identity=(profile == GENERAL)
            )
            if auto_enabled and not require_acceptance
            else ({}, [])
        )
        combined = dict(auto)
        # A user-accepted recommendation is an explicit, residue-aware rule.
        # It may refine automatic detection, but validated v8 presets and a
        # hand-written project rule remain higher-priority for compatibility.
        combined.update({key.upper(): value for key, value in accepted_recommendations.items()})
        combined.update({key.upper(): value for key, value in builtin.items()})
        combined.update({key.upper(): value for key, value in manual.items()})
        report = {
            "contact_unit_mode": public_contact_mode,
            "manual_unit_source": manual_source,
            "builtin_unit_source": builtin_source,
            "auto_detect_ligand_units": auto_enabled,
            "n_manual_atom_mappings": len(manual),
            "n_builtin_atom_mappings": len(builtin),
            "n_auto_atom_mappings": len(auto),
            "n_accepted_recommendation_atom_mappings": len(accepted_recommendations),
            "n_total_atom_mappings": len(combined),
            "unknown_ligand_atoms_policy": "keep_residue_aware_atom_label",
            "strict_ion_identity": profile == GENERAL,
            "equivalence_mode": mode,
            **recommendation_report,
        }
        return combined, report, pd.DataFrame(auto_audit)

    # In a newly configured strict analysis, automatic motif detection feeds
    # the recommendation table only.  It is not applied until the user accepts
    # a stable proposal ID.  Older strict YAMLs retain their historical
    # automatic behavior unless the new recommendations block is added.
    strict_auto_enabled = auto_enabled and not require_acceptance
    if strict_auto_enabled:
        # General automatic rules are chemistry-local and ensemble-consistent.
        # Strict graph symmetry supplements them for motifs not covered by the
        # functional-group catalogue, but can never override a group unit.
        functional_auto, functional_audit = detect_functional_group_consensus_units_from_pdbs(ligand_pdbs)
        symmetry_auto, symmetry_audit = detect_strict_symmetry_units_from_pdbs(ligand_pdbs)
        auto = dict(symmetry_auto)
        auto.update(functional_auto)
        auto_audit = functional_audit + symmetry_audit
    else:
        functional_auto, symmetry_auto, auto, auto_audit = {}, {}, {}, []

    # Precedence: explicit user/registry rule > automatic chemistry rule.
    combined = dict(auto)
    combined.update({key.upper(): value for key, value in accepted_recommendations.items()})
    combined.update({k.upper(): v for k, v in manual.items()})
    report = {
        "contact_unit_mode": public_contact_mode,
        "manual_unit_source": manual_source,
        "ligand_unit_registry_source": str(can.get("_ligand_unit_registry_source", "")),
        "ligand_unit_registry_ids": list(can.get("_ligand_unit_registry_ids", []) or []),
        "builtin_unit_source": builtin_source,
        "auto_detect_ligand_units": auto_enabled,
        "recommendations_require_acceptance": require_acceptance,
        "n_manual_atom_mappings": len(manual),
        "n_builtin_atom_mappings": len(builtin),
        "n_auto_functional_group_mappings": len(functional_auto),
        "n_auto_strict_symmetry_mappings": len(symmetry_auto),
        "n_auto_atom_mappings": len(auto),
        "n_accepted_recommendation_atom_mappings": len(accepted_recommendations),
        "n_total_atom_mappings": len(combined),
        "unknown_ligand_atoms_policy": "keep_residue_aware_atom_label",
        "strict_ion_identity": True,
        "automatic_rule": "manual_first_functional_group_ensemble_consensus_plus_explicit_CONECT_symmetry",
        "equivalence_mode": mode,
        **recommendation_report,
    }
    return combined, report, pd.DataFrame(auto_audit)


def build_ligand_unit_coverage_audit(
    can_df: pd.DataFrame, cfg: dict, target_id: str
) -> Tuple[pd.DataFrame, dict]:
    """Audit whether explicit chemistry rules actually matched observed PLIP atoms.

    A silent zero-match rule set is the most common reason a Step09 result has
    exactly as many merged classes as GED classes.  This table makes that
    condition reproducible and distinguishable from the valid case where rules
    matched but no two topology classes had an identical final signature.
    """
    manual, manual_source = collect_manual_units(cfg, target_id)
    rows: List[dict] = []
    if manual:
        matched_series = (
            can_df.get("ligand_unit_matched_key", pd.Series(dtype=str))
            .fillna("").astype(str).str.upper()
        )
        for key, unit in sorted(manual.items()):
            count = int((matched_series == str(key).upper()).sum())
            rows.append({
                "mapping_source": manual_source,
                "configured_lookup_key": str(key).upper(),
                "configured_unit": str(unit),
                "n_raw_interaction_records_matched": count,
                "status": "matched" if count else "unmatched_in_observed_PLIP_records",
            })
    else:
        rows.append({
            "mapping_source": "none",
            "configured_lookup_key": "",
            "configured_unit": "",
            "n_raw_interaction_records_matched": 0,
            "status": "no_explicit_manual_or_registry_rules",
        })

    auto_matched = int((
        can_df.get("ligand_unit_mapping_source", pd.Series(dtype=str))
        .fillna("").astype(str).eq("automatic_chemistry_consensus")
    ).sum())
    if auto_matched:
        rows.append({
            "mapping_source": "automatic_functional_group_or_strict_symmetry",
            "configured_lookup_key": "",
            "configured_unit": "",
            "n_raw_interaction_records_matched": auto_matched,
            "status": "matched_automatically",
        })

    audit = pd.DataFrame(rows)
    n_configured = int(len(manual))
    n_matched_keys = int((audit.get("status", pd.Series(dtype=str)) == "matched").sum())
    n_matched_records = int(audit.get("n_raw_interaction_records_matched", pd.Series(dtype=int)).sum())
    diagnostics: List[str] = []
    if n_configured and not n_matched_records:
        diagnostics.append(
            "Explicit ligand contact-unit rules matched 0 observed PLIP ligand atoms. "
            "Check that this exact YAML was used, and that its atom names/residue names match Step06 ligand_node_key values."
        )
    elif not n_configured:
        if auto_matched:
            diagnostics.append(
                "No explicit rules were configured; recurring functional-group/symmetry evidence automatically matched observed PLIP ligand atoms. Review auto_ligand_unit_audit.csv before reporting the merged classes."
            )
        else:
            diagnostics.append(
                "No explicit ligand contact-unit rules were configured and no automatic functional-group or strict-symmetry mapping matched observed PLIP ligand atoms. Step09 kept atom-level labels."
            )
    return audit, {
        "n_explicit_ligand_unit_mappings": n_configured,
        "n_explicit_ligand_unit_keys_matched": n_matched_keys,
        "n_raw_interactions_matched_by_explicit_units": n_matched_records,
        "chemical_equivalence_diagnostics": diagnostics,
    }


def collect_type_map(cfg: dict) -> Tuple[Dict[str, str], str]:
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) or {}
    tmap = {normalize_type_name(k): v for k, v in DEFAULT_TYPE_MAP.items()}
    user_map = can.get("interaction_type_map") or can.get("type_map") or {}
    for k, v in (user_map or {}).items():
        tmap[normalize_type_name(k)] = str(v)
    source = "Supplementary_Table_S5_default"
    if user_map:
        source += "+config_override"
    return tmap, source


def canonicalize_records(raw_records: List[dict], cfg: dict, target_id: str, ligand_pdbs: Iterable[Path]) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    step09 = cfg.get("step09", {}) or {}
    can = step09.get("canonicalization", {}) or {}
    profile = workflow_profile(cfg)
    mode = equivalence_mode(cfg)
    selected_contact_mode = contact_unit_mode(cfg)
    strip_chain = can.get("strip_single_chain_suffix", "auto")
    strip_chain_bool = (profile != GENERAL) if strip_chain == "auto" else read_bool(strip_chain, profile != GENERAL)
    preserve_chain_display = profile == GENERAL and not strip_chain_bool
    atom_to_unit, unit_report, auto_audit_df = collect_ligand_units(cfg, target_id, ligand_pdbs)
    explicit_units, _explicit_source = collect_manual_units(cfg, target_id)
    accepted_recommendation_keys = {
        str(key).upper()
        for key in unit_report.get("accepted_recommendation_mapping_keys", [])
    }
    type_map, type_source = collect_type_map(cfg)
    rows = []
    for r in raw_records:
        raw_type = r.get("raw_type", "")
        raw_key = normalize_type_name(raw_type)
        supertype = type_map.get(raw_key, raw_type.upper() if raw_type else "UNKNOWN")
        ligand_atom = r.get("ligand_atom", "")
        ligand_node_key = r.get("ligand_node_key", "")
        _resname, _chain, _resseq, atom = extract_ligand_identity(ligand_atom, ligand_node_key)
        unit = None
        matched_key = ""
        lookup_keys = ligand_lookup_keys(ligand_atom, ligand_node_key)
        if mode == "standard":
            # Preserve v8 lookup-key precedence exactly.  This matters when a
            # target-specific bare atom mapping coexists with a residue-aware
            # auto-detected key in the same docking ensemble.
            for key in lookup_keys:
                if key in atom_to_unit:
                    unit = atom_to_unit[key]
                    matched_key = key
                    break
        else:
            # In the explicitly selected strict mode, manual YAML/registry
            # rules override automatic residue-aware mappings.
            for key in lookup_keys:
                if key in explicit_units:
                    unit = explicit_units[key]
                    matched_key = key
                    break
            if not unit:
                for key in lookup_keys:
                    if key in atom_to_unit:
                        unit = atom_to_unit[key]
                        matched_key = key
                        break
        if not unit:
            if selected_contact_mode == CONTACT_UNIT_UNIVERSAL and atom:
                component = " ".join(
                    token for token in (
                        _resname,
                        f"{_chain}:{_resseq}" if _chain and _resseq else "",
                    ) if token
                )
                unit = f"{component} {atom} site".strip()
            else:
                unit = default_keep_ligand_unit(
                    ligand_atom,
                    ligand_node_key,
                    infer_ion_from_atom_name=(profile != GENERAL),
                )
        prot = protein_canonical(r.get("protein_raw", ""), strip_chain_bool)
        prot_display = protein_display_label(prot, preserve_chain=preserve_chain_display)
        edge = f"{prot_display} | {supertype} | {unit}"
        if matched_key and matched_key.upper() in explicit_units:
            mapping_source = "explicit_manual_or_registry"
        elif matched_key and selected_contact_mode == CONTACT_UNIT_UNIVERSAL:
            mapping_source = "universal_s4_rule"
        elif matched_key and matched_key.upper() in accepted_recommendation_keys:
            mapping_source = "accepted_recommendation"
        elif matched_key and mode == "standard":
            mapping_source = "standard_compatibility_mapping"
        elif matched_key:
            mapping_source = "automatic_chemistry_consensus"
        else:
            mapping_source = "atom_level_fallback"
        row = dict(r)
        row.update({
            "interaction_supertype": supertype,
            "protein_canonical": prot,
            "protein_display": prot_display,
            "ligand_label": atom,
            "canonical_ligand_unit": unit,
            "canonical_edge": edge,
            "canonical_edge_display": edge,
            "ligand_unit_matched_key": matched_key,
            "ligand_unit_mapping_source": mapping_source,
        })
        rows.append(row)
    report = {
        "interaction_type_source": type_source,
        "strip_single_chain_suffix": strip_chain_bool,
        "preserve_chain_in_canonical_edge": preserve_chain_display,
        "workflow_profile": profile,
        **unit_report,
    }
    return pd.DataFrame(rows), report, auto_audit_df


def activate_ligand_unit_registry(cfg: dict, config_path: Path) -> None:
    """Load reusable substrate rules once, then apply selected entries by ID.

    The registry is deliberately substrate-centric rather than enzyme-centric:
    a rule entry can be reused for every enzyme analysed with that substrate.
    Explicit ``manual_ligand_units`` in the project config still overrides the
    registry, and the coverage audit above verifies every selected rule.
    """
    can = ((cfg.get("step09", {}) or {}).get("canonicalization", {}) or {})
    path_text = can.get("ligand_unit_registry_file")
    selected = can.get("registry_ligand_ids", []) or []
    if isinstance(selected, str):
        selected = [value for value in re.split(r"[,;\\s]+", selected) if value]
    selected = [str(value) for value in selected]
    if not path_text and not selected:
        return
    if not path_text:
        raise ValueError("step09.canonicalization.registry_ligand_ids requires ligand_unit_registry_file")
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ligand-unit registry file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        registry = yaml.safe_load(text) if yaml is not None else json.loads(text)
    except Exception as exc:
        raise ValueError(f"Cannot parse ligand-unit registry {path}: {exc}") from exc
    entries = (registry or {}).get("ligands", registry or {})
    if not isinstance(entries, Mapping):
        raise ValueError("Ligand-unit registry must contain a 'ligands:' mapping.")
    if not selected:
        raise ValueError(
            "Ligand-unit registry was supplied but no registry_ligand_ids were selected. "
            "Select an explicit substrate entry to prevent cross-substrate atom-name collisions."
        )
    registry_units: Dict[str, object] = {}
    for ligand_id in selected:
        entry = entries.get(ligand_id)
        if not isinstance(entry, Mapping):
            raise KeyError(f"Ligand-unit registry has no mapping for {ligand_id!r}.")
        units = entry.get("units", entry.get("manual_ligand_units", {}))
        if not isinstance(units, Mapping):
            raise ValueError(f"Registry entry {ligand_id!r} needs a 'units:' mapping.")
        for unit, atoms in units.items():
            registry_units[str(unit)] = atoms
    explicit = can.get("manual_ligand_units", {}) or {}
    if not isinstance(explicit, Mapping):
        raise ValueError("step09.canonicalization.manual_ligand_units must be a mapping.")
    merged_units = dict(registry_units)
    merged_units.update(dict(explicit))
    can["manual_ligand_units"] = merged_units
    can["_ligand_unit_registry_source"] = str(path)
    can["_ligand_unit_registry_ids"] = selected


def edge_set_signature(edges: Iterable[str]) -> str:
    return ";".join(sorted(set(str(e) for e in edges if str(e))))


def split_nonidentical_classes(
    class_df: pd.DataFrame,
    struct_df: pd.DataFrame,
    can_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a Step08 topology class when chemistry-aware signatures differ.

    This is general-mode behavior only.  Step08 files and class assignments are
    never modified; the original label is retained in step08_class.
    """
    assignments: Dict[Tuple[str, str], str] = {}
    audit_rows = []
    for cls, group in struct_df.groupby("class", sort=False):
        counts = group["canonical_signature"].astype(str).value_counts()
        ordered = sorted(counts.index, key=lambda sig: (-int(counts[sig]), sig))
        for index, signature in enumerate(ordered, start=1):
            effective = str(cls) if len(ordered) == 1 else f"{cls}__chem{index}"
            members = group[group["canonical_signature"].astype(str) == signature]
            for sid in members["structure_id"].astype(str):
                assignments[(str(cls), sid)] = effective
            audit_rows.append(
                {
                    "step08_class": cls,
                    "effective_class": effective,
                    "n_signatures_in_step08_class": len(ordered),
                    "n_structures": len(members),
                    "structure_ids": ";".join(sorted(members["structure_id"].astype(str), key=natural_key)),
                    "canonical_signature": signature,
                    "action": "retained" if len(ordered) == 1 else "split_by_chemical_signature",
                }
            )

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["step08_class"] = out["class"].astype(str)
        out["class"] = [
            assignments.get((str(cls), str(sid)), str(cls))
            for cls, sid in zip(out["step08_class"], out["structure_id"])
        ]
        return out

    return apply(class_df), apply(struct_df), apply(can_df), pd.DataFrame(audit_rows)


# ---------------------------------------------------------------------------
# PDB export mapping
# ---------------------------------------------------------------------------


def choose_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["standard_structure_id", "structure_id", "complex_id", "model_id", "id", "name"]:
        if c in df.columns:
            return c
    return None


def choose_pdb_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "trimmed_complex_pdb_path", "trimmed_complex_path", "complex_pdb_path",
        "complex_path", "pdb_path", "structure_path", "output_complex_pdb",
    ]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ["pdb", "complex", "path"]):
            return c
    return None


def build_complex_pdb_index(output_root: Path, target_id: str, config_path: Path, plip_pdb_index: Mapping[str, Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = dict(plip_pdb_index)
    tables = [
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step07_passed_complex_index.csv",
        output_root / "04_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        output_root / "07_key_residue_screening" / "tables" / f"{target_id}_step07_passed_complex_index.csv",
        output_root / "05_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        output_root / "05_pocket_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        output_root / "04_ligand_atom_trimming" / "tables" / f"{target_id}_step04_trimmed_complex_index.csv",
        output_root / "03_clash_filter" / "tables" / f"{target_id}_step03_passed_complex_index.csv",
        output_root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_complex_index.csv",
    ]
    for table in tables:
        if not table.exists():
            continue
        try:
            df = pd.read_csv(table)
        except Exception:
            continue
        pdb_col = choose_pdb_column(df)
        id_col = choose_id_column(df)
        if not pdb_col:
            continue
        for _, row in df.iterrows():
            if pd.isna(row[pdb_col]) or not str(row[pdb_col]).strip():
                continue
            p = resolve_path(str(row[pdb_col]), config_path)
            if not p.exists():
                continue
            aliases = {p.stem, normalize_structure_id(p.stem)}
            if id_col and pd.notna(row[id_col]):
                aliases.add(str(row[id_col]))
                aliases.add(normalize_structure_id(row[id_col]))
            for a in aliases:
                if a:
                    index.setdefault(a, p)
    return index


def find_pdb_for_structure(sid: str, pdb_index: Mapping[str, Path]) -> Optional[Path]:
    candidates = [sid, normalize_structure_id(sid)]
    for c in candidates:
        if c in pdb_index:
            return pdb_index[c]
    norm = normalize_structure_id(sid)
    matches = [(k, v) for k, v in pdb_index.items() if k.endswith(norm) or norm.endswith(k)]
    if matches:
        return sorted(matches, key=lambda kv: (len(kv[0]), kv[0]))[0][1]
    return None


def export_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main Step09
# ---------------------------------------------------------------------------


def run_step09(config_path: str | Path, clean_output: bool = False) -> dict:
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)
    activate_ligand_unit_registry(cfg, config_path)
    target_id, output_root_raw = get_project(cfg)
    output_root = resolve_path(output_root_raw, config_path)
    step09 = cfg.get("step09", {}) or {}
    profile = workflow_profile(cfg)
    mode = equivalence_mode(cfg)
    if step09.get("enabled", True) is False:
        return {"target_id": target_id, "status": "skipped", "message": "step09.enabled is false."}

    output_subdir = public_output_subdir(
        step09.get("output_subdir"), "08_chemical_equivalence"
    )
    outdir = output_root / output_subdir
    if clean_output and outdir.exists():
        shutil.rmtree(outdir)
    tables_dir = outdir / "tables"
    reports_dir = outdir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_label_cfg = step09.get("output_labels", {}) or {}
    class_prefix = str(output_label_cfg.get("class_prefix", "Class") or "Class")

    table_path, class_dir = find_step08_class_table(cfg, config_path, output_root, target_id)
    class_df = read_class_table(table_path) if table_path else reconstruct_class_table(class_dir)  # type: ignore[arg-type]
    class_df["class"] = class_df["class"].map(standard_class_name)
    class_df["structure_id"] = class_df["structure_id"].map(normalize_structure_id)
    class_df = class_df.drop_duplicates(subset=["class", "structure_id"]).sort_values(
        ["class", "structure_id"], key=lambda col: col.map(lambda x: _class_sort_key(x) if re.search(r"^(class|type|family)", str(x), flags=re.I) else str(x))
    ).reset_index(drop=True)
    n_step08_classes = int(class_df["class"].nunique())

    step06_dir = resolve_step06_dir(cfg, config_path, output_root)
    plip_index = build_plip_index(step06_dir)
    plip_pdb_index = build_plip_pdb_index(step06_dir)
    pdb_index = build_complex_pdb_index(output_root, target_id, config_path, plip_pdb_index)

    # Independent Step07-passed population used by the publication-style
    # interaction-frequency table.  Do not derive this population from Step08:
    # doing so would silently change the denominator when GED input/output is
    # missing for a structure that legitimately passed Step07.
    step07_frequency_cfg = step09.get("step07_interaction_frequency", {}) or {}
    step07_frequency_enabled = read_bool(step07_frequency_cfg.get("enabled"), True)
    step07_passed_table = find_step07_passed_table(cfg, config_path, output_root, target_id) if step07_frequency_enabled else None
    step07_passed_structures = read_step07_passed_structures(step07_passed_table) if step07_passed_table else []
    step07_raw_records: List[dict] = []
    step07_missing_rows: List[dict] = []
    step07_ligand_pdbs: List[Path] = []
    if step07_passed_structures:
        step07_raw_records, step07_missing_rows, step07_ligand_pdbs = collect_step07_passed_interactions(
            step07_passed_structures, plip_index, pdb_index
        )

    raw_records = []
    missing = []
    ligand_pdbs: List[Path] = []
    for _, row in class_df.iterrows():
        sid = row["structure_id"]
        ipath = find_interaction_file(sid, plip_index)
        if not ipath:
            missing.append({"structure_id": sid, "class": row["class"], "graph_file": row.get("graph_file", ""), "reason": "missing_interaction_output"})
            continue
        raw_records.extend(parse_interaction_output(ipath, sid, row["class"], row.get("graph_file", ""), row.get("graph_path", "")))
        pdb = find_pdb_for_structure(sid, pdb_index)
        if pdb:
            ligand_pdbs.append(pdb)
        else:
            for p in ipath.parent.glob("*.pdb"):
                if _pdb_has_hetatm(p):
                    ligand_pdbs.append(p)
                    break

    raw_df = pd.DataFrame(raw_records)
    raw_out = tables_dir / f"{target_id}_step08_raw_interactions.csv"
    write_csv(raw_out, raw_df)
    missing_df = pd.DataFrame(missing, columns=["structure_id", "class", "graph_file", "reason"])
    missing_out = tables_dir / f"{target_id}_step08_missing_interaction_outputs.csv"
    write_csv(missing_out, missing_df)
    if raw_df.empty:
        raise RuntimeError("No raw interactions parsed; cannot perform Step09. Check step09.input_step06_dir and PLIP interaction_output.txt files.")

    # This review bundle is deliberately written before contact-unit grouping.  It
    # lets a new enzyme project inspect the actual HETATM identities and each
    # proposed contact unit without accepting any recommendation by default.
    substrate_inventory_df = build_substrate_inventory(cfg, ligand_pdbs)
    recommendations_df, _accepted_recommendation_mapping, recommendation_report = (
        build_ligand_unit_recommendations(
            cfg, ligand_pdbs, raw_records=raw_records, target_id=target_id
        )
    )
    substrate_inventory_out = tables_dir / f"{target_id}_step08_substrate_inventory.csv"
    recommendations_out = tables_dir / f"{target_id}_step08_ligand_unit_recommendations.csv"
    write_csv(substrate_inventory_out, substrate_inventory_df)
    write_csv(recommendations_out, recommendations_df)

    configured_acceptances = recommendation_report.get("accepted_recommendation_ids", []) or []
    matched_acceptances = recommendation_report.get("matched_accepted_recommendation_ids", []) or []
    if configured_acceptances and not matched_acceptances:
        unmatched = ", ".join(recommendation_report.get("unmatched_accepted_recommendation_ids", []) or [])
        raise RuntimeError(
            "Chemical-unit decisions were present in the config, but none matched the "
            "current ligand ensemble. No chemical merging was performed. Review "
            f"{recommendations_out} and update the accepted IDs. Unmatched IDs: {unmatched}"
        )

    can_df, can_report, auto_audit_df = canonicalize_records(raw_records, cfg, target_id, ligand_pdbs)
    # Keep the recommendation report even if no PLIP edge happened to use an
    # accepted unit in this particular run.
    can_report.update(recommendation_report)
    mapping_out = tables_dir / f"{target_id}_step08_raw_to_canonical_mapping.csv"
    write_csv(mapping_out, can_df)
    auto_audit_out = tables_dir / f"{target_id}_step08_auto_ligand_unit_audit.csv"
    write_csv(auto_audit_out, auto_audit_df)
    unit_coverage_df, unit_coverage_report = build_ligand_unit_coverage_audit(can_df, cfg, target_id)
    unit_coverage_out = tables_dir / f"{target_id}_step08_ligand_unit_coverage_audit.csv"
    write_csv(unit_coverage_out, unit_coverage_df)
    can_report.update(unit_coverage_report)
    for message in unit_coverage_report["chemical_equivalence_diagnostics"]:
        print(f"[Step09 diagnostic] {message}", flush=True)

    # All interactions captured among Step07-passed structures, independent of
    # Step08 class membership.  This is the source for the manuscript-style
    # Interaction / Frequency / Category table requested by the workflow.
    thresholds = step09.get("thresholds", {}) or {}
    common_min = float(thresholds.get("common_frequency_min", 0.80))
    rare_max = float(thresholds.get("rare_frequency_max", 0.10))
    step07_can_df = pd.DataFrame()
    if step07_raw_records:
        step07_can_df, _step07_can_report, _step07_auto_audit = canonicalize_records(
            step07_raw_records, cfg, target_id, step07_ligand_pdbs
        )
    step07_frequency_df, step07_frequency_pub_df = build_step07_interaction_frequency_tables(
        step07_can_df,
        [row["structure_id"] for row in step07_passed_structures],
        high_frequency_min=common_min,
        low_frequency_max=rare_max,
    )
    step07_frequency_out = tables_dir / f"{target_id}_step08_step07_passed_interaction_frequency.csv"
    step07_frequency_pub_out = tables_dir / f"{target_id}_step08_step07_passed_interaction_frequency_publication.csv"
    step07_frequency_missing_out = tables_dir / f"{target_id}_step08_step07_passed_missing_interaction_outputs.csv"
    write_csv(step07_frequency_out, step07_frequency_df)
    write_csv(step07_frequency_pub_out, step07_frequency_pub_df)
    write_csv(
        step07_frequency_missing_out,
        pd.DataFrame(step07_missing_rows, columns=["structure_id", "reason"]),
    )

    # Complete Interaction / Frequency / Category table requested for the
    # manuscript.  Prefer the full Step07-passed population because this is the
    # population entering edge-level analysis.  If an older project has no
    # Step07 passed table, fall back to the structures successfully parsed by
    # Step09 and record that source explicitly in the detailed table/report.
    if step07_passed_structures:
        all_frequency_records = step07_can_df
        all_frequency_population_ids = [
            row["structure_id"] for row in step07_passed_structures
        ]
        all_frequency_population_label = "Step07-passed structures"
        all_frequency_population_source = "step07_passed"
    else:
        all_frequency_records = can_df
        all_frequency_population_ids = sorted(
            {
                normalize_structure_id(value)
                for value in can_df.get("structure_id", pd.Series(dtype=str)).tolist()
                if normalize_structure_id(value)
            },
            key=natural_key,
        )
        all_frequency_population_label = "Step09 structures with parsed interactions"
        all_frequency_population_source = "step09_parsed_fallback"

    all_frequency_df, all_frequency_pub_df = build_all_collected_interaction_frequency_tables(
        all_frequency_records,
        all_frequency_population_ids,
        population_label=all_frequency_population_label,
        high_frequency_min=common_min,
        low_frequency_max=rare_max,
    )
    all_frequency_out = tables_dir / f"{target_id}_step08_all_collected_interactions_frequency.csv"
    all_frequency_pub_out = tables_dir / f"{target_id}_step08_all_collected_interactions_frequency_publication.csv"
    write_csv(all_frequency_out, all_frequency_df)
    write_csv(all_frequency_pub_out, all_frequency_pub_df)

    # PLIP interaction to class lookup, useful for manual manuscript checks.
    plip_class_rows = []
    if not can_df.empty:
        for (protein, ligand, raw_type, canonical), g in can_df.groupby(["protein_raw", "ligand_atom", "raw_type", "canonical_edge"], dropna=False):
            plip_class_rows.append({
                "raw_plip_interaction": f"{ligand}:{protein} {raw_type}",
                "canonical_edge": canonical,
                "classes": ";".join(sorted(set(g["class"].astype(str)), key=natural_key)),
                "n_classes": g["class"].nunique(),
                "structures": ";".join(sorted(set(g["structure_id"].astype(str)), key=natural_key)),
                "n_structures": g["structure_id"].nunique(),
            })
    plip_classes_out = tables_dir / f"{target_id}_step08_plip_to_classes.csv"
    write_csv(plip_classes_out, pd.DataFrame(plip_class_rows))

    # Structure-level canonical edge sets.
    struct_rows = []
    for sid, g in can_df.groupby("structure_id", sort=False):
        edges = sorted(set(g["canonical_edge"].dropna().astype(str)))
        cls_values = class_df.loc[class_df["structure_id"] == sid, "class"]
        cls = cls_values.iloc[0] if not cls_values.empty else ""
        struct_rows.append({
            "structure_id": sid,
            "class": cls,
            "n_canonical_edges": len(edges),
            "canonical_signature": edge_set_signature(edges),
            "canonical_edges": ";".join(edges),
        })
    struct_df = pd.DataFrame(struct_rows)
    split_audit_df = pd.DataFrame()
    # Class splitting belongs to the optional chemistry-evidence workflow.
    # The standard mode deliberately keeps the established class contract,
    # even when an older GUI wrote split_nonidentical_classes: true.
    chemistry_review = mode == "strict_chemistry"
    split_enabled = chemistry_review and read_bool(
        step09.get("split_nonidentical_classes"), True
    )
    if split_enabled:
        class_df, struct_df, can_df, split_audit_df = split_nonidentical_classes(class_df, struct_df, can_df)
        # General-mode mapping should expose the effective chemistry-aware class.
        write_csv(mapping_out, can_df)
    struct_out = tables_dir / f"{target_id}_step08_structure_canonical_edges.csv"
    write_csv(struct_out, struct_df)
    split_audit_out = tables_dir / f"{target_id}_step08_class_split_audit.csv"
    write_csv(split_audit_out, split_audit_df)

    requested_policy = str(step09.get("class_signature_policy", "most_common")).lower()
    use_ged_representative = bool(
        validated_contact_unit_target(target_id)
        and contact_unit_mode(cfg) == CONTACT_UNIT_GUIDED
    )
    policy = "ged_representative" if use_ged_representative else requested_policy
    fail_nonidentical = read_bool(
        step09.get("fail_if_class_members_nonidentical"), chemistry_review
    )
    class_rows = []
    audit_nonidentical = []
    for cls, g in struct_df.groupby("class", sort=False):
        sigs = sorted(set(g["canonical_signature"].astype(str)))
        if len(sigs) > 1:
            audit_nonidentical.append({
                "class": cls,
                "n_signatures": len(sigs),
                "structure_ids": ";".join(g["structure_id"].astype(str)),
                "signatures": " || ".join(sigs),
            })
            if fail_nonidentical:
                raise RuntimeError(
                    f"Class {cls} contains {len(sigs)} different canonical signatures. "
                    "Check Step06/Step08 mapping or set "
                    "step09.fail_if_class_members_nonidentical=false for audit mode."
                )
        if policy == "ged_representative":
            representative_ids = (
                class_df.loc[class_df["class"] == cls, "representative_structure_id"]
                .dropna().astype(str).tolist()
                if "representative_structure_id" in class_df.columns
                else []
            )
            representative_id = representative_ids[0] if representative_ids else ""
            representative_rows = g.loc[
                g["structure_id"].astype(str) == representative_id,
                "canonical_signature",
            ]
            chosen = (
                str(representative_rows.iloc[0])
                if not representative_rows.empty
                else str(g.sort_values("structure_id", key=lambda s: s.map(natural_key)).iloc[0]["canonical_signature"])
            )
        elif policy == "union":
            all_edges = set()
            for sig in sigs:
                all_edges.update(x for x in sig.split(";") if x)
            chosen = edge_set_signature(all_edges)
        elif policy == "intersection":
            sets = [set(x for x in sig.split(";") if x) for sig in sigs]
            chosen = edge_set_signature(set.intersection(*sets) if sets else set())
        else:  # most_common; "representative" remains a backward-compatible alias
            # Deterministic representative: most frequent signature, then lexical.
            counts = g["canonical_signature"].astype(str).value_counts()
            chosen = sorted(counts.index, key=lambda x: (-counts[x], x))[0] if len(counts) else ""
        class_rows.append({
            "class": cls,
            "n_structures": len(g),
            "structure_ids": ";".join(sorted(g["structure_id"].astype(str), key=natural_key)),
            "n_unique_canonical_signatures_in_class": len(sigs),
            "canonical_signature_edge_count": len([x for x in chosen.split(";") if x]),
            "canonical_signature": chosen,
        })
    class_sig_df = pd.DataFrame(class_rows).sort_values("class", key=lambda s: s.map(_class_sort_key)).reset_index(drop=True)
    class_sig_out = tables_dir / f"{target_id}_step08_class_signatures.csv"
    write_csv(class_sig_out, class_sig_df)
    audit_out = tables_dir / f"{target_id}_step08_nonidentical_class_signature_audit.csv"
    write_csv(audit_out, pd.DataFrame(audit_nonidentical))

    # Merge original GED classes with identical canonical signatures, then
    # relabel the ordered merged groups consecutively as class_1 ... class_N.
    # The original Step08 labels remain available in original_classes.
    merged_rows, orig_map_rows = build_sequential_merged_class_tables(
        class_sig_df,
        class_df,
    )
    orig_map_df = pd.DataFrame(orig_map_rows).sort_values(["merged_class", "original_class"], key=lambda col: col.map(_class_sort_key)).reset_index(drop=True)
    orig_map_out = tables_dir / f"{target_id}_step08_original_to_merged_classes.csv"
    write_csv(orig_map_out, orig_map_df)
    orig_map_pub_df = orig_map_df.copy()
    if not orig_map_pub_df.empty:
        orig_map_pub_df["Original class"] = orig_map_pub_df["original_class"].map(lambda x: class_display_name(x, class_prefix))
        orig_map_pub_df["Merged class"] = orig_map_pub_df["merged_class"].map(lambda x: class_display_name(x, class_prefix))
        orig_map_pub_df = orig_map_pub_df[["Original class", "Merged class", "canonical_signature"]].rename(columns={"canonical_signature": "Canonical signature"})
    orig_map_pub_out = tables_dir / f"{target_id}_step08_original_to_merged_classes_publication.csv"
    write_csv(orig_map_pub_out, orig_map_pub_df)

    # Frequency statistics across structure-level canonical edges.
    n_total_structures = struct_df["structure_id"].nunique()
    edge_to_structs: Dict[str, Set[str]] = defaultdict(set)
    edge_parts = {}
    for _, r in can_df.iterrows():
        edge = str(r["canonical_edge"])
        edge_to_structs[edge].add(str(r["structure_id"]))
        edge_parts[edge] = (r["protein_canonical"], r["interaction_supertype"], r["canonical_ligand_unit"])
    freq_rows = []
    common_edges = set()
    rare_edges = set()
    for edge in sorted(edge_to_structs.keys()):
        structs = sorted(edge_to_structs[edge], key=natural_key)
        freq = len(structs) / n_total_structures if n_total_structures else 0.0
        category = "Common" if freq >= common_min else ("Rare" if freq < rare_max else "Other")
        if category == "Common":
            common_edges.add(edge)
        elif category == "Rare":
            rare_edges.add(edge)
        prot, itype, unit = edge_parts[edge]
        freq_rows.append({
            "canonical_edge": edge,
            "protein_canonical": prot,
            "interaction_supertype": itype,
            "canonical_ligand_unit": unit,
            "n_structures_with_edge": len(structs),
            "n_total_structures": n_total_structures,
            "frequency": freq,
            "category": category,
            "structure_ids": ";".join(structs),
        })
    freq_df = pd.DataFrame(freq_rows).sort_values(["category", "frequency", "canonical_edge"], ascending=[True, False, True]).reset_index(drop=True)
    freq_out = tables_dir / f"{target_id}_step08_canonical_interaction_frequency.csv"
    write_csv(freq_out, freq_df)
    freq_pub_df = freq_df.copy()
    if not freq_pub_df.empty:
        freq_pub_df = freq_pub_df[["canonical_edge", "frequency", "category"]].rename(
            columns={"canonical_edge": "Interaction", "frequency": "Frequency", "category": "Category"}
        )
    freq_pub_out = tables_dir / f"{target_id}_step08_canonical_interaction_frequency_publication.csv"
    write_csv(freq_pub_out, freq_pub_df)

    # Candidate status by merged class and by structure.
    summary_rows = []
    candidate_class_rows = []
    candidate_struct_rows = []
    specific_support_rows = []
    for mrow in merged_rows:
        sig_edges = set(x for x in str(mrow["canonical_signature"]).split(";") if x)
        missing_common = sorted(common_edges - sig_edges)
        rare_present = sorted(rare_edges & sig_edges)
        is_candidate = (not missing_common) and bool(rare_present)
        n_structures_in_class = int(mrow.get("n_structures", 0) or 0)
        supported_rare = list(rare_present) if n_structures_in_class >= 2 else []
        if not is_candidate:
            confidence_tier = "Not candidate"
        elif len(supported_rare) >= 2:
            confidence_tier = "Tier A"
        elif len(supported_rare) == 1:
            confidence_tier = "Tier B"
        else:
            confidence_tier = "Tier C"
        for edge in rare_present:
            specific_support_rows.append(
                {
                    "merged_class": mrow["merged_class"],
                    "canonical_edge": edge,
                    "global_frequency": next((r["frequency"] for r in freq_rows if r["canonical_edge"] == edge), None),
                    "class_support_count": n_structures_in_class,
                    "class_size": n_structures_in_class,
                    "class_support_frequency": 1.0,
                    "support_status": "supported" if n_structures_in_class >= 2 else "singleton_evidence",
                }
            )
        row = dict(mrow)
        row.update({
            "merged_class_display": class_display_name(row.get("merged_class", ""), class_prefix),
            "original_classes_display": class_display_list(row.get("original_classes", ""), class_prefix),
            "n_common_edges_required": len(common_edges),
            "n_common_edges_missing": len(missing_common),
            "missing_common_edges": ";".join(missing_common),
            "n_rare_edges_present": len(rare_present),
            "rare_edges_present": ";".join(rare_present),
            "is_catalytic_candidate": is_candidate,
        })
        if profile == GENERAL:
            row.update({
                "legacy_candidate_rule_pass": is_candidate,
                "candidate_confidence_tier": confidence_tier,
                "n_supported_rare_edges": len(supported_rare),
                "supported_rare_edges": ";".join(supported_rare),
                "candidate_rank_rare_edge_count": len(rare_present),
                "candidate_rank_class_size": n_structures_in_class,
            })
        summary_rows.append(row)
        if is_candidate:
            candidate_class_rows.append(row)
        for sid in str(mrow["structure_ids"]).split(";"):
            if not sid:
                continue
            struct_edges = set()
            tmp = struct_df[struct_df["structure_id"] == sid]
            if not tmp.empty:
                struct_edges = set(x for x in str(tmp.iloc[0]["canonical_signature"]).split(";") if x)
            struct_missing_common = sorted(common_edges - struct_edges)
            struct_rare_present = sorted(rare_edges & struct_edges)
            struct_candidate = (not struct_missing_common) and bool(struct_rare_present)
            if struct_candidate:
                candidate_struct_rows.append({
                    "structure_id": sid,
                    "merged_class": row["merged_class"],
                    "merged_class_display": class_display_name(row["merged_class"], class_prefix),
                    "original_classes": row["original_classes"],
                    "original_classes_display": class_display_list(row["original_classes"], class_prefix),
                    "rare_edges_present": ";".join(struct_rare_present),
                    "source_pdb_path": str(find_pdb_for_structure(sid, pdb_index) or ""),
                    **({"candidate_confidence_tier": confidence_tier} if profile == GENERAL else {}),
                })
    merged_summary_df = pd.DataFrame(summary_rows).sort_values("merged_class", key=lambda col: col.map(_class_sort_key)).reset_index(drop=True)
    merged_summary_out = tables_dir / f"{target_id}_step08_merged_class_summary.csv"
    write_csv(merged_summary_out, merged_summary_df)
    n_merged_classes = int(merged_summary_df["merged_class"].nunique()) if not merged_summary_df.empty else 0
    can_report["n_original_classes_before_equivalence"] = int(n_step08_classes)
    can_report["n_merged_classes_after_equivalence"] = n_merged_classes
    can_report["n_original_class_reduction"] = int(n_step08_classes) - n_merged_classes
    if n_merged_classes >= n_step08_classes:
        can_report["chemical_equivalence_merge_status"] = "no_class_merge"
        can_report.setdefault("chemical_equivalence_diagnostics", []).append(
            "No GED classes were merged. This can be valid when canonical signatures truly differ; "
            "use ligand_unit_coverage_audit.csv to distinguish it from unmatched rules."
        )
    else:
        can_report["chemical_equivalence_merge_status"] = "merged"
    merged_summary_pub_df = merged_summary_df.copy()
    if not merged_summary_pub_df.empty:
        keep_cols = [
            "merged_class_display", "original_classes_display", "n_structures",
            "structure_ids", "canonical_signature", "is_catalytic_candidate",
            "rare_edges_present", "missing_common_edges",
        ]
        merged_summary_pub_df = merged_summary_pub_df[[c for c in keep_cols if c in merged_summary_pub_df.columns]].rename(columns={
            "merged_class_display": "Merged class",
            "original_classes_display": "Original classes",
            "n_structures": "Structures",
            "structure_ids": "Structure IDs",
            "canonical_signature": "Canonical signature",
            "is_catalytic_candidate": "Catalytic candidate",
            "rare_edges_present": "Rare edges present",
            "missing_common_edges": "Missing common edges",
        })
    merged_summary_pub_out = tables_dir / f"{target_id}_step08_merged_class_summary_publication.csv"
    write_csv(merged_summary_pub_out, merged_summary_pub_df)
    cand_class_df = pd.DataFrame(candidate_class_rows)
    cand_class_out = tables_dir / f"{target_id}_step08_candidate_catalytic_classes.csv"
    write_csv(cand_class_out, cand_class_df)
    cand_class_pub_df = cand_class_df.copy()
    if not cand_class_pub_df.empty:
        keep_cols = ["merged_class_display", "original_classes_display", "n_structures", "structure_ids", "rare_edges_present"]
        cand_class_pub_df = cand_class_pub_df[[c for c in keep_cols if c in cand_class_pub_df.columns]].rename(columns={
            "merged_class_display": "Merged class",
            "original_classes_display": "Original classes",
            "n_structures": "Structures",
            "structure_ids": "Structure IDs",
            "rare_edges_present": "Rare edges present",
        })
    cand_class_pub_out = tables_dir / f"{target_id}_step08_candidate_catalytic_classes_publication.csv"
    write_csv(cand_class_pub_out, cand_class_pub_df)
    cand_struct_df = pd.DataFrame(candidate_struct_rows)
    cand_struct_out = tables_dir / f"{target_id}_step08_candidate_structures.csv"
    write_csv(cand_struct_out, cand_struct_df)
    specific_support_out = tables_dir / f"{target_id}_step08_specific_edge_support.csv"
    write_csv(specific_support_out, pd.DataFrame(specific_support_rows))

    # Optional candidate export: copy/symlink PLIP interaction outputs, GED graphs, and PDB complexes.
    export_cfg = step09.get("export_candidates", {}) or {}
    exported_pdb_rows = []
    if read_bool(export_cfg.get("enabled"), False):
        export_dir = outdir / str(export_cfg.get("output_subdir", "candidate_exports"))
        mode = str(export_cfg.get("mode", "copy")).lower()
        for _, row in cand_struct_df.iterrows():
            sid = str(row["structure_id"])
            mclass = str(row["merged_class"])
            ipath = find_interaction_file(sid, plip_index)
            if ipath and ipath.exists():
                export_file(ipath, export_dir / "interaction_outputs" / mclass / f"{sid}_interaction_output.txt", mode)
            pdb_path = find_pdb_for_structure(sid, pdb_index)
            if pdb_path and pdb_path.exists():
                dst = export_dir / "pdbs" / mclass / f"{sid}.pdb"
                export_file(pdb_path, dst, mode)
                exported_pdb_rows.append({"structure_id": sid, "merged_class": mclass, "source_pdb_path": str(pdb_path), "exported_pdb_path": str(dst), "mode": mode})
            else:
                exported_pdb_rows.append({"structure_id": sid, "merged_class": mclass, "source_pdb_path": "", "exported_pdb_path": "", "mode": mode, "error": "missing_source_pdb"})
    exported_pdb_out = tables_dir / f"{target_id}_step08_exported_candidate_pdbs.csv"
    write_csv(exported_pdb_out, pd.DataFrame(exported_pdb_rows))

    report = {
        "target_id": target_id,
        "status": "completed",
        "config": str(config_path),
        "output_dir": str(outdir),
        "input_step08_class_table": str(table_path) if table_path else None,
        "input_step08_class_dir": str(class_dir) if class_dir else None,
        "input_step06_dir": str(step06_dir),
        "n_structures_in_step08_classes": int(class_df["structure_id"].nunique()),
        "n_structures_with_interactions": int(n_total_structures),
        "n_original_classes": n_step08_classes,
        "n_effective_classes_after_chemical_split": int(class_df["class"].nunique()),
        "n_merged_classes": n_merged_classes,
        "merged_class_numbering": "sequential_after_chemical_equivalence",
        "n_raw_interactions": int(len(raw_df)),
        "n_canonical_edges": int(len(edge_to_structs)),
        "n_common_edges": int(len(common_edges)),
        "n_rare_edges": int(len(rare_edges)),
        "n_candidate_merged_classes": int(len(cand_class_df)),
        "n_candidate_structures": int(len(cand_struct_df)),
        "n_step07_passed_structures_for_interaction_frequency": int(len(step07_passed_structures)),
        "n_step07_passed_structures_missing_interaction_output": int(len(step07_missing_rows)),
        "n_step07_passed_canonical_interactions": int(len(step07_frequency_df)),
        "n_all_collected_interactions_in_frequency_table": int(len(all_frequency_df)),
        "all_collected_interaction_frequency_population_source": all_frequency_population_source,
        "n_all_collected_interaction_frequency_population_structures": int(
            len(
                {
                    normalize_structure_id(value)
                    for value in all_frequency_population_ids
                    if normalize_structure_id(value)
                }
            )
        ),
        "n_missing_interaction_outputs": int(len(missing_df)),
        "n_exported_candidate_pdbs": int(len([r for r in exported_pdb_rows if r.get("exported_pdb_path")])),
        "canonicalization": {
            "workflow_profile": profile,
            "split_nonidentical_classes": split_enabled,
            "rule_set": (
                "Supplementary_Table_S4_universal_contact_units"
                if contact_unit_mode(cfg) == CONTACT_UNIT_UNIVERSAL
                else ("general_chemical_equivalence" if profile == GENERAL
                else "standard_chemical_equivalence")
            ) if mode == "standard" else "chemistry_evidence_review",
            "class_signature_policy": policy,
            "fail_if_class_members_nonidentical": fail_nonidentical,
            "common_frequency_min": common_min,
            "rare_frequency_max": rare_max,
            **can_report,
        },
        "tables": {
            "raw_interactions": str(raw_out),
            "substrate_inventory": str(substrate_inventory_out),
            "ligand_unit_recommendations": str(recommendations_out),
            "raw_to_canonical_mapping": str(mapping_out),
            "step07_passed_interaction_frequency": str(step07_frequency_out),
            "step07_passed_interaction_frequency_publication": str(step07_frequency_pub_out),
            "step07_passed_missing_interaction_outputs": str(step07_frequency_missing_out),
            "all_collected_interactions_frequency": str(all_frequency_out),
            "all_collected_interactions_frequency_publication": str(all_frequency_pub_out),
            "auto_ligand_unit_audit": str(auto_audit_out),
            "ligand_unit_coverage_audit": str(unit_coverage_out),
            "plip_to_classes": str(plip_classes_out),
            "structure_canonical_edges": str(struct_out),
            "class_signatures": str(class_sig_out),
            "nonidentical_class_signature_audit": str(audit_out),
            "class_split_audit": str(split_audit_out),
            "original_to_merged_classes": str(orig_map_out),
            "original_to_merged_classes_publication": str(orig_map_pub_out),
            "merged_class_summary": str(merged_summary_out),
            "merged_class_summary_publication": str(merged_summary_pub_out),
            "canonical_interaction_frequency": str(freq_out),
            "canonical_interaction_frequency_publication": str(freq_pub_out),
            "candidate_catalytic_classes": str(cand_class_out),
            "candidate_catalytic_classes_publication": str(cand_class_pub_out),
            "candidate_structures": str(cand_struct_out),
            "specific_edge_support": str(specific_support_out),
            "missing_interaction_outputs": str(missing_out),
            "exported_candidate_pdbs": str(exported_pdb_out),
        },
    }
    report_out = reports_dir / f"{target_id}_step08_report.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md = reports_dir / f"{target_id}_step08_summary.md"
    summary_md.write_text(
        "\n".join([
            f"# Step08 ligand contact-unit grouping summary: {target_id}",
            "",
            f"- Original GED classes: {report['n_original_classes']}",
            f"- Merged classes: {report['n_merged_classes']}",
            f"- Original-class reduction: {can_report['n_original_class_reduction']}",
            "- Merged-class numbering: sequential class_1 ... class_N after contact-unit grouping",
            f"- Structures with interactions: {report['n_structures_with_interactions']}",
            f"- Common canonical edges: {report['n_common_edges']}",
            f"- Rare canonical edges: {report['n_rare_edges']}",
            f"- Candidate merged classes: {report['n_candidate_merged_classes']}",
            f"- Candidate structures: {report['n_candidate_structures']}",
            f"- Step07-passed structures in interaction-frequency denominator: {report['n_step07_passed_structures_for_interaction_frequency']}",
            f"- Step07-passed structures missing PLIP output: {report['n_step07_passed_structures_missing_interaction_output']}",
            f"- All collected interactions in frequency table: {report['n_all_collected_interactions_in_frequency_table']}",
            f"- Interaction-frequency population source: {report['all_collected_interaction_frequency_population_source']}",
            f"- Exported candidate PDBs: {report['n_exported_candidate_pdbs']}",
            "",
            "The tables directory includes a complete Interaction / Frequency / Category table, raw-to-canonical mapping, class merging, candidate structures, and exported PDB audit tables.",
            "",
        ]),
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AF2-CatGraph Step08: ligand contact-unit grouping and common/rare screening")
    ap.add_argument("--config", required=True, help="Project YAML config")
    ap.add_argument("--clean", action="store_true", help="Remove previous Step09 output before running")
    args = ap.parse_args(argv)
    report = run_step09(args.config, clean_output=args.clean)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

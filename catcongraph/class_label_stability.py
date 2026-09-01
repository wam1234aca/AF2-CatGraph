"""Stable numeric class labels for reproducible workflows."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def class_number(value: object) -> int:
    numbers = re.findall(r"(\d+)", str(value))
    return int(numbers[-1]) if numbers else 10**12


def class_sort_key(value: object) -> tuple:
    return (class_number(value), str(value))


def offset_last_integer(value: object, offset: int) -> str:
    text = str(value or "").strip()
    if not text or not offset:
        return text
    matches = list(re.finditer(r"\d+", text))
    if not matches:
        return text
    match = matches[-1]
    raw = match.group(0)
    shifted = int(raw) + int(offset)
    width = len(raw) if raw.startswith("0") and len(raw) > 1 else 0
    replacement = f"{shifted:0{width}d}" if width else str(shifted)
    return text[: match.start()] + replacement + text[match.end() :]


def _choose_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(name).strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def read_reference_manifest(
    path: str | Path,
    *,
    structure_id_offset: int = 0,
) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stable-class reference manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        class_col = _choose_column(
            fieldnames, ["reference_class", "stable_class", "class"]
        )
        structure_col = _choose_column(
            fieldnames, ["structure_id", "normalized_structure_id", "graph_id"]
        )
        representative_col = _choose_column(
            fieldnames,
            [
                "reference_representative_structure_id",
                "representative_structure_id",
                "representative_id",
            ],
        )
        if class_col is None or structure_col is None:
            raise ValueError(
                "Reference manifest must contain a class column "
                "(reference_class/stable_class/class) and a structure ID column "
                "(structure_id/normalized_structure_id/graph_id)."
            )
        rows = []
        for row in reader:
            cls = str(row.get(class_col, "")).strip()
            sid = offset_last_integer(row.get(structure_col, ""), structure_id_offset)
            rep = (
                offset_last_integer(row.get(representative_col, ""), structure_id_offset)
                if representative_col
                else ""
            )
            if cls and sid:
                rows.append(
                    {
                        "reference_class": cls,
                        "structure_id": sid,
                        "reference_representative_structure_id": rep,
                    }
                )
    if not rows:
        raise ValueError(f"Reference manifest contains no usable rows: {path}")
    return rows


def build_reference_registry(class_rows: Iterable[dict]) -> List[dict]:
    return [
        {
            "reference_class": str(row.get("class", "")),
            "structure_id": str(row.get("structure_id", "")),
            "reference_representative_structure_id": str(
                row.get("representative_structure_id", "")
            ),
            "class_size": row.get("class_size", ""),
        }
        for row in class_rows
    ]


def _group_current(class_rows: Iterable[dict]) -> Tuple[Dict[str, set], Dict[str, str]]:
    members: Dict[str, set] = defaultdict(set)
    representatives: Dict[str, str] = {}
    for row in class_rows:
        cls = str(row.get("class", "")).strip()
        sid = str(row.get("structure_id", "")).strip()
        rep = str(row.get("representative_structure_id", "")).strip()
        if cls and sid:
            members[cls].add(sid)
        if cls and rep and cls not in representatives:
            representatives[cls] = rep
    return dict(members), representatives


def _group_reference(reference_rows: Iterable[dict]) -> Tuple[Dict[str, set], Dict[str, str]]:
    members: Dict[str, set] = defaultdict(set)
    representatives: Dict[str, str] = {}
    for row in reference_rows:
        cls = str(row.get("reference_class", row.get("class", ""))).strip()
        sid = str(row.get("structure_id", "")).strip()
        rep = str(
            row.get(
                "reference_representative_structure_id",
                row.get("representative_structure_id", ""),
            )
        ).strip()
        if cls and sid:
            members[cls].add(sid)
        if cls and rep and cls not in representatives:
            representatives[cls] = rep
    return dict(members), representatives


def _format_new_class(index: int, class_naming: Optional[dict]) -> str:
    cfg = class_naming or {}
    prefix = str(cfg.get("prefix", "class"))
    separator = str(cfg.get("separator", "_"))
    width = int(cfg.get("width", cfg.get("zero_pad", 0)) or 0)
    number = f"{index:0{width}d}" if width else str(index)
    return f"{prefix}{separator}{number}"


def match_stable_class_labels(
    class_rows: List[dict],
    reference_rows: List[dict],
    *,
    minimum_jaccard: float = 0.25,
    class_naming: Optional[dict] = None,
) -> Tuple[Dict[str, str], List[dict]]:
    current_members, current_reps = _group_current(class_rows)
    reference_members, reference_reps = _group_reference(reference_rows)
    if not current_members:
        return {}, []
    if not reference_members:
        raise ValueError("Stable class labeling is enabled, but reference is empty.")

    candidates = []
    for current_class, current_set in current_members.items():
        for reference_class, reference_set in reference_members.items():
            intersection = len(current_set & reference_set)
            if intersection == 0:
                continue
            union = len(current_set | reference_set)
            jaccard = intersection / union if union else 0.0
            exact = current_set == reference_set
            reference_rep = reference_reps.get(reference_class, "")
            representative_retained = bool(reference_rep and reference_rep in current_set)
            candidates.append(
                {
                    "current_class": current_class,
                    "reference_class": reference_class,
                    "exact_member_set": exact,
                    "reference_representative_retained": representative_retained,
                    "intersection_size": intersection,
                    "jaccard": jaccard,
                    "current_size": len(current_set),
                    "reference_size": len(reference_set),
                    "size_difference": abs(len(current_set) - len(reference_set)),
                    "reference_representative_structure_id": reference_rep,
                    "current_representative_structure_id": current_reps.get(current_class, ""),
                }
            )

    candidates.sort(
        key=lambda row: (
            -int(row["exact_member_set"]),
            -int(row["reference_representative_retained"]),
            -float(row["jaccard"]),
            -int(row["intersection_size"]),
            int(row["size_difference"]),
            class_sort_key(row["reference_class"]),
            class_sort_key(row["current_class"]),
        )
    )

    mapping: Dict[str, str] = {}
    audit_by_current: Dict[str, dict] = {}
    used_references = set()
    for candidate in candidates:
        cur = candidate["current_class"]
        ref = candidate["reference_class"]
        if cur in mapping or ref in used_references:
            continue
        acceptable = (
            candidate["exact_member_set"]
            or candidate["reference_representative_retained"]
            or float(candidate["jaccard"]) >= float(minimum_jaccard)
        )
        if not acceptable:
            continue
        mapping[cur] = ref
        used_references.add(ref)
        row = dict(candidate)
        row["match_type"] = (
            "exact_member_set"
            if candidate["exact_member_set"]
            else (
                "reference_representative_retained"
                if candidate["reference_representative_retained"]
                else "member_overlap"
            )
        )
        row["assigned_class"] = ref
        row["label_source"] = "reference"
        audit_by_current[cur] = row

    next_number = max(class_number(value) for value in reference_members) + 1
    occupied = set(reference_members)
    for cur in sorted(current_members, key=class_sort_key):
        if cur in mapping:
            continue
        while True:
            label = _format_new_class(next_number, class_naming)
            next_number += 1
            if label not in occupied:
                break
        occupied.add(label)
        mapping[cur] = label
        audit_by_current[cur] = {
            "current_class": cur,
            "reference_class": "",
            "assigned_class": label,
            "match_type": "new_class_appended",
            "label_source": "new",
            "exact_member_set": False,
            "reference_representative_retained": False,
            "intersection_size": 0,
            "jaccard": 0.0,
            "current_size": len(current_members[cur]),
            "reference_size": 0,
            "size_difference": len(current_members[cur]),
            "current_representative_structure_id": current_reps.get(cur, ""),
            "reference_representative_structure_id": "",
        }

    audit_rows = [
        audit_by_current[cur] for cur in sorted(current_members, key=class_sort_key)
    ]
    matched_refs = set(mapping.values()) & set(reference_members)
    for ref in sorted(reference_members, key=class_sort_key):
        if ref not in matched_refs:
            audit_rows.append(
                {
                    "current_class": "",
                    "reference_class": ref,
                    "assigned_class": "",
                    "match_type": "reference_class_not_present",
                    "label_source": "retired_or_absent",
                    "exact_member_set": False,
                    "reference_representative_retained": False,
                    "intersection_size": 0,
                    "jaccard": 0.0,
                    "current_size": 0,
                    "reference_size": len(reference_members[ref]),
                    "size_difference": len(reference_members[ref]),
                    "current_representative_structure_id": "",
                    "reference_representative_structure_id": reference_reps.get(ref, ""),
                }
            )
    return mapping, audit_rows


def apply_mapping_to_class_rows(class_rows: List[dict], mapping: Dict[str, str]) -> List[dict]:
    updated = []
    for row in class_rows:
        row = dict(row)
        provisional = str(row.get("class", ""))
        row["provisional_class"] = provisional
        row["class"] = mapping.get(provisional, provisional)
        updated.append(row)
    return updated


def rename_class_directories(class_dir: str | Path, mapping: Dict[str, str]) -> None:
    class_dir = Path(class_dir)
    staged = []
    for index, old_label in enumerate(sorted(mapping, key=class_sort_key), start=1):
        source = class_dir / old_label
        if not source.exists() or not source.is_dir():
            continue
        temp = class_dir / f".__stable_label_tmp_{index:06d}"
        if temp.exists():
            shutil.rmtree(temp)
        source.rename(temp)
        staged.append((temp, mapping[old_label]))
    for temp, new_label in staged:
        destination = class_dir / new_label
        if destination.exists():
            raise FileExistsError(f"Stable label destination already exists: {destination}")
        temp.rename(destination)


def rewrite_pairwise_tsv(path: str | Path, mapping: Dict[str, str]) -> None:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    for row in rows:
        old = str(row.get("class", ""))
        if old in mapping:
            row["class"] = mapping[old]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def rewrite_results_text(path: str | Path, mapping: Dict[str, str]) -> None:
    path = Path(path)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for old in sorted(mapping, key=lambda value: (-len(value), value)):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
            mapping[old],
            text,
        )
    path.write_text(text, encoding="utf-8")

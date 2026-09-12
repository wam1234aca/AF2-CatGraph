#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step08: GED-based conformation classification.

This module integrates the user's previous GED.py workflow into AF2-CatGraph while
keeping the core classification logic unchanged:
  - GED command: ged -d graph_g -q graph_q -m pair -p astar -l LSa -g
  - edge-count prefilter: unequal edge counts return infinity
  - representatives are processed in descending edge-count order
  - a graph joins the current class when GED < threshold
  - timeout protection and multithreaded comparisons are retained
  - optional memory guards limit GED subprocess memory and stream pairwise output

The main framework change is safety/reproducibility: Step08 first copies the
Step06 GED input files that passed Step07 into a working directory, then it moves
only those copied files during classification. Original Step06 files are never
modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from catcongraph.class_label_stability import (
    apply_mapping_to_class_rows,
    build_reference_registry,
    class_sort_key,
    match_stable_class_labels,
    read_reference_manifest,
    rename_class_directories,
    rewrite_pairwise_tsv,
    rewrite_results_text,
)
from catcongraph.config import load_yaml_config, public_output_subdir

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


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
    # In AF2-CatGraph examples config lives at projects/<target>/config.yaml.
    if config_path.name in {"config.yaml", "config.yml"} and config_path.parent.parent.name == "projects":
        return config_path.parent.parent.parent.resolve()
    return Path.cwd().resolve()


def resolve_path(value: Optional[str], root: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return None
    p = Path(text).expanduser()
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _listify(value) -> List[str]:
    """Return a YAML scalar/list value as a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _candidate_is_executable(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def resolve_ged_executable(
    executable_cfg: object,
    root: Path,
    executable_cwd: Optional[Path] = None,
    auto_search_paths: Optional[object] = None,
) -> str:
    """Resolve and validate the external LijunChang GED executable before classification.

    This intentionally happens before any classification starts. If the executable
    is missing, Step08 must fail fast instead of silently assigning every graph
    to singleton classes.
    """
    text = str(executable_cfg or "").strip() or "ged"
    raw_path = Path(text).expanduser()
    candidates: List[Path] = []

    if executable_cwd is not None:
        executable_cwd = executable_cwd.expanduser().resolve()
        if not executable_cwd.exists():
            raise FileNotFoundError(f"step08.ged.working_directory does not exist: {executable_cwd}")
        if not executable_cwd.is_dir():
            raise NotADirectoryError(f"step08.ged.working_directory is not a directory: {executable_cwd}")

    # 1) If user gives a command name such as "ged", check PATH first.
    if not raw_path.is_absolute() and raw_path.parent == Path("."):
        found = shutil.which(text)
        if found:
            found_path = Path(found).resolve()
            if _candidate_is_executable(found_path):
                return str(found_path)

    # 2) Check the path exactly as configured, plus AF2-CatGraph-root and cwd-relative variants.
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if executable_cwd is not None:
            candidates.append((executable_cwd / raw_path).resolve())
        candidates.append((root / raw_path).resolve())
        candidates.append((Path.cwd() / raw_path).resolve())

    # 3) Optional search roots make server-specific layouts easy without hard-coding.
    for item in _listify(auto_search_paths):
        base = resolve_path(item, root)
        if base is None:
            continue
        if base.is_file():
            candidates.append(base)
        else:
            # Common layouts for the LijunChang Graph_Edit_Distance repository.
            candidates.extend(
                [
                    base / text,
                    base / "ged",
                    base / "Graph_Edit_Distance" / "ged",
                    base / "build" / "ged",
                    base / "Graph_Edit_Distance" / "build" / "ged",
                ]
            )

    # Deduplicate while preserving order.
    seen = set()
    unique_candidates = []
    for c in candidates:
        c = c.expanduser().resolve()
        if str(c) not in seen:
            unique_candidates.append(c)
            seen.add(str(c))

    existing_non_executable = []
    for candidate in unique_candidates:
        if _candidate_is_executable(candidate):
            return str(candidate)
        if candidate.exists() and candidate.is_file():
            existing_non_executable.append(candidate)

    if existing_non_executable:
        raise PermissionError(
            "GED executable file exists but is not executable: "
            + "; ".join(str(p) for p in existing_non_executable)
            + ". Run: chmod +x <ged_path>"
        )

    searched = "\n  - ".join(str(p) for p in unique_candidates) or "(no candidate paths)"
    raise FileNotFoundError(
        "GED executable not found before Step08 classification.\n"
        f"Configured step08.ged.executable: {text}\n"
        "Searched:\n"
        f"  - {searched}\n"
        "Fix: set step08.ged.executable to the compiled LijunChang Graph_Edit_Distance binary, "
        "or add its folder to step08.ged.auto_search_paths. Do not use a placeholder path."
    )


def get_target_id(config: dict, config_path: Path) -> str:
    for key in ["target_id", "target", "name", "project_id"]:
        if key in config and config[key]:
            return str(config[key])
    project = config.get("project", {}) or {}
    for key in ["target_id", "target", "name", "project_id"]:
        if key in project and project[key]:
            return str(project[key])
    return config_path.parent.name


def get_output_root(config: dict, root: Path, target_id: str) -> Path:
    project = config.get("project", {}) or {}
    results = config.get("results", {}) or {}
    output = config.get("output", {}) or {}
    if not isinstance(project, dict):
        project = {}
    if not isinstance(results, dict):
        results = {}
    if not isinstance(output, dict):
        output = {}
    value = (
        project.get("output_root")
        or project.get("output_dir")
        or project.get("results_dir")
        or results.get("target_dir")
        or results.get("output_root")
        or results.get("output_dir")
        or output.get("target_dir")
        or output.get("output_dir")
        or config.get("output_root")
        or config.get("output_dir")
    )
    if value:
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = root / p
        # If config uses results/<target>, keep it. If it uses results, add target.
        if p.name != target_id and not (
            (p / "05_plip_interaction_graphs").exists() or (p / "06_plip_interaction_graphs").exists()
        ):
            maybe = p / target_id
            if maybe.exists() or p.name == "results":
                p = maybe
        return p.resolve()
    return (root / "results" / target_id).resolve()


def normalize_structure_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    stem = Path(text).stem
    suffixes = [
        "_final",
        "_trimmed_complex",
        "_filtered_complex",
        "_passed_complex",
        "_complex_complex_complex",
        "_complex_complex",
        "_complex",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def _natural_key(value: object) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def _last_integer(value: object) -> int:
    nums = re.findall(r"(\d+)", str(value))
    return int(nums[-1]) if nums else 10**12


def graph_order_key(path: Path, edge_counter) -> tuple:
    sid = normalize_structure_id(path.name)
    return (-edge_counter(path), _last_integer(sid), _natural_key(sid), path.name)


def format_class_name(index: int, class_naming: Optional[dict] = None) -> str:
    cfg = class_naming or {}
    prefix = str(cfg.get("prefix", "class"))
    separator = str(cfg.get("separator", "_"))
    width = int(cfg.get("width", cfg.get("zero_pad", 0)) or 0)
    num = f"{index:0{width}d}" if width > 0 else str(index)
    return f"{prefix}{separator}{num}"


def graph_id_from_path(path: Path) -> str:
    return normalize_structure_id(path.stem)


def read_csv_table(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_step06_graph_dir(output_root: Path, target_id: str, step_cfg: dict, root: Path) -> Path:
    explicit = step_cfg.get("input_graph_dir") or step_cfg.get("ged_input_dir")
    if explicit and str(explicit).lower() != "auto":
        p = resolve_path(str(explicit), root)
        if p and p.exists():
            return p
        raise FileNotFoundError(f"Step08 input graph directory not found: {explicit}")
    candidates = [
        output_root / "05_plip_interaction_graphs" / "ged_input_files",
        output_root / "05_plip_interaction_graphs" / "ged_inputs",
        output_root / "05_plip_interaction_graphs",
        # Historical output locations.
        output_root / "06_plip_interaction_graphs" / "ged_input_files",
        output_root / "06_plip_interaction_graphs" / "ged_inputs",
        output_root / "06_plip_interaction_graphs",
    ]
    for p in candidates:
        if p.exists() and list(p.glob("*_final.txt")):
            return p
    raise FileNotFoundError(
        "Could not find public Step05 GED input files. Expected results/<target>/05_plip_interaction_graphs/ged_input_files/*_final.txt"
    )


def find_step07_passed_table(output_root: Path, target_id: str, step_cfg: dict, root: Path) -> Path:
    explicit = step_cfg.get("step07_passed_table") or step_cfg.get("input_passed_table")
    if explicit and str(explicit).lower() != "auto":
        p = resolve_path(str(explicit), root)
        if p and p.exists():
            return p
        raise FileNotFoundError(f"Step07 passed table not found: {explicit}")
    candidates = [
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step06_passed_complex_index.csv",
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step06_structure_key_residue_filter.csv",
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step07_passed_complex_index.csv",
        output_root / "06_key_residue_screening" / "tables" / f"{target_id}_step07_structure_key_residue_filter.csv",
        output_root / "07_key_residue_screening" / "tables" / f"{target_id}_step07_passed_complex_index.csv",
        output_root / "07_key_residue_screening" / "tables" / f"{target_id}_step07_structure_key_residue_filter.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Step08 is configured to use Step07 filtering, but Step07 passed table was not found. "
        "Run Step07 first or set step08.use_step07_filter: false."
    )


def ids_from_step07_table(table_path: Path) -> Tuple[set, List[dict]]:
    rows = read_csv_table(table_path)
    passed_rows = []
    ids = set()
    for row in rows:
        pass_value = str(row.get("pass_key_residue_filter", "true")).strip().lower()
        if pass_value in {"false", "0", "no", "fail", "failed"}:
            continue
        passed_rows.append(row)
        for key in ["structure_id", "normalized_structure_id", "pdb_path", "complex_path", "trimmed_complex_path", "passed_complex_path"]:
            if row.get(key):
                ids.add(normalize_structure_id(row[key]))
                ids.add(Path(str(row[key])).stem)
    ids = {x for x in ids if x}
    return ids, passed_rows


def collect_graph_files(graph_dir: Path, allowed_ids: Optional[set]) -> Tuple[List[Path], List[dict]]:
    files = sorted({p.resolve() for p in graph_dir.rglob("*_final.txt") if p.is_file()})
    selected = []
    skipped = []
    for p in files:
        gid = graph_id_from_path(p)
        if allowed_ids is None or gid in allowed_ids or p.stem in allowed_ids:
            selected.append(p)
        else:
            skipped.append({"graph_file": str(p), "graph_id": gid, "reason": "not_passed_step07"})
    return selected, skipped


def count_edges(file_path: Path, cache: dict, lock: threading.Lock) -> int:
    with lock:
        if file_path in cache:
            return cache[file_path]
    count = 0
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("e "):
                count += 1
    with lock:
        cache[file_path] = count
    return count


def build_memory_limiter(memory_limit_mb: Optional[int]):
    """Return a POSIX preexec_fn that caps each GED subprocess memory.

    This uses RLIMIT_AS on Linux/Unix. If memory_limit_mb is None or <= 0, no
    subprocess memory limit is applied. When the external GED binary exceeds the
    limit it will usually terminate and the pair is treated as GED=inf.
    """
    if memory_limit_mb is None:
        return None
    try:
        limit_mb = int(memory_limit_mb)
    except Exception:
        return None
    if limit_mb <= 0:
        return None

    def _limit_memory():
        try:
            import resource
            limit_bytes = limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            # Avoid failing before exec on systems where resource is unavailable.
            pass

    return _limit_memory


def append_tsv_row(path: Path, row: dict, fieldnames: Sequence[str]) -> None:
    """Append one row to a TSV file, creating the header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def calculate_ged(
    graph_g: Path,
    graph_q: Path,
    executable: str,
    method: str,
    algorithm: str,
    lower_bound: str,
    timeout_seconds: int,
    logger: logging.Logger,
    cwd: Optional[Path] = None,
    memory_limit_mb: Optional[int] = None,
) -> float:
    command = [
        executable,
        "-d",
        str(graph_g),
        "-q",
        str(graph_q),
        "-m",
        method,
        "-p",
        algorithm,
        "-l",
        lower_bound,
        "-g",
    ]
    preexec_fn = build_memory_limiter(memory_limit_mb)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            cwd=str(cwd) if cwd else None,
            preexec_fn=preexec_fn,
        )
    except subprocess.TimeoutExpired:
        logger.error("GED calculation timeout: %s vs %s", graph_g, graph_q)
        return math.inf
    except FileNotFoundError:
        raise FileNotFoundError(
            f"GED executable not found: {executable}. Set step08.ged.executable to the compiled LijunChang Graph_Edit_Distance executable."
        )
    output = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    logger.info("Command: %s", " ".join(command))
    if output:
        logger.info("STDOUT for %s vs %s:\n%s", graph_g.name, graph_q.name, output)
    if stderr:
        logger.info("STDERR for %s vs %s:\n%s", graph_g.name, graph_q.name, stderr)
    if result.returncode != 0:
        logger.warning(
            "GED command returned non-zero exit code %s for %s vs %s. This may indicate timeout-like failure, bad input, or memory-limit termination.",
            result.returncode,
            graph_g.name,
            graph_q.name,
        )
    for line in output.splitlines():
        if "min_ged" in line:
            m = re.search(r"min_ged:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return math.inf


class GEDClassifier:
    def __init__(
        self,
        executable: str,
        method: str = "pair",
        algorithm: str = "astar",
        lower_bound: str = "LSa",
        threshold: float = 1,
        timeout_seconds: int = 300,
        max_workers: int = 6,
        require_same_edge_count: bool = True,
        logger: Optional[logging.Logger] = None,
        executable_cwd: Optional[Path] = None,
        memory_limit_mb: Optional[int] = None,
        cache_ged: bool = False,
        pair_batch_size: Optional[int] = None,
        class_naming: Optional[dict] = None,
    ) -> None:
        self.executable = executable
        self.method = method
        self.algorithm = algorithm
        self.lower_bound = lower_bound
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self.require_same_edge_count = require_same_edge_count
        self.logger = logger or logging.getLogger(__name__)
        self.executable_cwd = executable_cwd
        self.memory_limit_mb = memory_limit_mb
        self.cache_ged = cache_ged
        self.pair_batch_size = pair_batch_size
        self.class_naming = class_naming or {}
        self.ged_cache: Dict[Tuple[str, str], float] = {}
        self.ged_cache_lock = threading.Lock()
        self.edge_count_cache: Dict[Path, int] = {}
        self.edge_count_lock = threading.Lock()

    def edge_count(self, path: Path) -> int:
        return count_edges(path, self.edge_count_cache, self.edge_count_lock)

    def get_ged(self, graph_g: Path, graph_q: Path) -> float:
        if self.require_same_edge_count and self.edge_count(graph_g) != self.edge_count(graph_q):
            return math.inf
        key = tuple(sorted([str(graph_g), str(graph_q)]))
        if self.cache_ged:
            with self.ged_cache_lock:
                if key in self.ged_cache:
                    return self.ged_cache[key]
        value = calculate_ged(
            graph_g=graph_g,
            graph_q=graph_q,
            executable=self.executable,
            method=self.method,
            algorithm=self.algorithm,
            lower_bound=self.lower_bound,
            timeout_seconds=self.timeout_seconds,
            logger=self.logger,
            cwd=self.executable_cwd,
            memory_limit_mb=self.memory_limit_mb,
        )
        if self.cache_ged:
            with self.ged_cache_lock:
                self.ged_cache[key] = value
        return value

    def classify(self, base_folder: Path, output_folder: Path, pairwise_output_path: Optional[Path] = None, pairwise_fieldnames: Optional[Sequence[str]] = None) -> Tuple[List[dict], List[dict]]:
        remaining_files = [p for p in base_folder.iterdir() if p.is_file() and p.name.endswith("_final.txt")]
        # Deterministic order: highest edge count first, then AF2/source number.
        # This prevents class labels from changing across filesystems when edge-count ties occur.
        remaining_files.sort(key=lambda p: graph_order_key(p, self.edge_count))
        output_folder.mkdir(parents=True, exist_ok=True)
        class_rows: List[dict] = []
        pair_rows: List[dict] = []
        if pairwise_fieldnames is None:
            pairwise_fieldnames = [
        "representative_graph", "query_graph", "representative_id", "query_id",
        "representative_edge_count", "query_edge_count", "same_edge_count",
        "ged", "threshold", "assigned_to_current_class", "class",
    ]
        if pairwise_output_path is not None and pairwise_output_path.exists():
            pairwise_output_path.unlink()

        results_file = output_folder / "ged_classification_results.txt"
        class_counter = 1
        with results_file.open("w", encoding="utf-8") as result_output:
            while len(remaining_files) > 1:
                current_path = remaining_files[0]
                current_name = current_path.name
                class_name = format_class_name(class_counter, self.class_naming)
                class_folder = output_folder / class_name
                class_folder.mkdir(parents=True, exist_ok=True)
                result_output.write(f"#{class_name} start\n")
                self.logger.info("Starting classification for %s", current_name)

                current_class_names: List[str] = []
                compare_files = list(remaining_files[1:])
                batch_size = self.pair_batch_size or max(self.max_workers * 4, self.max_workers)
                batch_size = max(1, int(batch_size))
                for batch_start in range(0, len(compare_files), batch_size):
                    batch = [p for p in compare_files[batch_start: batch_start + batch_size] if p.name not in set(current_class_names)]
                    if not batch:
                        continue
                    tasks = {}
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        for other_path in batch:
                            future = executor.submit(self.get_ged, current_path, other_path)
                            tasks[future] = other_path
                        for future in as_completed(tasks):
                            other_path = tasks[future]
                            try:
                                ged_value = future.result()
                            except (FileNotFoundError, PermissionError, NotADirectoryError):
                                self.logger.exception("Fatal GED executable/setup error for %s vs %s", current_path, other_path)
                                raise
                            except Exception:
                                self.logger.exception("GED failed for %s vs %s", current_path, other_path)
                                ged_value = math.inf
                            same_edge_count = self.edge_count(current_path) == self.edge_count(other_path)
                            assigned = bool(ged_value < self.threshold)
                            pair_row = {
                                "representative_graph": current_name,
                                "query_graph": other_path.name,
                                "representative_id": graph_id_from_path(current_path),
                                "query_id": graph_id_from_path(other_path),
                                "representative_edge_count": self.edge_count(current_path),
                                "query_edge_count": self.edge_count(other_path),
                                "same_edge_count": same_edge_count,
                                "ged": "inf" if math.isinf(ged_value) else ged_value,
                                "threshold": self.threshold,
                                "assigned_to_current_class": assigned,
                                "class": class_name if assigned else "",
                            }
                            if pairwise_output_path is not None:
                                append_tsv_row(pairwise_output_path, pair_row, pairwise_fieldnames)
                            else:
                                pair_rows.append(pair_row)
                            if assigned and other_path.exists():
                                current_class_names.append(other_path.name)
                                shutil.move(str(other_path), str(class_folder / other_path.name))
                            result_output.write(f"{current_name},{other_path.name},max_ged: {'inf' if math.isinf(ged_value) else ged_value}\n")
                            result_output.flush()

                shutil.move(str(current_path), str(class_folder / current_name))
                class_members = [current_name] + current_class_names
                for member in class_members:
                    class_rows.append(
                        {
                            "class": class_name,
                            "graph_file": member,
                            "structure_id": normalize_structure_id(member),
                            "representative_graph": current_name,
                            "representative_structure_id": normalize_structure_id(current_name),
                            "class_size": len(class_members),
                        }
                    )
                result_output.write(f"#{class_name} complete\n\n")
                self.logger.info("%s has %d graphs", class_name, len(class_members))

                remaining_files = [p for p in remaining_files if p.name not in set(class_members)]
                remaining_files.sort(key=lambda p: graph_order_key(p, self.edge_count))
                class_counter += 1

            if len(remaining_files) == 1:
                last_path = remaining_files[0]
                class_name = format_class_name(class_counter, self.class_naming)
                class_folder = output_folder / class_name
                class_folder.mkdir(parents=True, exist_ok=True)
                shutil.move(str(last_path), str(class_folder / last_path.name))
                class_rows.append(
                    {
                        "class": class_name,
                        "graph_file": last_path.name,
                        "structure_id": normalize_structure_id(last_path.name),
                        "representative_graph": last_path.name,
                        "representative_structure_id": normalize_structure_id(last_path.name),
                        "class_size": 1,
                    }
                )
                result_output.write(f"#{class_name} start\n")
                result_output.write(f"{last_path.name} only protein in {class_name}\n")
                result_output.write(f"#{class_name} complete\n\n")
        return class_rows, pair_rows


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("catcongraph.step08")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def copy_graphs_to_workdir(graph_files: List[Path], work_graph_dir: Path) -> List[Path]:
    if work_graph_dir.exists():
        shutil.rmtree(work_graph_dir)
    work_graph_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    seen = set()
    for src in graph_files:
        dest = work_graph_dir / src.name
        if dest.name in seen:
            raise ValueError(f"Duplicate graph filename after filtering: {dest.name}")
        seen.add(dest.name)
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def class_summary_rows(class_rows: List[dict]) -> List[dict]:
    by_class: Dict[str, List[dict]] = {}
    for row in class_rows:
        by_class.setdefault(row["class"], []).append(row)
    out = []
    for cls, rows in sorted(by_class.items(), key=lambda kv: class_sort_key(kv[0])):
        rep = rows[0]["representative_graph"]
        out.append(
            {
                "class": cls,
                "class_size": len(rows),
                "representative_graph": rep,
                "representative_structure_id": normalize_structure_id(rep),
                "members": ";".join(r["graph_file"] for r in rows),
                "structure_ids": ";".join(r["structure_id"] for r in rows),
            }
        )
    return out


def run_step08(config_path: str | Path, max_structures: Optional[int] = None, clean_output: bool = False) -> dict:
    config_path = Path(config_path).resolve()
    root = get_project_root(config_path)
    config = load_config_dict(config_path)
    target_id = get_target_id(config, config_path)
    output_root = get_output_root(config, root, target_id)
    step_cfg = config.get("step08", {}) or {}
    if step_cfg.get("enabled", True) is False:
        return {"target_id": target_id, "status": "skipped", "message": "step08.enabled is false."}

    output_subdir = public_output_subdir(
        step_cfg.get("output_subdir"), "07_ged_classification"
    )
    out_dir = output_root / output_subdir
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    work_dir = out_dir / "work"
    class_dir = out_dir / "classes"
    for d in [tables_dir, reports_dir, work_dir, class_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
        for d in [tables_dir, reports_dir, work_dir, class_dir]:
            d.mkdir(parents=True, exist_ok=True)

    graph_dir = find_step06_graph_dir(output_root, target_id, step_cfg, root)
    use_step07_filter = bool(step_cfg.get("use_step07_filter", True))
    passed_table = None
    passed_ids = None
    passed_rows: List[dict] = []
    if use_step07_filter:
        passed_table = find_step07_passed_table(output_root, target_id, step_cfg, root)
        passed_ids, passed_rows = ids_from_step07_table(passed_table)

    graph_files, skipped_rows = collect_graph_files(graph_dir, passed_ids)
    if max_structures:
        # Keep deterministic debug behavior after filtering.
        extra = graph_files[max_structures:]
        skipped_rows.extend(
            {"graph_file": str(p), "graph_id": graph_id_from_path(p), "reason": "max_structures_debug_limit"}
            for p in extra
        )
        graph_files = graph_files[:max_structures]

    if not graph_files:
        raise ValueError("No GED graph files selected for Step08 after applying Step07 filter.")

    selected_rows = [
        {
            "graph_file": str(p),
            "graph_filename": p.name,
            "structure_id": graph_id_from_path(p),
            "edge_count": sum(1 for line in p.open("r", encoding="utf-8", errors="replace") if line.startswith("e ")),
            "source_graph_dir": str(graph_dir),
            "selected_by_step07_filter": use_step07_filter,
        }
        for p in graph_files
    ]
    selected_table = tables_dir / f"{target_id}_step07_selected_graphs.csv"
    skipped_table = tables_dir / f"{target_id}_step07_skipped_graphs.csv"
    write_csv(selected_table, selected_rows)
    write_csv(skipped_table, skipped_rows, fieldnames=["graph_file", "graph_id", "reason"])

    work_graph_dir = work_dir / "filtered_graphs"
    copied = copy_graphs_to_workdir(graph_files, work_graph_dir)

    ged_cfg = step_cfg.get("ged", {}) or {}
    executable_cwd_cfg = ged_cfg.get("working_directory", None)
    executable_cwd = resolve_path(str(executable_cwd_cfg), root) if executable_cwd_cfg else None
    executable_cfg = ged_cfg.get("executable", "ged")
    executable = resolve_ged_executable(
        executable_cfg=executable_cfg,
        root=root,
        executable_cwd=executable_cwd,
        auto_search_paths=ged_cfg.get("auto_search_paths"),
    )

    threshold = float(ged_cfg.get("threshold", step_cfg.get("threshold", 1)))
    max_workers = int(ged_cfg.get("max_workers", step_cfg.get("max_workers", 6)))
    timeout_seconds = int(ged_cfg.get("timeout_seconds", 300))
    require_same_edge_count = bool(ged_cfg.get("require_same_edge_count", True))
    method = str(ged_cfg.get("method", "pair"))
    algorithm = str(ged_cfg.get("algorithm", ged_cfg.get("path_algorithm", "astar")))
    lower_bound = str(ged_cfg.get("lower_bound", "LSa"))

    memory_cfg = ged_cfg.get("memory", {}) or {}
    memory_limit_raw = (
        ged_cfg.get("memory_limit_mb")
        or ged_cfg.get("subprocess_memory_limit_mb")
        or memory_cfg.get("per_process_mb")
        or memory_cfg.get("memory_limit_mb")
    )
    memory_limit_mb = int(memory_limit_raw) if memory_limit_raw not in {None, "", "null"} else None
    max_total_raw = ged_cfg.get("max_total_memory_mb") or memory_cfg.get("max_total_mb")
    max_total_memory_mb = int(max_total_raw) if max_total_raw not in {None, "", "null"} else None
    if memory_limit_mb and max_total_memory_mb:
        safe_workers = max(1, max_total_memory_mb // memory_limit_mb)
        if max_workers > safe_workers:
            max_workers = safe_workers

    pair_batch_size_raw = ged_cfg.get("pair_batch_size") or memory_cfg.get("pair_batch_size")
    pair_batch_size = int(pair_batch_size_raw) if pair_batch_size_raw not in {None, "", "null"} else None
    cache_ged = bool(ged_cfg.get("cache_ged", memory_cfg.get("cache_ged", False)))
    stream_pairwise_results = bool(ged_cfg.get("stream_pairwise_results", memory_cfg.get("stream_pairwise_results", True)))

    graph_classes_table = tables_dir / f"{target_id}_step07_graph_classes.csv"
    pairwise_table = tables_dir / f"{target_id}_step07_representative_pairwise_ged.tsv"
    class_summary_table = tables_dir / f"{target_id}_step07_class_summary.csv"
    representatives_table = tables_dir / f"{target_id}_step07_representatives.csv"
    pairwise_fieldnames = [
        "representative_graph", "query_graph", "representative_id", "query_id",
        "representative_edge_count", "query_edge_count", "same_edge_count",
        "ged", "threshold", "assigned_to_current_class", "class",
    ]

    logger = setup_logger(reports_dir / f"{target_id}_step07_ged_classification.log")
    logger.info(
        "Memory settings: max_workers=%s, subprocess_memory_limit_mb=%s, max_total_memory_mb=%s, pair_batch_size=%s, cache_ged=%s, stream_pairwise_results=%s",
        max_workers,
        memory_limit_mb,
        max_total_memory_mb,
        pair_batch_size,
        cache_ged,
        stream_pairwise_results,
    )
    classifier = GEDClassifier(
        executable=executable,
        method=method,
        algorithm=algorithm,
        lower_bound=lower_bound,
        threshold=threshold,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
        require_same_edge_count=require_same_edge_count,
        logger=logger,
        executable_cwd=executable_cwd,
        memory_limit_mb=memory_limit_mb,
        cache_ged=cache_ged,
        pair_batch_size=pair_batch_size,
        class_naming=step_cfg.get("class_naming", {}) or {},
    )

    class_rows, pair_rows = classifier.classify(
        work_graph_dir,
        class_dir,
        pairwise_output_path=pairwise_table if stream_pairwise_results else None,
        pairwise_fieldnames=pairwise_fieldnames,
    )

    stable_cfg = (
        step_cfg.get("stable_class_labels")
        or step_cfg.get("class_label_stability")
        or {}
    )
    if isinstance(stable_cfg, bool):
        stable_cfg = {"enabled": stable_cfg}
    stable_enabled = bool(stable_cfg.get("enabled", False))
    stable_mapping = {}
    stable_audit_rows = []
    stable_reference_path = None

    if stable_enabled:
        reference_value = (
            stable_cfg.get("reference_class_manifest")
            or stable_cfg.get("reference_class_table")
            or stable_cfg.get("reference_registry")
        )
        if not reference_value:
            raise ValueError(
                "step08.stable_class_labels.enabled is true, but no "
                "reference_class_manifest/reference_class_table was provided."
            )
        stable_reference_path = resolve_path(str(reference_value), root)
        if stable_reference_path is None:
            raise ValueError(f"Invalid stable class reference path: {reference_value}")
        reference_rows = read_reference_manifest(
            stable_reference_path,
            structure_id_offset=int(
                stable_cfg.get("reference_structure_id_integer_offset", 0) or 0
            ),
        )
        stable_mapping, stable_audit_rows = match_stable_class_labels(
            class_rows,
            reference_rows,
            minimum_jaccard=float(stable_cfg.get("minimum_jaccard", 0.25)),
            class_naming=step_cfg.get("class_naming", {}) or {},
        )
        rename_class_directories(class_dir, stable_mapping)
        class_rows = apply_mapping_to_class_rows(class_rows, stable_mapping)
        if stream_pairwise_results:
            rewrite_pairwise_tsv(pairwise_table, stable_mapping)
        else:
            for row in pair_rows:
                old_class = str(row.get("class", ""))
                if old_class in stable_mapping:
                    row["class"] = stable_mapping[old_class]
        rewrite_results_text(
            class_dir / "ged_classification_results.txt",
            stable_mapping,
        )
    else:
        for row in class_rows:
            row["provisional_class"] = row["class"]

    stable_audit_table = tables_dir / f"{target_id}_step07_stable_class_label_audit.csv"
    stable_registry_table = tables_dir / f"{target_id}_step07_class_label_registry.csv"
    write_csv(
        stable_audit_table,
        stable_audit_rows,
        fieldnames=[
            "current_class", "reference_class", "assigned_class",
            "match_type", "label_source", "exact_member_set",
            "reference_representative_retained", "intersection_size",
            "jaccard", "current_size", "reference_size", "size_difference",
            "current_representative_structure_id",
            "reference_representative_structure_id",
        ],
    )
    write_csv(
        stable_registry_table,
        build_reference_registry(class_rows),
        fieldnames=[
            "reference_class", "structure_id",
            "reference_representative_structure_id", "class_size",
        ],
    )

    write_csv(graph_classes_table, class_rows)
    if not stream_pairwise_results:
        write_csv(pairwise_table, pair_rows, fieldnames=pairwise_fieldnames)
    summary_rows = class_summary_rows(class_rows)
    write_csv(class_summary_table, summary_rows)
    write_csv(
        representatives_table,
        [
            {
                "class": r["class"],
                "representative_graph": r["representative_graph"],
                "representative_structure_id": r["representative_structure_id"],
                "class_size": r["class_size"],
            }
            for r in summary_rows
        ],
    )

    report = {
        "target_id": target_id,
        "status": "completed",
        "config": str(config_path),
        "output_dir": str(out_dir),
        "input_graph_dir": str(graph_dir),
        "use_step07_filter": use_step07_filter,
        "step07_passed_table": str(passed_table) if passed_table else None,
        "n_step07_passed_rows": len(passed_rows) if passed_rows else None,
        "n_graphs_selected": len(graph_files),
        "n_graphs_skipped": len(skipped_rows),
        "n_classes": len(summary_rows),
        "stable_class_labels": {
            "enabled": stable_enabled,
            "reference_manifest": str(stable_reference_path) if stable_reference_path else None,
            "minimum_jaccard": float(stable_cfg.get("minimum_jaccard", 0.25)) if stable_enabled else None,
            "n_reference_labels_reused": int(
                sum(1 for row in stable_audit_rows if row.get("label_source") == "reference")
            ),
            "n_new_labels_appended": int(
                sum(1 for row in stable_audit_rows if row.get("label_source") == "new")
            ),
            "n_reference_labels_absent": int(
                sum(1 for row in stable_audit_rows if row.get("label_source") == "retired_or_absent")
            ),
        },
        "ged": {
            "executable": executable,
            "method": method,
            "algorithm": algorithm,
            "lower_bound": lower_bound,
            "threshold": threshold,
            "timeout_seconds": timeout_seconds,
            "max_workers": max_workers,
            "require_same_edge_count": require_same_edge_count,
            "memory_limit_mb": memory_limit_mb,
            "max_total_memory_mb": max_total_memory_mb,
            "pair_batch_size": pair_batch_size,
            "cache_ged": cache_ged,
            "stream_pairwise_results": stream_pairwise_results,
            "classification_rule": "GED < threshold",
            "class_naming": step_cfg.get("class_naming", {}) or {"prefix": "class", "separator": "_"},
        },
        "tables": {
            "selected_graphs": str(selected_table),
            "skipped_graphs": str(skipped_table),
            "graph_classes": str(graph_classes_table),
            "representative_pairwise_ged": str(pairwise_table),
            "class_summary": str(class_summary_table),
            "representatives": str(representatives_table),
            "stable_class_label_audit": str(stable_audit_table),
            "class_label_registry": str(stable_registry_table),
        },
        "directories": {
            "working_filtered_graphs": str(work_graph_dir),
            "classes": str(class_dir),
        },
        "reports": {
            "log": str(reports_dir / f"{target_id}_step07_ged_classification.log"),
        },
    }
    report_path = reports_dir / f"{target_id}_step07_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step08 GED classification after Step07 key-residue filtering")
    parser.add_argument("--config", required=True, help="Path to project config.yaml")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional debug limit after Step07 filtering")
    parser.add_argument("--clean", action="store_true", help="Remove the Step08 output directory before running")
    args = parser.parse_args(argv)
    report = run_step08(args.config, max_structures=args.max_structures, clean_output=args.clean)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

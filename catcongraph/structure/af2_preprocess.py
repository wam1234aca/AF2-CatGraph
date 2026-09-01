"""Trim disordered AF2 termini and give an ensemble stable PDB names.

This is intentionally an input-preparation utility rather than a workflow
step.  It never changes the source PDBs or renumbers the residues that remain.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_RANGE_PART = re.compile(r"^(?P<start>-?\d+)\s*(?:-\s*(?P<end>-?\d+))?$")
_RANK_TOKEN = re.compile(r"(?:^|[_\-.])rank(?:ed)?(?:[_\-.])?(\d+)(?:$|[_\-.])", re.IGNORECASE)


@dataclass(frozen=True)
class ResidueRange:
    """Inclusive residue-number interval."""

    start: int
    end: int

    def contains(self, residue_number: int) -> bool:
        return self.start <= residue_number <= self.end


@dataclass(frozen=True)
class PreprocessJob:
    source: Path
    rank: int
    source_rank_token: int | None


def parse_residue_ranges(text: str) -> list[ResidueRange]:
    """Parse ``1-120,290-300`` into validated inclusive intervals."""
    ranges: list[ResidueRange] = []
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        match = _RANGE_PART.fullmatch(part)
        if not match:
            raise ValueError(
                f"Invalid residue range {part!r}. Use values such as '1-120,290-300,415'."
            )
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            raise ValueError(f"Residue range {part!r} ends before it starts.")
        ranges.append(ResidueRange(start, end))
    if not ranges:
        raise ValueError("At least one residue range is required, for example '1-120,290-300'.")
    return ranges


def format_residue_ranges(ranges: Iterable[ResidueRange]) -> str:
    return ",".join(
        str(item.start) if item.start == item.end else f"{item.start}-{item.end}"
        for item in ranges
    )


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def extract_source_rank(path: Path) -> int | None:
    """Return an AF2/ColabFold rank token, when the filename contains one."""
    padded = f"_{path.stem}_"
    match = _RANK_TOKEN.search(padded)
    return int(match.group(1)) if match else None


def discover_pdbs(input_dir: Path, recursive: bool = True) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    pdbs = [path for path in iterator if path.is_file() and path.suffix.lower() == ".pdb"]
    if not pdbs:
        raise FileNotFoundError(f"No PDB files were found in {input_dir}.")
    return pdbs


def build_jobs(paths: Iterable[Path], rank_start: int = 1) -> list[PreprocessJob]:
    """Sort AF2 rank files naturally and assign a contiguous output rank."""
    if rank_start < 1:
        raise ValueError("rank_start must be at least 1.")
    decorated = [(extract_source_rank(path), path) for path in paths]
    decorated.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0, _natural_key(item[1])))
    return [
        PreprocessJob(source=path, rank=rank_start + index, source_rank_token=source_rank)
        for index, (source_rank, path) in enumerate(decorated)
    ]


def _record_name(line: str) -> str:
    return line[:6].strip().upper()


def _residue_number(line: str) -> int | None:
    try:
        return int(line[22:26].strip())
    except ValueError:
        return None


def _atom_serial(line: str) -> int | None:
    try:
        return int(line[6:11].strip())
    except ValueError:
        return None


def _in_removed_region(line: str, ranges: Sequence[ResidueRange], chains: set[str] | None) -> bool:
    residue_number = _residue_number(line)
    if residue_number is None:
        return False
    chain = line[21].strip()
    return (chains is None or chain in chains) and any(item.contains(residue_number) for item in ranges)


def _filter_conect(line: str, removed_atom_serials: set[int]) -> str | None:
    try:
        serials = [int(part) for part in line[6:].split()]
    except ValueError:
        # A nonstandard CONECT record is safer to retain than to rewrite.
        return line
    retained = [serial for serial in serials if serial not in removed_atom_serials]
    if len(retained) < 2:
        return None
    return "CONECT" + "".join(f"{serial:>5}" for serial in retained) + "\n"


def trim_pdb(
    source: Path,
    destination: Path,
    ranges: Sequence[ResidueRange],
    chains: set[str] | None = None,
) -> dict[str, object]:
    """Write a trimmed PDB while retaining original residue numbering."""
    removed_serials: set[int] = set()
    removed_residues: set[tuple[str, int, str, str]] = set()
    retained_coordinate_records = 0
    staged: list[str | None] = []

    for line in source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        record = _record_name(line)
        if record in {"ATOM", "HETATM", "ANISOU", "TER"} and _in_removed_region(line, ranges, chains):
            serial = _atom_serial(line)
            if serial is not None and record in {"ATOM", "HETATM", "ANISOU"}:
                removed_serials.add(serial)
            residue_number = _residue_number(line)
            if residue_number is not None:
                removed_residues.add((line[21].strip(), residue_number, line[26].strip(), line[17:20].strip()))
            continue
        if record in {"ATOM", "HETATM"}:
            retained_coordinate_records += 1
        staged.append(line)

    if retained_coordinate_records == 0:
        raise ValueError(f"Trimming {source.name} would remove every ATOM/HETATM record.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for line in staged:
            if line is None:
                continue
            if _record_name(line) == "CONECT":
                line = _filter_conect(line, removed_serials)
            if line is not None:
                handle.write(line)

    return {
        "source": str(source),
        "output": str(destination),
        "removed_atom_records": len(removed_serials),
        "removed_residue_count": len(removed_residues),
        "removed_residues": ";".join(
            f"{chain or '-'}:{residue}{icode or ''}:{resname}" for chain, residue, icode, resname in sorted(removed_residues)
        ),
        "retained_coordinate_records": retained_coordinate_records,
    }


def preprocess_af2_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    prefix: str,
    residue_ranges: str | Sequence[ResidueRange],
    *,
    chains: str | Iterable[str] | None = None,
    recursive: bool = True,
    rank_start: int = 1,
    report_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Trim every PDB below ``input_dir`` and write ``prefix.AF2rankN.pdb``."""
    source_root = Path(input_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_root}")
    if destination_root == source_root or source_root in destination_root.parents:
        raise ValueError("output_dir must be outside input_dir so source PDB files can never be overwritten.")
    clean_prefix = str(prefix).strip()
    if not clean_prefix or any(character in clean_prefix for character in "/\\"):
        raise ValueError("prefix must be a non-empty filename stem, for example '6B90'.")

    ranges = parse_residue_ranges(residue_ranges) if isinstance(residue_ranges, str) else list(residue_ranges)
    chain_set = None
    if chains:
        chain_values = str(chains).split(",") if isinstance(chains, str) else chains
        chain_set = {str(chain).strip() for chain in chain_values if str(chain).strip()}
        if not chain_set:
            chain_set = None

    jobs = build_jobs(discover_pdbs(source_root, recursive=recursive), rank_start=rank_start)
    rows: list[dict[str, object]] = []
    for job in jobs:
        output = destination_root / f"{clean_prefix}.AF2rank{job.rank}.pdb"
        row = trim_pdb(job.source, output, ranges, chains=chain_set)
        row.update({
            "assigned_rank": job.rank,
            "source_rank_token": "" if job.source_rank_token is None else job.source_rank_token,
            "removed_ranges": format_residue_ranges(ranges),
            "chains": "all" if chain_set is None else ",".join(sorted(chain_set)),
        })
        rows.append(row)

    csv_path = Path(report_path).expanduser().resolve() if report_path else destination_root / "af2_preprocess_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "source", "output", "assigned_rank", "source_rank_token", "removed_ranges", "chains",
            "removed_atom_records", "removed_residue_count", "removed_residues", "retained_coordinate_records",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trim specified residue ranges from AF2 PDBs and rename them predictably."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing AF2 PDB files; nested folders are included by default.")
    parser.add_argument("--output-dir", required=True, help="New folder for trimmed PDB files; must not be inside input-dir.")
    parser.add_argument("--name", required=True, help="Output prefix, for example 6B90.")
    parser.add_argument("--remove", required=True, help="Inclusive residue ranges, for example 1-120,290-300.")
    parser.add_argument("--chains", default="", help="Optional comma-separated chain IDs; default removes from every chain.")
    parser.add_argument("--no-recursive", action="store_true", help="Only inspect PDB files directly in input-dir.")
    parser.add_argument("--rank-start", type=int, default=1, help="First output rank; default 1.")
    parser.add_argument("--report", default="", help="Optional CSV report path; default is output-dir/af2_preprocess_manifest.csv.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = preprocess_af2_directory(
        args.input_dir,
        args.output_dir,
        args.name,
        args.remove,
        chains=args.chains or None,
        recursive=not args.no_recursive,
        rank_start=args.rank_start,
        report_path=args.report or None,
    )
    print(f"Processed {len(rows)} AF2 PDB file(s).")
    for row in rows:
        print(f"  {row['source']} -> {row['output']}")


if __name__ == "__main__":  # pragma: no cover
    main()

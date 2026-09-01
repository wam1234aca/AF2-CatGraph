from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re


@dataclass(frozen=True)
class PDBAtomRecord:
    line_index: int
    record_name: str
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    x: float
    y: float
    z: float
    b_factor: float
    raw_line: str

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id.strip() or "_",
            self.residue_number,
            self.insertion_code.strip() or "",
            self.residue_name.strip(),
        )

    @property
    def atom_key(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id.strip() or "_",
            self.residue_number,
            self.insertion_code.strip() or "",
            self.atom_name.strip(),
        )


def parse_pdb_atoms(
    pdb_path: str | Path,
    *,
    include_hetatm: bool = False,
    atom_name: str | None = None,
) -> list[PDBAtomRecord]:
    """Parse ATOM records from a PDB file.

    AlphaFold/ColabFold stores pLDDT in the B-factor column of the PDB file.
    By default, only ATOM records are used because Step 01 is protein-only.
    """
    pdb_path = Path(pdb_path)
    atom_name_filter = atom_name.strip().upper() if atom_name else None
    records: list[PDBAtomRecord] = []

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            record_name = line[0:6].strip()
            if record_name == "ATOM":
                pass
            elif record_name == "HETATM" and include_hetatm:
                pass
            else:
                continue

            try:
                this_atom_name = line[12:16].strip()
                if atom_name_filter and this_atom_name.upper() != atom_name_filter:
                    continue

                records.append(
                    PDBAtomRecord(
                        line_index=idx,
                        record_name=record_name,
                        atom_name=this_atom_name,
                        altloc=line[16:17].strip(),
                        residue_name=line[17:20].strip(),
                        chain_id=line[21:22].strip(),
                        residue_number=int(line[22:26]),
                        insertion_code=line[26:27].strip(),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        b_factor=float(line[60:66]),
                        raw_line=line.rstrip("\n"),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Failed to parse PDB atom line in {pdb_path}: {line!r}") from exc

    return records


def extract_colabfold_metadata(filename: str) -> dict[str, int | str | None]:
    """Extract common ColabFold metadata from a file name when available."""
    stem = Path(filename).stem
    patterns = {
        "rank": r"_rank_(\d+)",
        "model": r"_model_(\d+)",
        "seed": r"_seed_(\d+)",
    }
    result: dict[str, int | str | None] = {
        "original_structure_id": stem,
        "rank": None,
        "model": None,
        "seed": None,
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, stem)
        if m:
            result[key] = int(m.group(1))
    return result


def replace_xyz_in_pdb_line(line: str, x: float, y: float, z: float) -> str:
    """Return a PDB line with updated XYZ coordinates."""
    if len(line) < 54:
        line = line.rstrip("\n").ljust(54)
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def write_transformed_pdb(
    input_pdb: str | Path,
    output_pdb: str | Path,
    transformed_coords_by_line_index: dict[int, tuple[float, float, float]],
) -> None:
    """Write a transformed PDB while preserving all non-coordinate fields."""
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    with input_pdb.open("r", encoding="utf-8", errors="replace") as inp, output_pdb.open(
        "w", encoding="utf-8"
    ) as out:
        for idx, line in enumerate(inp):
            raw = line.rstrip("\n")
            if idx in transformed_coords_by_line_index:
                x, y, z = transformed_coords_by_line_index[idx]
                out.write(replace_xyz_in_pdb_line(raw, x, y, z) + "\n")
            else:
                out.write(raw + "\n")

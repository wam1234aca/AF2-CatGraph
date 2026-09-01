"""Step07: conserved key-residue screening.

This step applies four reproducible requirements to each protein residue:

1. ConSurf conservation grade, parsed directly from a ConSurf result ZIP,
   website TAR.GZ/GZ download, or normalized CSV/TSV file.
2. Minimum heavy-atom distance from the residue to substrate / ligand atoms in
   each accepted complex.
3. Occurrence frequency of close residue--ligand contacts across accepted
   complexes.
4. Occurrence frequency of the residue in Step06 PLIP interaction outputs.

All four are required for a residue to be reported as a key residue.  The PLIP
requirement is intentionally not optional: a merely close residue should not
be promoted to a catalytic-contact claim without an interaction record.

The implementation intentionally avoids changing Step06 GED labels or PLIP
classification behavior. It consumes Step06 outputs, but does not rewrite them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import io
import json
import math
import re
import tarfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from catcongraph.config import load_yaml_config, public_output_subdir

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL", "X": "UNK", "-": "GAP",
}
AA3_TO_AA1 = {v: k for k, v in AA1_TO_AA3.items()}

WATER_RESNAMES = {"HOH", "WAT", "H2O"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


@dataclass(frozen=True)
class AtomRecord:
    record_name: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str

    @property
    def residue_key(self) -> str:
        return make_residue_key(self.resname, self.chain, self.resseq, self.icode)


def make_residue_key(resname: str, chain: str, resseq: int, icode: str = "") -> str:
    chain = (chain or "_").strip() or "_"
    icode = (icode or "").strip()
    return f"{resname.strip().upper()}{int(resseq)}{chain}{icode}"


def resolve_path(path_like: Optional[str], root: Path) -> Optional[Path]:
    if path_like is None:
        return None
    text = str(path_like).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    p = Path(text)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _load_config_dict(config_path: Path) -> dict:
    return load_yaml_config(config_path)


def get_project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    # Standard layout: <repo>/projects/<target_id>/config.yaml
    if config_path.parent.parent.name == "projects":
        return config_path.parent.parent.parent
    return Path.cwd().resolve()


def get_target_id(config: dict, config_path: Path) -> str:
    project = config.get("project", {}) or {}
    if isinstance(project, dict) and project.get("target_id"):
        return str(project.get("target_id"))
    if config.get("target_id"):
        return str(config.get("target_id"))
    return config_path.parent.name


def get_output_root(config: dict, root: Path, target_id: str) -> Path:
    out = config.get("project", {}).get("output_root") or config.get("output_root")
    if out:
        return resolve_path(out, root) or (root / "results" / target_id)
    return root / "results" / target_id


def parse_pdb_atoms(pdb_path: Path) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                record = line[0:6].strip()
                serial = int(line[6:11])
                atom_name = line[12:16].strip()
                altloc = line[16:17].strip()
                resname = line[17:20].strip().upper()
                chain = line[21:22].strip() or "_"
                resseq = int(line[22:26])
                icode = line[26:27].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip().upper() if len(line) >= 78 else ""
                if not element:
                    element = re.sub(r"[^A-Za-z]", "", atom_name)[:2].upper()
                atoms.append(
                    AtomRecord(
                        record_name=record,
                        serial=serial,
                        atom_name=atom_name,
                        altloc=altloc,
                        resname=resname,
                        chain=chain,
                        resseq=resseq,
                        icode=icode,
                        x=x,
                        y=y,
                        z=z,
                        element=element,
                    )
                )
            except Exception:
                # Keep Step07 robust for imperfect PDB files.
                continue
    return atoms


def is_hydrogen(atom: AtomRecord) -> bool:
    return atom.element.upper() == "H" or atom.atom_name.upper().startswith("H")


def atom_distance(a: AtomRecord, b: AtomRecord) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def residue_min_distance(
    residue_atoms: Sequence[AtomRecord], ligand_atoms: Sequence[AtomRecord]
) -> float:
    best = math.inf
    for a in residue_atoms:
        ax, ay, az = a.x, a.y, a.z
        for b in ligand_atoms:
            d2 = (ax - b.x) ** 2 + (ay - b.y) ** 2 + (az - b.z) ** 2
            if d2 < best:
                best = d2
    return math.sqrt(best) if best < math.inf else math.inf


def split_protein_ligand_atoms(
    atoms: Sequence[AtomRecord],
    ligand_cfg: dict,
    distance_cfg: dict,
) -> Tuple[List[AtomRecord], List[AtomRecord]]:
    exclude_resnames = set(ligand_cfg.get("exclude_resnames") or []) | WATER_RESNAMES
    exclude_resnames = {x.upper() for x in exclude_resnames}
    include_resnames = ligand_cfg.get("include_resnames")
    include_resnames_set = {x.upper() for x in include_resnames} if include_resnames else None
    heavy_only = bool(distance_cfg.get("use_heavy_atoms_only", True))
    protein_atom_scope = str(distance_cfg.get("protein_atom_scope", "all")).lower()

    protein_atoms: List[AtomRecord] = []
    ligand_atoms: List[AtomRecord] = []
    for atom in atoms:
        if heavy_only and is_hydrogen(atom):
            continue
        if atom.record_name == "ATOM":
            if protein_atom_scope == "sidechain" and atom.atom_name.strip().upper() in BACKBONE_ATOMS:
                continue
            protein_atoms.append(atom)
        elif atom.record_name == "HETATM":
            if atom.resname.upper() in exclude_resnames:
                continue
            if include_resnames_set is not None and atom.resname.upper() not in include_resnames_set:
                continue
            ligand_atoms.append(atom)
    return protein_atoms, ligand_atoms


def _looks_like_consurf_grade_file(name: str) -> bool:
    """Return whether an archive member is a ConSurf residue-score table."""
    lower = name.lower()
    basename = Path(lower).name
    return (
        ("consurf" in lower and "grade" in lower and lower.endswith((".txt", ".tsv", ".csv")))
        or basename.endswith("consurf_grades.txt")
    )


def _choose_consurf_grade_name(names: Sequence[str], preferred_name: Optional[str], archive_path: Path) -> str:
    """Choose the authoritative ConSurf grade table from an archive member list."""
    files = [str(name) for name in names if str(name) and not str(name).endswith("/")]
    if preferred_name:
        preferred = str(preferred_name).strip()
        for name in files:
            if Path(name).name == preferred or name == preferred:
                return name
        raise FileNotFoundError(
            f"Configured ConSurf grade file {preferred!r} was not found inside {archive_path}."
        )
    candidates = [name for name in files if _looks_like_consurf_grade_file(name)]
    if not candidates:
        raise FileNotFoundError(
            f"No ConSurf grade file found inside {archive_path}. Expected a file like '*consurf_grades.txt'."
        )
    # Prefer the canonical name, then a shallow member path.  This avoids
    # accidentally selecting visualization scripts that happen to mention
    # ConSurf in their filename.
    candidates.sort(
        key=lambda name: (
            0 if Path(name).name.lower().endswith("consurf_grades.txt") else 1,
            len(Path(name).parts),
            len(name),
            name,
        )
    )
    return candidates[0]


def choose_consurf_file_from_zip(zip_path: Path, preferred_name: Optional[str] = None) -> Tuple[str, str]:
    """Read a grade table from the legacy ConSurf ZIP download format."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        chosen = _choose_consurf_grade_name(zf.namelist(), preferred_name, zip_path)
        return chosen, zf.read(chosen).decode("utf-8", errors="replace")


def choose_consurf_file_from_tar(tar_path: Path, preferred_name: Optional[str] = None) -> Tuple[str, str]:
    """Read a grade table from ConSurf website ``.tar.gz`` / ``.gz`` downloads."""
    with tarfile.open(tar_path, "r:*") as tf:
        members = [member for member in tf.getmembers() if member.isfile()]
        chosen = _choose_consurf_grade_name([member.name for member in members], preferred_name, tar_path)
        member = next(member for member in members if member.name == chosen)
        handle = tf.extractfile(member)
        if handle is None:  # pragma: no cover - protected by member.isfile()
            raise FileNotFoundError(f"Could not read {chosen} from {tar_path}.")
        return chosen, handle.read().decode("utf-8", errors="replace")


def read_consurf_gzip_text(gzip_path: Path) -> str:
    """Read a gzipped plain-text grades file (not a tar archive)."""
    with gzip.open(gzip_path, "rt", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def parse_consurf_atom_annotation(annotation: str) -> Optional[Tuple[str, int, str, str]]:
    """Parse a website grade-table PDB mapping such as ``THR:132:A``.

    ConSurf writes ``-`` for an aligned query position that is absent from the
    submitted PDB.  Such a position must not be silently joined to a structure
    residue by sequence position: it has no coordinate-level correspondence.
    """
    text = str(annotation or "").strip()
    if not text or text in {"-", "?", "NA", "N/A"}:
        return None
    match = re.fullmatch(
        r"(?P<resname>[A-Za-z]{3}):(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?):(?P<chain>[^:\s]+)",
        text,
    )
    if match:
        return (
            match.group("resname").upper(),
            int(match.group("resseq")),
            match.group("chain").strip() or "_",
            match.group("icode") or "",
        )
    # Be permissive for occasional chain-first exports: ``A:THR:132``.
    match = re.fullmatch(
        r"(?P<chain>[^:\s]+):(?P<resname>[A-Za-z]{3}):(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)",
        text,
    )
    if match:
        return (
            match.group("resname").upper(),
            int(match.group("resseq")),
            match.group("chain").strip() or "_",
            match.group("icode") or "",
        )
    return None


def parse_consurf_grades_text(text: str, default_chain: str = "A") -> pd.DataFrame:
    """Parse ConSurf's '*consurf_grades.txt' text output.

    The standard ConSurf file contains explanatory text followed by rows like:
        13    K   -1.110   9    e   f   149/150   K 98%, ...

    Current website downloads add an ``ATOM`` column between ``SEQ`` and
    ``SCORE``.  Its values (for example ``THR:132:A``) are the authoritative
    mapping from alignment position to the submitted PDB's residue numbering.
    The parser preserves that mapping whenever it is available; it falls back
    to alignment positions only for older score files with no such column.
    """
    rows: List[dict] = []
    # Legacy compact format. COLOR can be "9" or "9*".
    compact_re = re.compile(
        r"^\s*(?P<pos>\d+)\s+"
        r"(?P<seq>[A-Z\-])\s+"
        r"(?P<score>[-+]?\d+(?:\.\d+)?)\s+"
        r"(?P<grade>\d)\*?\s+"
        r"(?P<be>[be\-]?)\s*"
        r"(?P<fs>[fs\-]?)\s+"
        r"(?P<msa>\d+\/\d+)\s*"
        r"(?P<variety>.*)$"
    )
    # Website format with the PDB mapping column:
    #  18 C CYS:134:A -1.190 9 -1.303, -1.135 9,9 b s 96/150 C 98%, ...
    website_re = re.compile(
        r"^\s*(?P<pos>\d+)\s+"
        r"(?P<seq>[A-Z\-])\s+"
        r"(?P<atom>\S+)\s+"
        r"(?P<score>[-+]?\d+(?:\.\d+)?)\s+"
        r"(?P<grade>\d)\*?\s+"
        r"(?P<rest>.*)$"
    )
    parsed_website_rows = 0
    parsed_annotation_rows = 0
    skipped_unmapped_rows = 0
    for line in text.splitlines():
        site_match = website_re.match(line)
        compact_match = compact_re.match(line) if site_match is None else None
        if site_match is None and compact_match is None:
            continue
        match = site_match or compact_match
        assert match is not None
        pos = int(match.group("pos"))
        seq = match.group("seq")
        grade = int(match.group("grade"))
        score = float(match.group("score"))
        atom_annotation = ""
        chain = default_chain
        resseq = pos
        icode = ""
        resname = AA1_TO_AA3.get(seq, "UNK")
        mapping_source = "alignment_position"
        if site_match is not None:
            parsed_website_rows += 1
            atom_annotation = site_match.group("atom")
            annotation = parse_consurf_atom_annotation(atom_annotation)
            if annotation is not None:
                resname, resseq, chain, icode = annotation
                mapping_source = "pdb_annotation"
                parsed_annotation_rows += 1
            else:
                # Defer the decision about unmapped rows until we know whether
                # this file contains any PDB mappings at all.  Pure alignment
                # files still retain their position-based legacy behaviour.
                mapping_source = "unmapped_pdb_position"
        rest = site_match.group("rest") if site_match is not None else ""
        msa_match = re.search(r"\b\d+\/\d+\b", rest) if rest else None
        be_match = re.search(r"(?:^|\s)([be])(?:\s|$)", rest, flags=re.IGNORECASE) if rest else None
        fs_match = re.search(r"(?:^|\s)([fs])(?:\s|$)", rest, flags=re.IGNORECASE) if rest else None
        rows.append(
            {
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "seq": seq,
                "resname": resname,
                "consurf_score": score,
                "consurf_grade": grade,
                "buried_exposed": (be_match.group(1).lower() if be_match else (match.group("be") or "") if compact_match is not None else ""),
                "functional_structural": (fs_match.group(1).lower() if fs_match else (match.group("fs") or "") if compact_match is not None else ""),
                "msa_data": (msa_match.group(0) if msa_match else (match.group("msa") or "") if compact_match is not None else ""),
                "residue_variety": (rest[msa_match.end():].strip() if msa_match else (match.group("variety") or "").strip() if compact_match is not None else rest.strip()),
                "consurf_position": pos,
                "pdb_residue_annotation": atom_annotation,
                "residue_mapping_source": mapping_source,
            }
        )
    if not rows:
        raise ValueError("Could not parse any residue rows from ConSurf grades text.")
    # In website grade files, ``-`` means an alignment position absent from the
    # submitted PDB.  Retaining it as (for example) residue 1 would create a
    # false conservation match if the analysed model happens to use numbering
    # that starts at 1.  Only filter these rows when at least one real mapping
    # proves that the file is PDB-aware.
    if parsed_annotation_rows:
        retained_rows = [row for row in rows if row["residue_mapping_source"] != "unmapped_pdb_position"]
        skipped_unmapped_rows = len(rows) - len(retained_rows)
        rows = retained_rows
    if not rows:
        raise ValueError("ConSurf grade file contains no residues mapped to the submitted PDB.")
    df = pd.DataFrame(rows)
    df["residue_key"] = [make_residue_key(r.resname, r.chain, int(r.resseq), r.icode) for r in df.itertuples()]
    df.attrs["consurf_parse_metadata"] = {
        "grades_format": "website_pdb_mapped" if parsed_website_rows else "legacy_compact",
        "n_rows_parsed": int(len(df)),
        "n_rows_with_pdb_annotation": int(parsed_annotation_rows),
        "n_rows_skipped_unmapped_pdb_position": int(skipped_unmapped_rows),
    }
    return df


def parse_normalized_conservation_table(path: Path, default_chain: str = "A") -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    lower_map = {c.lower().strip(): c for c in df.columns}
    def col(*names: str) -> Optional[str]:
        for n in names:
            if n in lower_map:
                return lower_map[n]
        return None
    resseq_col = col("resseq", "resnum", "residue_number", "pos", "position")
    grade_col = col("consurf_grade", "grade", "color", "conservation_grade")
    if not resseq_col or not grade_col:
        raise ValueError("Conservation table must contain residue number and ConSurf grade columns.")
    chain_col = col("chain", "chain_id")
    resname_col = col("resname", "residue", "aa3")
    seq_col = col("seq", "aa", "aa1")
    score_col = col("consurf_score", "score")
    icode_col = col("icode", "insertion_code")
    out = pd.DataFrame()
    out["chain"] = df[chain_col].fillna(default_chain).astype(str) if chain_col else default_chain
    out["resseq"] = df[resseq_col].astype(int)
    out["icode"] = df[icode_col].fillna("").astype(str) if icode_col else ""
    if resname_col:
        out["resname"] = df[resname_col].astype(str).str.upper().str[:3]
    elif seq_col:
        out["seq"] = df[seq_col].astype(str).str.upper().str[0]
        out["resname"] = out["seq"].map(AA1_TO_AA3).fillna("UNK")
    else:
        out["resname"] = "UNK"
    out["consurf_grade"] = df[grade_col].astype(str).str.extract(r"(\d)")[0].astype(int)
    out["consurf_score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else math.nan
    out["residue_key"] = [make_residue_key(r.resname, r.chain, int(r.resseq), r.icode) for r in out.itertuples()]
    return out


def load_consurf_scores(conservation_path: Path, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    default_chain = str(cfg.get("default_chain") or cfg.get("chain") or "A")
    preferred_name = cfg.get("preferred_grade_filename")
    metadata = {
        "source": str(conservation_path),
        "source_type": "table_or_text",
        "selected_file": None,
        "default_chain": default_chain,
    }
    if zipfile.is_zipfile(conservation_path):
        selected_name, text = choose_consurf_file_from_zip(conservation_path, preferred_name=preferred_name)
        metadata["source_type"] = "zip_archive"
        metadata["selected_file"] = selected_name
        df = parse_consurf_grades_text(text, default_chain=default_chain)
        metadata.update(df.attrs.get("consurf_parse_metadata", {}))
        return df, metadata
    if tarfile.is_tarfile(conservation_path):
        selected_name, text = choose_consurf_file_from_tar(conservation_path, preferred_name=preferred_name)
        metadata["source_type"] = "tar_archive"
        metadata["selected_file"] = selected_name
        df = parse_consurf_grades_text(text, default_chain=default_chain)
        metadata.update(df.attrs.get("consurf_parse_metadata", {}))
        return df, metadata
    if conservation_path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        df = parse_normalized_conservation_table(conservation_path, default_chain=default_chain)
        metadata["source_type"] = "normalized_table"
        metadata["selected_file"] = conservation_path.name
        return df, metadata
    if conservation_path.suffix.lower() == ".gz":
        text = read_consurf_gzip_text(conservation_path)
        metadata["source_type"] = "gzip_text"
    else:
        text = conservation_path.read_text(encoding="utf-8", errors="replace")
        metadata["source_type"] = "grades_text"
    metadata["selected_file"] = conservation_path.name
    df = parse_consurf_grades_text(text, default_chain=default_chain)
    metadata.update(df.attrs.get("consurf_parse_metadata", {}))
    return df, metadata


# ---------------------------------------------------------------------------
# ConSurf -> analysed-model residue mapping
# ---------------------------------------------------------------------------
# ConSurf commonly reports the numbering of the submitted experimental PDB,
# while AF2 preprocessing may trim a construct and renumber it from 1 (or use
# a different chain identifier).  The old implementation compared only the
# literal ``resname + number + chain`` key.  That made every score appear
# missing for perfectly valid truncated constructs such as NEW6.  The helpers
# below first honour an explicit mapping, then use exact keys, and finally use
# a sequence-aware local alignment.  Unmapped rows are deliberately removed
# from the score key, so they can never be mistaken for low conservation.


def _protein_residue_sequences(records: pd.DataFrame) -> Dict[str, List[dict]]:
    """Return ordered protein residues for the first usable complex PDB."""
    for value in records.get("pdb_path", pd.Series(dtype=str)).dropna().astype(str):
        path = Path(value)
        if not path.is_file():
            continue
        residues: Dict[str, List[dict]] = {}
        seen: set[Tuple[str, int, str]] = set()
        for atom in parse_pdb_atoms(path):
            if atom.record_name != "ATOM":
                continue
            token = (atom.chain, atom.resseq, atom.icode)
            if token in seen:
                continue
            seen.add(token)
            aa1 = AA3_TO_AA1.get(atom.resname.upper(), "X")
            residues.setdefault(atom.chain, []).append(
                {
                    "residue_key": atom.residue_key,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "icode": atom.icode,
                    "resname": atom.resname.upper(),
                    "seq": aa1,
                }
            )
        if residues:
            return residues
    return {}


def _needleman_wunsch_local_pairs(source: str, target: str) -> Tuple[List[Tuple[int, int]], float]:
    """Smith-Waterman alignment returning source/target index pairs and score."""
    if not source or not target:
        return [], 0.0
    n, m = len(source), len(target)
    # A compact integer matrix is sufficient for the short protein domains
    # encountered here and avoids adding another dependency.
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 1 diag, 2 up, 3 left
    best = (0.0, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = scores[i - 1][j - 1] + (3.0 if source[i - 1] == target[j - 1] else -2.0)
            up = scores[i - 1][j] - 3.0
            left = scores[i][j - 1] - 3.0
            value = max(0.0, diag, up, left)
            scores[i][j] = value
            if value == 0.0:
                trace[i][j] = 0
            elif value == diag:
                trace[i][j] = 1
            elif value == up:
                trace[i][j] = 2
            else:
                trace[i][j] = 3
            if value > best[0]:
                best = (value, i, j)
    _score, i, j = best
    pairs: List[Tuple[int, int]] = []
    while i > 0 and j > 0 and scores[i][j] > 0:
        direction = trace[i][j]
        if direction == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break
    pairs.reverse()
    return pairs, float(best[0])


def _mapping_cfg(cfg: dict) -> dict:
    value = cfg.get("mapping", {}) if isinstance(cfg, dict) else {}
    return value if isinstance(value, dict) else {}


def map_consurf_scores_to_complexes(
    consurf_df: pd.DataFrame,
    records: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, dict, List[str]]:
    """Map ConSurf rows onto the residue numbering used by analysed complexes.

    The returned frame keeps ``source_residue_key`` for auditability and uses
    ``residue_key`` only for rows that were actually mapped to a target PDB.
    """
    mapping = _mapping_cfg(cfg)
    enabled = bool(mapping.get("enabled", True))
    out = consurf_df.copy().reset_index(drop=True)
    out["source_residue_key"] = out.get("residue_key", pd.Series([pd.NA] * len(out)))
    out["source_chain"] = out.get("chain", pd.Series([pd.NA] * len(out)))
    out["source_resseq"] = out.get("resseq", pd.Series([pd.NA] * len(out)))
    out["source_icode"] = out.get("icode", pd.Series([""] * len(out)))
    out["consurf_mapping_status"] = "disabled" if not enabled else "missing"
    out["consurf_mapping_method"] = "disabled" if not enabled else "unmapped"
    out["consurf_mapping_identity"] = pd.NA

    target_by_chain = _protein_residue_sequences(records)
    target_keys = {r["residue_key"] for rows in target_by_chain.values() for r in rows}
    target_by_tuple = {
        (r["resname"], r["chain"], int(r["resseq"]), r["icode"]): r["residue_key"]
        for rows in target_by_chain.values() for r in rows
    }
    warnings: List[str] = []
    if not enabled:
        audit = {
            "enabled": False,
            "status": "disabled",
            "n_source_rows": int(len(out)),
            "n_target_residues": int(len(target_keys)),
            "n_mapped_rows": 0,
            "n_unmapped_rows": int(len(out)),
            "target_residue_coverage": None,
            "source_row_coverage": None,
        }
        return out, audit, warnings
    if not target_by_chain:
        warnings.append("ConSurf mapping could not inspect any protein residues in the complex PDBs.")
        out["residue_key"] = pd.NA
        audit = {
            "enabled": True,
            "status": "no_target_structure",
            "n_source_rows": int(len(out)),
            "n_target_residues": 0,
            "n_mapped_rows": 0,
            "n_unmapped_rows": int(len(out)),
            "target_residue_coverage": 0.0,
            "source_row_coverage": 0.0,
        }
        return out, audit, warnings

    chain_map = mapping.get("chain_map", {}) or {}
    offset_cfg = mapping.get("residue_number_offset", {}) or {}
    if isinstance(offset_cfg, (int, float)):
        offset_cfg = {"*": int(offset_cfg)}
    if not isinstance(chain_map, dict):
        chain_map = {}
    if not isinstance(offset_cfg, dict):
        offset_cfg = {}

    # Explicit offsets are the only mapping mode that is allowed to take
    # precedence over sequence evidence.  This matters for constructs such as
    # NEW6, where the ConSurf numbering is intentionally shifted.  For every
    # other case we first align the complete ConSurf chain to each target chain;
    # an exact residue number is only a fallback when an alignment is not
    # possible.  This prevents NEW26-style same-number collisions from silently
    # assigning scores to the wrong residues.
    candidates: Dict[int, Tuple[str, str, float]] = {}
    explicit_rows: set[int] = set()
    for idx, row in out.iterrows():
        source_chain = str(row.get("source_chain") or "_").strip() or "_"
        source_resseq = row.get("source_resseq")
        source_icode = str(row.get("source_icode") or "")
        resname = str(row.get("resname") or "UNK").upper()[:3]
        target_chain = str(chain_map.get(source_chain, source_chain)).strip() or source_chain
        raw_offset = offset_cfg.get(source_chain, offset_cfg.get("*", 0))
        try:
            offset = int(raw_offset or 0)
        except (TypeError, ValueError):
            offset = 0
        if not offset or source_resseq is None or pd.isna(source_resseq):
            continue
        try:
            shifted = int(source_resseq) + offset
            target = target_by_tuple.get((resname, target_chain, shifted, source_icode))
        except (TypeError, ValueError):
            target = None
        if target:
            candidates[int(idx)] = (target, "manual_offset", 1.0)
            explicit_rows.add(int(idx))

    # Sequence-aware mapping for every row not handled by an explicit offset.
    # Build one source sequence per ConSurf chain, preserving source dataframe
    # row indices.  The alignment pairs are intrinsically one-to-one.
    sequence_col = "seq" if "seq" in out.columns else None
    if sequence_col is None:
        out["seq"] = out["resname"].map(lambda x: AA3_TO_AA1.get(str(x).upper(), "X"))
        sequence_col = "seq"
    sequence_checked_chains: set[str] = set()
    for source_chain, group in out.groupby(out["source_chain"].astype(str), sort=False):
        group = group.sort_values("consurf_position" if "consurf_position" in group.columns else "source_resseq")
        source_rows: List[int] = []
        source_seq: List[str] = []
        for idx, row in group.iterrows():
            aa = str(row.get(sequence_col, "X") or "X").upper()[:1]
            if aa in AA1_TO_AA3 and aa != "-":
                source_rows.append(int(idx))
                source_seq.append(aa)
        if len(source_seq) < 3:
            continue
        sequence_checked_chains.add(str(source_chain))
        forced_target = str(chain_map.get(source_chain, "")).strip()
        target_chains = [forced_target] if forced_target in target_by_chain else list(target_by_chain)
        if all(int(i) in explicit_rows for i in source_rows):
            continue
        best = None
        for target_chain_name in target_chains:
            target_rows = target_by_chain[target_chain_name]
            target_seq = "".join(r["seq"] for r in target_rows)
            pairs, score = _needleman_wunsch_local_pairs("".join(source_seq), target_seq)
            if not pairs:
                continue
            identities = sum(source_seq[i] == target_rows[j]["seq"] for i, j in pairs)
            identity = identities / float(max(len(pairs), 1))
            target_cov = len({j for _i, j in pairs}) / float(max(len(target_rows), 1))
            source_cov = len({i for i, _j in pairs}) / float(max(len(source_seq), 1))
            candidate = (identity, target_cov, source_cov, score, target_chain_name, pairs, target_rows)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
        if best is None:
            continue
        identity, target_cov, source_cov, _score, target_chain_name, pairs, target_rows = best
        min_identity = float(mapping.get("min_sequence_identity", 0.70))
        min_target_cov = float(mapping.get("min_target_coverage", 0.50))
        if identity < min_identity or target_cov < min_target_cov:
            warnings.append(
                f"ConSurf chain {source_chain} did not meet sequence mapping thresholds "
                f"(identity={identity:.3f}, target_coverage={target_cov:.3f})."
            )
            continue
        for source_i, target_i in pairs:
            row_idx = source_rows[source_i]
            if row_idx in candidates:
                continue
            target = target_rows[target_i]
            candidates[row_idx] = (target["residue_key"], "sequence_alignment", float(identity))

        # If a chain was aligned successfully, do not let a literal residue
        # number fill alignment gaps: an unrelated residue with the same number
        # is exactly the failure mode this mapping stage must prevent.

    # Conservative fallback for files without usable sequence columns.  Exact
    # keys are accepted only for rows that still have no candidate and only when
    # the source residue type agrees with the target residue type.
    for idx, row in out.iterrows():
        if int(idx) in candidates:
            continue
        source_key = str(row.get("source_residue_key") or "").strip()
        source_chain = str(row.get("source_chain") or "_").strip() or "_"
        if source_chain in sequence_checked_chains:
            continue
        if source_key and source_key in target_keys:
            target = next((r for rows in target_by_chain.values() for r in rows if r["residue_key"] == source_key), None)
            if target is not None and str(row.get("resname") or "UNK").upper()[:3] == target["resname"]:
                candidates[int(idx)] = (source_key, "exact_key_fallback", 1.0)

    # Never permit one target residue to receive multiple ConSurf rows.  Mark
    # every member of a collision as ``conflict`` and leave it unmapped rather
    # than allowing row order to decide which score survives.
    target_to_rows: Dict[str, List[int]] = {}
    for idx, (target_key, _method, _identity) in candidates.items():
        target_to_rows.setdefault(target_key, []).append(idx)
    conflicts = {idx for rows in target_to_rows.values() if len(rows) > 1 for idx in rows}
    if conflicts:
        for idx in conflicts:
            candidates.pop(idx, None)
            out.at[idx, "consurf_mapping_status"] = "conflict"
        warnings.append(
            f"Rejected {len(conflicts)} ConSurf rows involved in non-unique target-residue mappings."
        )

    for idx, (target_key, method, identity) in candidates.items():
        target = next((r for rows in target_by_chain.values() for r in rows if r["residue_key"] == target_key), None)
        if target is None:
            continue
        out.at[idx, "residue_key"] = target["residue_key"]
        out.at[idx, "chain"] = target["chain"]
        out.at[idx, "resseq"] = target["resseq"]
        out.at[idx, "icode"] = target["icode"]
        out.at[idx, "resname"] = target["resname"]
        out.at[idx, "consurf_mapping_status"] = "mapped"
        out.at[idx, "consurf_mapping_method"] = method
        out.at[idx, "consurf_mapping_identity"] = identity

    mapped_keys = set(out.loc[out["consurf_mapping_status"] == "mapped", "residue_key"].dropna().astype(str))
    # Unmapped or conflicted rows must not retain their source number; keeping
    # it would recreate the original false match when an AF2 model starts at
    # residue 1.
    out.loc[out["consurf_mapping_status"] != "mapped", "residue_key"] = pd.NA
    target_coverage = len(mapped_keys) / float(max(len(target_keys), 1))
    source_coverage = len(mapped_keys) / float(max(len(out), 1))
    audit = {
        "enabled": True,
        "status": "mapped" if mapped_keys else "no_mapping",
        "n_source_rows": int(len(out)),
        "n_target_residues": int(len(target_keys)),
        "n_mapped_rows": int((out["consurf_mapping_status"] == "mapped").sum()),
        "n_unmapped_rows": int((out["consurf_mapping_status"] != "mapped").sum()),
        "n_mapped_target_residues": int(len(mapped_keys)),
        "n_conflict_rows": int(len(conflicts)),
        "target_residue_coverage": float(target_coverage),
        "source_row_coverage": float(source_coverage),
        "mapping_methods": out.loc[out["consurf_mapping_status"] == "mapped", "consurf_mapping_method"].value_counts().to_dict(),
        "chain_map": {str(k): str(v) for k, v in chain_map.items()},
        "residue_number_offset": offset_cfg,
    }
    if not mapped_keys:
        warnings.append("No ConSurf rows could be mapped to the analysed complex numbering.")
    return out, audit, warnings


def assess_consurf_mapping_against_complexes(
    consurf_df: pd.DataFrame, records: pd.DataFrame
) -> Tuple[dict, List[str]]:
    """Audit whether ConSurf residue labels overlap the analysed complex PDBs.

    Website exports should normally match by exact PDB residue key.  A zero or
    very low overlap is scientifically more important than a parser exception:
    it usually means ConSurf was run on a differently numbered construct.
    """
    warnings: List[str] = []
    protein_keys: set[str] = set()
    inspected_path = ""
    for value in records.get("pdb_path", pd.Series(dtype=str)).dropna().astype(str):
        pdb_path = Path(value)
        if not pdb_path.is_file():
            continue
        protein_keys = {atom.residue_key for atom in parse_pdb_atoms(pdb_path) if atom.record_name == "ATOM"}
        inspected_path = str(pdb_path)
        if protein_keys:
            break
    consurf_keys = set(consurf_df.get("residue_key", pd.Series(dtype=str)).dropna().astype(str))
    exact_matches = protein_keys & consurf_keys
    protein_bases = {residue_key_base(key) for key in protein_keys}
    consurf_bases = {residue_key_base(key) for key in consurf_keys}
    base_matches = protein_bases & consurf_bases
    protein_coverage = len(exact_matches) / float(max(len(protein_keys), 1))
    consurf_coverage = len(exact_matches) / float(max(len(consurf_keys), 1))
    audit = {
        "inspected_complex_pdb": inspected_path,
        "n_protein_residues_in_inspected_complex": int(len(protein_keys)),
        "n_consurf_residues": int(len(consurf_keys)),
        "n_exact_residue_key_matches": int(len(exact_matches)),
        "n_chain_insensitive_matches": int(len(base_matches)),
        "protein_residue_coverage_exact": protein_coverage,
        "consurf_residue_coverage_exact": consurf_coverage,
    }
    if "consurf_mapping_status" in consurf_df.columns:
        status = consurf_df["consurf_mapping_status"].astype(str)
        mapped_rows = int((status == "mapped").sum())
        audit.update(
            {
                "n_consurf_rows_mapped": mapped_rows,
                "n_consurf_rows_unmapped": int(len(consurf_df) - mapped_rows),
                "mapping_methods": consurf_df.loc[status == "mapped", "consurf_mapping_method"].value_counts().to_dict()
                if "consurf_mapping_method" in consurf_df.columns else {},
            }
        )
    if protein_keys and consurf_keys and not exact_matches:
        if base_matches:
            warnings.append(
                "ConSurf residue numbers/residue names match the complex only after removing chain IDs. "
                "Check that the ConSurf job and complex PDB use the same chain labels."
            )
        else:
            warnings.append(
                "No ConSurf residue labels match the analysed complex PDB. Check that ConSurf was run on "
                "the same construct and that AF2 preprocessing preserved original residue numbering."
            )
    elif protein_keys and consurf_keys and protein_coverage < 0.25:
        warnings.append(
            f"Only {protein_coverage:.1%} of residues in the inspected complex have an exact ConSurf match. "
            "Review construct boundaries, chain IDs, and residue numbering before interpreting key residues."
        )
    return audit, warnings


def enforce_consurf_contact_coverage(
    consurf_df: pd.DataFrame,
    contact_df: pd.DataFrame,
    mapping_cfg: dict,
) -> dict:
    """Fail closed when ligand-neighbour residues lack ConSurf scores.

    The coverage is measured on residues actually observed at the configured
    distance cutoff, not on the whole source PDB.  This allows a full-length
    ConSurf job to be compared with a deliberately truncated AF2 domain while
    still preventing a false "all residues are non-conserved" conclusion.
    """
    mapping = _mapping_cfg(mapping_cfg)
    enabled = bool(mapping.get("enabled", True))
    fail = bool(mapping.get("fail_on_low_contact_coverage", True))
    minimum = float(mapping.get("min_contact_coverage", 0.90))
    if contact_df.empty:
        return {
            "enabled": enabled,
            "n_contact_residues": 0,
            "n_mapped_contact_residues": 0,
            "contact_residue_coverage": None,
            "minimum_required": minimum,
            "status": "not_assessed",
        }
    contact = contact_df.copy()
    if "within_distance_cutoff" in contact.columns:
        nearby = contact[contact["within_distance_cutoff"] == True]  # noqa: E712
        if not nearby.empty:
            contact = nearby
    contact_keys = set(contact["residue_key"].dropna().astype(str))
    if "consurf_mapping_status" in consurf_df.columns and enabled:
        mapped_keys = set(
            consurf_df.loc[consurf_df["consurf_mapping_status"] == "mapped", "residue_key"]
            .dropna().astype(str)
        )
    else:
        mapped_keys = set(consurf_df["residue_key"].dropna().astype(str))
    covered = contact_keys & mapped_keys
    coverage = len(covered) / float(max(len(contact_keys), 1))
    audit = {
        "enabled": enabled,
        "n_contact_residues": int(len(contact_keys)),
        "n_mapped_contact_residues": int(len(covered)),
        "n_missing_contact_residues": int(len(contact_keys - mapped_keys)),
        "contact_residue_coverage": float(coverage),
        "minimum_required": minimum,
        "status": "passed" if coverage >= minimum else "failed",
        "missing_contact_residues": sorted(contact_keys - mapped_keys),
    }
    if coverage < minimum and fail:
        raise RuntimeError(
            "ConSurf mapping coverage is below the safety threshold for ligand-neighbour residues: "
            f"{coverage:.1%} < {minimum:.1%}. This is a numbering/construct mapping problem, "
            "not evidence that the residues are non-conserved. Review the Step07 mapping audit "
            "or set step07.conservation.mapping.residue_number_offset/chain_map explicitly."
        )
    return audit


def find_auto_complex_table(output_root: Path, target_id: str) -> Path:
    candidates = [
        # Prefer the current Step05 QC output. This keeps Step07, Step08, and
        # downstream GED analysis on the same post-QC ensemble used by Step06.
        output_root / "04_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step04_passed_complex_index.csv",
        output_root / "04_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        # Historical output location.
        output_root / "05_catalytic_region_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        # Backward-compatible legacy Step05 directory names.
        output_root / "05_pocket_plddt_qc" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        output_root / "05_pocket_plddt_filter" / "tables" / f"{target_id}_step05_passed_complex_index.csv",
        output_root / "04_ligand_atom_trimming" / "tables" / f"{target_id}_step04_trimmed_complex_index.csv",
        output_root / "04_ligand_atom_trimming" / "tables" / f"{target_id}_step04_complex_index.csv",
        output_root / "03_clash_filter" / "tables" / f"{target_id}_step03_passed_complex_index.csv",
        output_root / "02_template_guided_docking" / "tables" / f"{target_id}_step02_complex_index.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate an input complex table automatically. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def choose_pdb_column(df: pd.DataFrame, requested: str = "auto") -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"Requested PDB column '{requested}' not present in input table.")
        return requested
    preferred = [
        "trimmed_complex_pdb_path", "trimmed_complex_path", "complex_pdb_path",
        "complex_path", "pdb_path", "structure_path", "standardized_pdb_path",
    ]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if "pdb" in c.lower() or "complex" in c.lower() or "path" in c.lower():
            return c
    raise KeyError("Could not auto-detect PDB path column in input complex table.")


def choose_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["structure_id", "complex_id", "model_id", "id", "name"]:
        if c in df.columns:
            return c
    return None


def load_complex_records(config: dict, root: Path, output_root: Path, target_id: str, step_cfg: dict) -> Tuple[pd.DataFrame, Path, str]:
    requested_table = step_cfg.get("input_complex_table", "auto")
    if requested_table == "auto":
        table_path = find_auto_complex_table(output_root, target_id)
    else:
        table_path = resolve_path(requested_table, root)
        if table_path is None:
            raise FileNotFoundError("step07.input_complex_table is empty.")
    df = pd.read_csv(table_path)
    pdb_col = choose_pdb_column(df, step_cfg.get("input_complex_column", "auto"))
    id_col = choose_id_column(df)
    rows = []
    for idx, row in df.iterrows():
        pdb_path = resolve_path(str(row[pdb_col]), root)
        if pdb_path is None:
            continue
        sid = str(row[id_col]) if id_col else pdb_path.stem
        rows.append({"structure_id": sid, "pdb_path": str(pdb_path), "source_row_index": idx})
    records = pd.DataFrame(rows)
    if records.empty:
        raise ValueError(f"No complex PDB records found in {table_path} using column {pdb_col}.")
    return records, table_path, pdb_col


def parse_residue_label(label: str) -> Optional[Tuple[str, int, str, str]]:
    text = label.strip()
    # Preferred labels from Step06 debug files: LYS13A, ARG90A, ASP9A.
    m = re.fullmatch(r"([A-Z]{3})(-?\d+)([A-Za-z_]?)([A-Za-z]?)", text)
    if m:
        resname, resseq, chain, icode = m.group(1), int(m.group(2)), m.group(3) or "_", m.group(4) or ""
        return resname, resseq, chain, icode
    # More explicit variants: LYS_A_13 or A:LYS:13.
    m = re.fullmatch(r"([A-Z]{3})[_: -]([A-Za-z_])[_: -](-?\d+)([A-Za-z]?)", text)
    if m:
        return m.group(1), int(m.group(3)), m.group(2) or "_", m.group(4) or ""
    m = re.fullmatch(r"([A-Za-z_])[:_ -]([A-Z]{3})[:_ -](-?\d+)([A-Za-z]?)", text)
    if m:
        return m.group(2), int(m.group(3)), m.group(1) or "_", m.group(4) or ""
    return None


def residue_key_from_label(label: str) -> Optional[str]:
    parsed = parse_residue_label(label)
    if not parsed:
        return None
    resname, resseq, chain, icode = parsed
    return make_residue_key(resname, chain, resseq, icode)


def residue_key_base(label_or_key: str) -> str:
    """Return a chain-insensitive residue key such as CYS215.

    PLIP text files sometimes report protein residues without a chain suffix
    (CYS215) while ConSurf/distance tables use CYS215A.  The base key allows
    robust matching without changing the fully traceable residue_key output.
    """
    try:
        if label_or_key is None or bool(pd.isna(label_or_key)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(label_or_key or "").strip()
    parsed = parse_residue_label(text)
    if parsed:
        resname, resseq, _chain, icode = parsed
        return f"{resname}{int(resseq)}{icode or ''}"
    m = re.match(r"^([A-Z]{3})(-?\d+)([A-Za-z]?)", text)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return text


def key_set_present_by_alias(key_set: set[str], observed: Iterable[str]) -> set[str]:
    """Return members of key_set matched by exact key or chain-insensitive key."""
    observed_set = {str(x) for x in observed}
    observed_base = {residue_key_base(x) for x in observed_set}
    present: set[str] = set()
    for key in key_set:
        if key in observed_set or residue_key_base(key) in observed_base:
            present.add(key)
    return present


def structure_id_from_interaction_path(path: Path) -> str:
    stem = path.stem
    for suffix in ["_interaction_output", "_interactions", "_interaction"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem == "interaction_output" and path.parent.name:
        return path.parent.name
    return stem


def find_interaction_output_files(output_root: Path, target_id: str, step_cfg: dict) -> List[Path]:
    explicit = step_cfg.get("interaction_output_dir") or step_cfg.get("step06_interaction_output_dir")
    base = resolve_path(explicit, Path.cwd()) if explicit and explicit != "auto" else output_root / "05_plip_interaction_graphs"
    if (not base or not base.exists()) and not explicit:
        base = output_root / "06_plip_interaction_graphs"
    if not base or not base.exists():
        return []
    patterns = [
        "**/*interaction_output*.txt",
        "**/*interactions*.txt",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(base.glob(pat))
    # Avoid parsing final GED graph files as interaction debug files.
    files = [p for p in files if "ged_input" not in str(p).lower() and p.is_file()]
    unique = sorted({p.resolve() for p in files})
    return unique


def parse_interaction_output_file(path: Path) -> Tuple[str, List[str]]:
    residues = set()
    sid = structure_id_from_interaction_path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # Legacy/debug line: Type: ... Protein: LYS13A ...
            m = re.search(r"Protein:\s*([^\t\s]+)", line)
            if m:
                raw_label = m.group(1)
                key = residue_key_from_label(raw_label)
                if key:
                    residues.add(key)
                    residues.add(residue_key_base(key))
                else:
                    residues.add(residue_key_base(raw_label))
                continue
            # TSV/CSV-like fallback: scan all residue-like tokens.
            for token in re.findall(r"\b[A-Z]{3}-?\d+[A-Za-z_]?\b", line):
                key = residue_key_from_label(token)
                if key:
                    residues.add(key)
                    residues.add(residue_key_base(key))
                else:
                    residues.add(residue_key_base(token))
    return sid, sorted(residues)


def collect_graph_occurrence(output_root: Path, target_id: str, step_cfg: dict) -> Tuple[Dict[str, set], List[str]]:
    files = find_interaction_output_files(output_root, target_id, step_cfg)
    by_structure: Dict[str, set] = {}
    warnings: List[str] = []
    for f in files:
        sid, residues = parse_interaction_output_file(f)
        if residues:
            by_structure.setdefault(sid, set()).update(residues)
    if not by_structure:
        warnings.append(
            "No Step06 interaction_output files with residue labels were found. "
            "graph_occurrence_frequency will be reported as NA."
        )
    return by_structure, warnings


def residue_label_to_base(label: object) -> str:
    """Convert residue labels to chain-insensitive AA3+number labels.

    Accepts PLIP labels such as ASN317A, internal keys such as ASN317A, and
    manuscript labels such as N317.  The returned value is chain-insensitive,
    e.g. ASN317, which is the safest key for comparing AF2, ConSurf, and PLIP
    outputs when all structures use a single protein chain.
    """
    try:
        if label is None or bool(pd.isna(label)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(label or "").strip()
    if not text:
        return ""
    parsed = parse_residue_label(text)
    if parsed:
        resname, resseq, _chain, icode = parsed
        return f"{resname}{int(resseq)}{icode or ''}"
    m = re.fullmatch(r"([A-Z])(-?\d+)([A-Za-z]?)", text.upper())
    if m:
        resname = AA1_TO_AA3.get(m.group(1), "UNK")
        return f"{resname}{int(m.group(2))}{m.group(3) or ''}"
    m = re.match(r"^([A-Z]{3})(-?\d+)([A-Za-z]?)", text.upper())
    if m:
        return f"{m.group(1)}{int(m.group(2))}{m.group(3) or ''}"
    return text


def residue_display_label(label: object, display_format: str = "one_letter_number") -> str:
    base = residue_label_to_base(label)
    parsed = parse_residue_label(base)
    if not parsed:
        try:
            if label is None or bool(pd.isna(label)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(label or "").strip()
    resname, resseq, _chain, icode = parsed
    if str(display_format or "one_letter_number").lower() in {"aa3", "three_letter", "three_letter_number"}:
        return f"{resname}{int(resseq)}{icode or ''}"
    aa = AA3_TO_AA1.get(resname.upper(), resname[:1].upper())
    return f"{aa}{int(resseq)}{icode or ''}"


def _ordered_unique_residue_bases(labels: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for label in labels:
        base = residue_label_to_base(label)
        if base and base not in seen:
            seen.add(base)
            out.append(base)
    return out


def find_step06_interaction_edges_table(output_root: Path, target_id: str, step_cfg: dict) -> Optional[Path]:
    explicit = step_cfg.get("interaction_edges_table") or step_cfg.get("step06_interaction_edges_table")
    if explicit and str(explicit).strip().lower() not in {"auto", "none", "null"}:
        p = resolve_path(str(explicit), Path.cwd())
        if p and p.exists():
            return p
    candidates = [
        output_root / "05_plip_interaction_graphs" / "tables" / f"{target_id}_step05_interaction_edges.csv",
        output_root / "05_plip_interaction_graphs" / "tables" / f"{target_id}_step06_interaction_edges.csv",
        output_root / "06_plip_interaction_graphs" / "tables" / f"{target_id}_step06_interaction_edges.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_interaction_output_file_ordered(path: Path) -> Tuple[str, List[str]]:
    """Parse a Step06 interaction_output.txt while preserving first-seen residue order."""
    sid = structure_id_from_interaction_path(path)
    residues: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = re.search(r"Protein:\s*([^\t\s]+)", line)
            if m:
                residues.append(m.group(1))
                continue
            for token in re.findall(r"\b[A-Z]{3}-?\d+[A-Za-z_]?\b", line):
                residues.append(token)
    return sid, _ordered_unique_residue_bases(residues)


def collect_graph_contact_nodes_ordered(output_root: Path, target_id: str, step_cfg: dict) -> Tuple[Dict[str, List[str]], str, List[str]]:
    """Return PLIP substrate-contact residue nodes for each structure.

    This powers the manuscript-style node-level grouping table (Table S3/S9/S13
    style).  The preferred source is the Step06 interaction edge table because it
    preserves the PLIP row order and is less fragile than reparsing text files.
    If that table is unavailable, the function falls back to per-structure
    interaction_output.txt files.
    """
    warnings: List[str] = []
    table = find_step06_interaction_edges_table(output_root, target_id, step_cfg)
    if table is not None:
        try:
            df = pd.read_csv(table)
            if {"standard_structure_id", "protein_node"}.issubset(df.columns):
                by_structure: Dict[str, List[str]] = {}
                for sid, g in df.groupby("standard_structure_id", sort=False):
                    by_structure[str(sid)] = _ordered_unique_residue_bases(g["protein_node"].dropna().astype(str).tolist())
                if by_structure:
                    return by_structure, str(table), warnings
        except Exception as exc:
            warnings.append(f"Could not parse Step06 interaction edge table for node grouping: {table}; {exc}")

    files = find_interaction_output_files(output_root, target_id, step_cfg)
    by_structure = {}
    for f in files:
        sid, residues = parse_interaction_output_file_ordered(f)
        if residues:
            by_structure.setdefault(sid, residues)
    if not by_structure:
        warnings.append(
            "No ordered Step06 PLIP contact nodes were found. The node-level grouping table will be empty unless distance fallback is enabled."
        )
    return by_structure, "interaction_output_files", warnings


def _nodes_for_structure(graph_nodes_ordered: Mapping[str, List[str]], structure_id: str, pdb_path: str) -> List[str]:
    candidates = [
        str(structure_id),
        Path(str(pdb_path)).stem,
        normalize_structure_id(structure_id),
        normalize_structure_id(pdb_path),
    ]
    normalized: Dict[str, List[str]] = {}
    for sid, nodes in graph_nodes_ordered.items():
        normalized.setdefault(str(sid), list(nodes))
        normalized.setdefault(normalize_structure_id(sid), list(nodes))
    for c in candidates:
        if c in normalized:
            return list(normalized[c])
    return []


def _distance_nodes_by_structure(contact_df: pd.DataFrame) -> Dict[str, List[str]]:
    if contact_df.empty:
        return {}
    out: Dict[str, List[str]] = {}
    sub = contact_df[contact_df["within_distance_cutoff"] == True].copy()  # noqa: E712
    if sub.empty:
        return out
    sub["_base"] = sub["residue_key"].map(residue_label_to_base)
    # Distance-derived nodes do not have PLIP row order, so use sequence order.
    sub["_resseq"] = pd.to_numeric(sub.get("resseq", pd.Series([0] * len(sub))), errors="coerce").fillna(0).astype(int)
    for sid, g in sub.sort_values(["_resseq", "_base"]).groupby("structure_id", sort=False):
        out[str(sid)] = _ordered_unique_residue_bases(g["_base"].tolist())
    return out


def _manual_required_residues(step_cfg: dict) -> List[str]:
    """Return manually supplied required nodes for advanced/external use only.

    The normal CatConGraph manuscript workflow must not know the key residues
    before Step07.  Therefore node grouping uses the Step07-discovered
    key-residue table by default.  Manual residues are honored only when the
    user explicitly sets required_residue_source/source to manual/config.
    """
    grouping_cfg = step_cfg.get("node_grouping", {}) or {}
    raw = grouping_cfg.get("manual_required_residues") or grouping_cfg.get("required_residues") or grouping_cfg.get("required_nodes")
    if not raw:
        return []
    if isinstance(raw, str):
        raw = re.split(r"[,;\s]+", raw)
    return [r for r in _ordered_unique_residue_bases(raw) if r]


def _required_residue_source(step_cfg: dict) -> str:
    grouping_cfg = step_cfg.get("node_grouping", {}) or {}
    source = grouping_cfg.get("required_residue_source", grouping_cfg.get("source", "auto"))
    source = str(source or "auto").lower()
    if source in {"automatic", "step07", "derived", "discovered"}:
        source = "auto"
    if source in {"configured", "config", "manual_override"}:
        source = "manual"
    if source not in {"auto", "manual"}:
        raise ValueError("step07.node_grouping.required_residue_source must be auto or manual")
    return source


def _ordered_required_residues(key_df: pd.DataFrame, step_cfg: dict) -> Tuple[List[str], str]:
    source = _required_residue_source(step_cfg)
    if source == "manual":
        return _manual_required_residues(step_cfg), "manual_config"
    if key_df.empty or "residue_key" not in key_df.columns:
        return [], "auto_step06_key_residues"
    return _ordered_unique_residue_bases(key_df["residue_key"].dropna().astype(str).tolist()), "auto_step06_key_residues"


def _order_group_nodes(nodes: Iterable[str], grouping_cfg: dict) -> List[str]:
    nodes = _ordered_unique_residue_bases(nodes)
    # Publication default: residue number ascending (Y46, D48, F182, ...).
    # first_seen remains available as an explicit compatibility option.
    mode = str(grouping_cfg.get("residue_order", "sequence") or "sequence").lower()
    manual = grouping_cfg.get("residue_order_list") or grouping_cfg.get("publication_residue_order") or []
    manual_bases = _ordered_unique_residue_bases(manual if not isinstance(manual, str) else re.split(r"[,;\s]+", manual))
    if mode in {"manual", "publication"} and manual_bases:
        priority = {b: i for i, b in enumerate(manual_bases)}
        return sorted(nodes, key=lambda b: (priority.get(b, 10**6), _residue_number_for_sort(b), b))
    if mode in {"sequence", "resseq", "numeric"}:
        return sorted(nodes, key=lambda b: (_residue_number_for_sort(b), b))
    return nodes


def _residue_number_for_sort(label: object) -> int:
    parsed = parse_residue_label(residue_label_to_base(label))
    return int(parsed[1]) if parsed else 10**9


def _natural_text_key(value: object) -> list:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]


def build_node_level_grouping_table(
    records: pd.DataFrame,
    contact_df: pd.DataFrame,
    key_df: pd.DataFrame,
    graph_nodes_ordered: Mapping[str, List[str]],
    step_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build Table S3/S9/S13-style node-level grouping outputs."""
    grouping_cfg = step_cfg.get("node_grouping", {}) or {}
    contact_definition = str(grouping_cfg.get("contact_definition") or (step_cfg.get("structure_filter", {}) or {}).get("contact_definition", "graph")).lower()
    if contact_definition not in {"graph", "distance", "graph_or_distance"}:
        raise ValueError("step07.node_grouping.contact_definition must be graph, distance, or graph_or_distance")
    display_format = str(grouping_cfg.get("display_format", "one_letter_number") or "one_letter_number")
    required, required_source = _ordered_required_residues(key_df, step_cfg)
    required_set = set(required)
    distance_nodes = _distance_nodes_by_structure(contact_df)

    per_structure_rows: List[dict] = []
    grouped: Dict[Tuple[str, ...], dict] = {}
    for rec in records.itertuples():
        sid = str(rec.structure_id)
        pdb_path = str(rec.pdb_path)
        graph_nodes = _nodes_for_structure(graph_nodes_ordered, sid, pdb_path)
        dist_nodes = []
        for c in {sid, normalize_structure_id(sid), Path(pdb_path).stem, normalize_structure_id(pdb_path)}:
            dist_nodes.extend(distance_nodes.get(str(c), []))
        dist_nodes = _ordered_unique_residue_bases(dist_nodes)

        if contact_definition == "distance":
            nodes = dist_nodes
        elif contact_definition == "graph_or_distance":
            nodes = graph_nodes or dist_nodes
        else:
            nodes = graph_nodes
        nodes = _order_group_nodes(nodes, grouping_cfg)
        node_set = set(nodes)
        missing = sorted(required_set - node_set, key=lambda b: (_residue_number_for_sort(b), b))
        screen = "Retained" if not missing else "Removed"
        key = tuple(sorted(node_set))
        if key not in grouped:
            grouped[key] = {
                "nodes": nodes,
                "screen": screen,
                "missing": missing,
                "structure_ids": [],
                "representative_structure_id": sid,
            }
        grouped[key]["structure_ids"].append(sid)
        per_structure_rows.append({
            "structure_id": sid,
            "pdb_path": pdb_path,
            "contact_definition": contact_definition,
            "residue_nodes": "; ".join(residue_display_label(n, display_format) for n in nodes),
            "required_residues": "; ".join(residue_display_label(n, display_format) for n in required),
            "required_residue_source": required_source,
            "missing_required_residues": "; ".join(residue_display_label(n, display_format) for n in missing),
            "screen": screen,
        })

    # Publication default: Type1 has the largest residue-node set. Ties use
    # structure count and then natural representative ID for stable output.
    order_mode = str(grouping_cfg.get("type_order", "residue_count_desc") or "residue_count_desc").lower()
    items = list(grouped.values())
    if order_mode in {"residue_count_desc", "node_count_desc", "nodes_desc", "residues_desc"}:
        items.sort(
            key=lambda item: (
                -len(item["nodes"]),
                -len(item["structure_ids"]),
                _natural_text_key(item["representative_structure_id"]),
            )
        )
    elif order_mode in {"count_desc", "size_desc", "structures_desc"}:
        items.sort(key=lambda item: (-len(item["structure_ids"]), item["representative_structure_id"]))

    rows: List[dict] = []
    for idx, item in enumerate(items, start=1):
        nodes = item["nodes"]
        missing = item["missing"]
        rows.append({
            "Type": f"Type{idx}",
            "Residue": ", ".join(residue_display_label(n, display_format) for n in nodes),
            "Screen": item["screen"],
            "Structures": len(item["structure_ids"]),
            "Required residues": "; ".join(residue_display_label(n, display_format) for n in required),
            "Required residue source": required_source,
            "Missing required residues": "; ".join(residue_display_label(n, display_format) for n in missing),
            "Representative structure": item["representative_structure_id"],
            "Structure IDs": ";".join(item["structure_ids"]),
            "Contact definition": contact_definition,
        })
    return pd.DataFrame(rows), pd.DataFrame(per_structure_rows)


def write_node_level_grouping_outputs(
    table_df: pd.DataFrame,
    per_structure_df: pd.DataFrame,
    *,
    target_id: str,
    tables_dir: Path,
    reports_dir: Path,
) -> Dict[str, str]:
    full_csv = tables_dir / f"{target_id}_step06_node_level_grouping.csv"
    pub_csv = tables_dir / f"{target_id}_step06_node_level_grouping_publication.csv"
    per_structure_csv = tables_dir / f"{target_id}_step06_node_level_grouping_per_structure.csv"
    html_path = reports_dir / f"{target_id}_step06_node_level_grouping.html"
    md_path = reports_dir / f"{target_id}_step06_node_level_grouping.md"

    table_df.to_csv(full_csv, index=False)
    pub_cols = [c for c in ["Type", "Residue", "Screen", "Structures"] if c in table_df.columns]
    table_df[pub_cols].to_csv(pub_csv, index=False)
    per_structure_df.to_csv(per_structure_csv, index=False)

    lines = ["# Step07 node-level grouping", "", "| Type | Residue | Screen | Structures |", "|---|---|---|---|"]
    for _, row in table_df.iterrows():
        lines.append(f"| {row.get('Type', '')} | {row.get('Residue', '')} | {row.get('Screen', '')} | {row.get('Structures', '')} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_rows = []
    for _, row in table_df.iterrows():
        removed = str(row.get("Screen", "")).lower() == "removed"
        style = ' style="color:#0070C0;"' if removed else ""
        cells = "".join(f"<td{style}>{html.escape(str(row.get(c, '')))}</td>" for c in pub_cols)
        html_rows.append(f"<tr>{cells}</tr>")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in pub_cols)
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(target_id)} Step07 node-level grouping</title>
<style>body{{font-family:Arial, sans-serif;}} table{{border-collapse:collapse;}} th,td{{border:1px solid #333;padding:6px 10px;vertical-align:top;}}</style></head>
<body><h1>{html.escape(target_id)} Step07 node-level grouping</h1>
<p>Required high-frequency nodes are derived automatically from the Step07 key-residue table unless an advanced manual override was explicitly requested. Rows in blue lack one or more required nodes and should be excluded from edge-level classification.</p>
<table><thead><tr>{head}</tr></thead><tbody>{''.join(html_rows)}</tbody></table></body></html>"""
    html_path.write_text(html_text, encoding="utf-8")

    return {
        "node_level_grouping": str(full_csv),
        "node_level_grouping_publication": str(pub_csv),
        "node_level_grouping_per_structure": str(per_structure_csv),
        "node_level_grouping_html": str(html_path),
        "node_level_grouping_markdown": str(md_path),
    }


def compute_distance_contacts(
    records: pd.DataFrame,
    ligand_cfg: dict,
    distance_cfg: dict,
    max_structures: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    rows: List[dict] = []
    used = records.head(max_structures) if max_structures else records
    for rec in used.itertuples():
        pdb_path = Path(rec.pdb_path)
        if not pdb_path.exists():
            warnings.append(f"PDB not found: {pdb_path}")
            continue
        atoms = parse_pdb_atoms(pdb_path)
        protein_atoms, ligand_atoms = split_protein_ligand_atoms(atoms, ligand_cfg, distance_cfg)
        if not ligand_atoms:
            warnings.append(f"No ligand/substrate atoms found in {pdb_path}")
            continue
        by_residue: Dict[str, List[AtomRecord]] = {}
        for a in protein_atoms:
            by_residue.setdefault(a.residue_key, []).append(a)
        for residue_key, residue_atoms in by_residue.items():
            d = residue_min_distance(residue_atoms, ligand_atoms)
            first = residue_atoms[0]
            rows.append(
                {
                    "structure_id": rec.structure_id,
                    "pdb_path": str(pdb_path),
                    "residue_key": residue_key,
                    "chain": first.chain,
                    "resseq": first.resseq,
                    "icode": first.icode,
                    "resname": first.resname,
                    "min_distance_to_ligand": d,
                    "within_distance_cutoff": d <= float(distance_cfg.get("residue_ligand_distance_max", 3.0)),
                }
            )
    return pd.DataFrame(rows), warnings


def summarize_residues(
    contact_df: pd.DataFrame,
    consurf_df: pd.DataFrame,
    graph_occurrence: Dict[str, set],
    thresholds: dict,
    n_structures_total: int,
) -> pd.DataFrame:
    cons = consurf_df.copy()
    cons_columns = [
        "residue_key", "chain", "resseq", "icode", "resname", "consurf_grade", "consurf_score",
        "consurf_mapping_status", "consurf_mapping_method", "source_residue_key",
    ]
    for column in cons_columns:
        if column not in cons.columns:
            cons[column] = pd.NA
    cons_small = cons[cons_columns].drop_duplicates("residue_key")
    if contact_df.empty:
        summary = cons_small.copy()
        summary["n_structures_with_distance"] = 0
        summary["distance_contact_count"] = 0
        summary["distance_contact_frequency"] = math.nan
        summary["min_distance_min"] = math.nan
        summary["min_distance_median"] = math.nan
        summary["min_distance_mean"] = math.nan
    else:
        grouped = contact_df.groupby("residue_key", dropna=False)
        dist = grouped.agg(
            n_structures_with_distance=("structure_id", "nunique"),
            distance_contact_count=("within_distance_cutoff", "sum"),
            min_distance_min=("min_distance_to_ligand", "min"),
            min_distance_median=("min_distance_to_ligand", "median"),
            min_distance_mean=("min_distance_to_ligand", "mean"),
        ).reset_index()
        dist["distance_contact_frequency"] = dist["distance_contact_count"] / float(max(n_structures_total, 1))
        # Keep residues seen in PDB even if ConSurf is missing, but ConSurf is required for key_residue.
        residue_meta = contact_df[["residue_key", "chain", "resseq", "icode", "resname"]].drop_duplicates("residue_key")
        summary = residue_meta.merge(dist, on="residue_key", how="left").merge(
            cons_small[["residue_key", "consurf_grade", "consurf_score"]], on="residue_key", how="left"
        )
        # Include conserved residues not present in current PDB records as diagnostics.
        missing_from_pdb = cons_small[~cons_small["residue_key"].isin(summary["residue_key"])]
        if not missing_from_pdb.empty:
            filler = missing_from_pdb.copy()
            filler["n_structures_with_distance"] = 0
            filler["distance_contact_count"] = 0
            filler["distance_contact_frequency"] = 0.0
            filler["min_distance_min"] = math.nan
            filler["min_distance_median"] = math.nan
            filler["min_distance_mean"] = math.nan
            summary = pd.concat([summary, filler], ignore_index=True, sort=False)

    # Graph occurrence from Step06 debug interaction files.
    if graph_occurrence:
        graph_count: Dict[str, int] = {}
        for _sid, residues in graph_occurrence.items():
            bases_seen_in_structure = {residue_key_base(r) for r in residues}
            # Count each residue once per structure, even if both exact and base
            # aliases were parsed from the same interaction_output.txt.
            for r in bases_seen_in_structure:
                graph_count[r] = graph_count.get(r, 0) + 1
        summary["graph_occurrence_count"] = summary["residue_key"].map(lambda k: graph_count.get(residue_key_base(k), 0)).astype(int)
        summary["graph_occurrence_frequency"] = summary["graph_occurrence_count"] / float(max(n_structures_total, 1))
    else:
        summary["graph_occurrence_count"] = pd.NA
        summary["graph_occurrence_frequency"] = pd.NA

    cons_min = float(thresholds.get("consurf_grade_min", 8))
    dist_freq_min = float(thresholds.get("distance_contact_frequency_min", thresholds.get("occurrence_frequency_min", 0.80)))
    graph_freq_min = thresholds.get("graph_occurrence_frequency_min", None)
    if graph_freq_min is None:
        graph_freq_min = thresholds.get("occurrence_frequency_min", 0.80)
    graph_freq_min = float(graph_freq_min)
    # Key-residue discovery is deliberately based on conservation + 3 Å
    # distance-frequency by default.  PLIP/graph occurrence is useful as an
    # audit column, but requiring it here can miss true catalytic residues when
    # PLIP omits a chain suffix (for example CYS215 vs CYS215A).  Structure
    # filtering below uses PLIP interactions by default.
    # A key residue is defined by four concurrent criteria: conservation,
    # geometric contact recurrence, PLIP contact recurrence, and availability
    # in the analysed ensemble.  Do not silently relax the PLIP criterion for
    # a project merely because a checkbox was left unticked.
    require_graph = True

    summary["conservation_pass"] = pd.to_numeric(summary["consurf_grade"], errors="coerce") >= cons_min
    summary["distance_frequency_pass"] = pd.to_numeric(summary["distance_contact_frequency"], errors="coerce") >= dist_freq_min
    if graph_occurrence:
        summary["graph_frequency_pass"] = pd.to_numeric(summary["graph_occurrence_frequency"], errors="coerce") >= graph_freq_min
    else:
        summary["graph_frequency_pass"] = pd.NA

    if require_graph and graph_occurrence:
        summary["key_residue"] = summary["conservation_pass"] & summary["distance_frequency_pass"] & summary["graph_frequency_pass"]
    elif require_graph and not graph_occurrence:
        summary["key_residue"] = False
    else:
        summary["key_residue"] = summary["conservation_pass"] & summary["distance_frequency_pass"]

    ordered_cols = [
        "residue_key", "chain", "resseq", "icode", "resname", "consurf_grade", "consurf_score",
        "consurf_mapping_status", "consurf_mapping_method", "source_residue_key",
        "n_structures_with_distance", "distance_contact_count", "distance_contact_frequency",
        "min_distance_min", "min_distance_median", "min_distance_mean",
        "graph_occurrence_count", "graph_occurrence_frequency",
        "conservation_pass", "distance_frequency_pass", "graph_frequency_pass", "key_residue",
    ]
    for c in ordered_cols:
        if c not in summary.columns:
            summary[c] = pd.NA
    summary = summary[ordered_cols]
    summary = summary.sort_values(
        by=["key_residue", "consurf_grade", "distance_contact_frequency", "graph_occurrence_frequency", "resseq"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return summary


def _distance_cutoff_tag(distance_cutoff: float) -> str:
    """Return a filesystem-safe cutoff tag such as ``3A`` or ``3p5A``."""
    value = float(distance_cutoff)
    if value.is_integer():
        text = str(int(value))
    else:
        text = (f"{value:.3f}").rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}A"


def _selection_status(row: pd.Series, require_graph: bool) -> str:
    def is_true(value: object) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    if is_true(row.get("key_residue", False)):
        return "Selected key residue"
    failed: List[str] = []
    mapping_status = str(row.get("consurf_mapping_status", "")).strip().lower()
    if mapping_status in {"missing", "unmapped", "nan", "none", "<na>"}:
        failed.append("ConSurf mapping missing")
    elif not is_true(row.get("conservation_pass", False)):
        failed.append("conservation below threshold")
    if not is_true(row.get("distance_frequency_pass", False)):
        failed.append("distance frequency below threshold")
    graph_pass = row.get("graph_frequency_pass", pd.NA)
    if require_graph and not pd.isna(graph_pass) and not is_true(graph_pass):
        failed.append("PLIP graph frequency below threshold")
    return "; ".join(failed) if failed else "Not selected"


def build_distance_cutoff_residue_details(
    summary_df: pd.DataFrame,
    contact_df: pd.DataFrame,
    graph_occurrence: Mapping[str, set],
    thresholds: Mapping[str, object],
    n_structures_total: int,
    distance_cutoff: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build detailed and publication-style tables for residues entering the cutoff.

    Unlike the compact key-residue table, this output keeps *every* residue that
    occurs within the configured ligand-distance cutoff in at least one accepted
    structure.  It exposes conservation, distance-contact frequency, PLIP graph
    frequency, distance statistics, threshold decisions, and the supporting
    structure IDs so users can inspect the complete 3 Å residue population.
    """
    details = summary_df.copy()
    if details.empty:
        return details, details.copy()

    cutoff_contacts = contact_df.copy()
    if not cutoff_contacts.empty and "within_distance_cutoff" in cutoff_contacts.columns:
        cutoff_contacts = cutoff_contacts[cutoff_contacts["within_distance_cutoff"] == True].copy()  # noqa: E712
    else:
        cutoff_contacts = pd.DataFrame(columns=["residue_key", "structure_id"])

    structure_ids_by_residue: Dict[str, str] = {}
    if not cutoff_contacts.empty:
        for residue_key, group in cutoff_contacts.groupby("residue_key", sort=False):
            structure_ids = sorted(
                {str(x) for x in group["structure_id"].dropna().astype(str)},
                key=lambda x: [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", x)],
            )
            structure_ids_by_residue[str(residue_key)] = ";".join(structure_ids)

    graph_structure_ids_by_base: Dict[str, List[str]] = {}
    for sid, residues in graph_occurrence.items():
        for base in {residue_key_base(r) for r in residues}:
            graph_structure_ids_by_base.setdefault(base, []).append(str(sid))
    graph_structure_ids_by_base = {
        base: sorted(set(ids), key=lambda x: [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", x)])
        for base, ids in graph_structure_ids_by_base.items()
    }

    details["residue_label"] = details["residue_key"].map(lambda x: residue_display_label(x, "one_letter_number"))
    details["n_structures_total"] = int(n_structures_total)
    details["distance_cutoff_angstrom"] = float(distance_cutoff)
    details["structure_ids_within_cutoff"] = details["residue_key"].map(structure_ids_by_residue).fillna("")
    details["graph_occurrence_structure_ids"] = details["residue_key"].map(
        lambda x: ";".join(graph_structure_ids_by_base.get(residue_key_base(x), []))
    )

    # Keep the detailed audit table on the same mandatory four-criterion rule
    # as ``summarize_residues`` above.
    require_graph = True
    details["selection_status"] = details.apply(lambda row: _selection_status(row, bool(require_graph)), axis=1)

    details = details[pd.to_numeric(details["distance_contact_count"], errors="coerce").fillna(0) > 0].copy()
    detailed_columns = [
        "residue_key", "residue_label", "chain", "resseq", "icode", "resname",
        "consurf_grade", "consurf_score", "n_structures_total",
        "distance_cutoff_angstrom", "n_structures_with_distance",
        "distance_contact_count", "distance_contact_frequency",
        "structure_ids_within_cutoff", "min_distance_min", "min_distance_median",
        "min_distance_mean", "graph_occurrence_count", "graph_occurrence_frequency",
        "graph_occurrence_structure_ids", "conservation_pass", "distance_frequency_pass",
        "graph_frequency_pass", "key_residue", "selection_status",
    ]
    for column in detailed_columns:
        if column not in details.columns:
            details[column] = pd.NA
    details = details[detailed_columns].sort_values(
        ["key_residue", "distance_contact_frequency", "consurf_grade", "graph_occurrence_frequency", "resseq"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    cutoff_text = f"{float(distance_cutoff):g} Å"
    publication = details.rename(
        columns={
            "residue_label": "Residue",
            "residue_key": "Residue key",
            "chain": "Chain",
            "resseq": "Residue number",
            "resname": "Residue name",
            "consurf_grade": "ConSurf grade",
            "consurf_score": "ConSurf score",
            "distance_contact_count": f"Structures within {cutoff_text}",
            "n_structures_total": "Total structures",
            "distance_contact_frequency": f"Frequency within {cutoff_text}",
            "structure_ids_within_cutoff": f"Structure IDs within {cutoff_text}",
            "graph_occurrence_count": "PLIP graph occurrence count",
            "graph_occurrence_frequency": "PLIP graph occurrence frequency",
            "min_distance_min": "Minimum distance (Å)",
            "min_distance_median": "Median distance (Å)",
            "min_distance_mean": "Mean distance (Å)",
            "conservation_pass": "Conservation pass",
            "distance_frequency_pass": "Distance-frequency pass",
            "graph_frequency_pass": "PLIP-frequency pass",
            "key_residue": "Key residue",
            "selection_status": "Selection status",
        }
    )
    publication_columns = [
        "Residue", "Residue key", "Chain", "Residue number", "Residue name",
        "ConSurf grade", "ConSurf score", f"Structures within {cutoff_text}",
        "Total structures", f"Frequency within {cutoff_text}",
        "PLIP graph occurrence count", "PLIP graph occurrence frequency",
        "Minimum distance (Å)", "Median distance (Å)", "Mean distance (Å)",
        "Conservation pass", "Distance-frequency pass", "PLIP-frequency pass",
        "Key residue", "Selection status", f"Structure IDs within {cutoff_text}",
    ]
    publication = publication[publication_columns]
    return details, publication


def write_distance_cutoff_residue_detail_outputs(
    detailed_df: pd.DataFrame,
    publication_df: pd.DataFrame,
    target_id: str,
    tables_dir: Path,
    reports_dir: Path,
    distance_cutoff: float,
) -> Dict[str, str]:
    tag = _distance_cutoff_tag(distance_cutoff)
    detailed_path = tables_dir / f"{target_id}_step06_within_{tag}_residue_details.csv"
    publication_path = tables_dir / f"{target_id}_step06_within_{tag}_residue_details_publication.csv"
    html_path = reports_dir / f"{target_id}_step06_within_{tag}_residue_details.html"
    detailed_df.to_csv(detailed_path, index=False)
    publication_df.to_csv(publication_path, index=False)

    title = f"{html.escape(target_id)} residues observed within {float(distance_cutoff):g} Å of the ligand"
    table_html = publication_df.to_html(index=False, border=0, classes="residue-detail-table", na_rep="")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}"
        "h1{font-size:20px}p{max-width:980px;line-height:1.45}"
        "table{border-collapse:collapse;font-size:12px;width:100%}"
        "th,td{border:1px solid #d9d9d9;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#f3f5f7;position:sticky;top:0}tbody tr:nth-child(even){background:#fafafa}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        "<p>This table includes every residue observed inside the configured distance cutoff at least once, "
        "not only residues that passed the final key-residue thresholds.</p>"
        f"{table_html}</body></html>",
        encoding="utf-8",
    )
    return {
        "distance_cutoff_residue_details": str(detailed_path),
        "distance_cutoff_residue_details_publication": str(publication_path),
        "distance_cutoff_residue_details_html": str(html_path),
    }


def normalize_structure_id(value: object) -> str:
    """Normalize structure IDs from PDB paths, Step06 graph names, and index tables.

    This keeps Step07/Step08 matching robust when one step uses
    2PBR_AF2_0001, another uses 2PBR_AF2_0001_trimmed_complex, and Step06 GED
    files use 2PBR_AF2_0001_final.txt.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    stem = Path(text).stem
    for suffix in [
        "_final",
        "_trimmed_complex",
        "_filtered_complex",
        "_passed_complex",
        "_complex_complex_complex",
        "_complex_complex",
        "_complex",
        "_model",
    ]:
        changed = True
        while changed and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            changed = True
            break
    return stem


def graph_occurrence_for_structure(
    graph_occurrence: Dict[str, set],
    structure_id: str,
    pdb_path: str,
) -> set:
    if not graph_occurrence:
        return set()
    candidates = {
        str(structure_id),
        Path(str(pdb_path)).stem,
        normalize_structure_id(structure_id),
        normalize_structure_id(pdb_path),
    }
    normalized_to_residues: Dict[str, set] = {}
    for sid, residues in graph_occurrence.items():
        normalized_to_residues.setdefault(normalize_structure_id(sid), set()).update(residues)
        normalized_to_residues.setdefault(str(sid), set()).update(residues)
    out = set()
    for c in candidates:
        if c in graph_occurrence:
            out.update(graph_occurrence[c])
        nc = normalize_structure_id(c)
        if nc in normalized_to_residues:
            out.update(normalized_to_residues[nc])
    return out


def build_structure_key_residue_filter(
    records: pd.DataFrame,
    contact_df: pd.DataFrame,
    key_df: pd.DataFrame,
    graph_occurrence: Dict[str, set],
    step_cfg: dict,
) -> pd.DataFrame:
    """Filter conformations after key residues are identified.

    Default rule:
      - key residues are selected by residue-level conservation + distance frequency;
      - a conformation passes only if all key residues are present in that
        structure's Step06 PLIP interaction output.
    Users can switch to distance/distance_or_graph/distance_and_graph in config.
    """
    filter_cfg = step_cfg.get("structure_filter", {}) or {}
    enabled = bool(filter_cfg.get("enabled", True))
    mode = str(filter_cfg.get("require_key_residues", filter_cfg.get("mode", "all"))).lower()
    # For structure-level screening, require key residues to appear in the PLIP
    # interaction graph by default.  The residue list itself is discovered by
    # conservation + 3 Å distance, but the conformer filter should match the
    # actual interaction graph passed to GED.
    contact_definition = str(filter_cfg.get("contact_definition", "graph")).lower()
    if contact_definition not in {"distance", "graph", "distance_or_graph", "distance_and_graph"}:
        raise ValueError(
            "step07.structure_filter.contact_definition must be one of: "
            "distance, graph, distance_or_graph, distance_and_graph"
        )
    if mode not in {"all", "any"}:
        raise ValueError("step07.structure_filter.require_key_residues must be 'all' or 'any'.")

    key_residues = [str(x) for x in key_df.get("residue_key", pd.Series(dtype=str)).dropna().astype(str).tolist()]
    key_set = set(key_residues)

    distance_present_by_sid: Dict[str, set] = {}
    if not contact_df.empty and key_set:
        sub = contact_df[(contact_df["residue_key"].astype(str).isin(key_set)) & (contact_df["within_distance_cutoff"] == True)]  # noqa: E712
        for sid, g in sub.groupby("structure_id", dropna=False):
            distance_present_by_sid[str(sid)] = set(g["residue_key"].astype(str).tolist())

    rows = []
    for rec in records.itertuples():
        sid = str(rec.structure_id)
        pdb_path = str(rec.pdb_path)
        norm_sid = normalize_structure_id(sid)
        norm_pdb = normalize_structure_id(pdb_path)
        distance_present = set()
        for candidate in {sid, norm_sid, Path(pdb_path).stem, norm_pdb}:
            distance_present.update(distance_present_by_sid.get(str(candidate), set()))
            # Also compare by normalized keys.
            for stored_sid, residues in distance_present_by_sid.items():
                if normalize_structure_id(stored_sid) == normalize_structure_id(candidate):
                    distance_present.update(residues)

        graph_present_raw = graph_occurrence_for_structure(graph_occurrence, sid, pdb_path)
        distance_present_keys = key_set_present_by_alias(key_set, distance_present)
        graph_present_keys = key_set_present_by_alias(key_set, graph_present_raw)
        if contact_definition == "distance":
            present = set(distance_present_keys)
        elif contact_definition == "graph":
            present = set(graph_present_keys)
        elif contact_definition == "distance_or_graph":
            present = set(distance_present_keys) | set(graph_present_keys)
        else:
            present = set(distance_present_keys) & set(graph_present_keys)

        missing = sorted(key_set - present)
        present_keys = sorted(present)
        if not enabled:
            passed = True
            reason = "structure_filter_disabled"
        elif not key_set:
            passed = False
            reason = "no_key_residues_selected"
        elif mode == "all":
            passed = len(missing) == 0
            reason = "all_key_residues_present" if passed else "missing_required_key_residues"
        else:
            passed = len(present_keys) > 0
            reason = "at_least_one_key_residue_present" if passed else "no_key_residue_present"

        rows.append(
            {
                "structure_id": sid,
                "normalized_structure_id": norm_sid,
                "pdb_path": pdb_path,
                "key_residue_count": len(key_residues),
                "present_key_residue_count": len(present_keys),
                "missing_key_residue_count": len(missing),
                "present_key_residues": ";".join(present_keys),
                "missing_key_residues": ";".join(missing),
                "distance_present_key_residues": ";".join(sorted(distance_present_keys)),
                "graph_present_key_residues": ";".join(sorted(graph_present_keys)),
                "contact_definition": contact_definition,
                "require_key_residues": mode,
                "pass_key_residue_filter": bool(passed),
                "filter_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def run_step07(config_path: str | Path, max_structures: Optional[int] = None, clean_output: bool = False) -> dict:
    config_path = Path(config_path).resolve()
    root = get_project_root(config_path)
    config = _load_config_dict(config_path)
    target_id = get_target_id(config, config_path)
    output_root = get_output_root(config, root, target_id)
    step_cfg = config.get("step07", {}) or {}
    if step_cfg.get("enabled", True) is False:
        return {"target_id": target_id, "status": "skipped", "message": "step07.enabled is false."}

    output_subdir = public_output_subdir(
        step_cfg.get("output_subdir"), "06_key_residue_screening"
    )
    out_dir = output_root / output_subdir
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    lists_dir = out_dir / "lists"
    if clean_output and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    for d in [tables_dir, reports_dir, lists_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cons_cfg = step_cfg.get("conservation", {}) or {}
    conservation_path_text = step_cfg.get("conservation_file") or cons_cfg.get("file") or cons_cfg.get("zip")
    if not conservation_path_text:
        raise ValueError(
            "Step07 needs a ConSurf input file. Set step07.conservation_file to a ConSurf result "
            "ZIP, website TAR.GZ/GZ download, or normalized CSV/TSV table."
        )
    conservation_path = resolve_path(conservation_path_text, root)
    if conservation_path is None or not conservation_path.exists():
        raise FileNotFoundError(f"ConSurf conservation input not found: {conservation_path_text}")

    consurf_df, consurf_meta = load_consurf_scores(conservation_path, cons_cfg)
    consurf_table = tables_dir / f"{target_id}_step06_consurf_scores_normalized.csv"
    # Keep the parser's normalized source table immutable.  The mapped table is
    # a separate audit artifact; writing both paths from the same post-mapping
    # frame made it impossible to distinguish source numbering from target PDB
    # numbering and caused downstream reports to reuse the wrong table.
    normalized_consurf_df = consurf_df.copy()
    normalized_consurf_df.to_csv(consurf_table, index=False)
    # Align ConSurf/PDB numbering before any residue-level threshold is
    # evaluated.  This is the critical NEW6 fix and is applied to every target.

    records, complex_table_path, pdb_col = load_complex_records(config, root, output_root, target_id, step_cfg)
    if max_structures:
        records = records.head(max_structures).copy()
    consurf_df, consurf_mapping, consurf_mapping_warnings = map_consurf_scores_to_complexes(
        consurf_df, records, cons_cfg
    )
    consurf_mapping_table = tables_dir / f"{target_id}_step06_consurf_mapping.csv"
    consurf_df.to_csv(consurf_mapping_table, index=False)
    consurf_mapping_audit, consurf_overlap_warnings = assess_consurf_mapping_against_complexes(
        consurf_df, records
    )
    ligand_cfg = step_cfg.get("ligand", {}) or {}
    thresholds = step_cfg.get("thresholds", {}) or {}
    distance_cfg = step_cfg.get("distance", {}) or {}
    # Accept distance cutoff from either section.
    distance_cfg = dict(distance_cfg)
    if "residue_ligand_distance_max" not in distance_cfg:
        distance_cfg["residue_ligand_distance_max"] = thresholds.get("residue_ligand_distance_max", 3.0)

    contact_df, distance_warnings = compute_distance_contacts(records, ligand_cfg, distance_cfg)
    per_structure_table = tables_dir / f"{target_id}_step06_per_structure_residue_contacts.csv"
    contact_df.to_csv(per_structure_table, index=False)
    contact_mapping_audit = enforce_consurf_contact_coverage(consurf_df, contact_df, cons_cfg)

    graph_nodes_ordered, graph_node_source, graph_node_warnings = collect_graph_contact_nodes_ordered(output_root, target_id, step_cfg)
    graph_occurrence = {
        sid: set(nodes) | {residue_key_base(n) for n in nodes}
        for sid, nodes in graph_nodes_ordered.items()
    }
    graph_warnings = list(graph_node_warnings)
    if not graph_occurrence:
        graph_warnings.extend(collect_graph_occurrence(output_root, target_id, step_cfg)[1])
    summary_df = summarize_residues(
        contact_df=contact_df,
        consurf_df=consurf_df,
        graph_occurrence=graph_occurrence,
        thresholds=thresholds,
        n_structures_total=len(records),
    )
    residue_scores_table = tables_dir / f"{target_id}_step06_residue_scores.csv"
    summary_df.to_csv(residue_scores_table, index=False)

    distance_cutoff = float(distance_cfg.get("residue_ligand_distance_max", 3.0))
    cutoff_detail_df, cutoff_publication_df = build_distance_cutoff_residue_details(
        summary_df=summary_df,
        contact_df=contact_df,
        graph_occurrence=graph_occurrence,
        thresholds=thresholds,
        n_structures_total=len(records),
        distance_cutoff=distance_cutoff,
    )
    cutoff_detail_outputs = write_distance_cutoff_residue_detail_outputs(
        detailed_df=cutoff_detail_df,
        publication_df=cutoff_publication_df,
        target_id=target_id,
        tables_dir=tables_dir,
        reports_dir=reports_dir,
        distance_cutoff=distance_cutoff,
    )

    key_df = summary_df[summary_df["key_residue"] == True].copy()  # noqa: E712
    key_table = tables_dir / f"{target_id}_step06_key_residues.csv"
    key_df.to_csv(key_table, index=False)
    key_list = lists_dir / f"{target_id}_step06_key_residues.txt"
    key_list.write_text("\n".join(key_df["residue_key"].astype(str).tolist()) + ("\n" if len(key_df) else ""), encoding="utf-8")

    structure_filter_df = build_structure_key_residue_filter(
        records=records,
        contact_df=contact_df,
        key_df=key_df,
        graph_occurrence=graph_occurrence,
        step_cfg=step_cfg,
    )
    structure_filter_table = tables_dir / f"{target_id}_step06_structure_key_residue_filter.csv"
    structure_filter_df.to_csv(structure_filter_table, index=False)

    passed_df = structure_filter_df[structure_filter_df["pass_key_residue_filter"] == True].copy()  # noqa: E712
    failed_df = structure_filter_df[structure_filter_df["pass_key_residue_filter"] != True].copy()  # noqa: E712
    passed_table = tables_dir / f"{target_id}_step06_passed_complex_index.csv"
    failed_table = tables_dir / f"{target_id}_step06_failed_complex_index.csv"
    passed_df.to_csv(passed_table, index=False)
    failed_df.to_csv(failed_table, index=False)
    passed_list = lists_dir / f"{target_id}_step06_passed_structure_ids.txt"
    failed_list = lists_dir / f"{target_id}_step06_failed_structure_ids.txt"
    passed_list.write_text("\n".join(passed_df["structure_id"].astype(str).tolist()) + ("\n" if len(passed_df) else ""), encoding="utf-8")
    failed_list.write_text("\n".join(failed_df["structure_id"].astype(str).tolist()) + ("\n" if len(failed_df) else ""), encoding="utf-8")

    node_grouping_outputs: Dict[str, str] = {}
    node_grouping_cfg = step_cfg.get("node_grouping", {}) or {}
    if node_grouping_cfg.get("enabled", True) is not False:
        node_grouping_df, node_grouping_per_structure_df = build_node_level_grouping_table(
            records=records,
            contact_df=contact_df,
            key_df=key_df,
            graph_nodes_ordered=graph_nodes_ordered,
            step_cfg=step_cfg,
        )
        node_grouping_outputs = write_node_level_grouping_outputs(
            node_grouping_df,
            node_grouping_per_structure_df,
            target_id=target_id,
            tables_dir=tables_dir,
            reports_dir=reports_dir,
        )

    report = {
        "target_id": target_id,
        "status": "completed",
        "config": str(config_path),
        "output_dir": str(out_dir),
        "input_complex_table": str(complex_table_path),
        "input_complex_column": pdb_col,
        "graph_contact_node_source": graph_node_source,
        "n_structures": int(len(records)),
        "conservation": consurf_meta,
        "conservation_mapping_audit": {
            **consurf_mapping,
            **consurf_mapping_audit,
            "contact_coverage": contact_mapping_audit,
        },
        "thresholds": {
            "consurf_grade_min": thresholds.get("consurf_grade_min", 8),
            "residue_ligand_distance_max": distance_cfg.get("residue_ligand_distance_max", 3.0),
            "distance_contact_frequency_min": thresholds.get("distance_contact_frequency_min", thresholds.get("occurrence_frequency_min", 0.80)),
            "graph_occurrence_frequency_min": thresholds.get("graph_occurrence_frequency_min", thresholds.get("occurrence_frequency_min", 0.80)),
            "require_graph_occurrence": True,
        },
        "structure_filter": step_cfg.get("structure_filter", {
            "enabled": True,
            "require_key_residues": "all",
            "contact_definition": "graph",
        }),
        "n_consurf_residues": int(len(consurf_df)),
        "n_residues_scored": int(len(summary_df)),
        "n_residues_within_distance_cutoff": int(len(cutoff_detail_df)),
        "n_key_residues": int(len(key_df)),
        "key_residues": key_df["residue_key"].astype(str).tolist(),
        "n_structures_passed_key_residue_filter": int(len(passed_df)),
        "n_structures_failed_key_residue_filter": int(len(failed_df)),
        "tables": {
            "consurf_scores_normalized": str(consurf_table),
            "consurf_mapping": str(consurf_mapping_table),
            "per_structure_residue_contacts": str(per_structure_table),
            "residue_scores": str(residue_scores_table),
            **cutoff_detail_outputs,
            "key_residues": str(key_table),
            "structure_key_residue_filter": str(structure_filter_table),
            "passed_complex_index": str(passed_table),
            "failed_complex_index": str(failed_table),
            **node_grouping_outputs,
        },
        "lists": {
            "key_residues": str(key_list),
            "passed_structure_ids": str(passed_list),
            "failed_structure_ids": str(failed_list),
        },
        "warnings": consurf_mapping_warnings + consurf_overlap_warnings + distance_warnings + graph_warnings,
    }
    report_path = reports_dir / f"{target_id}_step06_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step07 conserved key-residue screening")
    parser.add_argument("--config", required=True, help="Path to CatConGraph config.yaml")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--clean", action="store_true", help="Remove the Step07 output directory before running")
    args = parser.parse_args(argv)
    report = run_step07(args.config, max_structures=args.max_structures, clean_output=args.clean)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()

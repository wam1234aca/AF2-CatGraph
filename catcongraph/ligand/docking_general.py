from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol

from catcongraph.ligand import dock_py_original as dockpy


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


@dataclass(frozen=True)
class DockingCompatibility:
    legacy_compatible: bool
    reason: str
    template_chain_count: int
    target_chain_count: int
    template_ca_count: int
    target_ca_count: int
    ordered_sequence_identical: bool


@dataclass(frozen=True)
class DockingAlignmentAudit:
    alignment_mode: str
    legacy_compatible: bool
    compatibility_reason: str
    template_chain_count: int
    target_chain_count: int
    template_ca_count: int
    target_ca_count: int
    matched_ca_count: int
    sequence_identity: float
    alignment_coverage: float
    alignment_rmsd: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceAwareCompatibility:
    compatible: bool
    reason: str
    matched_ca_count: int
    sequence_identity: float
    alignment_coverage: float
    chain_mapping: str


def _pdb_ca_sequences(path: str | Path) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain = line[21:22].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain, []).append(AA3_TO_AA1.get(line[17:20].strip().upper(), "X"))
    return chains


def assess_legacy_compatibility(template_pdb: str | Path, target_pdb: str | Path) -> DockingCompatibility:
    template = _pdb_ca_sequences(template_pdb)
    target = _pdb_ca_sequences(target_pdb)
    t_seq = next(iter(template.values()), []) if len(template) == 1 else []
    q_seq = next(iter(target.values()), []) if len(target) == 1 else []
    same = bool(t_seq) and t_seq == q_seq
    if len(template) != 1 or len(target) != 1:
        reason = "requires_single_protein_chain"
    elif not t_seq or not q_seq:
        reason = "missing_ca_atoms"
    elif len(t_seq) != len(q_seq):
        reason = "different_ca_count"
    elif not same:
        reason = "different_ordered_sequence"
    else:
        reason = "single_chain_same_length_same_ordered_sequence"
    return DockingCompatibility(
        legacy_compatible=bool(same and len(template) == 1 and len(target) == 1),
        reason=reason,
        template_chain_count=len(template),
        target_chain_count=len(target),
        template_ca_count=sum(len(x) for x in template.values()),
        target_ca_count=sum(len(x) for x in target.values()),
        ordered_sequence_identical=same,
    )


def _ca_atoms_by_chain(mol: Chem.Mol) -> dict[str, list[tuple[int, str]]]:
    chains: dict[str, list[tuple[int, str]]] = {}
    seen: set[tuple[str, int, str]] = set()
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetName().strip() != "CA":
            continue
        chain = info.GetChainId().strip() or "_"
        key = (chain, int(info.GetResidueNumber()), info.GetInsertionCode().strip())
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain, []).append((idx, AA3_TO_AA1.get(info.GetResidueName().strip().upper(), "X")))
    return chains


def _global_align(a: Sequence[str], b: Sequence[str]) -> tuple[list[tuple[int, int]], int]:
    """Needleman-Wunsch alignment returning paired residue indices and matches."""
    n, m = len(a), len(b)
    gap = -2
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
        trace[i][0] = "U"
    for j in range(1, m + 1):
        score[0][j] = j * gap
        trace[0][j] = "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (2 if a[i - 1] == b[j - 1] else -1)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            best = max(diag, up, left)
            score[i][j] = best
            trace[i][j] = "D" if diag == best else ("U" if up == best else "L")
    pairs: list[tuple[int, int]] = []
    matches = 0
    i, j = n, m
    while i or j:
        step = trace[i][j]
        if i and j and step == "D":
            pairs.append((i - 1, j - 1))
            matches += int(a[i - 1] == b[j - 1])
            i -= 1
            j -= 1
        elif i and (not j or step == "U"):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, matches


def assess_sequence_aware_compatibility(
    template_pdb: str | Path,
    target_pdb: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> SequenceAwareCompatibility:
    """Estimate whether the sequence-aware RDKit alignment will be admissible."""
    cfg = dict(config or {})
    min_identity = float(cfg.get("min_sequence_identity", 0.70))
    min_coverage = float(cfg.get("min_alignment_coverage", 0.70))
    min_matched = int(cfg.get("min_matched_ca", 30))
    template = _pdb_ca_sequences(template_pdb)
    target = _pdb_ca_sequences(target_pdb)
    if not template or not target:
        return SequenceAwareCompatibility(False, "missing_ca_atoms", 0, 0.0, 0.0, "")

    candidates = []
    for target_chain, target_seq in target.items():
        for template_chain, template_seq in template.items():
            pairs, matches = _global_align(target_seq, template_seq)
            identity = matches / max(len(pairs), 1)
            coverage = len(pairs) / max(min(len(target_seq), len(template_seq)), 1)
            candidates.append(
                (identity, coverage, len(pairs), target_chain, template_chain, matches)
            )
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    used_target: set[str] = set()
    used_template: set[str] = set()
    total_pairs = 0
    total_matches = 0
    chain_pairs = []
    for identity, _coverage, pairs, target_chain, template_chain, matches in candidates:
        if target_chain in used_target or template_chain in used_template:
            continue
        if identity < min_identity:
            continue
        used_target.add(target_chain)
        used_template.add(template_chain)
        total_pairs += pairs
        total_matches += matches
        chain_pairs.append(f"{target_chain}->{template_chain}")

    identity = total_matches / max(total_pairs, 1)
    coverage = total_pairs / max(
        min(sum(map(len, target.values())), sum(map(len, template.values()))), 1
    )
    if total_pairs < min_matched:
        reason = "insufficient_matched_ca"
    elif coverage < min_coverage:
        reason = "insufficient_alignment_coverage"
    elif identity < min_identity:
        reason = "insufficient_sequence_identity"
    else:
        reason = "sequence_aware_thresholds_passed"
    return SequenceAwareCompatibility(
        compatible=reason == "sequence_aware_thresholds_passed",
        reason=reason,
        matched_ca_count=total_pairs,
        sequence_identity=identity,
        alignment_coverage=coverage,
        chain_mapping=";".join(chain_pairs),
    )


def _sequence_aware_atom_map(
    moving: Chem.Mol,
    reference: Chem.Mol,
    *,
    min_sequence_identity: float,
    min_alignment_coverage: float,
    min_matched_ca: int,
) -> tuple[list[tuple[int, int]], float, float, int, int]:
    moving_chains = _ca_atoms_by_chain(moving)
    reference_chains = _ca_atoms_by_chain(reference)
    if not moving_chains or not reference_chains:
        raise ValueError("Sequence-aware docking requires CA atoms in both template and target proteins.")

    candidates = []
    for mc, ma in moving_chains.items():
        for rc, ra in reference_chains.items():
            pairs, matches = _global_align([x[1] for x in ma], [x[1] for x in ra])
            identity = matches / max(len(pairs), 1)
            coverage = len(pairs) / max(min(len(ma), len(ra)), 1)
            candidates.append((identity, coverage, len(pairs), mc, rc, pairs))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    used_m: set[str] = set()
    used_r: set[str] = set()
    atom_map: list[tuple[int, int]] = []
    total_matches = 0
    total_pairs = 0
    for identity, _coverage, _n, mc, rc, pairs in candidates:
        if mc in used_m or rc in used_r:
            continue
        if identity < min_sequence_identity:
            continue
        used_m.add(mc)
        used_r.add(rc)
        ma, ra = moving_chains[mc], reference_chains[rc]
        atom_map.extend((ma[i][0], ra[j][0]) for i, j in pairs)
        total_matches += sum(ma[i][1] == ra[j][1] for i, j in pairs)
        total_pairs += len(pairs)

    if len(atom_map) < min_matched_ca:
        raise ValueError(
            f"Only {len(atom_map)} CA atoms could be sequence-aligned; "
            f"step02.general_alignment.min_matched_ca={min_matched_ca}."
        )
    identity = total_matches / max(total_pairs, 1)
    coverage = len(atom_map) / max(min(sum(map(len, moving_chains.values())), sum(map(len, reference_chains.values()))), 1)
    if coverage < min_alignment_coverage:
        raise ValueError(
            f"Sequence-alignment coverage is {coverage:.3f}; "
            f"step02.general_alignment.min_alignment_coverage={min_alignment_coverage:.3f}."
        )
    return atom_map, identity, coverage, len(moving_chains), len(reference_chains)


def rigid_docking_sequence_aware(
    template_protein: Chem.Mol,
    ligand: Chem.Mol,
    mutant_protein: Chem.Mol,
    *,
    compatibility: DockingCompatibility,
    config: Mapping[str, Any] | None = None,
) -> tuple[Chem.Mol, DockingAlignmentAudit]:
    cfg = dict(config or {})
    min_identity = float(cfg.get("min_sequence_identity", 0.70))
    min_coverage = float(cfg.get("min_alignment_coverage", 0.70))
    min_matched = int(cfg.get("min_matched_ca", 30))
    atom_map, identity, coverage, n_moving_chains, n_reference_chains = _sequence_aware_atom_map(
        mutant_protein,
        template_protein,
        min_sequence_identity=min_identity,
        min_alignment_coverage=min_coverage,
        min_matched_ca=min_matched,
    )
    rmsd = float(AlignMol(mutant_protein, template_protein, atomMap=atom_map))
    if dockpy.has_collision(mutant_protein, ligand):
        ligand = dockpy.avoid_collision_strategic(
            mutant_protein,
            ligand,
            max_attempts=50,
            collision_threshold=2.0,
            max_translation=5.0,
            max_rotation=30.0,
        )
    complex_mol = Chem.CombineMols(mutant_protein, ligand)
    audit = DockingAlignmentAudit(
        alignment_mode="sequence_aware",
        legacy_compatible=compatibility.legacy_compatible,
        compatibility_reason=compatibility.reason,
        template_chain_count=n_reference_chains,
        target_chain_count=n_moving_chains,
        template_ca_count=compatibility.template_ca_count,
        target_ca_count=compatibility.target_ca_count,
        matched_ca_count=len(atom_map),
        sequence_identity=identity,
        alignment_coverage=coverage,
        alignment_rmsd=rmsd,
    )
    return complex_mol, audit


def legacy_audit(compatibility: DockingCompatibility, rmsd: float | None = None) -> DockingAlignmentAudit:
    return DockingAlignmentAudit(
        alignment_mode="dock_py_exact",
        legacy_compatible=compatibility.legacy_compatible,
        compatibility_reason=compatibility.reason,
        template_chain_count=compatibility.template_chain_count,
        target_chain_count=compatibility.target_chain_count,
        template_ca_count=compatibility.template_ca_count,
        target_ca_count=compatibility.target_ca_count,
        matched_ca_count=min(compatibility.template_ca_count, compatibility.target_ca_count),
        sequence_identity=1.0 if compatibility.ordered_sequence_identical else 0.0,
        alignment_coverage=1.0 if compatibility.ordered_sequence_identical else 0.0,
        alignment_rmsd=float("nan") if rmsd is None else float(rmsd),
    )

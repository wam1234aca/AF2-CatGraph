# Configuration reference

AF2-CatGraph uses YAML. The GUI writes the same configuration consumed by
the CLI. Save the exact YAML used for every reported result.

## Required inputs

| Field | Purpose |
|---|---|
| `project.target_id` | Filesystem-safe project identifier |
| `project.output_root` | Root of generated step outputs |
| `inputs.af2_dir` | Protein-only AF2/ColabFold PDB ensemble |
| `inputs.template_protein_pdb` | Protein portion of the reference complex |
| `inputs.ligand_pdb` / `ligand_pdbs` | Ordered substrate/cofactor/metal PDB files |
| `step06.conservation_file` | ConSurf archive or CSV/TSV input |
| `step07.ged.executable` | External Graph_Edit_Distance executable |

Use absolute paths for batch execution where practical. If using relative paths,
run commands from the repository root and confirm the resolved paths in
Preflight.

## Workflow sections

Configurations use `config_schema: public_01_09`; every numbered YAML section
matches the CLI and GUI number exactly.

| Step | Configuration section | Scientific role |
|---:|---|---|
| 01 | `step01` | Global pLDDT filter; new-project default is `80.0` |
| 02 | `step02` | Template-guided ligand placement |
| 03 | `step03.clash` | Protein–ligand VDW clash filter |
| 04 | `step04.catalytic_region` | Catalytic-region pLDDT QC |
| 05 | `step05` | PLIP interaction graphs |
| 06 | `step06` | Four-criterion key-residue screening |
| 07 | `step07.ged` | GED topology classification |
| 08 | `step08.canonicalization` | Ligand contact-unit grouping and candidate selection |
| 09 | `step09.plot` / `step09.layout` | Class plots and summary outputs |
| 10 (optional) | `step10_optional` | Substrate/ligand atom selection and trimming |

`step10_optional` is selectable in the CLI and GUI as optional Step10. It is
disabled by default. When enabled, the main workflow runs it at the configured
point before the downstream step consumes its output.

## Ligand contact-unit grouping policies

Step08 provides two user-facing routes with the same GED and candidate-screening
logic:

- `universal_rules` automatically applies the twelve Supplementary Table S4
  rules to all ligand components. It is the default for new projects. The
  automatic mapping and ensemble-consensus audit are written by Step08.
- `conservative_manual` uses only `manual_ligand_units`; an empty mapping
  retains atom-level labels.

Universal-rules example:

```yaml
step08:
  canonicalization:
    contact_unit_mode: universal_rules
    use_builtin_ligand_units: false
    auto_detect_ligand_units: true
    recommendations:
      enabled: true
      require_acceptance: false
      consensus_min_fraction: 0.80
```

Conservative example:

```yaml
step08:
  canonicalization:
    contact_unit_mode: conservative_manual
    manual_ligand_units: {}
```

Manual groups should use residue-aware atom names when more than one ligand is
present:

```yaml
step08:
  canonicalization:
    manual_ligand_units:
      ATP_terminal_phosphate_O:
        - ATP:O9
        - ATP:O10
        - ATP:O11
```

Atom names must exactly match the PDB. Group atoms only after checking chemical
connectivity, symmetry, protonation/charge, and their enzyme contacts.

## Result-changing parameters

These settings can change membership, filtering, or class assignment and need
scientific justification before modification:

- pLDDT thresholds and score scope;
- alignment method and coverage/identity requirements;
- clash count and VDW delta;
- catalytic-region radius;
- PLIP ligand identity/connectivity rules;
- conservation, distance, and occurrence thresholds;
- GED threshold and equal-edge-count policy;
- common/rare interaction frequencies;
- ligand contact-unit policy and manual mappings;
- class-signature and class-splitting policies;
- optional ligand-atom selection.

## Figure-only parameters

Step09 figure size, DPI, grid columns, entity spacing/rotation, node size, label
size, and manual plotting offsets change rendered figures only. The GUI shows SVG/PNG output selection. Additional figure settings can be edited directly in project YAML when needed.

## External dependency record

For a final analysis, record Python, NumPy, pandas, SciPy, Biopython, NetworkX,
RDKit, Matplotlib, PyYAML, PLIP/Open Babel, the GED executable commit/checksum,
and the conservation input source/date. The tested versions are recommendations
for reproduction, not hard restrictions on new analyses.

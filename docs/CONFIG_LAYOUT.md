# Configuration layout and workflow steps

AF2-CatGraph project configurations use YAML. New configurations include:

```yaml
config_schema: public_01_09
```

The numbered YAML sections correspond directly to the CLI commands, GUI steps,
and result directories.

| CLI command | YAML section | Result directory | Purpose |
|---|---|---|---|
| `step01` | `step01` | `01_af2_prepare` | Global pLDDT filtering |
| `step02` | `step02` | `02_template_guided_docking` | Reference-guided ligand placement |
| `step03` | `step03` | `03_clash_filter` | Protein–ligand VDW clash filtering |
| `step04` | `step04` | `04_catalytic_region_plddt_qc` | Catalytic-region pLDDT quality control |
| `step05` | `step05` | `05_plip_interaction_graphs` | PLIP interaction-graph generation |
| `step06` | `step06` | `06_key_residue_screening` | Core-residue screening |
| `step07` | `step07` | `07_ged_classification` | GED classification |
| `step08` | `step08` | `08_chemical_equivalence` | Ligand contact-unit grouping and candidate selection |
| `step09` | `step09` | `09_class_visualization` | Class summaries and figures |

Optional `step10_optional` selects or removes specified ligand atoms at either
`before_docking` or `after_clash`. It is disabled by default and should be
enabled only when the analysis requires a defined ligand-atom subset.

## Create a project

```bash
af2-catgraph init --target-id MY_TARGET --out projects/MY_TARGET
```

This creates `projects/MY_TARGET/config.yaml`. Add the input PDB paths, ligand
residue names, conservation file, and GED executable, then validate the project:

```bash
af2-catgraph preflight --config projects/MY_TARGET/config.yaml
```

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for the main fields and
result-changing parameters.

## Ligand contact-unit grouping policy

The GUI presents two analysis routes: `universal_rules` for S4 ligand
contact-unit grouping and `conservative_manual` with an empty mapping for
atom-level GED classes. A third policy remains available only for compatibility
with existing YAML files:

- `universal_rules` automatically applies the twelve S4 rules to every ligand
  component and is the default for new projects.
- `guided_review` generates reviewable contact-unit proposals for legacy projects. A proposal is
  applied only after it is accepted, saved to the YAML, and Step08 is rerun.
- `conservative_manual` generates no proposals and applies only
  `manual_ligand_units`. An empty mapping retains atom-level ligand labels.

All policies retain interaction-label grouping and the same
candidate-screening thresholds.

Example manual definition:

```yaml
step08:
  canonicalization:
    contact_unit_mode: conservative_manual
    manual_ligand_units:
      ATP_terminal_phosphate_O:
        - ATP:O9
        - ATP:O10
        - ATP:O11
```

See [CONTACT_UNIT_REVIEW.md](CONTACT_UNIT_REVIEW.md) before defining or
accepting ligand contact units.

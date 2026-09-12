# Ligand contact-unit grouping

After atom-level GED classification, Step08 can group selected ligand atoms or
atom groups into common contact units. Mapping several atoms to one contact
unit means that those labels are treated as one comparison unit in the
interaction graphs; it does not imply that the underlying physical
interactions are identical.

The available policies use the same GED results, interaction-frequency
calculation, candidate-selection criteria, and output logic. They differ only
in how ligand contact-unit mappings are defined.

## Universal S4 rules (recommended for the additional enzymes)

Set `contact_unit_mode: universal_rules` to apply the common rule set
automatically. Rules are evaluated within each ligand component; atoms from
different components are never combined only because their names match.

| Rule | Automatic assignment and naming |
|---|---|
| Carboxylate oxygen | O atoms bonded to the same carboxylate centre → `[ligand] carboxylate oxygen [position] unit` |
| Phosphate/phosphonate non-bridging oxygen | P centre plus directly bonded non-bridging O, excluding P–O–P bridges → `[ligand] phosphate/phosphonate [position] unit` |
| Phosphate-bridging oxygen | Each O bonded to two P centres remains separate → `[ligand] phosphate [i]–[j] bridging oxygen` |
| Sulfate/sulfonate oxygen | Terminal O atoms on the same S centre → `[ligand] sulfate/sulfonate oxygen [position] unit` |
| Nitro oxygen | Terminal O atoms on the same nitro N → `[ligand] nitro oxygen [position] unit` |
| Sulfonyl oxygen | The two terminal O atoms on the same sulfonyl S → `[ligand] sulfonyl oxygen [position] unit` |
| Delocalized nitrogen | Terminal N atoms on the same guanidinium-/amidinium-like centre; ring N remains separate → `[ligand] delocalized nitrogen [position] unit` |
| Aromatic ring fragment | Carbon atoms in the same detected ring system; heteroatoms remain separate → `[ligand] aromatic ring [position] unit` |
| Alkyl fragment | Contiguous non-ring, non-functional-centre carbon fragment → `[ligand] aliphatic carbon [position] unit` |
| Sugar-ring carbon fragment | Carbon atoms in the same detected monosaccharide-like ring; O atoms remain separate → `[ligand] sugar-ring carbon [position] unit` |
| Metal ion | Each single-atom metal component remains separate → `Mg2+ ion`, `Zn2+ ion`, etc. |
| Individual atom | Every atom not assigned above remains a ligand/component-aware site → `[ligand] [atom] site` |

Groups must recur in the configured fraction of ligand-bearing structures
(default `0.80`). The rule audit records the component, atom list, inferred
connectivity evidence, ensemble support, final name, and whether the rule was
applied. The same detector is applied to GAC, AqTMK, PTP1B, and the additional enzyme systems.

Step08 builds these rules from the ligand heavy-atom graph. Hydrogen and
deuterium records introduced by PLIP/Open Babel protonation are ignored in the
substrate inventory, connectivity tests, contact-unit mapping, and audit.

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
    manual_ligand_units: {}
```

## Conservative manual policy

Set `contact_unit_mode: conservative_manual` to use only explicit
`manual_ligand_units`:

- no built-in ligand mapping is loaded;
- no contact-unit proposal is generated;
- an empty manual mapping retains separate labels for individual ligand atoms.

Interaction-label grouping, interaction-frequency calculation,
candidate screening, and output generation still run. The policy disables only
ligand-atom grouping.

## YAML examples

Conservative manual policy without ligand-atom grouping:

```yaml
step08:
  canonicalization:
    contact_unit_mode: conservative_manual
    use_builtin_ligand_units: false
    auto_detect_ligand_units: false
    recommendations:
      enabled: false
      require_acceptance: true
    manual_ligand_units: {}
```

Conservative manual policy with an explicit contact unit:

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

Atom names must exactly match the ligand PDB entering PLIP analysis. When
several ligands contain atoms with the same name, use residue-aware labels such
as `ATP:O9`. Do not group atoms across ligands solely because their names or
positions are similar.

## Review criteria

| Local chemical group | Review guidance |
|---|---|
| Two oxygen atoms attached to the same carboxyl/carboxylate center | Grouping may be considered after checking stereospecific contacts and mechanistic context. |
| Non-bridging oxygen atoms attached to the same phosphate center | Grouping may be considered; do not group bridging oxygen atoms, atoms from different phosphate centers, or the central phosphorus by name alone. |
| Connected sulfur–oxygen or nitro oxygen atoms | Check protonation, charge, and reaction role before acceptance. |
| Multi-nitrogen groups, aromatic or sugar rings, or ligands with uncertain atom naming | Retain atom-level labels unless an explicit definition is supported independently. |
| Metal ions | Retain the metal identity; do not group it with ligand atoms. |

Before a full Step08 run, inspect the mapping audits from a small representative
subset of structures. Retain the final mapping in the project YAML used for the
reported analysis.

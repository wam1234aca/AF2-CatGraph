# Changelog

## 1.2.2

- Simplified the GUI to expose the confirmed scientific controls while
  preserving hidden values from existing YAML projects.
- Limited Step02 to automatic or sequence-aware Cα alignment in the GUI and
  displayed Step01-passed structures as the standard input source.
- Reduced Step08 to the universal S4 grouping and atom-level analysis routes,
  and reduced Step09 to SVG/PNG output selection.
- Adopted **ligand contact-unit grouping** and **interaction-label grouping**
  throughout user-facing labels and documentation.
- Kept optional Step10 ligand-atom selection visible and unchanged.

## 1.2.1

- Excluded hydrogen and deuterium records added by ligand protonation from
  Step08 contact-unit detection, substrate inventory, and automatic audit.
- Built every Step08 functional-group terminality check from the ligand
  heavy-atom graph, so protonation cannot suppress an otherwise valid S4 unit.
- Kept the universal phosphate policy unchanged: P–O–C connector oxygens remain
  in their phosphate unit, while P–O–P bridging oxygens remain separate.

## 1.2.0

- Removed target-specific runtime mappings from `universal_rules`; GAC,
  AqTMK, PTP1B, and all additional enzymes now use the same detector.
- Standardized all carboxylate labels to `Carboxylate oxygen unit`.
- Added context-sensitive general naming: simple names for a single ligand
  unit and ligand/position-qualified names only when needed.
- Retained the manuscript cases solely as regression benchmarks.

## 1.1.0

- Added `universal_rules`, the default contact-unit policy for new projects.
- Implemented the twelve Supplementary Table S4 ligand contact-unit rules with
  component-aware names and ensemble-consensus auditing.
- Preserved the exact GAC, AqTMK, and PTP1B atom groups and unit names.
- Retained `conservative_manual` for post-GED atom-level candidate screening
  and `guided_review` for backward-compatible review workflows.
- Added GUI controls, YAML defaults, documentation, and regression tests for
  the new policy.

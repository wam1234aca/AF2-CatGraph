# Step 02 summary: AqTMK

Workflow policy: one general, sequence-aware workflow.
Configured alignment mode: `auto`.

Input source: `table:results/AqTMK/01_af2_prepare/tables/AqTMK_step01_passed_structure_index.csv`
Input structures: 320
Complexes written: 320
Failed or skipped: 0

Naming rule:
- Default short rule: `<PDB>_<rank_or_index>_complex.pdb`
- Old basename rule is still available with `step02.output_naming: source_stem`

Multi-ligand rule:
- `inputs.ligand_pdbs are docked sequentially by default: ligand 1 -> ligand 2 -> ligand 3, preserving dock.py rigid_docking calls and user-defined ligand order. Set step02.multi_ligand_mode: rigid_assembly only for intentional one-piece assemblies.`

Main outputs:
- `results/AqTMK/02_template_guided_docking/tables/AqTMK_step02_template_guided_docking_summary.csv`
- `results/AqTMK/02_template_guided_docking/tables/AqTMK_step02_complex_index.csv`
- `results/AqTMK/02_template_guided_docking/tables/AqTMK_step02_failed_structures.csv`
- Ligand-order manifest: `results/AqTMK/02_template_guided_docking/tables/AqTMK_step02_ligand_order_manifest.csv`
- `results/AqTMK/02_template_guided_docking/complexes`

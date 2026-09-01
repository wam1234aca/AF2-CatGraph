# Step 02 summary: GAC

Workflow policy: one general, sequence-aware workflow.
Configured alignment mode: `auto`.

Input source: `table:results/GAC/01_af2_prepare/tables/GAC_step01_passed_structure_index.csv`
Input structures: 264
Complexes written: 264
Failed or skipped: 0

Naming rule:
- Default short rule: `<PDB>_<rank_or_index>_complex.pdb`
- Old basename rule is still available with `step02.output_naming: source_stem`

Multi-ligand rule:
- `inputs.ligand_pdbs are docked sequentially by default: ligand 1 -> ligand 2 -> ligand 3, preserving dock.py rigid_docking calls and user-defined ligand order. Set step02.multi_ligand_mode: rigid_assembly only for intentional one-piece assemblies.`

Main outputs:
- `results/GAC/02_template_guided_docking/tables/GAC_step02_template_guided_docking_summary.csv`
- `results/GAC/02_template_guided_docking/tables/GAC_step02_complex_index.csv`
- `results/GAC/02_template_guided_docking/tables/GAC_step02_failed_structures.csv`
- Ligand-order manifest: `results/GAC/02_template_guided_docking/tables/GAC_step02_ligand_order_manifest.csv`
- `results/GAC/02_template_guided_docking/complexes`

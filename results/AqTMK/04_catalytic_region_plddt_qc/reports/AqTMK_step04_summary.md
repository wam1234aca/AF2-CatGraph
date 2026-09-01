# Step05 catalytic-region pLDDT QC summary: AqTMK

- Input complexes: 297
- Passed complexes: 176
- Failed complexes: 121
- Catalytic region: protein residues with any atom within `6 Å` of selected ligand atoms
- pLDDT per residue: `CA` atom, with residue atom-mean fallback if missing
- pLDDT source: AF2/ColabFold PDB from `step05.plddt_pdb_column` or upstream AF2 path columns when available; complex PDB fallback otherwise.
- Pass rule: `mean pLDDT > 90` and `10th percentile pLDDT > 80`

## Output files

- `results/AqTMK/04_catalytic_region_plddt_qc/tables/AqTMK_step04_catalytic_region_plddt_summary.csv`
- `results/AqTMK/04_catalytic_region_plddt_qc/tables/AqTMK_step04_catalytic_region_residues.csv`
- `results/AqTMK/04_catalytic_region_plddt_qc/tables/AqTMK_step04_passed_complex_index.csv`
- `results/AqTMK/04_catalytic_region_plddt_qc/tables/AqTMK_step04_failed_complex_index.csv`

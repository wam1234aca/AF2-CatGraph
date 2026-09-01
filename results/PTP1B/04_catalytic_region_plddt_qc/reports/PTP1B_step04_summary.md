# Step05 catalytic-region pLDDT QC summary: PTP1B

- Input complexes: 309
- Passed complexes: 306
- Failed complexes: 3
- Catalytic region: protein residues with any atom within `6 Å` of selected ligand atoms
- pLDDT per residue: `CA` atom, with residue atom-mean fallback if missing
- pLDDT source: AF2/ColabFold PDB from `step05.plddt_pdb_column` or upstream AF2 path columns when available; complex PDB fallback otherwise.
- Pass rule: `mean pLDDT > 90` and `10th percentile pLDDT > 80`

## Output files

- `results/PTP1B/04_catalytic_region_plddt_qc/tables/PTP1B_step04_catalytic_region_plddt_summary.csv`
- `results/PTP1B/04_catalytic_region_plddt_qc/tables/PTP1B_step04_catalytic_region_residues.csv`
- `results/PTP1B/04_catalytic_region_plddt_qc/tables/PTP1B_step04_passed_complex_index.csv`
- `results/PTP1B/04_catalytic_region_plddt_qc/tables/PTP1B_step04_failed_complex_index.csv`

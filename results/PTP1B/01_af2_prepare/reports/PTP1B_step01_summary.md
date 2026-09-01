# Step 01 summary: PTP1B

## Scope

Step 01 is limited to protein-only AF2/ColabFold structures.
It performs structure indexing, optional standardized naming, optional protein alignment, and global pLDDT filtering.

It does not perform docking, ligand standardization, clash filtering, pocket pLDDT, PLIP, or interaction-graph analysis.

## Results

- Input PDB files: 320
- Passed global pLDDT: 320
- Failed or unreadable: 0
- Global pLDDT threshold: 80.0
- Requested pLDDT filter scope: source_if_available
- Step 0A manifest: /home/wms/AF2-test/AF2-CatGraph/data/PTP1B/01_af2_new/af2_preprocess_manifest.csv (ok)
- Full-length source scores used: 320
- Input/trimmed scores used: 0
- Minimum observed global mean pLDDT: 86.5053177257525
- Maximum observed global mean pLDDT: 93.85916387959865
- Average observed global mean pLDDT: 92.22912050585285

## Main outputs

- `results/PTP1B/01_af2_prepare/tables/PTP1B_step01_global_plddt_summary.csv`
- `results/PTP1B/01_af2_prepare/tables/PTP1B_step01_structure_id_mapping.csv`
- `results/PTP1B/01_af2_prepare/lists/PTP1B_step01_passed_structure_ids.txt`
- `results/PTP1B/01_af2_prepare/pdb/standardized`

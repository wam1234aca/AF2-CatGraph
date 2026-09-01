# Step 01 summary: GAC

## Scope

Step 01 is limited to protein-only AF2/ColabFold structures.
It performs structure indexing, optional standardized naming, optional protein alignment, and global pLDDT filtering.

It does not perform docking, ligand standardization, clash filtering, pocket pLDDT, PLIP, or interaction-graph analysis.

## Results

- Input PDB files: 264
- Passed global pLDDT: 264
- Failed or unreadable: 0
- Global pLDDT threshold: 75.0
- Requested pLDDT filter scope: source_if_available
- Step 0A manifest: /home/wms/AF2-test/AF2-CatGraph-test5/data/GAC/01_af2_new/af2_preprocess_manifest.csv (ok)
- Full-length source scores used: 264
- Input/trimmed scores used: 0
- Minimum observed global mean pLDDT: 90.25735099337749
- Maximum observed global mean pLDDT: 95.24814569536424
- Average observed global mean pLDDT: 93.40428544551476

## Main outputs

- `results/GAC/01_af2_prepare/tables/GAC_step01_global_plddt_summary.csv`
- `results/GAC/01_af2_prepare/tables/GAC_step01_structure_id_mapping.csv`
- `results/GAC/01_af2_prepare/lists/GAC_step01_passed_structure_ids.txt`
- `results/GAC/01_af2_prepare/pdb/standardized`

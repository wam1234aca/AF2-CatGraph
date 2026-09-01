# Step06 PLIP interaction graph summary: GAC

- Reproducibility mode: `interaction_profiler_v3_4`
- Input complexes: 218
- Successfully profiled: 218
- Failed: 0
- Interaction records: 2162
- Ligand atoms inventoried: 2180
- Ligand connectivity edges: 1962
- Global graph nodes: 21
- Ligand node ID mode: `residue_aware`
- Connectivity source: `input_conect_then_rdkit_same_residue`

## Main outputs

- `results/GAC/05_plip_interaction_graphs/tables/GAC_step05_graph_summary.csv`
- `results/GAC/05_plip_interaction_graphs/tables/GAC_step05_interaction_edges.csv`
- `results/GAC/05_plip_interaction_graphs/tables/GAC_step05_ligand_atom_inventory.csv`
- `results/GAC/05_plip_interaction_graphs/tables/GAC_step05_ligand_connectivity_edges.csv`
- `results/GAC/05_plip_interaction_graphs/tables/GAC_step05_global_node_mapping.csv`
- `results/GAC/05_plip_interaction_graphs/tables/GAC_step05_graph_index.csv`
- `results/GAC/05_plip_interaction_graphs/ged_input_files`

Each per-structure PLIP work directory also contains `interaction_output.txt`, `interaction_output.csv`, `substrate_connectivity.txt`, `substrate_connectivity_edges.csv`, and `ligand_atom_inventory.csv` for manual inspection.

Step06 does not modify or rewrite complex structures. It reads complex paths from the input index table and exports PLIP/graph-derived analysis files.

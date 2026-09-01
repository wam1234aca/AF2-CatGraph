# Step06 PLIP interaction graph summary: PTP1B

- Reproducibility mode: `interaction_profiler_v3_4`
- Input complexes: 306
- Successfully profiled: 306
- Failed: 0
- Interaction records: 3927
- Ligand atoms inventoried: 5202
- Ligand connectivity edges: 5202
- Global graph nodes: 31
- Ligand node ID mode: `residue_aware`
- Connectivity source: `input_conect_then_rdkit_same_residue`

## Main outputs

- `results/PTP1B/05_plip_interaction_graphs/tables/PTP1B_step05_graph_summary.csv`
- `results/PTP1B/05_plip_interaction_graphs/tables/PTP1B_step05_interaction_edges.csv`
- `results/PTP1B/05_plip_interaction_graphs/tables/PTP1B_step05_ligand_atom_inventory.csv`
- `results/PTP1B/05_plip_interaction_graphs/tables/PTP1B_step05_ligand_connectivity_edges.csv`
- `results/PTP1B/05_plip_interaction_graphs/tables/PTP1B_step05_global_node_mapping.csv`
- `results/PTP1B/05_plip_interaction_graphs/tables/PTP1B_step05_graph_index.csv`
- `results/PTP1B/05_plip_interaction_graphs/ged_input_files`

Each per-structure PLIP work directory also contains `interaction_output.txt`, `interaction_output.csv`, `substrate_connectivity.txt`, `substrate_connectivity_edges.csv`, and `ligand_atom_inventory.csv` for manual inspection.

Step06 does not modify or rewrite complex structures. It reads complex paths from the input index table and exports PLIP/graph-derived analysis files.

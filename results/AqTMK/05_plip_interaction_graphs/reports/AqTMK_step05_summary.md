# Step06 PLIP interaction graph summary: AqTMK

- Reproducibility mode: `interaction_profiler_v3_4`
- Input complexes: 176
- Successfully profiled: 176
- Failed: 0
- Interaction records: 2803
- Ligand atoms inventoried: 3696
- Ligand connectivity edges: 3168
- Global graph nodes: 34
- Ligand node ID mode: `residue_aware`
- Connectivity source: `input_conect_then_rdkit_same_residue`

## Main outputs

- `results/AqTMK/05_plip_interaction_graphs/tables/AqTMK_step05_graph_summary.csv`
- `results/AqTMK/05_plip_interaction_graphs/tables/AqTMK_step05_interaction_edges.csv`
- `results/AqTMK/05_plip_interaction_graphs/tables/AqTMK_step05_ligand_atom_inventory.csv`
- `results/AqTMK/05_plip_interaction_graphs/tables/AqTMK_step05_ligand_connectivity_edges.csv`
- `results/AqTMK/05_plip_interaction_graphs/tables/AqTMK_step05_global_node_mapping.csv`
- `results/AqTMK/05_plip_interaction_graphs/tables/AqTMK_step05_graph_index.csv`
- `results/AqTMK/05_plip_interaction_graphs/ged_input_files`

Each per-structure PLIP work directory also contains `interaction_output.txt`, `interaction_output.csv`, `substrate_connectivity.txt`, `substrate_connectivity_edges.csv`, and `ligand_atom_inventory.csv` for manual inspection.

Step06 does not modify or rewrite complex structures. It reads complex paths from the input index table and exports PLIP/graph-derived analysis files.

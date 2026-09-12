# Step command wrappers

These scripts allow individual workflow steps to be launched directly with
Python. For normal use, the equivalent `af2-catgraph step01` through
`af2-catgraph step09` commands are recommended.

`00_af2_trim_and_rename.py` is the separate input-preparation utility for
trimming specified AF2 residue ranges and assigning stable PDB names.

Example:

```bash
python scripts/08_chemical_equivalence.py --config projects/MY_TARGET/config.yaml --clean
python scripts/09_class_visualization.py --config projects/MY_TARGET/config.yaml --clean
```

`08_chemical_equivalence.py` runs ligand contact-unit grouping and candidate selection.

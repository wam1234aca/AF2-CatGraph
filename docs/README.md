# AF2-CatGraph user documentation

The root [README](../README.md) provides system requirements, installation,
input preparation, quick-start commands, outputs, and troubleshooting.

Detailed guides:

- [GUI guide](GUI_GUIDE.md)
- [Configuration layout and workflow steps](CONFIG_LAYOUT.md)
- [Configuration reference](CONFIG_REFERENCE.md)
- [AF2 structure preprocessing](AF2_PREPROCESS.md)
- [ConSurf archive and residue-mapping guide](CONSURF_WEBSITE_GZ.md)
- [Ligand contact-unit grouping](CONTACT_UNIT_REVIEW.md)

## First project

Create a project configuration:

```bash
af2-catgraph init \
  --target-id MY_ENZYME \
  --out projects/MY_ENZYME
```

Prepare the required inputs and edit `projects/MY_ENZYME/config.yaml`. At
minimum, define:

- the protein-only AlphaFold2/ColabFold ensemble;
- the protein portion of a reference complex;
- separate substrate, cofactor, and/or metal PDB files;
- ligand residue names;
- a ConSurf archive or normalized conservation table;
- the GED executable.

Validate the project before starting the complete analysis:

```bash
af2-catgraph preflight \
  --config projects/MY_ENZYME/config.yaml \
  --max-structures 5
```

Use a small representative subset to inspect ligand placement, clashes,
catalytic-region pLDDT, PLIP ligand recognition, ConSurf residue mapping, and
graph outputs. Then rerun Preflight without the structure limit and execute the
complete workflow:

```bash
af2-catgraph run \
  --config projects/MY_ENZYME/config.yaml \
  --stage all
```

Retain the input files, project YAML, accepted or manual ligand contact-unit
mappings, preprocessing manifests, and software-environment record with the
final results.

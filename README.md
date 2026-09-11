# AF2-CatGraph

AF2-CatGraph is a workflow for identifying catalytically relevant enzyme conformations from AlphaFold2-MSA perturbation ensembles. The workflow integrates reference-guided protein–ligand complex construction, global and local structural screening, protein–ligand interaction-graph generation, core-residue selection, graph edit distance (GED)-based classification, ligand contact-unit grouping, and candidate selection based on recurrent ligand-binding interactions and low-frequency reaction-center contacts. AF2-CatGraph provides an interpretable and auditable approach for identifying candidate catalytic conformations from predicted enzyme structural ensembles.

## Workflow

| Step | Purpose |
|---:|---|
| 00 | Preflight validation of inputs, paths, dependencies, and placement compatibility |
| 01 | Prepare predicted models and apply the global pLDDT threshold |
| 02 | Place substrate(s), cofactor(s), and/or metal(s) into each model |
| 03 | Remove complexes with excessive protein–ligand steric clashes |
| 04 | Assess pLDDT around the catalytic region |
| 05 | Run PLIP and generate interaction/GED graph files |
| 06 | Screen residues by conservation, distance, recurrence, and PLIP contacts |
| 07 | Classify interaction graphs with GED |
| 08 | Group ligand contact units and identify candidates |
| 09 | Export class tables, representative structures, and figures |
| 10 (optional) | Select or trim ligand atoms before docking or after clash filtering |


## System requirements

AF2-CatGraph is currently tested on Linux and requires an existing Conda installation. The installer creates a dedicated AF2-CatGraph environment and installs the workflow dependencies. Conda, ColabFold, and system build tools are not installed automatically.

The `make` and `g++` tools are required to compile the bundled graph edit distance (GED) source. Check that they are available before installation:

```bash
conda --version
make --version
g++ --version
```

AlphaFold2/ColabFold ensembles must be generated separately before running AF2-CatGraph. ColabFold should be maintained in a separate environment to avoid dependency conflicts. A GPU is required only for AlphaFold2/ColabFold prediction and is not required for the downstream AF2-CatGraph workflow.


## Installation

### Method A: recommended installation

Run the automated installer from the repository root:

```bash
cd /path/to/AF2-CatGraph
bash install_af2_catgraph.sh
```

The installer creates a new `af2_catgraph` Conda environment using the dependency versions specified in `requirements-conda-tested.txt`, installs AF2-CatGraph and its workflow dependencies, compiles the bundled [Graph_Edit_Distance](https://github.com/LijunChang/Graph_Edit_Distance) source, and verifies the installation. Conda downloads use a repository-local package cache, extended timeouts, and automatic retries to reduce failures on unstable connections. GED installation does not require access to GitHub. ColabFold is not installed and should be maintained in a separate environment.

Optional installer arguments are available for changing the environment name or GED installation:

```bash
# Use a different environment name
bash install_af2_catgraph.sh --env af2_catgraph_test

# Compile GED from another compatible source directory
bash install_af2_catgraph.sh --ged-dir /absolute/path/Graph_Edit_Distance

# Skip GED compilation when an existing installation will be used
bash install_af2_catgraph.sh --skip-ged
```

The installer does not overwrite an existing Conda environment. The bundled GED source is retained under `external/Graph_Edit_Distance`; its MIT license and upstream README are included with the source.

After installation, activate the environment and verify the command-line interface:

```bash
conda activate af2_catgraph
python -m catcongraph.cli --help
```

By default, the compiled GED executable is located at:

```text
/path/to/AF2-CatGraph/external/Graph_Edit_Distance/ged
```

Use this absolute path when configuring the GED executable in the GUI or YAML configuration.


### Method B: compatible-version installation (advanced)

Use this method if the pinned environment in Method A cannot be resolved on your system, or if you need newer dependency versions within the declared compatibility ranges.

```bash
cd /path/to/AF2-CatGraph

mkdir -p .conda_package_cache

CONDA_PKGS_DIRS="$PWD/.conda_package_cache" \
CONDA_REMOTE_CONNECT_TIMEOUT_SECS=60 \
CONDA_REMOTE_READ_TIMEOUT_SECS=300 \
CONDA_REMOTE_MAX_RETRIES=10 \
conda create --yes --name af2_catgraph \
  --override-channels --channel conda-forge \
  --file requirements-conda-compatible.txt

conda activate af2_catgraph
python -m pip install "streamlit>=1.32,<2.0"
python -m pip install --no-deps --editable .
```

The repository-local package cache prevents incomplete downloads from affecting the user's global Conda cache. The timeout and retry variables apply only to this command, as does `--override-channels`; none of them modifies the user's global Conda configuration. The compatible specification includes a recent Conda `libstdc++` runtime for compiled dependencies on older Linux systems.

Compile the GED program in the default repository location:

```bash
cd /path/to/AF2-CatGraph
make -C external/Graph_Edit_Distance clean
make -C external/Graph_Edit_Distance
```

Verify the installation:

```bash
python scripts/verify_install.py \
  --ged /absolute/path/to/AF2-CatGraph/external/Graph_Edit_Distance/ged

af2-catgraph --help
```

This method installs the same AF2-CatGraph source as Method A but allows Conda to select dependency versions within the ranges specified in `requirements-conda-compatible.txt`. Not every possible version combination has been individually validated, and changes to PLIP or OpenBabel may affect the detected protein–ligand interactions. 



### ColabFold

ColabFold is not required when protein-only predicted PDB files already exist.
To generate new models, use a separate local ColabFold environment or the
[ColabFold AlphaFold2 notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb),
then provide the resulting PDB files to Step01.

## Input files

A recommended project layout is:

```text
AF2-CatGraph/
├── data/
│   └── MY_ENZYME/
│       ├── 01_af2/
│       │   └── predicted_model_*.pdb
│       ├── 02_templates/
│       │   ├── template_protein.pdb
│       │   ├── substrate.pdb
│       │   └── cofactor_or_metal.pdb
│       └── 07_conservation/
│           └── MY_ENZYME_conservation.csv
├── projects/
│   └── MY_ENZYME/
│       └── config.yaml
└── results/
    └── MY_ENZYME/
```

### Predicted protein ensemble

Provide protein-only AlphaFold2 or ColabFold PDB models in the directory specified by `inputs.af2_dir`, normally `data/MY_ENZYME/01_af2/`. Place the PDB files directly in this directory.

AF2-CatGraph reads residue-level pLDDT values from 
Specified low-confidence or disordered terminal regions can optionally be removed before the main workflow. This preprocessing creates new PDB files and does not modify the original models. Instructions are provided in [docs/AF2_PREPROCESS.md](docs/AF2_PREPROCESS.md).

### Reference complex and ligands

Provide the protein portion of a reference complex and a separate PDB file for each substrate, cofactor, or metal. All ligand files must be in the same coordinate frame as the reference protein because Step02 performs reference-guided rigid placement rather than de novo molecular docking.

The reference binding mode may be obtained from an experimental substrate-, cofactor-, inhibitor-, or homolog-bound structure, or from a separately prepared complex model such as an AlphaFold3 prediction or molecular-docking model.

For multiple components, list the ligand files in their Step02 placement order:

```yaml
inputs:
  template_protein_pdb: data/MY_ENZYME/02_templates/template_protein.pdb
  ligand_pdbs:
    - data/MY_ENZYME/02_templates/substrate.pdb
    - data/MY_ENZYME/02_templates/cofactor.pdb
    - data/MY_ENZYME/02_templates/metal.pdb
```

Keep ligand residue and atom names consistent across the PDB files and the corresponding catalytic-region, ligand-selection, PLIP, and contact-unit settings.

### Residue conservation

Step06 requires a residue-conservation input. AF2-CatGraph accepts ConSurf result archives (`.zip`, `.tar.gz`, `.tgz`, or `.gz`), uncompressed `*_consurf_grades.txt` files, and normalized CSV, TSV, or TAB tables.

A minimal normalized table is:

```csv
chain,resseq,resname,conservation_grade
A,46,TYR,9
A,215,CYS,9
A,216,SER,8
```

`resseq` and `conservation_grade` are required. Include `chain` for multichain structures; otherwise, the configured default chain (`A` by default) is used. `resname` and insertion-code fields are optional.

ConSurf results can be obtained from the [ConSurf server](https://consurf.tau.ac.il/) or the [ConSurf Colab notebook](https://colab.research.google.com/github/BenTalLab/consurf/blob/master/consurf.ipynb).

Additional archive and residue-mapping guidance is provided in [docs/CONSURF_WEBSITE_GZ.md](docs/CONSURF_WEBSITE_GZ.md). Use Preflight to confirm that the conservation residue numbering and chain identifiers map correctly to the predicted structures.


## Quick start

### GUI

Activate the environment and launch the graphical interface:

```bash
conda activate af2_catgraph
cd /path/to/AF2-CatGraph
af2-catgraph gui --fresh
```

To load an existing project configuration when the GUI starts:

```bash
af2-catgraph gui --fresh \
  --config /absolute/path/to/config.yaml
```

The `--fresh` option replaces the recorded AF2-CatGraph GUI process on the same port and starts a clean interface state. A scientific calculation begins only when it is started from the interface.

### Command line

Create a project configuration template:

```bash
af2-catgraph init \
  --target-id MY_ENZYME \
  --out projects/MY_ENZYME
```

This creates `projects/MY_ENZYME/config.yaml`. Prepare the required input files and edit the generated configuration before continuing.

Validate the input files, external programs, residue mapping, and template-to-model alignment:

```bash
af2-catgraph preflight \
  --config projects/MY_ENZYME/config.yaml
```

For a quick initial check, add `--max-structures 5`. Run Preflight without this limit before the complete analysis.

Run the full Step01–Step09 workflow:

```bash
af2-catgraph run \
  --config projects/MY_ENZYME/config.yaml \
  --stage all
```

## Advanced usage

### Run selected stages

The `run` command can execute the complete workflow or a predefined subset:

| Stage | Steps | Purpose |
|---|---:|---|
| `prepare` | 01–04 | Structure preparation and quality control |
| `interactions` | 05 | PLIP interaction-graph generation |
| `classify` | 06–08 | Core-residue screening, GED classification, and candidate selection |
| `visualize` | 09 | Class summaries, structures, and figures |
| `analysis` | 05–09 | Interaction analysis through visualization |
| `all` | 01–09 | Complete main workflow |

For example, to run the analysis stages after preparation is complete:

```bash
af2-catgraph run \
  --config projects/MY_ENZYME/config.yaml \
  --stage analysis
```

### Resume or rerun part of a workflow

Use `--from-step` and `--to-step` when the required upstream outputs already exist:

```bash
af2-catgraph run \
  --config projects/MY_ENZYME/config.yaml \
  --stage all \
  --from-step 05 \
  --to-step 09
```

Add `--clean` to replace existing outputs for the selected steps that support clean reruns.

Each main step can also be run separately. For example:

```bash
af2-catgraph step06 \
  --config /home/wms/AF2-test/AF2-CatGraph/projects/GAC/config.gui.yaml \
  --clean
```

Individual commands are available as `step01` through `step09`. Use
`af2-catgraph step05 --help` or the corresponding step number to inspect its
options.

### Optional ligand-atom selection

Optional Step10 selects or removes specified ligand atoms either before Step02
(`before_docking`) or after Step03 (`after_clash`). Configure and enable
`step10_optional` in the YAML before use. When enabled, the main workflow runs
it automatically immediately before the downstream step that consumes its
output.

To run the optional operation separately for inspection or regeneration:

```bash
af2-catgraph step10 \
  --config projects/MY_ENZYME/config.yaml \
  --timing before_docking
```

The alternative timing is `after_clash`. Normally, Step10 does not need to be
run manually before the main workflow. Its atom-selection rules change the
downstream analysis and should be retained with the results.


## Ligand contact-unit grouping

After atom-level GED classification, Step08 can group selected ligand atoms or
atom groups into common contact units. Interaction labels are then grouped as
Polar, Hydrophobic, π, or Metal contacts. Classes with identical interaction
patterns after grouping are merged before interaction-frequency analysis and
candidate selection.

The GUI presents two analysis routes. New projects default to
`universal_rules`; `guided_review` remains available in YAML for older projects:

- `universal_rules` automatically applies the twelve Supplementary Table S4
  rules to every ligand component, writes the complete mapping and evidence
  audit, consolidates matching post-GED signatures, and then performs the
  normal interaction-frequency and candidate screening. GAC, AqTMK, PTP1B,
  and all additional enzymes pass through this same structure-derived engine;
  the three manuscript cases are retained only as regression benchmarks.
- `conservative_manual` performs no automatic ligand-atom merging. With an
  empty `manual_ligand_units` mapping, the original GED classes proceed directly
  to atom-level frequency analysis and catalytic-conformation screening.

All policies use the same interaction-label grouping, frequency
thresholds, and candidate-screening logic. They differ only in the source of
the ligand endpoint labels.

Recommended configuration for the additional enzyme systems:

```yaml
step08:
  canonicalization:
    contact_unit_mode: universal_rules
    equivalence_mode: standard
    use_builtin_ligand_units: false
    auto_detect_ligand_units: true
    recommendations:
      enabled: true
      require_acceptance: false
      consensus_min_fraction: 0.80
    manual_ligand_units: {}
```

Manual example:

```yaml
step08:
  canonicalization:
    contact_unit_mode: conservative_manual
    manual_ligand_units:
      ATP_terminal_phosphate_O:
        - ATP:O9
        - ATP:O10
        - ATP:O11
```

Atom names must exactly match the ligand PDB. Do not group atoms solely because
their names or spatial positions are similar. Contact-unit mapping changes graph
labels for comparison; it does not modify the coordinates. See
[docs/CONTACT_UNIT_REVIEW.md](docs/CONTACT_UNIT_REVIEW.md) and
[docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md).

## Main outputs

| Output | Default directory |
|---|---|
| Preflight report | `results/MY_ENZYME/00_preflight/` |
| Prepared structures | `results/MY_ENZYME/01_af2_prepare/` |
| Placed complexes | `results/MY_ENZYME/02_template_guided_docking/` |
| Clash-filter results | `results/MY_ENZYME/03_clash_filter/` |
| Catalytic-region pLDDT | `results/MY_ENZYME/04_catalytic_region_plddt_qc/` |
| PLIP tables and graphs | `results/MY_ENZYME/05_plip_interaction_graphs/` |
| Key-residue screening | `results/MY_ENZYME/06_key_residue_screening/` |
| GED classes | `results/MY_ENZYME/07_ged_classification/` |
| Ligand contact-unit grouping and candidates | `results/MY_ENZYME/08_chemical_equivalence/` |
| Class summaries and figures | `results/MY_ENZYME/09_class_visualization/` |
| Optional ligand-atom selection | `results/MY_ENZYME/10_optional_ligand_atom_trimming/` |

The results page can display tables, figures, representative PDB files, and an
offline 3Dmol.js view with cartoon, stick, and interaction-link rendering.
PyMOL is not required for the browser viewer; downloadable `.pml` files are
provided for users who want to continue in PyMOL.

## Troubleshooting

### PLIP finds no ligand or interactions

Confirm that the ligand is written as `HETATM`, its residue name matches the
YAML, and the Step04 complex contains both protein and ligand atoms. Test
`plip -h` inside the AF2-CatGraph environment. Ligand connectivity,
protonation, hydrogen handling, and PLIP/OpenBabel versions can change detected
contacts.


### ConSurf residues do not map to the structures

Check the chain, residue number, insertion code, and construct boundaries
against the analysed PDB files. Run Preflight and inspect the conservation
mapping audit before Step06.

## Reproducibility

- Save the exact YAML used for each analysis.
- Record all changes to thresholds, alignment, ligand definitions, PLIP settings, GED settings, contact-unit mappings, and optional trimming rules.
- Save a final environment record with:

```bash
conda env export --no-builds > af2-catgraph-environment.yml
python -m pip freeze > af2-catgraph-pip-freeze.txt
```

## Repository structure

```text
catcongraph/       Python package, workflow steps, and GUI
scripts/           installation checks and command wrappers
examples/          minimal configuration template
docs/              configuration and user guides
tests/             regression tests
external/          bundled GED source and locally compiled executable
data/              local inputs (ignored by Git)
projects/          local project YAML files (ignored by Git)
results/           generated outputs (ignored by Git)
```

## License

See [LICENSE](LICEN
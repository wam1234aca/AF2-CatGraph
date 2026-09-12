# AF2-CatGraph GUI guide

Launch the interface from the repository root:

```bash
conda activate af2_catgraph
af2-catgraph gui --fresh
```

Use `--no-browser` on a server without a graphical desktop.

## 1. Overview

The Overview page summarizes required inputs, available Step01–Step09 output
directories, the current run mode, and the suggested next action. These checks
are interface indicators only; use Preflight for formal input and dependency
validation.

## 2. Project setup

Provide:

- project ID and result directory;
- the directory containing protein-only AlphaFold2/ColabFold PDB files;
- the protein PDB from the reference complex;
- separate substrate, cofactor, and/or metal PDB files;
- the ConSurf archive or normalized conservation table;
- the GED executable;
- ligand residue names used in the analysis.

Enter one ligand file per line. Their order is the sequential placement order
used in Step02. Select **Apply project settings**, then save the project YAML. The
GUI does not move input files.

When loading an existing analysis, retain the original YAML used to generate
reported results.

## 3. Preflight

Run Preflight before the first scientific calculation. It checks input paths,
PDB readability, dependencies, external programs, configuration consistency,
and template-to-model placement compatibility.

A successful Preflight confirms technical readiness. Users should still verify
the reference complex, ligand identities, chain identifiers, residue numbering,
and system-specific thresholds.

## 4. Pilot run

Start with three to five structures and inspect:

1. ligand placement in the target structures;
2. protein–ligand VDW clash results;
3. the residues included in catalytic-region pLDDT quality control;
4. ligand recognition and contacts reported by PLIP;
5. ConSurf residue mapping and coverage;
6. GED classes and ligand contact-unit audit outputs.

Continue with the complete ensemble after these checks are satisfactory.

## 5. Step parameters and ligand contact-unit grouping

The compact form shows the scientific settings most often adjusted between
enzyme systems. Step02 offers automatic or sequence-aware Cα alignment, uses
the structures passed by Step01, and retains both multiple-component placement
modes. Step05–06 shows the conservation, distance, and occurrence thresholds.
Step09 shows only SVG/PNG output selection. Optional Step10 ligand-atom
selection remains visible and disabled by default.

Step08 presents two routes. Select `universal_rules` to apply the twelve S4
ligand contact-unit grouping rules, or select `conservative_manual` to retain
atom-level GED classes.

See [CONTACT_UNIT_REVIEW.md](CONTACT_UNIT_REVIEW.md) for the review procedure.

## 6. Run and monitor

The page displays the current job status. Expand **Detailed command and run
log** to inspect the command and recent log output. This panel opens
automatically after a failed task.

The **Clean prior output for this step** option replaces existing output for
steps that support clean reruns. Preserve required results or use a new output
directory before enabling it.

## 7. Results browser

The Results browser summarizes final classes, structures per class, and
completed steps. Depending on available outputs, class cards can provide:

- representative PDB structures;
- an embedded 3Dmol.js pocket view;
- PLIP contact residues and interaction links;
- downloadable PyMOL `.pml` scripts;
- class tables, reports, and figures.

By default, the browser indexes final tables, reports, figures, and candidate
structures. Enable **Include large intermediate files** only when those
files are needed.

Interactive HTML output may contain scripts and is not executed by default.
Enable interactive preview only for files produced by a trusted
AF2-CatGraph run.

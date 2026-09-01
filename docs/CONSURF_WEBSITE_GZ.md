# Using ConSurf result archives in Step06

AF2-CatGraph Step06 can read result archives downloaded from the ConSurf
website, including files commonly named `new1.gz`. Manual extraction or CSV
conversion is not required.

## Accepted inputs

- ConSurf `.zip` result archives;
- ConSurf `.tar.gz`, `.tgz`, or `.gz` result archives;
- a gzip-compressed `*_consurf_grades.txt` file;
- an uncompressed `*_consurf_grades.txt` file;
- normalized `.csv`, `.tsv`, or `.tab` conservation tables.

Archive type is detected from file content. When a website result archive is
used, AF2-CatGraph selects the contained `*_consurf_grades.txt` table and
ignores visualization scripts, PyMOL sessions, and alignment files.

## YAML configuration

```yaml
step06:
  conservation_file: "data/NEW1/07_conservation/new1.gz"
  conservation:
    # Used only when the input does not provide PDB residue mapping.
    default_chain: "A"
    # Set only when an archive contains multiple grade tables.
    # preferred_grade_filename: "1PZT_1_consurf_grades.txt"
```

Validate the file before running Step06:

```bash
af2-catgraph preflight --config projects/NEW1/config.yaml
```

Preflight reports the detected source type, selected grade table, number of
parsed residues, and available PDB residue mapping. After validation, run the
complete workflow or rerun Step06:

```bash
af2-catgraph run --config projects/NEW1/config.yaml --stage all

# When Step01–Step05 outputs already exist:
af2-catgraph step06 --config projects/NEW1/config.yaml --clean
```

## GUI inspection

Enter the archive path in the **Conservation file** field under **Project
setup**. Expand **Check ConSurf conservation input** and select **Parse and
check current file**. The GUI displays:

- the archive type and selected `*_consurf_grades.txt` file;
- the number of parsed residues;
- a preview of PDB residue mapping, internal residue keys, and grades.

Save the YAML after confirming the mapping. The GUI and CLI use the same
ConSurf parser.

## Residue-numbering checks

ConSurf grade tables may include an `ATOM` field such as `CYS:134:A`. This
refers to the residue number and chain in the structure submitted to ConSurf,
not to a simple sequence position. AF2-CatGraph converts this entry to the
residue key used for matching the complex and PLIP records.

Rows with `-` in the structural mapping are alignment positions that are not
present in the submitted PDB and are excluded. They are not assigned sequential
structure numbers.

After Step06, inspect
`06_key_residue_screening/reports/*_step07_report.json`, especially
`conservation_mapping_audit`. The value
`n_exact_residue_key_matches` should be greater than zero. A value of zero
indicates a likely mismatch in residue numbering, chain identifiers, or
construct boundaries.

The optional AF2 preprocessing utility preserves the original numbering of
retained residues. If ConSurf was run on a different construct, chain, or
renumbered model, rerun ConSurf using a structure consistent with the complex
used for AF2-CatGraph rather than relying on inferred renumbering.

For batch projects, assign the appropriate ConSurf file to each project YAML
and run Preflight for every enabled project before starting the complete batch.

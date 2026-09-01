# AF2 structure preprocessing

Use `scripts/00_af2_trim_and_rename.py` to remove specified residue ranges
from a directory of AlphaFold2 or ColabFold PDB files before the main
AF2-CatGraph workflow. The utility writes new files, preserves the original
residue numbering of retained residues, and does not modify the source PDB
files.

## Command-line example

```bash
python scripts/00_af2_trim_and_rename.py \
  --input-dir data/MY_ENZYME/01_af2 \
  --output-dir data/MY_ENZYME/01_af2_trimmed \
  --remove 1-120,290-300 \
  --name MY_ENZYME
```

Example outputs:

```text
data/MY_ENZYME/01_af2_trimmed/MY_ENZYME.AF2rank1.pdb
data/MY_ENZYME/01_af2_trimmed/MY_ENZYME.AF2rank2.pdb
data/MY_ENZYME/01_af2_trimmed/af2_preprocess_manifest.csv
```

The script searches subdirectories by default. It prioritizes recognizable
`rank_001` or `ranked_0` names and otherwise uses natural filename order.
Output ranks are consecutive and start at `AF2rank1` unless another starting
rank is requested.

## GUI procedure

Open **Step 0A · AF2/ColabFold preprocessing** in the GUI:

1. Select the **Source AF2 PDB folder**.
2. Choose a different **New preprocessed output folder**. Do not place the
   output folder inside the input folder.
3. Enter an output name prefix and the residue intervals to remove. To restrict
   removal to selected chains, enter their chain identifiers; leave the field
   blank to process every chain.
4. Select **Start preprocessing**.
5. Review `af2_preprocess_manifest.csv`, then select **Use preprocessed folder
   as current AF2 input** and save the project YAML.

The GUI and command-line utility perform the same preprocessing operation.

## Options

- `--remove 1-120,290-300`: remove inclusive ranges. Single residues are also
  accepted, for example `1-120,290-300,415`.
- `--chains A,B`: apply removal only to the listed chains. Omit this option to
  process all chains.
- `--no-recursive`: read PDB files only from the top level of the input
  directory.
- `--rank-start 1`: set the first output rank.
- `--report /path/report.csv`: write the audit manifest to a specified path.

`CONECT` records involving removed atoms are removed from the processed PDB.
The manifest records the source and output files, requested residue ranges,
removed residues, and retained atom-record counts. Retain this file with the
analysis inputs.

After reviewing the processed structures, set `inputs.af2_dir` in the project
YAML to the new output directory and begin the main workflow with Step01.

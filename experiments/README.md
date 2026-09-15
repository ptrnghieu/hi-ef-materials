# Source-folder-held-out experiments

This directory contains the frozen split protocol for the Hi-EF experiments.
It intentionally does **not** use the earlier component-safe split.

## Protocol

The prefix in each clip id is treated as an anonymized source-folder id. For
example, all MCIS samples containing clips under `01/...` stay in one split.
The released 53 source folders are allocated as 37 train, 8 validation, and 8
test folders. The assignment is optimized for MCIS count and clip-III/clip-IV
emotion balance.

The source-folder semantics are not stated in the public metadata. Evidence in
the preprocessing code suggests that the folders are episode-level sources,
but this remains a proxy until the dataset authors confirm it.

## Rebuild the frozen manifest

Run from the repository root:

```bash
python experiments/build_source_folder_split.py
```

Generated files:

- `experiments/manifests/source_folder_split_seed42.csv`
- `experiments/manifests/source_folder_split_seed42_audit.json`

The audit must report zero shared source folders, clips, and adjacent clip
pairs for every pair of splits.

## Run tests

```bash
PYTHONPATH=experiments python -m unittest experiments/test_source_folder_split.py
```

## Use on Kaggle

Open `experiments/kaggle_source_folder_split.ipynb` on Kaggle. Its bootstrap
cell clones the `experiments` branch, rebuilds the manifest deterministically,
and checks all leakage invariants. Kaggle Internet must be enabled for the
clone cell. Alternatively, upload the repository and change `REPO_DIR`.

Load the frozen assignment instead of independently recomputing a split inside
a training notebook:

```python
import pandas as pd

manifest = pd.read_csv(
    "/kaggle/working/hi-ef-materials/experiments/manifests/"
    "source_folder_split_seed42.csv"
)

train_rows = manifest[manifest["split"] == "train"]
val_rows = manifest[manifest["split"] == "val"]
test_rows = manifest[manifest["split"] == "test"]
```

Do not select seeds, checkpoints, or model variants using the test partition.
Select checkpoints using validation UAR and evaluate the frozen test partition
only after the experiment configuration is fixed.
